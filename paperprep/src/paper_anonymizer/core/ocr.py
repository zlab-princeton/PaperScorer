"""MinerU OCR — content list loading and MinerU CLI runner.

Loading functions work without GPU. The run_mineru* functions require
the `gpu` optional dependency (`pip install paper-anonymizer[gpu]`)
and a CUDA GPU.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Content list loading (no GPU required)
# ---------------------------------------------------------------------------

def load_content_list(path: Path) -> list[dict]:
    """Load a MinerU content_list.json file.

    Args:
        path: Path to the content_list.json file.

    Returns:
        List of content item dicts.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_content_list(mineru_output_dir: Path) -> Path | None:
    """Find the content_list.json in a MinerU output directory.

    MinerU outputs to: {output_dir}/{folder_name}/vlm/{folder_name}_content_list.json

    Args:
        mineru_output_dir: The MinerU output directory.

    Returns:
        Path to content_list.json, or None if not found.
    """
    for cl in mineru_output_dir.rglob("*_content_list.json"):
        return cl
    return None


def find_middle_json(mineru_output_dir: Path) -> Path | None:
    """Find the middle.json in a MinerU output directory.

    Args:
        mineru_output_dir: The MinerU output directory.

    Returns:
        Path to middle.json, or None if not found.
    """
    for mj in mineru_output_dir.rglob("*_middle.json"):
        return mj
    return None


# ---------------------------------------------------------------------------
# Batch preparation (no GPU required)
# ---------------------------------------------------------------------------

def prepare_batch(pdf_paths: list[Path], batch_dir: Path) -> Path:
    """Create a batch directory with symlinks to PDFs for MinerU processing.

    Args:
        pdf_paths: List of PDF file paths.
        batch_dir: Directory to create with symlinks.

    Returns:
        The batch_dir path.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdf_paths:
        link = batch_dir / pdf.name
        if not link.exists():
            link.symlink_to(pdf.resolve())
    return batch_dir


# ---------------------------------------------------------------------------
# MinerU model management (run on login node with internet)
# ---------------------------------------------------------------------------

MINERU_PIPELINE_REPO = "opendatalab/PDF-Extract-Kit-1.0"
MINERU_VLM_REPO = "opendatalab/MinerU2.5-2509-1.2B"


def _get_hf_cache_dir() -> str:
    """Get the HuggingFace cache directory. Respects HF_HOME."""
    import os
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def check_mineru_cli() -> tuple[bool, str]:
    """Check if `mineru` CLI is available.

    Returns:
        (available, version_or_error)
    """
    try:
        result = subprocess.run(
            ["mineru", "--version"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version
        return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "mineru CLI not found (install with: pip install paper-anonymizer[gpu])"
    except Exception as e:
        return False, str(e)


def verify_mineru_models() -> dict:
    """Check if MinerU models are cached locally. No internet required.

    Returns:
        Dict with 'ready' (bool), 'pipeline_cached', 'vlm_cached',
        'cache_dir', 'missing' (list of repo IDs).
    """
    cache_dir = _get_hf_cache_dir()
    cache_path = Path(cache_dir)

    result = {
        'ready': False,
        'cache_dir': cache_dir,
        'pipeline_cached': False,
        'vlm_cached': False,
        'missing': [],
    }

    # Check for model dirs in HF cache
    # HF cache structure: hub/models--{org}--{repo}/snapshots/...
    pipeline_dir = cache_path / f"models--{MINERU_PIPELINE_REPO.replace('/', '--')}"
    vlm_dir = cache_path / f"models--{MINERU_VLM_REPO.replace('/', '--')}"

    result['pipeline_cached'] = pipeline_dir.exists() and any(pipeline_dir.rglob("*.bin")) or any(pipeline_dir.rglob("*.safetensors")) or any(pipeline_dir.rglob("*.onnx"))
    result['vlm_cached'] = vlm_dir.exists() and any(vlm_dir.rglob("*.safetensors")) or any(vlm_dir.rglob("*.bin"))

    if not result['pipeline_cached']:
        result['missing'].append(MINERU_PIPELINE_REPO)
    if not result['vlm_cached']:
        result['missing'].append(MINERU_VLM_REPO)

    result['ready'] = result['pipeline_cached'] and result['vlm_cached']
    return result


def download_mineru_models() -> dict:
    """Download MinerU models to HF cache. Respects HF_HOME env var.

    Run this on a login node with internet access before dispatching
    batch jobs to compute nodes.

    Returns:
        Dict with download status for each model.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {'success': False, 'error': 'huggingface_hub not installed (pip install huggingface_hub)'}

    result = {'success': True, 'models': {}}

    for repo in [MINERU_PIPELINE_REPO, MINERU_VLM_REPO]:
        print(f"Downloading {repo}...")
        try:
            path = snapshot_download(repo)
            result['models'][repo] = {'cached': True, 'path': path}
            print(f"  OK: {path}")
        except Exception as e:
            result['models'][repo] = {'cached': False, 'error': str(e)}
            result['success'] = False
            print(f"  FAILED: {e}")

    return result


# ---------------------------------------------------------------------------
# MinerU runner (requires GPU + mineru[core,vllm])
# ---------------------------------------------------------------------------

def run_mineru(
    input_dir: Path,
    output_dir: Path,
    backend: str = "vlm-vllm-engine",
    max_pages: int = 14,
    server_url: str | None = None,
) -> bool:
    """Run MinerU OCR on a directory of PDFs.

    Wraps the `mineru` CLI. Requires GPU and `mineru[core,vllm]` installed.
    Sets TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 to prevent model
    downloads on compute nodes — run `paper-anonymize setup` first.

    When ``server_url`` is supplied, the backend is forced to
    ``vlm-http-client`` and the call hits a long-running mineru-vllm-server
    instead of loading vLLM in-process — used by ``paperprep serve``.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Where to write MinerU output.
        backend: MinerU backend (default: vlm-vllm-engine).
        max_pages: Maximum pages to process per PDF.
        server_url: Optional URL of a running mineru-vllm-server. When set,
            overrides backend to ``vlm-http-client``.

    Returns:
        True on success, False on failure.
    """
    import os

    output_dir.mkdir(parents=True, exist_ok=True)

    if server_url:
        cmd = [
            "mineru",
            "-p", str(input_dir),
            "-o", str(output_dir),
            "-b", "vlm-http-client",
            "-u", server_url,
            "--end", str(max_pages),
        ]
    else:
        cmd = [
            "mineru",
            "-p", str(input_dir),
            "-o", str(output_dir),
            "-b", backend,
            "--end", str(max_pages),
        ]

    # Force offline mode — compute nodes should not download models
    env = {**os.environ, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}

    print(f"Running MinerU: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def run_mineru_single(
    pdf_path: Path,
    output_dir: Path,
    backend: str = "vlm-vllm-engine",
    max_pages: int = 14,
    server_url: str | None = None,
) -> dict:
    """Run MinerU on a single PDF.

    Creates a temporary batch directory with a symlink to the PDF,
    runs MinerU, then returns paths to the output files.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Where to write MinerU output.
        backend: MinerU backend.
        max_pages: Maximum pages to process.
        server_url: Optional URL of a running mineru-vllm-server. When set,
            switches to ``vlm-http-client`` and skips the local model-cache
            preflight (the server holds the weights).

    Returns:
        Dict with 'success', 'content_list_path', 'middle_json_path', 'error'.
    """
    result = {
        'success': False,
        'content_list_path': None,
        'middle_json_path': None,
        'error': None,
    }

    try:
        # Preflight: only required for the in-process backend. With a
        # remote server, the client subprocess doesn't load weights itself.
        if not server_url:
            model_status = verify_mineru_models()
            if not model_status['ready']:
                result['error'] = (
                    f"MinerU models not cached at {model_status['cache_dir']}. "
                    f"Missing: {model_status['missing']}. "
                    "Run `paper-anonymize setup` on a node with internet first."
                )
                return result

        # Create temp batch dir with symlink
        batch_dir = output_dir / "_batch"
        prepare_batch([pdf_path], batch_dir)

        # Run MinerU (offline mode enforced)
        success = run_mineru(batch_dir, output_dir, backend, max_pages, server_url=server_url)
        if not success:
            result['error'] = "MinerU returned non-zero exit code"
            return result

        # Clean up batch dir
        shutil.rmtree(batch_dir, ignore_errors=True)

        # Find output files
        cl_path = find_content_list(output_dir)
        mj_path = find_middle_json(output_dir)

        if cl_path is None:
            result['error'] = "MinerU completed but content_list.json not found"
            return result

        result['success'] = True
        result['content_list_path'] = cl_path
        result['middle_json_path'] = mj_path

    except Exception as e:
        result['error'] = str(e)

    return result
