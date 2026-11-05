#!/usr/bin/env python3
"""
Standalone Auto-Inference Checkpoint Watcher

A background script that monitors a training output directory for new checkpoints
and automatically submits sbatch inference jobs for intermediate checkpoints.

This script is designed to be launched from a training sbatch script and run in
the background. It will automatically terminate when training completes (detected
via the presence of all_results.json).

Usage:
    # Start in background from training sbatch
    python scripts/auto_infer_watcher.py \
        --save_dir saves/final_sweep_v7_pli/balanced_vision \
        --results_dir results/final_sweep_v7_pli/balanced_vision \
        --dataset iclr_2020_2025_split7_balanced_vision_binary_noreviews_v7 \
        --template qwen2_vl \
        --cutoff_len 24480 \
        --max_new_tokens 1280 \
        --image_min_pixels 784 \
        --image_max_pixels 1003520 \
        --poll_interval 60 &

Features:
    - Self-terminating: Exits when all_results.json appears (training done)
    - Idempotent: Touch files (.infer.touch) prevent duplicate submissions
    - Fault-tolerant: Errors logged but don't crash the watcher
    - Skip final checkpoint: Final checkpoint handled by training sbatch PHASE 2
    - Dry-run mode: Test without actually submitting jobs
"""

import argparse
import json as json_module
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None) -> None:
    """Configure logging to both console and a file."""
    log_format = "[%(asctime)s] [AutoInfer] %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger setup - clear any existing handlers first
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        root_logger.addHandler(file_handler)

# Base directory for sbatch scripts (relative to repo root)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SBATCH_DIR = REPO_ROOT / "sbatch" / "inference"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor training directory for checkpoints and submit inference jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Training output directory to monitor for checkpoints",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (without _test suffix, will be added automatically)",
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Template name (qwen or qwen2_vl)",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=24480,
        help="Cutoff length for inference (default: 24480)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1280,
        help="Max new tokens for generation (default: 1280)",
    )
    parser.add_argument(
        "--image_min_pixels",
        type=int,
        default=None,
        help="Min image pixels for vision models (default: None)",
    )
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=None,
        help="Max image pixels for vision models (default: None)",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=60,
        help="Seconds between polling for new checkpoints (default: 60)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually submitting jobs",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path. If not specified, defaults to {save_dir}/auto_infer_watcher.log",
    )
    parser.add_argument(
        "--compute_train_accuracy",
        action="store_true",
        help="Compute training accuracy metrics on each checkpoint (default: False)",
    )
    parser.add_argument(
        "--train_accuracy_max_samples",
        type=int,
        default=2000,
        help="Max samples for training accuracy computation (default: 2000)",
    )
    parser.add_argument(
        "--train_accuracy_batch_size",
        type=int,
        default=2,
        help="Batch size for training accuracy computation (default: 2)",
    )
    parser.add_argument(
        "--delete_safetensors_posteval",
        action="store_true",
        help="Delete safetensors from checkpoint dir after inference completes (saves disk space)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature for inference (default: 0)",
    )
    parser.add_argument(
        "--save_logprobs",
        action="store_true",
        help="Save token logprobs in inference output (default: False)",
    )
    parser.add_argument(
        "--keep_safetensors_epochs",
        type=str,
        default="",
        help="DEPRECATED: Comma-separated checkpoint steps to keep safetensors for (skip deletion)",
    )
    parser.add_argument(
        "--keep_safetensors_epoch_idx",
        type=str,
        default="",
        help="Comma-separated epoch numbers (1-indexed) to keep safetensors for (skip deletion). "
             "Checkpoints are sorted by step; the i-th checkpoint = epoch i.",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="Override train dataset name for --compute_train_accuracy "
             "(default: derived from --dataset by replacing _test with _train)",
    )

    return parser.parse_args()


def is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint has all required model files (full FT or LoRA)."""
    # LoRA checkpoint: adapter_model*.safetensors + adapter_config.json
    lora_files = list(checkpoint_dir.glob("adapter_model*.safetensors"))
    lora_config = checkpoint_dir / "adapter_config.json"
    if lora_files and lora_config.exists():
        return True

    # Full FT checkpoint: model*.safetensors + config.json
    model_files = list(checkpoint_dir.glob("model*.safetensors"))
    config_file = checkpoint_dir / "config.json"
    if model_files and config_file.exists():
        return True

    return False


def is_lora_checkpoint(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint directory contains LoRA adapter weights."""
    return (checkpoint_dir / "adapter_config.json").exists()


def get_checkpoint_step(checkpoint_dir: Path) -> Optional[int]:
    """Extract the step number from checkpoint-XXXX directory name."""
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[1])
        except (IndexError, ValueError):
            return None
    return None


def get_all_checkpoints(save_dir: Path) -> list[Path]:
    """Get all checkpoint directories in save_dir, sorted by step."""
    checkpoints = []
    for ckpt in save_dir.glob("checkpoint-*"):
        if ckpt.is_dir():
            step = get_checkpoint_step(ckpt)
            if step is not None:
                checkpoints.append((step, ckpt))

    checkpoints.sort(key=lambda x: x[0])
    return [ckpt for _, ckpt in checkpoints]



def submit_sbatch(
    sbatch_script: Path,
    env: dict[str, str],
    dry_run: bool = False,
) -> Optional[str]:
    """
    Submit an sbatch job with the given environment variables.
    Returns the job ID if successful, None otherwise.
    """
    if dry_run:
        logger.info(f"[DRY-RUN] Would submit: {sbatch_script}")
        logger.info(f"[DRY-RUN] Environment: {env}")
        return "DRY-RUN-JOB-ID"

    try:
        full_env = os.environ.copy()
        full_env.update(env)

        result = subprocess.run(
            ["sbatch", str(sbatch_script)],
            env=full_env,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            # Parse job ID from "Submitted batch job 12345"
            job_output = result.stdout.strip()
            job_id = job_output.split()[-1] if job_output else None
            return job_id
        else:
            logger.warning(f"sbatch failed: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"Error submitting sbatch: {e}")
        return None


def main():
    """Main entry point for the auto-inference watcher."""
    args = parse_args()

    # Setup logging - default to {save_dir}/auto_infer_watcher.log
    log_file = args.log_file if args.log_file else args.save_dir / "auto_infer_watcher.log"
    setup_logging(log_file)

    # Validate paths
    if not args.save_dir.exists():
        logger.error(f"Save directory does not exist: {args.save_dir}")
        sys.exit(1)

    # Determine if this is a vision model
    is_vision = "qwen2_vl" in args.template.lower()

    # Select sbatch script
    if is_vision:
        sbatch_script = SBATCH_DIR / "vision_inference.sbatch"
    else:
        sbatch_script = SBATCH_DIR / "text_inference.sbatch"

    if not sbatch_script.exists():
        logger.error(f"Sbatch script not found: {sbatch_script}")
        sys.exit(1)

    # Build image params string for vision models
    image_params = ""
    if is_vision and args.image_min_pixels is not None and args.image_max_pixels is not None:
        image_params = f"--image_min_pixels {args.image_min_pixels} --image_max_pixels {args.image_max_pixels}"

    # Add _test suffix to dataset if not already present
    test_dataset = args.dataset if args.dataset.endswith("_test") else f"{args.dataset}_test"

    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Log startup info
    logger.info("=" * 60)
    logger.info("Auto-Inference Watcher Started")
    logger.info("=" * 60)
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Dataset: {test_dataset}")
    logger.info(f"Template: {args.template}")
    logger.info(f"Cutoff length: {args.cutoff_len}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    if image_params:
        logger.info(f"Image params: {image_params}")
    logger.info(f"Poll interval: {args.poll_interval}s")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Delete safetensors post-eval: {args.delete_safetensors_posteval}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Save logprobs: {args.save_logprobs}")
    logger.info(f"Keep safetensors epochs (steps): {args.keep_safetensors_epochs}")
    logger.info(f"Keep safetensors epoch idx: {args.keep_safetensors_epoch_idx}")
    logger.info(f"Log file: {log_file}")

    # Parse keep_safetensors_epochs (legacy: exact step numbers)
    keep_steps = set()
    if args.keep_safetensors_epochs:
        keep_steps = set(int(s) for s in args.keep_safetensors_epochs.split(",") if s.strip())

    # Parse keep_safetensors_epoch_idx (new: 1-indexed epoch positions)
    keep_epoch_idx = set()
    if args.keep_safetensors_epoch_idx:
        keep_epoch_idx = set(int(s) for s in args.keep_safetensors_epoch_idx.split(",") if s.strip())
    logger.info("=" * 60)

    # Main polling loop
    while True:
        try:
            training_complete = (args.save_dir / "all_results.json").exists()

            # Scan for unprocessed checkpoints (including final checkpoint)
            for ckpt_dir in args.save_dir.glob("checkpoint-*"):
                try:
                    if not ckpt_dir.is_dir():
                        continue

                    touch_file = ckpt_dir / ".infer.touch"
                    if touch_file.exists():
                        continue  # Already submitted

                    if not is_checkpoint_complete(ckpt_dir):
                        continue  # Still being written

                    ckpt_step = get_checkpoint_step(ckpt_dir)
                    if ckpt_step is None:
                        continue

                    # Prepare environment variables for sbatch
                    env = {
                        "CHECKPOINT_DIR": str(ckpt_dir),
                        "DATASET": test_dataset,
                        "TEMPLATE": args.template,
                        "CKPT_STEP": str(ckpt_step),
                        "RESULTS_DIR": str(args.results_dir),
                        "CUTOFF_LEN": str(args.cutoff_len),
                        "MAX_NEW_TOKENS": str(args.max_new_tokens),
                        "COMPUTE_TRAIN_ACCURACY": "1" if args.compute_train_accuracy else "0",
                        "TRAIN_ACCURACY_MAX_SAMPLES": str(args.train_accuracy_max_samples),
                        "TRAIN_ACCURACY_BATCH_SIZE": str(args.train_accuracy_batch_size),
                        "TEMPERATURE": str(args.temperature),
                        "SAVE_LOGPROBS": "1" if args.save_logprobs else "0",
                    }

                    # LoRA checkpoint: pass base model path + adapter path + rank
                    ckpt_is_lora = is_lora_checkpoint(ckpt_dir)
                    if ckpt_is_lora:
                        env["IS_LORA"] = "1"
                        adapter_cfg = json_module.loads((ckpt_dir / "adapter_config.json").read_text())
                        env["BASE_MODEL_PATH"] = adapter_cfg["base_model_name_or_path"]
                        env["LORA_RANK"] = str(adapter_cfg.get("r", 64))
                        logger.info(f"LoRA checkpoint detected: base_model={env['BASE_MODEL_PATH']}")

                    if args.train_dataset:
                        env["TRAIN_DATASET"] = args.train_dataset

                    if image_params:
                        env["IMAGE_PARAMS"] = image_params

                    # Skip safetensors deletion for LoRA (adapters are small, keep them)
                    # TODO: Remove the `False and` override once queue pressure is resolved.
                    if not ckpt_is_lora and False and args.delete_safetensors_posteval:
                        should_keep = ckpt_step in keep_steps
                        if not should_keep and keep_epoch_idx:
                            # Sort all checkpoints by step, find 1-indexed position
                            all_steps = sorted(
                                get_checkpoint_step(d)
                                for d in args.save_dir.glob("checkpoint-*")
                                if d.is_dir() and get_checkpoint_step(d) is not None
                            )
                            epoch_num = all_steps.index(ckpt_step) + 1
                            should_keep = epoch_num in keep_epoch_idx
                        if not should_keep:
                            env["DELETE_SAFETENSORS"] = "1"

                    # Submit job
                    job_id = submit_sbatch(sbatch_script, env, dry_run=args.dry_run)

                    if job_id:
                        # Write touch file to prevent duplicate submissions
                        touch_file.write_text(f"{job_id}\n")
                        logger.info(f"Submitted job {job_id} for {ckpt_dir.name}")
                    else:
                        logger.warning(f"Failed to submit job for {ckpt_dir.name}")

                except Exception as e:
                    # Log error but continue processing other checkpoints
                    logger.warning(f"Error processing {ckpt_dir}: {e}")
                    continue

        except Exception as e:
            # Catch-all: log error, don't crash the watcher
            logger.warning(f"Watcher loop error (will retry): {e}")

        # Exit after final scan once training is done
        if training_complete:
            logger.info("Training complete (all_results.json found), all checkpoints processed, exiting watcher")
            break

        # Wait for next poll
        time.sleep(args.poll_interval)

    logger.info("Auto-Inference Watcher stopped")


if __name__ == "__main__":
    main()
