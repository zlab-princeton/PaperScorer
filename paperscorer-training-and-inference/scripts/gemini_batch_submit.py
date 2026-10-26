#!/usr/bin/env python3
"""
Gemini Batch API - Submit batch jobs for ICLR review prediction using Vertex AI.

Uses Vertex AI batch prediction with GCS URIs for images (50% cost savings).
Supports balanced_deepreview datasets: clean (text), clean+images, vision.

Usage:
    # Text-only inference
    python scripts/gemini_batch_submit.py \
        --dataset iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3 \
        --split test \
        --project YOUR_PROJECT \
        --output results/gemini/clean/gemini-2.5-flash.jsonl

    # Vision inference with page images
    python scripts/gemini_batch_submit.py \
        --dataset iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3 \
        --split test \
        --project YOUR_PROJECT \
        --output results/gemini/vision/gemini-2.5-flash.jsonl

    # Dry run (test with 10 samples)
    python scripts/gemini_batch_submit.py \
        --dataset ... --dry_run 10 --project YOUR_PROJECT

    # Submit only (don't wait for results)
    python scripts/gemini_batch_submit.py \
        --dataset ... --submit_only --project YOUR_PROJECT
"""

import argparse
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_GCS_IMAGES_BASE = "gs://autoreviewer-data/autoreviewer_data/images"
LOCAL_IMAGES_PREFIX = "data/images/"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"


def get_short_name(dataset_name: str) -> str:
    """Get short name for dataset (clean, clean+images, vision)."""
    if "clean+images" in dataset_name or "clean_images" in dataset_name:
        return "clean+images"
    elif "vision" in dataset_name:
        return "vision"
    else:
        return "clean"


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load dataset from LLaMA Factory format.

    Args:
        data_dir: Base data directory
        dataset_name: Dataset name (without _train/_test suffix)
        split: 'train' or 'test'

    Returns:
        List of dataset entries with conversations, images, _metadata
    """
    dataset_path = data_dir / f"{dataset_name}_{split}" / "data.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} samples")
    return data


def parse_conversations(entry: dict) -> dict:
    """Parse conversations to extract system, human, and gpt messages.

    Args:
        entry: Dataset entry with 'conversations' field

    Returns:
        Dict with 'system', 'human', 'gpt' messages
    """
    result = {"system": None, "human": None, "gpt": None}

    for msg in entry.get("conversations", []):
        role = msg.get("from", "")
        value = msg.get("value", "")

        if role == "system":
            result["system"] = value
        elif role == "human":
            result["human"] = value
        elif role == "gpt":
            result["gpt"] = value

    return result


def convert_image_path_to_gcs(local_path: str, gcs_base: str) -> str:
    """Convert local image path to GCS URI.

    Args:
        local_path: Local path like 'data/images/xxx/page_1.png'
        gcs_base: GCS base URI like 'gs://bucket/path/images'

    Returns:
        GCS URI like 'gs://bucket/path/images/xxx/page_1.png'
    """
    if local_path.startswith(LOCAL_IMAGES_PREFIX):
        relative_path = local_path[len(LOCAL_IMAGES_PREFIX):]
        return f"{gcs_base.rstrip('/')}/{relative_path}"
    return local_path


def get_mime_type(path: str) -> str:
    """Get MIME type from file extension."""
    ext = path.lower().split(".")[-1]
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return mime_types.get(ext, "image/png")


# ============================================================================
# BATCH REQUEST CREATION
# ============================================================================

def build_content_parts(human_message: str, images: list[str], gcs_base: str) -> list[dict]:
    """Build content parts array with text and images.

    The human message contains <image> placeholders that should be replaced
    with actual image parts in order.

    Args:
        human_message: Human message with <image> placeholders
        images: List of local image paths
        gcs_base: GCS base URI for images

    Returns:
        List of parts (text and fileData)
    """
    parts = []

    # Split message by <image> placeholders
    segments = re.split(r"<image>", human_message)
    image_idx = 0

    for i, segment in enumerate(segments):
        # Add text segment if non-empty
        if segment.strip():
            parts.append({"text": segment})

        # Add image part after each segment (except the last)
        if i < len(segments) - 1 and image_idx < len(images):
            gcs_uri = convert_image_path_to_gcs(images[image_idx], gcs_base)
            mime_type = get_mime_type(images[image_idx])
            parts.append({
                "fileData": {
                    "fileUri": gcs_uri,
                    "mimeType": mime_type
                }
            })
            image_idx += 1

    # If no segments or all empty, add the full message
    if not parts:
        parts.append({"text": human_message})

    return parts


def create_batch_request(
    entry: dict,
    gcs_base: str,
    temperature: float = 0.0,
    max_tokens: int = 250,
    thinking_budget: int = 2000,
) -> dict:
    """Create a batch request for a single entry.

    Args:
        entry: Dataset entry
        gcs_base: GCS base URI for images
        temperature: Sampling temperature
        max_tokens: Max output tokens
        thinking_budget: Thinking budget tokens (0 to disable)

    Returns:
        Batch request dict with 'request' key
    """
    messages = parse_conversations(entry)
    images = entry.get("images", [])

    # Build content parts
    parts = build_content_parts(messages["human"] or "", images, gcs_base)

    # Build request with format reminder
    request = {
        "contents": [
            {"role": "user", "parts": parts},
            {"role": "user", "parts": [{"text": "PLEASE FORMAT ANSWER AS: \\boxed{Accept} or \\boxed{Reject}"}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {
                "thinkingBudget": thinking_budget
            }
        }
    }

    # Add system instruction if present
    if messages["system"]:
        request["systemInstruction"] = {
            "parts": [{"text": messages["system"]}]
        }

    return {"request": request}


def create_batch_jsonl(
    data: list[dict],
    output_path: str,
    gcs_base: str,
    temperature: float = 0.0,
    max_tokens: int = 250,
    thinking_budget: int = 2000,
) -> tuple[str, list[dict]]:
    """Create JSONL file for batch processing.

    Args:
        data: List of dataset entries
        output_path: Path to output JSONL file
        gcs_base: GCS base URI for images
        temperature: Sampling temperature
        max_tokens: Max output tokens
        thinking_budget: Thinking budget tokens (0 to disable)

    Returns:
        Tuple of (output_path, metadata_list)
        metadata_list contains original entry info for result matching
    """
    print(f"Creating batch JSONL: {output_path}")

    metadata_list = []

    with open(output_path, "w") as f:
        for idx, entry in enumerate(data):
            # Create batch request
            request = create_batch_request(
                entry, gcs_base, temperature, max_tokens, thinking_budget
            )

            # Write request line
            f.write(json.dumps(request) + "\n")

            # Store metadata for result matching
            messages = parse_conversations(entry)
            metadata_list.append({
                "idx": idx,
                "submission_id": entry.get("_metadata", {}).get("submission_id", ""),
                "label": messages["gpt"] or "",
                "human_message": messages["human"] or "",
            })

    file_size = os.path.getsize(output_path)
    print(f"  Created {len(data)} requests, file size: {file_size / 1024 / 1024:.2f} MB")

    return output_path, metadata_list


# ============================================================================
# BATCH JOB MANAGEMENT
# ============================================================================

def get_client(project: str, location: str) -> genai.Client:
    """Get Gemini client for Vertex AI.

    Args:
        project: GCP project ID
        location: GCP region

    Returns:
        Configured genai.Client
    """
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1")
    )


def upload_to_gcs(local_path: str, gcs_uri: str) -> str:
    """Upload local file to GCS.

    Args:
        local_path: Local file path
        gcs_uri: GCS URI (gs://bucket/path)

    Returns:
        GCS URI
    """
    from google.cloud import storage

    # Parse GCS URI
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    # Upload
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    print(f"  Uploaded to: {gcs_uri}")
    return gcs_uri


def submit_batch_job(
    client: genai.Client,
    model: str,
    input_uri: str,
    output_uri: str,
    display_name: str = None,
) -> any:
    """Submit a batch prediction job.

    Args:
        client: Gemini client
        model: Model name (e.g., gemini-2.5-flash)
        input_uri: GCS URI for input JSONL
        output_uri: GCS URI prefix for output
        display_name: Optional job display name

    Returns:
        Batch job object
    """
    if display_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_name = f"autoreviewer_{timestamp}"

    print(f"\nSubmitting batch job...")
    print(f"  Model: {model}")
    print(f"  Input: {input_uri}")
    print(f"  Output: {output_uri}")

    job = client.batches.create(
        model=model,
        src=input_uri,
        config=CreateBatchJobConfig(
            dest=output_uri,
            display_name=display_name,
        )
    )

    print(f"  Job submitted: {job.name}")
    print(f"  State: {job.state}")

    return job


def poll_until_complete(
    client: genai.Client,
    job_name: str,
    interval: int = 300,
    max_wait: int = 86400,
) -> any:
    """Poll job until completion.

    Args:
        client: Gemini client
        job_name: Batch job name
        interval: Polling interval in seconds
        max_wait: Max wait time in seconds (default 24h)

    Returns:
        Completed job object

    Raises:
        RuntimeError: If job fails or times out
    """
    print(f"\nPolling job status (interval: {interval}s)...")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise RuntimeError(f"Job timed out after {elapsed/3600:.1f} hours")

        job = client.batches.get(name=job_name)
        state = str(job.state)

        elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
        print(f"  [{elapsed_str}] State: {state}")

        if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
            print(f"\n  Job completed successfully!")
            return job

        if "FAILED" in state:
            raise RuntimeError(f"Batch job failed: {state}")

        if "CANCELLED" in state:
            raise RuntimeError(f"Batch job cancelled: {state}")

        time.sleep(interval)


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """Download file from GCS.

    Args:
        gcs_uri: GCS URI
        local_path: Local destination path

    Returns:
        Local path
    """
    from google.cloud import storage

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

    return local_path


def list_gcs_files(gcs_prefix: str) -> list[str]:
    """List files in GCS with prefix.

    Args:
        gcs_prefix: GCS URI prefix

    Returns:
        List of GCS URIs
    """
    from google.cloud import storage

    parts = gcs_prefix[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]


# ============================================================================
# RESULT PROCESSING
# ============================================================================

def parse_prediction(response: dict) -> str:
    """Extract prediction text from batch response.

    Args:
        response: Response object from batch result

    Returns:
        Generated text
    """
    try:
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def process_batch_results(
    output_uri: str,
    metadata_list: list[dict],
    output_path: str,
) -> int:
    """Process batch results and save in vllm_infer.py format.

    Args:
        output_uri: GCS URI prefix where results are stored
        metadata_list: List of metadata for each request
        output_path: Local path to save results JSONL

    Returns:
        Number of successfully processed results
    """
    print(f"\nProcessing batch results from: {output_uri}")

    # List result files
    result_files = list_gcs_files(output_uri)
    result_files = [f for f in result_files if f.endswith(".jsonl")]

    if not result_files:
        print("  No result files found!")
        return 0

    print(f"  Found {len(result_files)} result file(s)")

    # Download and parse results
    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gcs_file in result_files:
            local_file = os.path.join(tmpdir, os.path.basename(gcs_file))
            download_from_gcs(gcs_file, local_file)

            with open(local_file) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

    print(f"  Parsed {len(all_results)} results")

    # Match results with metadata and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = 0
    with open(output_path, "w") as f:
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break

            meta = metadata_list[idx]

            # Extract prediction
            response = result.get("response", {})
            prediction = parse_prediction(response)

            # Write in vllm_infer.py format (full prompt, no truncation)
            output_entry = {
                "prompt": meta["human_message"],
                "predict": prediction,
                "label": meta["label"],
            }
            f.write(json.dumps(output_entry) + "\n")
            processed += 1

    print(f"  Saved {processed} results to: {output_path}")
    return processed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch API for ICLR review prediction"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (without _train/_test suffix)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory (default: data)"
    )

    # GCP arguments
    parser.add_argument(
        "--project",
        type=str,
        default='ringed-inn-474523-u3',
        help="GCP project ID"
    )
    parser.add_argument(
        "--location",
        type=str,
        default=DEFAULT_LOCATION,
        help=f"GCP region (default: {DEFAULT_LOCATION})"
    )
    parser.add_argument(
        "--gcs_base",
        type=str,
        default=DEFAULT_GCS_IMAGES_BASE,
        help=f"GCS base URI for images (default: {DEFAULT_GCS_IMAGES_BASE})"
    )
    parser.add_argument(
        "--gcs_staging",
        type=str,
        default=None,
        help="GCS URI for staging batch input/output (default: gs://autoreviewer-data/batch_staging)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=250,
        help="Max output tokens (default: 250)"
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=2000,
        help="Thinking budget tokens (default: 2000, set to 0 to disable)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path"
    )

    # Control arguments
    parser.add_argument(
        "--dry_run",
        type=int,
        default=None,
        help="Test with N samples (default: None = all)"
    )
    parser.add_argument(
        "--submit_only",
        action="store_true",
        help="Submit job and exit without waiting for results"
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=300,
        help="Polling interval in seconds (default: 300)"
    )

    args = parser.parse_args()

    # Set default staging URI
    if args.gcs_staging is None:
        args.gcs_staging = "gs://autoreviewer-data/batch_staging"

    # Print configuration
    print("=" * 70)
    print("Gemini Batch API - ICLR Review Prediction")
    print("=" * 70)
    print(f"Dataset: {args.dataset}_{args.split}")
    print(f"Model: {args.model}")
    print(f"Thinking Budget: {args.thinking_budget} tokens")
    print(f"Max Output Tokens: {args.max_tokens}")
    print(f"Project: {args.project}")
    print(f"Location: {args.location}")
    print(f"GCS Images: {args.gcs_base}")
    print(f"Output: {args.output}")
    if args.dry_run:
        print(f"DRY RUN: Using {args.dry_run} samples")
    if args.submit_only:
        print("SUBMIT ONLY: Will not wait for results")
    print()

    # Load dataset
    data = load_dataset(Path(args.data_dir), args.dataset, args.split)

    # Apply dry run limit
    if args.dry_run:
        data = data[:args.dry_run]
        print(f"Using {len(data)} samples for dry run")

    # Create batch JSONL (include microseconds to avoid collisions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    short_name = get_short_name(args.dataset)
    local_jsonl = f"/tmp/batch_input_{timestamp}.jsonl"
    _, metadata_list = create_batch_jsonl(
        data, local_jsonl, args.gcs_base,
        args.temperature, args.max_tokens, args.thinking_budget
    )

    # Upload to GCS (use short name + timestamp)
    input_uri = f"{args.gcs_staging}/{short_name}_{timestamp}_input.jsonl"
    output_uri = f"{args.gcs_staging}/{short_name}_{timestamp}_output/"

    print(f"\nUploading batch input to GCS...")
    upload_to_gcs(local_jsonl, input_uri)

    # Get client and submit job
    client = get_client(args.project, args.location)
    # Use short name for display
    display_name = f"{short_name}_{timestamp}"
    job = submit_batch_job(
        client, args.model, input_uri, output_uri,
        display_name=display_name
    )

    # Save job info
    job_info = {
        "job_name": job.name,
        "model": args.model,
        "dataset": f"{args.dataset}_{args.split}",
        "num_samples": len(data),
        "input_uri": input_uri,
        "output_uri": output_uri,
        "submitted_at": timestamp,
    }
    job_info_path = args.output.replace(".jsonl", "_job_info.json")
    os.makedirs(os.path.dirname(job_info_path), exist_ok=True)
    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)
    print(f"Job info saved: {job_info_path}")

    # Save metadata for later result processing
    metadata_path = args.output.replace(".jsonl", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f)
    print(f"Metadata saved: {metadata_path}")

    if args.submit_only:
        print("\n" + "=" * 70)
        print("Job submitted. To retrieve results later, run:")
        print(f"  python scripts/gemini_batch_retrieve.py --job_name {job.name} --output {args.output}")
        print("=" * 70)
        return

    # Poll until complete
    try:
        completed_job = poll_until_complete(
            client, job.name,
            interval=args.poll_interval
        )
    except RuntimeError as e:
        print(f"\nError: {e}")
        return

    # Process results
    num_processed = process_batch_results(
        output_uri, metadata_list, args.output
    )

    # Update job info
    job_info["status"] = "completed"
    job_info["completed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_info["num_results"] = num_processed
    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)
    print(f"Results: {args.output}")
    print(f"Processed: {num_processed} / {len(data)}")

    # Cleanup local temp file
    if os.path.exists(local_jsonl):
        os.remove(local_jsonl)


if __name__ == "__main__":
    main()
