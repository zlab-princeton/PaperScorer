#!/usr/bin/env python3
"""
Gemini Batch API - Retrieve results from a previously submitted batch job.

Usage:
    # Check job status
    python scripts/gemini_batch_retrieve.py \
        --job_name projects/.../batchPredictionJobs/... \
        --project YOUR_PROJECT \
        --status_only

    # Retrieve completed results
    python scripts/gemini_batch_retrieve.py \
        --job_name projects/.../batchPredictionJobs/... \
        --output results/gemini/clean/gemini-2.5-flash.jsonl \
        --project YOUR_PROJECT
"""

import argparse
import json
import os
import tempfile

from google import genai
from google.genai.types import HttpOptions


DEFAULT_LOCATION = "us-central1"


def get_client(project: str, location: str) -> genai.Client:
    """Get Gemini client for Vertex AI."""
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1")
    )


def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """Download file from GCS."""
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
    """List files in GCS with prefix."""
    from google.cloud import storage

    parts = gcs_prefix[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]


def parse_prediction(response: dict) -> str:
    """Extract prediction text from batch response."""
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
    """Process batch results and save in vllm_infer.py format."""
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
            response = result.get("response", {})
            prediction = parse_prediction(response)

            output_entry = {
                "prompt": meta["human_message"][:500] + "..." if len(meta["human_message"]) > 500 else meta["human_message"],
                "predict": prediction,
                "label": meta["label"],
            }
            f.write(json.dumps(output_entry) + "\n")
            processed += 1

    print(f"  Saved {processed} results to: {output_path}")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve results from Gemini batch job"
    )

    parser.add_argument(
        "--job_name",
        type=str,
        required=True,
        help="Batch job name (from submit output)"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--location",
        type=str,
        default=DEFAULT_LOCATION,
        help=f"GCP region (default: {DEFAULT_LOCATION})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (required if not --status_only)"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to metadata JSON (default: output.replace('.jsonl', '_metadata.json'))"
    )
    parser.add_argument(
        "--status_only",
        action="store_true",
        help="Only check job status, don't retrieve results"
    )

    args = parser.parse_args()

    if not args.status_only and not args.output:
        parser.error("--output is required when not using --status_only")

    # Get client
    client = get_client(args.project, args.location)

    # Get job status
    print(f"Checking job: {args.job_name}")
    job = client.batches.get(name=args.job_name)

    state = str(job.state)
    print(f"  State: {state}")

    if args.status_only:
        return

    # Check if completed
    if "SUCCEEDED" not in state and state != "JOB_STATE_SUCCEEDED":
        print(f"\nJob not completed yet. Current state: {state}")
        print("Use --status_only to check status without retrieving results.")
        return

    # Load metadata
    if args.metadata_path:
        metadata_path = args.metadata_path
    else:
        metadata_path = args.output.replace(".jsonl", "_metadata.json")

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        print("Please provide --metadata_path")
        return

    with open(metadata_path) as f:
        metadata_list = json.load(f)

    print(f"Loaded {len(metadata_list)} metadata entries")

    # Get output URI from job info
    job_info_path = args.output.replace(".jsonl", "_job_info.json")
    if os.path.exists(job_info_path):
        with open(job_info_path) as f:
            job_info = json.load(f)
        output_uri = job_info.get("output_uri", "")
    else:
        # Try to extract from job object
        output_uri = getattr(job, "output_uri", None)
        if not output_uri:
            print("Error: Cannot determine output URI")
            print("Please check job_info file or provide output location")
            return

    # Process results
    num_processed = process_batch_results(
        output_uri, metadata_list, args.output
    )

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)
    print(f"Results: {args.output}")
    print(f"Processed: {num_processed} / {len(metadata_list)}")


if __name__ == "__main__":
    main()
