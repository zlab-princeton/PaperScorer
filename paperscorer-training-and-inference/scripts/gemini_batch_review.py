#!/usr/bin/env python3
"""
Gemini Batch Review Generation — Generate AI reviews (strengths, weaknesses,
critical disputes) for ICLR papers using Gemini batch API (50% cost savings)
or online mode (immediate results).

Two generation modes:
  blind    — Gemini sees only the paper (no decision, no human reviews)
  informed — Gemini sees paper + real human reviews as inspiration (no decision)

Usage:
    # Blind text, 3 candidates (batch mode — default)
    python scripts/gemini_batch_review.py \
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
        --split train --mode blind --candidate_count 3 --temperature 1.0 \
        --output results/reviews/blind_text_train.jsonl

    # Blind vision
    python scripts/gemini_batch_review.py \
        --dataset iclr_..._vision_... \
        --split train --mode blind --candidate_count 3 \
        --output results/reviews/blind_vision_train.jsonl

    # Informed (needs metadata_dir for human reviews)
    python scripts/gemini_batch_review.py \
        --dataset ...text... --split train --mode informed \
        --metadata_dir data/massive_metadata_v7_5 \
        --output results/reviews/informed_text_train.jsonl

    # Online mode — direct API calls with rate limiting
    python scripts/gemini_batch_review.py \
        --dataset ... --split train --mode blind --candidate_count 1 \
        --nobatch --qps 10 \
        --output results/reviews/blind_text_train.jsonl

    # Dry run + diversity analysis
    python scripts/gemini_batch_review.py \
        --dataset ... --dry_run 5 --candidate_count 3 --analyze_diversity
"""

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_GCS_IMAGES_BASE = "gs://autoreviewer-data/autoreviewer_data/images"
LOCAL_IMAGES_PREFIX = "data/images/"
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_LOCATION = "us-central1"

# Fields to strip from human reviews to avoid decision leakage
REVIEW_FIELDS_TO_STRIP = {"rating", "decision"}


# ============================================================================
# PYDANTIC SCHEMA — used for native structured output
# ============================================================================

class ReviewOutput(BaseModel):
    strengths: list[str] = Field(description="Top 3 strengths, 3-4 sentences each")
    weaknesses: list[str] = Field(description="Top 3 weaknesses, 3-4 sentences each")
    critical_disputes: str = Field(
        description="What reviewers would likely disagree about, 3-4 sentences"
    )
    most_important_strength: str = Field(
        description="Single most compelling aspect, 3-4 sentences"
    )
    most_important_weakness: str = Field(
        description="Single biggest concern, 3-4 sentences"
    )
    relation_to_other_work: str = Field(
        description="How this paper relates to and builds on prior work, 3-4 sentences"
    )
    paper_quality: str = Field(
        description="Overall 'paper gestalt' — the gut feeling of quality. Consider: Is the writing polished with full, well-structured paragraphs? Are figures professional, clear, and easy to understand? Are tables well-formatted and informative? Does the paper feel like a mature, carefully prepared submission or a rushed draft? 3-4 sentences"
    )


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_BLIND_TEXT = """\
You are an expert reviewer at a top ML conference in {year}.
It is currently {year}. Evaluate this paper ONLY using knowledge available up to {year}.
Do NOT use hindsight or reference any work published after {year}.
Do NOT predict the acceptance decision.
Note: This paper was converted from PDF to Markdown, so ignore any formatting artifacts from the conversion process — they are not reflective of the authors' original submission quality.
Provide exactly 3 strengths and 3 weaknesses, each 3-4 sentences. All other fields: 3-4 sentences.

{paper_text}"""

PROMPT_BLIND_VISION = """\
You are an expert reviewer at a top ML conference in {year}.
It is currently {year}. Evaluate this paper ONLY using knowledge available up to {year}.
Do NOT use hindsight or reference any work published after {year}.
Do NOT predict the acceptance decision.
Consider visual elements (figures, tables, diagrams) in your analysis where relevant.
Provide exactly 3 strengths and 3 weaknesses, each 3-4 sentences. All other fields: 3-4 sentences.

{paper_text}"""

PROMPT_INFORMED = """\
You are an expert reviewer at a top ML conference in {year}.
It is currently {year}. Evaluate this paper ONLY using knowledge available up to {year}.
Do NOT use hindsight or reference any work published after {year}.
You are also given real peer reviews for reference. Use them to inform your analysis \
but write your own independent assessment. Do NOT predict the acceptance decision.
Note: This paper was converted from PDF to Markdown, so ignore any formatting artifacts from the conversion process — they are not reflective of the authors' original submission quality.
Provide exactly 3 strengths and 3 weaknesses, each 3-4 sentences. All other fields: 3-4 sentences.

{paper_text}

## Peer Reviews (ratings redacted)
{peer_reviews}"""


# ============================================================================
# PARTIAL PREDICTIONS LOADING
# ============================================================================

def load_completed_ids(predictions_path: str, metadata_path: str) -> set[str]:
    """Load submission_ids that have successful responses from a partial predictions file.

    The predictions JSONL has one line per request (matched by order to the metadata).
    Lines with a 'response' key containing non-empty 'candidates' are considered complete.

    Returns:
        Set of submission_ids that were successfully completed.
    """
    with open(metadata_path) as f:
        metadata_list = json.load(f)

    completed = set()
    verified = 0
    empty = 0

    with open(predictions_path) as f:
        for idx, line in enumerate(f):
            if idx >= len(metadata_list):
                break
            d = json.loads(line)
            response = d.get("response", {})
            candidates = response.get("candidates", [])
            if candidates:
                # Verify at least one candidate has parseable JSON
                for cand in candidates:
                    try:
                        text = cand.get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            json.loads(text)  # validate JSON
                            verified += 1
                            completed.add(metadata_list[idx]["submission_id"])
                            break
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        pass
            else:
                empty += 1

    print(f"  Partial predictions: {len(metadata_list)} total, {len(completed)} completed "
          f"({verified} verified), {empty} empty/incomplete")
    return completed


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load dataset from LLaMA Factory format (data.json)."""
    dataset_path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")
    return data


def get_short_name(dataset_name: str) -> str:
    """Get short name for dataset (clean, clean+images, vision)."""
    if "clean+images" in dataset_name or "clean_images" in dataset_name:
        return "clean+images"
    elif "vision" in dataset_name:
        return "vision"
    else:
        return "clean"


def is_vision_dataset(dataset_name: str) -> bool:
    """Check if dataset contains images."""
    return "vision" in dataset_name or "clean_images" in dataset_name or "clean+images" in dataset_name


def extract_paper_text(entry: dict) -> str:
    """Extract the paper text from the human message in conversations.

    Strips the system instruction prefix (accept/reject prompt) to get raw paper text.
    """
    for msg in entry.get("conversations", []):
        if msg.get("from") == "human":
            text = msg["value"]
            # Strip the instruction prefix up to the paper content
            # Typical prefix ends before the paper title (starts with #)
            match = re.search(r"\n(#\s)", text)
            if match:
                return text[match.start():].strip()
            return text
    return ""


def extract_year(entry: dict) -> int:
    """Extract year from entry metadata."""
    return entry.get("_metadata", {}).get("year", 2024)


def extract_submission_id(entry: dict) -> str:
    """Extract submission_id from entry metadata."""
    return entry.get("_metadata", {}).get("submission_id", "")


# ============================================================================
# HUMAN REVIEW LOADING (informed mode)
# ============================================================================

def load_metadata_reviews(metadata_dir: str) -> dict[str, str]:
    """Load human reviews from massive_metadata Arrow dataset.

    Returns:
        Dict mapping submission_id → formatted review text (ratings stripped).
    """
    from datasets import load_from_disk

    print(f"Loading metadata reviews from: {metadata_dir}")
    ds = load_from_disk(metadata_dir)
    print(f"  Loaded {len(ds)} metadata entries")

    reviews_lookup = {}
    for row in ds:
        sid = row["submission_id"]
        raw = row.get("original_reviews")
        if not raw:
            continue
        try:
            reviews_list = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        formatted = format_reviews_for_prompt(reviews_list)
        if formatted:
            reviews_lookup[sid] = formatted

    print(f"  Built review lookup for {len(reviews_lookup)} papers")
    return reviews_lookup


def format_reviews_for_prompt(reviews_list: list[dict]) -> str:
    """Format human reviews for injection into the prompt, stripping rating fields."""
    parts = []
    for i, review in enumerate(reviews_list, 1):
        lines = [f"### Review {i}"]
        for key, value in review.items():
            if key.lower() in REVIEW_FIELDS_TO_STRIP:
                continue
            if isinstance(value, str) and value.strip():
                lines.append(f"**{key}**: {value}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ============================================================================
# IMAGE HANDLING
# ============================================================================

def convert_image_path_to_gcs(local_path: str, gcs_base: str) -> str:
    """Convert local image path to GCS URI."""
    if local_path.startswith(LOCAL_IMAGES_PREFIX):
        relative_path = local_path[len(LOCAL_IMAGES_PREFIX):]
        return f"{gcs_base.rstrip('/')}/{relative_path}"
    return local_path


def get_mime_type(path: str) -> str:
    """Get MIME type from file extension."""
    ext = path.lower().split(".")[-1]
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/png")


# ============================================================================
# PROMPT + CONTENT BUILDING
# ============================================================================

def build_prompt_text(
    entry: dict,
    mode: str,
    is_vision: bool,
    reviews_lookup: dict[str, str] | None = None,
) -> str:
    """Build the prompt text for a single entry.

    Args:
        entry: Dataset entry
        mode: 'blind' or 'informed'
        is_vision: Whether this is a vision dataset
        reviews_lookup: submission_id → review text (for informed mode)

    Returns:
        Formatted prompt string
    """
    paper_text = extract_paper_text(entry)
    year = extract_year(entry)
    sid = extract_submission_id(entry)

    if mode == "informed" and reviews_lookup:
        peer_reviews = reviews_lookup.get(sid, "(No peer reviews available)")
        return PROMPT_INFORMED.format(
            year=year, paper_text=paper_text, peer_reviews=peer_reviews
        )
    elif is_vision:
        return PROMPT_BLIND_VISION.format(year=year, paper_text=paper_text)
    else:
        return PROMPT_BLIND_TEXT.format(year=year, paper_text=paper_text)


def build_content_parts(
    prompt_text: str,
    images: list[str],
    gcs_base: str,
) -> list[dict]:
    """Build content parts for batch request (text + images).

    For vision datasets, images are interleaved at <image> placeholders
    in the original human message. For review generation, we put the prompt
    first, then append all images.
    """
    parts = []

    if images:
        # For vision: split text by <image> placeholders if present
        segments = re.split(r"<image>", prompt_text)
        image_idx = 0
        for i, segment in enumerate(segments):
            if segment.strip():
                parts.append({"text": segment})
            if i < len(segments) - 1 and image_idx < len(images):
                gcs_uri = convert_image_path_to_gcs(images[image_idx], gcs_base)
                mime_type = get_mime_type(images[image_idx])
                parts.append({
                    "fileData": {"fileUri": gcs_uri, "mimeType": mime_type}
                })
                image_idx += 1
        # Append remaining images not matched to placeholders
        while image_idx < len(images):
            gcs_uri = convert_image_path_to_gcs(images[image_idx], gcs_base)
            mime_type = get_mime_type(images[image_idx])
            parts.append({
                "fileData": {"fileUri": gcs_uri, "mimeType": mime_type}
            })
            image_idx += 1
    else:
        parts.append({"text": prompt_text})

    if not parts:
        parts.append({"text": prompt_text})

    return parts


def build_generation_config(args: argparse.Namespace) -> dict:
    """Build generationConfig for the API request."""
    config = {
        "temperature": args.temperature,
        "maxOutputTokens": args.max_tokens,
        "candidateCount": args.candidate_count,
        "responseMimeType": "application/json",
        "responseSchema": ReviewOutput.model_json_schema(),
    }
    if args.thinking_budget is not None:
        config["thinkingConfig"] = {"thinkingBudget": args.thinking_budget}
    return config


# ============================================================================
# BATCH REQUEST CREATION
# ============================================================================

def create_batch_request(
    entry: dict,
    mode: str,
    is_vision: bool,
    gcs_base: str,
    generation_config: dict,
    reviews_lookup: dict[str, str] | None = None,
) -> dict:
    """Create a single batch request dict."""
    prompt_text = build_prompt_text(entry, mode, is_vision, reviews_lookup)
    images = entry.get("images", [])
    parts = build_content_parts(prompt_text, images, gcs_base)

    request = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    return {"request": request}


def create_batch_jsonl(
    data: list[dict],
    output_path: str,
    mode: str,
    is_vision: bool,
    gcs_base: str,
    generation_config: dict,
    reviews_lookup: dict[str, str] | None = None,
) -> tuple[str, list[dict]]:
    """Create JSONL file for batch processing.

    Returns:
        Tuple of (output_path, metadata_list)
    """
    print(f"Creating batch JSONL: {output_path}")
    metadata_list = []

    with open(output_path, "w") as f:
        for idx, entry in enumerate(data):
            request = create_batch_request(
                entry, mode, is_vision, gcs_base, generation_config, reviews_lookup
            )
            f.write(json.dumps(request) + "\n")

            metadata_list.append({
                "idx": idx,
                "submission_id": extract_submission_id(entry),
                "year": extract_year(entry),
                "split": entry.get("_metadata", {}).get("_split", ""),
            })

    file_size = os.path.getsize(output_path)
    print(f"  Created {len(data)} requests, file size: {file_size / 1024 / 1024:.2f} MB")
    return output_path, metadata_list


# ============================================================================
# GCS + BATCH JOB (reused from gemini_batch_submit.py)
# ============================================================================

def get_client(project: str, location: str):
    """Get Gemini client for Vertex AI."""
    from google import genai
    from google.genai.types import HttpOptions

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )


def upload_to_gcs(local_path: str, gcs_uri: str) -> str:
    """Upload local file to GCS."""
    from google.cloud import storage

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    print(f"  Uploaded to: {gcs_uri}")
    return gcs_uri


def submit_batch_job(client, model: str, input_uri: str, output_uri: str, display_name: str | None = None):
    """Submit a batch prediction job."""
    from google.genai.types import CreateBatchJobConfig

    if display_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_name = f"review_gen_{timestamp}"

    print(f"\nSubmitting batch job...")
    print(f"  Model: {model}")
    print(f"  Input: {input_uri}")
    print(f"  Output: {output_uri}")

    job = client.batches.create(
        model=model,
        src=input_uri,
        config=CreateBatchJobConfig(dest=output_uri, display_name=display_name),
    )

    print(f"  Job submitted: {job.name}")
    print(f"  State: {job.state}")
    return job


def poll_until_complete(client, job_name: str, interval: int = 300, max_wait: int = 86400):
    """Poll job until completion."""
    print(f"\nPolling job status (interval: {interval}s)...")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise RuntimeError(f"Job timed out after {elapsed / 3600:.1f} hours")

        job = client.batches.get(name=job_name)
        state = str(job.state)

        elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
        print(f"  [{elapsed_str}] State: {state}")

        if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
            print("  Job completed successfully!")
            return job
        if "FAILED" in state:
            raise RuntimeError(f"Batch job failed: {state}")
        if "CANCELLED" in state:
            raise RuntimeError(f"Batch job cancelled: {state}")

        time.sleep(interval)


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


# ============================================================================
# RESULT PROCESSING (batch mode)
# ============================================================================

def parse_batch_results(
    output_uri: str,
    metadata_list: list[dict],
    output_path: str,
    candidate_count: int,
) -> int:
    """Process batch results and save as JSONL.

    Handles candidateCount > 1 by emitting one row per candidate.

    Returns:
        Number of successfully processed results.
    """
    print(f"\nProcessing batch results from: {output_uri}")

    result_files = list_gcs_files(output_uri)
    result_files = [f for f in result_files if f.endswith(".jsonl")]

    if not result_files:
        print("  No result files found!")
        return 0

    print(f"  Found {len(result_files)} result file(s)")

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed = 0
    empty_count = 0

    with open(output_path, "w") as f:
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break
            meta = metadata_list[idx]
            response = result.get("response", {})
            candidates = response.get("candidates", [])

            if not candidates:
                empty_count += 1
                continue

            for cand_idx, candidate in enumerate(candidates):
                raw_text = ""
                try:
                    parts = candidate.get("content", {}).get("parts", [])
                    if parts:
                        raw_text = parts[0].get("text", "")
                except (KeyError, IndexError, TypeError):
                    pass

                review_obj = None
                if raw_text:
                    try:
                        review_obj = json.loads(raw_text)
                    except json.JSONDecodeError:
                        pass

                output_entry = {
                    "submission_id": meta["submission_id"],
                    "year": meta["year"],
                    "split": meta.get("split", ""),
                    "candidate_idx": cand_idx,
                    "review": review_obj,
                    "raw_text": raw_text,
                }
                f.write(json.dumps(output_entry) + "\n")
                processed += 1

    if empty_count:
        print(f"  WARNING: {empty_count} empty/missing responses")
    print(f"  Saved {processed} review entries to: {output_path}")
    return processed


# ============================================================================
# ONLINE MODE (--nobatch)
# ============================================================================

def get_online_client(args: argparse.Namespace):
    """Get genai Client — uses GEMINI_API_KEY if set, otherwise Vertex AI."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or args.api_key
    if api_key:
        print("  Auth: API key")
        return genai.Client(api_key=api_key)
    else:
        from google.genai.types import HttpOptions
        print(f"  Auth: Vertex AI (project={args.project}, location={args.location})")
        return genai.Client(
            vertexai=True,
            project=args.project,
            location=args.location,
            http_options=HttpOptions(api_version="v1"),
        )


async def run_online(
    data: list[dict],
    args: argparse.Namespace,
    is_vision: bool,
    reviews_lookup: dict[str, str] | None,
) -> int:
    """Run online mode — direct API calls with rate-limited concurrency."""
    from tqdm.asyncio import tqdm as atqdm

    client = get_online_client(args)

    from google.genai.types import GenerateContentConfig

    thinking_config = None
    if args.thinking_budget is not None:
        from google.genai.types import ThinkingConfig
        thinking_config = ThinkingConfig(thinking_budget=args.thinking_budget)

    gen_config = GenerateContentConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        candidate_count=args.candidate_count,
        response_mime_type="application/json",
        response_schema=ReviewOutput,
        thinking_config=thinking_config,
    )

    semaphore = asyncio.Semaphore(args.qps)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed = 0
    errors = 0
    results = []

    async def process_one(entry: dict, idx: int) -> list[dict]:
        """Process a single entry with retries."""
        prompt_text = build_prompt_text(entry, args.mode, is_vision, reviews_lookup)
        sid = extract_submission_id(entry)
        year = extract_year(entry)
        split = entry.get("_metadata", {}).get("_split", "")

        # Build contents for the SDK
        images = entry.get("images", [])
        if images:
            # Vision: build multimodal content with local image files
            from google.genai.types import Content, Part
            import re as _re

            parts = []
            segments = _re.split(r"<image>", prompt_text)
            img_idx = 0
            for seg_i, segment in enumerate(segments):
                if segment.strip():
                    parts.append(Part.from_text(text=segment))
                if seg_i < len(segments) - 1 and img_idx < len(images):
                    img_path = images[img_idx]
                    mime = get_mime_type(img_path)
                    with open(img_path, "rb") as img_f:
                        img_bytes = img_f.read()
                    parts.append(Part.from_bytes(data=img_bytes, mime_type=mime))
                    img_idx += 1
            # Append any remaining images
            while img_idx < len(images):
                img_path = images[img_idx]
                mime = get_mime_type(img_path)
                with open(img_path, "rb") as img_f:
                    img_bytes = img_f.read()
                parts.append(Part.from_bytes(data=img_bytes, mime_type=mime))
                img_idx += 1
            contents = Content(role="user", parts=parts)
        else:
            contents = prompt_text

        max_retries = 3
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    response = await client.aio.models.generate_content(
                        model=args.model,
                        contents=contents,
                        config=gen_config,
                    )

                    entries = []
                    for cand_idx, candidate in enumerate(response.candidates):
                        raw_text = ""
                        if candidate.content and candidate.content.parts:
                            raw_text = candidate.content.parts[0].text or ""
                        review_obj = None
                        if raw_text:
                            try:
                                review_obj = json.loads(raw_text)
                            except json.JSONDecodeError:
                                pass

                        entries.append({
                            "submission_id": sid,
                            "year": year,
                            "split": split,
                            "candidate_idx": cand_idx,
                            "review": review_obj,
                            "raw_text": raw_text,
                        })
                    return entries

                except Exception as e:
                    error_str = str(e)
                    # Retry on transient errors
                    if attempt < max_retries - 1 and any(
                        code in error_str for code in ("429", "503", "RESOURCE_EXHAUSTED")
                    ):
                        wait = 2 ** (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    # Log and return empty on final failure
                    print(f"\n  ERROR [{sid}]: {error_str[:200]}")
                    return []

        return []

    # Run all tasks with progress bar
    tasks = [process_one(entry, i) for i, entry in enumerate(data)]
    all_entries = []

    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating reviews"):
        result = await coro
        if result:
            all_entries.extend(result)
            processed += 1
        else:
            errors += 1

    # Write all results
    with open(args.output, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n  Processed: {processed}, Errors: {errors}")
    print(f"  Saved {len(all_entries)} review entries to: {args.output}")
    return len(all_entries)


# ============================================================================
# DIVERSITY ANALYSIS
# ============================================================================

def analyze_diversity(output_path: str):
    """Analyze diversity across candidates for the same paper."""
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS")
    print("=" * 70)

    # Group by submission_id
    papers: dict[str, list[dict]] = {}
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sid = entry["submission_id"]
            papers.setdefault(sid, []).append(entry)

    multi_candidate_papers = {k: v for k, v in papers.items() if len(v) > 1}

    if not multi_candidate_papers:
        print("  No multi-candidate papers found. Use --candidate_count > 1.")
        return

    print(f"  Papers with multiple candidates: {len(multi_candidate_papers)}")

    jaccard_scores = []
    overlap_scores = []

    for sid, candidates in multi_candidate_papers.items():
        print(f"\n  Paper: {sid} ({len(candidates)} candidates)")

        # Extract keyword sets for strengths/weaknesses
        keyword_sets = []
        raw_texts = []
        for cand in candidates:
            review = cand.get("review") or {}
            words = set()
            for s in review.get("strengths", []):
                words.update(w.lower() for w in s.split() if len(w) > 3)
            for w in review.get("weaknesses", []):
                words.update(w.lower() for w in w.split() if len(w) > 3)
            keyword_sets.append(words)
            raw_texts.append(cand.get("raw_text", ""))

        # Pairwise Jaccard similarity
        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                a, b = keyword_sets[i], keyword_sets[j]
                if a | b:
                    jaccard = len(a & b) / len(a | b)
                else:
                    jaccard = 1.0
                jaccard_scores.append(jaccard)
                print(f"    Candidates {i} vs {j}: Jaccard={jaccard:.3f}")

        # Text overlap ratio (simple character-level)
        for i in range(len(raw_texts)):
            for j in range(i + 1, len(raw_texts)):
                a, b = raw_texts[i], raw_texts[j]
                if not a or not b:
                    continue
                common = sum(1 for c in a if c in b)
                ratio = common / max(len(a), len(b))
                overlap_scores.append(ratio)

        # Show side-by-side preview of strengths
        print("    --- Strength previews ---")
        for cand_idx, cand in enumerate(candidates):
            review = cand.get("review") or {}
            strengths = review.get("strengths", [])
            preview = strengths[0][:120] + "..." if strengths else "(none)"
            print(f"    Candidate {cand_idx}: {preview}")

    # Summary
    print(f"\n  === Summary ===")
    if jaccard_scores:
        avg_j = sum(jaccard_scores) / len(jaccard_scores)
        print(f"  Avg Jaccard similarity (keywords): {avg_j:.3f}")
        if avg_j < 0.3:
            print(f"  → Candidates are genuinely diverse (low keyword overlap)")
        elif avg_j < 0.6:
            print(f"  → Candidates are moderately diverse")
        else:
            print(f"  → Candidates are largely similar (high keyword overlap)")
    if overlap_scores:
        avg_o = sum(overlap_scores) / len(overlap_scores)
        print(f"  Avg text overlap ratio: {avg_o:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch Review Generation for Data Augmentation"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (without _train/_test/_validation suffix)",
    )
    parser.add_argument(
        "--split", type=str, nargs="+", default=["train", "test"],
        help="Dataset split(s) to process (default: train test)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Data directory (default: data)",
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, default="blind",
        choices=["blind", "informed"],
        help="Generation mode: blind (paper only) or informed (paper + human reviews)",
    )
    parser.add_argument(
        "--metadata_dir", type=str, default="data/massive_metadata_v7_5",
        help="Path to massive_metadata Arrow dataset (for informed mode)",
    )

    # Auth / GCP arguments
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--project", type=str, default="project-ab4d0323-a7e8-4f58-93e", help="GCP project ID")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="GCP region")
    parser.add_argument(
        "--gcs_base", type=str, default=DEFAULT_GCS_IMAGES_BASE,
        help="GCS base URI for images",
    )
    parser.add_argument(
        "--gcs_staging", type=str, default=None,
        help="GCS URI for staging batch input/output",
    )

    # Model arguments
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max output tokens (default: 2048)")
    parser.add_argument("--candidate_count", type=int, default=1, help="Number of candidates per request (default: 1)")
    parser.add_argument("--thinking_budget", type=int, default=512, help="Thinking budget tokens (default: 512, set to 0 to disable)")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")

    # Control
    parser.add_argument("--dry_run", type=int, default=None, help="Test with N samples")
    parser.add_argument("--submit_only", action="store_true", help="Submit batch job and exit without waiting")
    parser.add_argument("--poll_interval", type=int, default=300, help="Polling interval in seconds (default: 300)")
    parser.add_argument("--nobatch", action="store_true", help="Online mode: direct API calls instead of batch")
    parser.add_argument("--qps", type=int, default=10, help="Max concurrent requests in online mode (default: 10)")
    parser.add_argument("--analyze_diversity", action="store_true", help="Analyze candidate diversity after generation")

    # Sharding — split data across parallel batch jobs
    parser.add_argument("--num_shards", type=int, default=1, help="Number of parallel batch jobs to submit (default: 1 = no sharding)")

    # Merge / extract
    parser.add_argument("--merge", type=str, nargs="+", default=None,
                        help="Merge multiple result JSONL files (shard outputs + extracted partial) into one. "
                             "Usage: --merge shard0.jsonl shard1.jsonl ... -o merged.jsonl")
    parser.add_argument("--extract_partial", action="store_true",
                        help="Extract completed results from --partial_predictions into output JSONL format, then exit.")

    # Retry from failed SID list
    parser.add_argument("--retry_file", type=str, default=None,
                        help="File with submission_ids to retry (one per line). "
                             "Only papers matching these IDs will be processed.")

    # Resume from partial results
    parser.add_argument("--partial_predictions", type=str, default=None,
                        help="Path to partial predictions JSONL from a cancelled batch job. "
                             "Skips submission_ids that already have successful responses.")
    parser.add_argument("--partial_metadata", type=str, default=None,
                        help="Path to metadata JSON matching the partial predictions (maps line index to submission_id). "
                             "Required with --partial_predictions.")

    args = parser.parse_args()

    # ---- Merge mode: combine multiple JSONL files and exit ----
    if args.merge:
        output = args.output or "results/reviews/merged.jsonl"
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        total = 0
        seen_ids = set()
        with open(output, "w") as out_f:
            for fpath in args.merge:
                count = 0
                with open(fpath) as in_f:
                    for line in in_f:
                        entry = json.loads(line)
                        key = (entry.get("submission_id", ""), entry.get("candidate_idx", 0))
                        if key not in seen_ids:
                            seen_ids.add(key)
                            out_f.write(line)
                            count += 1
                print(f"  {fpath}: {count} entries")
                total += count
        print(f"\nMerged {total} unique entries → {output}")
        return

    # ---- Extract partial: convert raw predictions to output format and exit ----
    if args.extract_partial:
        if not args.partial_predictions or not args.partial_metadata:
            parser.error("--extract_partial requires --partial_predictions and --partial_metadata")
        output = args.output or "results/reviews/extracted_partial.jsonl"
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        with open(args.partial_metadata) as f:
            metadata_list = json.load(f)

        extracted = 0
        with open(output, "w") as out_f, open(args.partial_predictions) as in_f:
            for idx, line in enumerate(in_f):
                if idx >= len(metadata_list):
                    break
                d = json.loads(line)
                response = d.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    continue
                meta = metadata_list[idx]
                for cand_idx, candidate in enumerate(candidates):
                    try:
                        text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            review_obj = json.loads(text)
                            out_f.write(json.dumps({
                                "submission_id": meta["submission_id"],
                                "year": meta["year"],
                                "split": meta.get("split", ""),
                                "candidate_idx": cand_idx,
                                "review": review_obj,
                                "raw_text": text,
                            }) + "\n")
                            extracted += 1
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        pass

        print(f"Extracted {extracted} entries from partial predictions → {output}")
        return

    # Defaults
    if args.gcs_staging is None:
        args.gcs_staging = "gs://autoreviewer-data/batch_staging"
    splits_str = "+".join(args.split)
    if args.output is None:
        args.output = f"results/reviews/{args.mode}_{get_short_name(args.dataset)}_{splits_str}.jsonl"

    vision = is_vision_dataset(args.dataset)

    # Print configuration
    print("=" * 70)
    print("Gemini Batch Review Generation")
    print("=" * 70)
    print(f"Dataset:         {args.dataset}")
    print(f"Splits:          {args.split}")
    print(f"Mode:            {args.mode}")
    print(f"Vision:          {vision}")
    print(f"Model:           {args.model}")
    print(f"Temperature:     {args.temperature}")
    print(f"Max Tokens:      {args.max_tokens}")
    print(f"Candidates:      {args.candidate_count}")
    print(f"Thinking Budget: {args.thinking_budget}")
    print(f"Batch:           {'no (online)' if args.nobatch else 'yes'}")
    print(f"Output:          {args.output}")
    if args.mode == "informed":
        print(f"Metadata:        {args.metadata_dir}")
    if args.retry_file:
        print(f"Retry file:      {args.retry_file}")
    if args.partial_predictions:
        print(f"Partial:         {args.partial_predictions}")
    if args.num_shards > 1:
        print(f"Shards:          {args.num_shards}")
    if args.dry_run:
        print(f"DRY RUN:         {args.dry_run} samples")
    if args.submit_only:
        print(f"SUBMIT ONLY:     will not wait for results")
    print()

    # Load dataset(s) — concatenate all requested splits
    data = []
    for split in args.split:
        split_data = load_dataset(Path(args.data_dir), args.dataset, split)
        # Tag each entry with its split
        for entry in split_data:
            entry.setdefault("_metadata", {})["_split"] = split
        data.extend(split_data)
    print(f"  Total across splits: {len(data)} samples\n")

    if args.dry_run:
        data = data[: args.dry_run]
        print(f"Using {len(data)} samples for dry run\n")

    # Filter to only retry-file submission IDs
    if args.retry_file:
        with open(args.retry_file) as f:
            retry_sids = {line.strip() for line in f if line.strip()}
        before = len(data)
        data = [e for e in data if extract_submission_id(e) in retry_sids]
        print(f"  Retry filter: {before} → {len(data)} papers (from {len(retry_sids)} retry IDs)\n")

    # Filter out already-completed submissions from partial predictions
    if args.partial_predictions:
        if not args.partial_metadata:
            parser.error("--partial_predictions requires --partial_metadata")
        print(f"Loading partial predictions: {args.partial_predictions}")
        completed_ids = load_completed_ids(args.partial_predictions, args.partial_metadata)
        before = len(data)
        data = [e for e in data if extract_submission_id(e) not in completed_ids]
        print(f"  Filtered: {before} → {len(data)} (skipped {before - len(data)} already completed)\n")

    # Load human reviews for informed mode
    reviews_lookup = None
    if args.mode == "informed":
        reviews_lookup = load_metadata_reviews(args.metadata_dir)
        # Report coverage
        sids = {extract_submission_id(e) for e in data}
        covered = sids & set(reviews_lookup.keys())
        print(f"  Review coverage: {len(covered)}/{len(sids)} ({100*len(covered)/len(sids):.1f}%)\n")

    # ---- Online mode ----
    if args.nobatch:
        print("Running in ONLINE mode (direct API calls)...\n")
        count = asyncio.run(run_online(data, args, vision, reviews_lookup))

        if args.analyze_diversity and os.path.exists(args.output):
            analyze_diversity(args.output)

        print("\n" + "=" * 70)
        print("COMPLETED (online mode)")
        print("=" * 70)
        print(f"Results: {args.output}")
        print(f"Entries: {count}")
        return

    # ---- Batch mode ----
    generation_config = build_generation_config(args)
    client = get_client(args.project, args.location)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    short_name = f"review_{args.mode}_{get_short_name(args.dataset)}"
    num_shards = args.num_shards

    # Split data into shards
    shards = []
    if num_shards <= 1:
        shards = [data]
    else:
        shard_size = len(data) // num_shards
        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else len(data)
            shards.append(data[start:end])

    print(f"Submitting {len(shards)} batch job(s)...\n")

    # Submit each shard as a separate batch job
    all_jobs = []
    for shard_idx, shard_data in enumerate(shards):
        shard_tag = f"_shard{shard_idx}" if num_shards > 1 else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        local_jsonl = f"/tmp/batch_review_{timestamp}{shard_tag}.jsonl"
        _, metadata_list = create_batch_jsonl(
            shard_data, local_jsonl, args.mode, vision, args.gcs_base,
            generation_config, reviews_lookup,
        )

        input_uri = f"{args.gcs_staging}/{short_name}_{timestamp}{shard_tag}_input.jsonl"
        output_uri = f"{args.gcs_staging}/{short_name}_{timestamp}{shard_tag}_output/"

        if num_shards > 1:
            print(f"\n--- Shard {shard_idx}/{num_shards} ({len(shard_data)} samples) ---")

        print(f"Uploading batch input to GCS...")
        upload_to_gcs(local_jsonl, input_uri)

        display_name = f"{short_name}_{timestamp}{shard_tag}"
        job = submit_batch_job(client, args.model, input_uri, output_uri, display_name=display_name)

        # Determine output paths for this shard
        base, ext = os.path.splitext(args.output)
        shard_output = f"{base}{shard_tag}{ext}"
        job_info_path = shard_output.replace(".jsonl", "_job_info.json")
        metadata_path = shard_output.replace(".jsonl", "_metadata.json")

        job_info = {
            "job_name": job.name,
            "model": args.model,
            "dataset": args.dataset,
            "splits": args.split,
            "mode": args.mode,
            "candidate_count": args.candidate_count,
            "temperature": args.temperature,
            "num_samples": len(shard_data),
            "shard_idx": shard_idx if num_shards > 1 else None,
            "num_shards": num_shards if num_shards > 1 else None,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "submitted_at": timestamp,
        }
        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)
        print(f"Job info saved: {job_info_path}")

        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f)
        print(f"Metadata saved: {metadata_path}")

        all_jobs.append({
            "shard_idx": shard_idx,
            "job": job,
            "job_info": job_info,
            "job_info_path": job_info_path,
            "metadata_list": metadata_list,
            "output_uri": output_uri,
            "shard_output": shard_output,
        })

        # Clean up local temp
        if os.path.exists(local_jsonl):
            os.remove(local_jsonl)

    if args.submit_only:
        print("\n" + "=" * 70)
        print(f"Submitted {len(all_jobs)} batch job(s). To check status:")
        for j in all_jobs:
            tag = f" (shard {j['shard_idx']})" if num_shards > 1 else ""
            print(f"  {j['job'].name}{tag}")
        if num_shards > 1:
            print(f"\nAfter all complete, merge with:")
            base, ext = os.path.splitext(args.output)
            print(f"  cat {base}_shard*{ext} > {base}_merged{ext}")
        print("=" * 70)
        return

    # Poll all jobs until complete (sequential — one at a time)
    for j in all_jobs:
        tag = f" (shard {j['shard_idx']})" if num_shards > 1 else ""
        print(f"\nWaiting for job{tag}: {j['job'].name}")
        try:
            poll_until_complete(client, j["job"].name, interval=args.poll_interval)
        except RuntimeError as e:
            print(f"\nError on job{tag}: {e}")
            continue

        num_processed = parse_batch_results(
            j["output_uri"], j["metadata_list"], j["shard_output"], args.candidate_count
        )

        j["job_info"]["status"] = "completed"
        j["job_info"]["completed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        j["job_info"]["num_results"] = num_processed
        with open(j["job_info_path"], "w") as f:
            json.dump(j["job_info"], f, indent=2)

        print(f"  Shard {j['shard_idx']}: {num_processed} results → {j['shard_output']}")

    if args.analyze_diversity:
        if num_shards <= 1 and os.path.exists(args.output):
            analyze_diversity(args.output)

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)
    print(f"Results: {args.output}")
    print(f"Processed: {num_processed} / {len(data)}")

    # Cleanup
    if os.path.exists(local_jsonl):
        os.remove(local_jsonl)


if __name__ == "__main__":
    main()
