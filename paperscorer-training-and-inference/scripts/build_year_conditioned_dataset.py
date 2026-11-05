#!/usr/bin/env python3
"""Build year-conditioned datasets by injecting submission year into the prompt.

Reads existing datasets and modifies the human message to include the
submission year and correct conference name from each paper's metadata.

Example usage:
    python scripts/build_year_conditioned_dataset.py \
        --source-datasets \
            iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
            iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480 \
        --extra-eval-datasets \
            all_conferences_text_v7_extra_test \
            all_conferences_vision_v7_extra_test \
        --output-suffix _yearcond
"""

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONFERENCE_MAP = {
    "iclr": "ICLR",
    "nips": "NeurIPS",
    "ICML": "ICML",
    "icml": "ICML",
    "COLM": "COLM",
    "colm": "COLM",
}

# The exact string we expect in the original prompt.
# We match a flexible conference name so it works even if the source already
# mentions a specific venue (though typically it says "ICLR").
PROMPT_PATTERN = re.compile(
    r"I am giving you a paper\. I want to predict its acceptance outcome at ([A-Za-z]+)\."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _sanitize_surrogates(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(v) for v in obj]
    return obj


def load_source_data(dataset_name, split):
    path = DATA_DIR / f"{dataset_name}_{split}" / "data.json"
    with open(path) as f:
        return json.load(f)


def load_extra_eval_data(dataset_name):
    """Load an extra eval dataset that already has the split suffix in its name."""
    path = DATA_DIR / dataset_name / "data.json"
    with open(path) as f:
        return json.load(f)


def write_dataset(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _sanitize_surrogates(data)
    with open(output_dir / "data.json", "w") as f:
        json.dump(data, f, indent=None, ensure_ascii=False)


def update_dataset_info(new_entries):
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info.update(new_entries)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_dataset_info_entry(dir_name, is_vision):
    entry = {
        "file_name": f"{dir_name}/data.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
    if is_vision:
        entry["columns"]["images"] = "images"
    return entry


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def is_vision_dataset(name):
    """Heuristic: dataset is vision if 'vision' appears in its name."""
    return "vision" in name


def condition_entry(entry):
    """Modify a single dataset entry to include the year in the prompt."""
    metadata = entry.get("_metadata", {})
    year = metadata.get("year")
    conference_raw = metadata.get("conference", "")
    conference = CONFERENCE_MAP.get(conference_raw, conference_raw.upper())

    if year is None:
        raise ValueError(
            f"Entry missing _metadata.year: submission_id={metadata.get('submission_id', '?')}"
        )

    conversations = entry["conversations"]
    modified = False
    for msg in conversations:
        if msg["from"] == "human":
            original = msg["value"]
            m = PROMPT_PATTERN.search(original)
            if m:
                old_str = m.group(0)
                new_str = (
                    f"I am giving you a paper. I want to predict its acceptance "
                    f"outcome at {conference} {year}."
                )
                msg["value"] = original.replace(old_str, new_str, 1)
                modified = True
                break

    if not modified:
        sid = metadata.get("submission_id", "?")
        print(f"  WARNING: Could not find prompt pattern in entry {sid}, skipping modification")

    return entry


def process_dataset(dataset_name, split, suffix):
    """Process a single split of a source dataset."""
    data = load_source_data(dataset_name, split)
    output_name = f"{dataset_name}{suffix}_{split}"
    print(f"  Processing {dataset_name}_{split} -> {output_name} ({len(data)} entries)")

    for entry in data:
        condition_entry(entry)

    output_dir = DATA_DIR / output_name
    write_dataset(data, output_dir)
    return output_name


def process_extra_eval(dataset_name, suffix):
    """Process an extra eval dataset (single split, name already includes _test)."""
    data = load_extra_eval_data(dataset_name)
    # Insert suffix before the last component (e.g., before _test)
    # all_conferences_text_v7_extra_test -> all_conferences_text_v7_extra_yearcond_test
    parts = dataset_name.rsplit("_", 1)  # split off "_test"
    if len(parts) == 2 and parts[1] == "test":
        output_name = f"{parts[0]}{suffix}_{parts[1]}"
    else:
        output_name = f"{dataset_name}{suffix}"

    print(f"  Processing {dataset_name} -> {output_name} ({len(data)} entries)")

    for entry in data:
        condition_entry(entry)

    output_dir = DATA_DIR / output_name
    write_dataset(data, output_dir)
    return output_name


def main():
    parser = argparse.ArgumentParser(
        description="Build year-conditioned datasets by injecting year into prompts."
    )
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        default=[],
        help="Base names of source datasets (without _train/_test/_validation suffix).",
    )
    parser.add_argument(
        "--extra-eval-datasets",
        nargs="+",
        default=[],
        help="Extra eval dataset names (already include split suffix, e.g. ..._extra_test).",
    )
    parser.add_argument(
        "--output-suffix",
        required=True,
        help="Suffix to append to output dataset names (e.g. _yearcond).",
    )
    args = parser.parse_args()

    splits = ["train", "validation", "test"]
    new_info_entries = {}

    # Process standard datasets (train/validation/test splits)
    for ds_name in args.source_datasets:
        print(f"\nDataset: {ds_name}")
        vision = is_vision_dataset(ds_name)
        for split in splits:
            output_name = process_dataset(ds_name, split, args.output_suffix)
            new_info_entries[output_name] = make_dataset_info_entry(output_name, vision)

    # Process extra eval datasets (single split)
    for ds_name in args.extra_eval_datasets:
        print(f"\nExtra eval dataset: {ds_name}")
        vision = is_vision_dataset(ds_name)
        output_name = process_extra_eval(ds_name, args.output_suffix)
        new_info_entries[output_name] = make_dataset_info_entry(output_name, vision)

    # Update dataset_info.json
    if new_info_entries:
        print(f"\nRegistering {len(new_info_entries)} new dataset entries in dataset_info.json")
        update_dataset_info(new_info_entries)

    print("\nDone.")


if __name__ == "__main__":
    main()
