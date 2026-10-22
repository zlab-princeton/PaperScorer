#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate a checkpoint on training data to compute accuracy metrics.
Used for offline training metrics when using fused CE loss during training.

Usage:
    python scripts/eval_training_ckpt.py \
        --model_name_or_path saves/exp/checkpoint-1000 \
        --dataset dataset_train \
        --template qwen2_vl \
        --cutoff_len 24480 \
        --save_name results/exp/train-ckpt-1000.json \
        --sft_accuracy_format boxed \
        --sft_positive_token Accept \
        --sft_negative_token Reject
"""

import os
# Disable wandb before any imports
os.environ["WANDB_DISABLED"] = "true"

import json
import time
from pathlib import Path
from typing import Literal, Optional

import fire
import torch

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments, TrainingArguments
from llamafactory.model import load_model, load_tokenizer


def count_dataset_samples(dataset_name: str, dataset_dir: str = "data") -> int:
    """Count number of samples in dataset by reading the JSON file directly."""
    # Try common file patterns
    for pattern in [
        os.path.join(dataset_dir, dataset_name, "train.json"),
        os.path.join(dataset_dir, dataset_name, "data.json"),
        os.path.join(dataset_dir, f"{dataset_name}.json"),
    ]:
        if os.path.exists(pattern):
            with open(pattern, "r") as f:
                data = json.load(f)
                return len(data)

    # Fallback: try to count lines in jsonl
    for pattern in [
        os.path.join(dataset_dir, dataset_name, "train.jsonl"),
        os.path.join(dataset_dir, dataset_name, "data.jsonl"),
        os.path.join(dataset_dir, f"{dataset_name}.jsonl"),
    ]:
        if os.path.exists(pattern):
            with open(pattern, "r") as f:
                return sum(1 for _ in f)

    return -1  # Unknown


def eval_training_ckpt(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    save_name: str = "train_metrics.json",
    sft_accuracy_format: Literal["boxed", "yesno"] = "boxed",
    sft_positive_token: str = "Accept",
    sft_negative_token: str = "Reject",
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    per_device_eval_batch_size: int = 2,
    preprocessing_num_workers: int = 16,
    test_dataset: str = None,
    bf16: bool = True,
    max_samples: int = None,  # Limit number of samples for testing
):
    """Evaluate checkpoint on training data and save accuracy metrics."""
    # HARDCODED OVERRIDE: Force conservative values to avoid OOM
    # TODO: Remove this override once all queued jobs have completed
    per_device_eval_batch_size = 2
    if max_samples is None or max_samples > 2000:
        max_samples = 2000

    print("=" * 70)
    print("Eval Training Checkpoint")
    print("=" * 70)
    print(f"Model: {model_name_or_path}")
    print(f"Dataset: {dataset}")
    print(f"Template: {template}")
    print(f"Cutoff len: {cutoff_len}")
    print(f"Save name: {save_name}")
    print(f"Accuracy format: {sft_accuracy_format}")
    print(f"Positive token: {sft_positive_token}")
    print(f"Negative token: {sft_negative_token}")
    print(f"Batch size: {per_device_eval_batch_size}")
    print("=" * 70)

    start_time = time.time()

    # Build model args
    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
        use_kv_cache=False,  # Not doing generation, just forward pass for logits
        enable_liger_kernel=True,  # Fused ops reduce memory during inference
    )

    # Set compute dtype for bf16
    if bf16:
        model_args.compute_dtype = torch.bfloat16

    # Set device map for single GPU eval
    model_args.device_map = "auto"
    model_args.model_max_length = cutoff_len

    # Build data args - use eval_dataset to point to our train data
    data_args = DataArguments(
        dataset=None,  # No train dataset
        eval_dataset=dataset,  # Our "train" data goes here for eval
        dataset_dir=dataset_dir,
        template=template,
        cutoff_len=cutoff_len,
        preprocessing_num_workers=preprocessing_num_workers,
        max_samples=max_samples,  # Limit samples for testing
    )

    # Build finetuning args with accuracy tracking enabled
    finetuning_args = FinetuningArguments(
        stage="sft",
        finetuning_type="full",  # Not actually finetuning, just eval
        sft_train_accuracy=True,
        sft_train_accuracy_format=sft_accuracy_format,
        sft_positive_token=sft_positive_token,
        sft_negative_token=sft_negative_token,
    )

    # Build training args for eval only (use LlamaFactory's TrainingArguments for fp8 support)
    training_args = TrainingArguments(
        output_dir="dummy_eval_dir",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=per_device_eval_batch_size,
        bf16=bf16,
        remove_unused_columns=False,  # Important for multimodal
        dataloader_num_workers=preprocessing_num_workers,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Load tokenizer and template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load model (is_trainable=False for eval)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)

    # Load dataset
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "sft", **tokenizer_module)

    # Get the eval dataset (our train data)
    eval_dataset = dataset_module.get("eval_dataset")

    if eval_dataset is None:
        raise ValueError("No dataset loaded. Check dataset configuration.")

    num_samples = len(eval_dataset)
    print(f"Loaded {num_samples} samples for evaluation")

    # Import trainer here to avoid circular imports
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    # Create data collator for proper batching
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template_obj,
        model=None,  # Not needed for eval without generate
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
    )

    # Create trainer with accuracy tracking enabled
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        processor=tokenizer_module.get("processor"),
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Run evaluation
    print(f"\nEvaluating {model_name_or_path} on {dataset}...")
    eval_results = trainer.evaluate()

    eval_time = time.time() - start_time

    # Extract epoch/step from checkpoint
    epoch, step = extract_epoch_step(model_name_or_path)

    # Build output metrics
    metrics = {
        "epoch": epoch,
        "step": step,
        "train_ds": dataset,
        "test_ds": test_dataset,
        "num_samples": num_samples,
        "eval_runtime_seconds": round(eval_time, 2),
        "eval_samples_per_second": round(num_samples / eval_time, 2) if eval_time > 0 and num_samples > 0 else 0,
    }

    # Add eval metrics from trainer
    for key, value in eval_results.items():
        # Rename eval_ prefix metrics to match expected format
        if key.startswith("eval_"):
            # Keep the eval_loss as is
            metrics[key] = value
            # Also add sft_ metrics without eval_ prefix for convenience
            short_key = key[5:]  # Remove "eval_" prefix
            if short_key.startswith("sft_"):
                metrics[short_key] = value
        else:
            metrics[key] = value

    # Save metrics
    Path(save_name).parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved training metrics to {save_name}")
    print("=" * 70)
    print(json.dumps(metrics, indent=2))
    print("=" * 70)

    return metrics


def extract_epoch_step(ckpt_path: str) -> tuple[float, int]:
    """Extract epoch and step from checkpoint's trainer_state.json."""
    state_file = Path(ckpt_path) / "trainer_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        return state.get("epoch", 0), state.get("global_step", 0)

    # Fallback: parse step from path
    ckpt_name = Path(ckpt_path).name
    if "checkpoint-" in ckpt_name:
        try:
            step = int(ckpt_name.replace("checkpoint-", ""))
            return 0.0, step
        except ValueError:
            pass

    return 0.0, 0


if __name__ == "__main__":
    fire.Fire(eval_training_ckpt)
