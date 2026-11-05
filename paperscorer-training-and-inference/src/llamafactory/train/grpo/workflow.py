# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/main/examples/scripts/grpo.py
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

import json
import os
from typing import TYPE_CHECKING, Any, Optional

from datasets import Dataset

from ...extras import logging
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .rewards import extract_boxed_answer, get_reward_funcs
from .trainer import CustomGRPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def load_grpo_dataset(
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
    data_dir: str,
) -> Dataset:
    """Load dataset in format suitable for TRL's GRPOTrainer.

    TRL's GRPOTrainer expects:
    - "prompt": List of messages in chat format [{"role": "user", "content": "..."}]
    - Other columns (like "answer") are passed to reward functions as kwargs

    Args:
        data_args: Data arguments containing dataset name
        data_dir: Base directory for datasets

    Returns:
        HuggingFace Dataset with "prompt" and "ground_truth" columns
    """
    # Get dataset name from data_args
    dataset_names = data_args.dataset
    if isinstance(dataset_names, str):
        dataset_name = dataset_names.split(",")[0].strip()
    else:
        dataset_name = dataset_names[0] if dataset_names else None

    if not dataset_name:
        raise ValueError("No dataset specified for GRPO training")

    # Load dataset_info.json to get file path
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if dataset_name not in dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_info.json")

    # Get file path
    file_name = dataset_info[dataset_name].get("file_name", f"{dataset_name}.json")
    file_path = os.path.join(data_dir, file_name)

    logger.info_rank0(f"Loading GRPO dataset from: {file_path}")

    # Load raw data
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert to TRL format
    processed_data = []
    for item in raw_data:
        # Extract prompt from conversations (exclude assistant response)
        # Extract ground truth from assistant's \boxed{} response
        conversations = item.get("conversations", [])
        prompt_messages = []
        ground_truth = ""

        for msg in conversations:
            role = msg.get("from", "")
            content = msg.get("value", "")

            # Map roles: system -> system, human -> user, gpt -> assistant
            if role == "system":
                prompt_messages.append({"role": "system", "content": content})
            elif role == "human":
                content_to_add = content
                if finetuning_args.grpo_append_reasoning_explicit:
                    content_to_add += "\n - Reason about the outcome before providing your boxed answer"
                prompt_messages.append({"role": "user", "content": content_to_add})
            elif role == "gpt":
                # Extract ground truth from \boxed{} in assistant response
                ground_truth = extract_boxed_answer(content) or ""

        if not prompt_messages:
            logger.warning_rank0(f"Skipping item with no prompt messages")
            continue

        if not ground_truth:
            logger.warning_rank0(f"Skipping item with no ground truth in \\boxed{{}}")
            continue

        processed_data.append({
            "prompt": prompt_messages,
            "ground_truth": ground_truth,
        })

    logger.info_rank0(f"Loaded {len(processed_data)} examples for GRPO training")

    return Dataset.from_list(processed_data)


def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    r"""Run GRPO (Group Relative Policy Optimization) training.

    GRPO is an online RL algorithm that optimizes the policy using group-relative
    advantages computed from multiple completions per prompt. Unlike PPO, GRPO does
    not require a value head or separate reward model - it uses custom reward functions.
    """
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # Load dataset in TRL format (NOT using LlamaFactory's data processing)
    # TRL's GRPOTrainer expects raw text prompts, not tokenized data
    data_dir = data_args.dataset_dir or "data"
    train_dataset = load_grpo_dataset(data_args, finetuning_args, data_dir)

    # Load model (NO value head needed for GRPO, unlike PPO)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=False)

    # Set padding side to left for generation
    tokenizer.padding_side = "left"

    # Get reward functions from registry
    reward_funcs, reward_weights = get_reward_funcs(
        finetuning_args.grpo_reward_funcs,
        finetuning_args.grpo_reward_weights,
    )

    # Initialize GRPO trainer
    grpo_trainer = CustomGRPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        model=model,
        reward_funcs=reward_funcs,
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
        train_dataset=train_dataset,
        eval_dataset=None,  # TODO: support eval dataset
        callbacks=callbacks,
        reward_weights=reward_weights,
    )

    # Training
    if training_args.do_train:
        train_result = grpo_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        grpo_trainer.save_model()
        grpo_trainer.log_metrics("train", train_result.metrics)
        grpo_trainer.save_metrics("train", train_result.metrics)
        grpo_trainer.save_state()

        if grpo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward"])

    # Create model card
    create_modelcard_and_push(grpo_trainer, model_args, data_args, training_args, finetuning_args)
