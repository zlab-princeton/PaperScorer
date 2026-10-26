# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
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

import warnings
from types import MethodType
from typing import TYPE_CHECKING, Callable, Optional, Union

from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer
from typing_extensions import override

from ...extras import logging
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomGRPOTrainer(GRPOTrainer):
    r"""Custom GRPO trainer that wraps TRL's GRPOTrainer with LlamaFactory patterns."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        model: Union["PreTrainedModel", str],
        reward_funcs: Union[Callable, list[Callable]],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        train_dataset: "Dataset",
        eval_dataset: Optional["Dataset"] = None,
        callbacks: Optional[list] = None,
        reward_weights: Optional[list[float]] = None,
    ) -> None:
        # Build GRPOConfig from finetuning_args and training_args
        grpo_config = GRPOConfig(
            # GRPO-specific parameters (TRL 0.18+ compatible)
            beta=finetuning_args.grpo_beta,
            num_iterations=finetuning_args.grpo_num_iterations,
            epsilon=finetuning_args.grpo_epsilon,
            scale_rewards=finetuning_args.grpo_scale_rewards,
            loss_type=finetuning_args.grpo_loss_type,
            temperature=finetuning_args.grpo_temperature,
            max_completion_length=finetuning_args.grpo_max_new_tokens,
            num_generations=finetuning_args.grpo_num_generations,
            # vLLM settings (TRL 0.18+ compatible)
            use_vllm=finetuning_args.grpo_use_vllm,
            vllm_mode=finetuning_args.grpo_vllm_mode,
            vllm_gpu_memory_utilization=finetuning_args.grpo_vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=finetuning_args.grpo_vllm_tensor_parallel_size,
            # Training arguments
            output_dir=training_args.output_dir,
            learning_rate=training_args.learning_rate,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            num_train_epochs=training_args.num_train_epochs,
            max_steps=training_args.max_steps,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            bf16=training_args.bf16,
            fp16=training_args.fp16,
            logging_steps=training_args.logging_steps,
            save_steps=training_args.save_steps,
            eval_steps=training_args.eval_steps if training_args.eval_steps else training_args.save_steps,
            warmup_ratio=training_args.warmup_ratio,
            warmup_steps=training_args.warmup_steps,
            lr_scheduler_type=training_args.lr_scheduler_type,
            weight_decay=training_args.weight_decay,
            report_to=training_args.report_to[0] if training_args.report_to else None,
            logging_dir=training_args.logging_dir,
            # Reward weights
            reward_weights=reward_weights,
        )

        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args

        # Initialize GRPOTrainer
        GRPOTrainer.__init__(
            self,
            model=model,
            args=grpo_config,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        warnings.simplefilter("ignore")  # suppress warnings

        # Add processor callback if needed
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # Add BAdam callback if enabled
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
