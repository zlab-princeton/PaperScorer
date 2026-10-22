# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, _cosine_then_constant_lambda, _cosine_drop_linear_lambda, _exponential_decay_lambda
from .train_accuracy import TrainAccuracyTracker, create_output_format_handler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        if training_args.fp8 and hasattr(self, "accelerator"): # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

        # Initialize train accuracy tracker if enabled
        self._train_accuracy_tracker = None
        if finetuning_args.sft_train_accuracy:
            handler = create_output_format_handler(
                format_type=finetuning_args.sft_train_accuracy_format,
                tokenizer=self.processing_class,
                positive_token=finetuning_args.sft_positive_token,
                negative_token=finetuning_args.sft_negative_token,
            )
            self._train_accuracy_tracker = TrainAccuracyTracker(handler)
            logger.info_rank0(
                f"SFT train accuracy tracking enabled with format '{finetuning_args.sft_train_accuracy_format}', "
                f"positive='{finetuning_args.sft_positive_token}', negative='{finetuning_args.sft_negative_token}'"
            )

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
        scheduler_kwargs = self.args.lr_scheduler_kwargs or {}
        if scheduler_kwargs.get("custom_scheduler") == "cosine_then_constant":
            from torch.optim.lr_scheduler import LambdaLR
            from functools import partial

            num_warmup_steps = int(scheduler_kwargs["num_warmup_steps"])
            num_decay_steps = int(scheduler_kwargs["num_decay_steps"])
            num_total_steps = int(scheduler_kwargs["num_total_steps"])
            min_lr_rate = float(scheduler_kwargs["min_lr_rate"])
            self.lr_scheduler = LambdaLR(
                optimizer if optimizer is not None else self.optimizer,
                partial(
                    _cosine_then_constant_lambda,
                    num_warmup_steps=num_warmup_steps,
                    num_decay_steps=num_decay_steps,
                    num_total_steps=num_total_steps,
                    min_lr_rate=min_lr_rate,
                ),
            )
            return self.lr_scheduler
        if scheduler_kwargs.get("custom_scheduler") == "cosine_drop_linear":
            from torch.optim.lr_scheduler import LambdaLR
            from functools import partial

            num_warmup_steps = int(scheduler_kwargs["num_warmup_steps"])
            num_cosine_steps = int(scheduler_kwargs["num_cosine_steps"])
            num_drop_step = int(scheduler_kwargs["num_drop_step"])
            num_total_steps = int(scheduler_kwargs["num_total_steps"])
            min_lr_rate = float(scheduler_kwargs["min_lr_rate"])
            self.lr_scheduler = LambdaLR(
                optimizer if optimizer is not None else self.optimizer,
                partial(
                    _cosine_drop_linear_lambda,
                    num_warmup_steps=num_warmup_steps,
                    num_cosine_steps=num_cosine_steps,
                    num_drop_step=num_drop_step,
                    num_total_steps=num_total_steps,
                    min_lr_rate=min_lr_rate,
                ),
            )
            return self.lr_scheduler
        if scheduler_kwargs.get("custom_scheduler") == "exponential_decay":
            from torch.optim.lr_scheduler import LambdaLR
            from functools import partial

            num_warmup_steps = int(scheduler_kwargs["num_warmup_steps"])
            num_decay_steps = int(scheduler_kwargs["num_decay_steps"])
            min_lr_rate = float(scheduler_kwargs["min_lr_rate"])
            self.lr_scheduler = LambdaLR(
                optimizer if optimizer is not None else self.optimizer,
                partial(
                    _exponential_decay_lambda,
                    num_warmup_steps=num_warmup_steps,
                    num_decay_steps=num_decay_steps,
                    min_lr_rate=min_lr_rate,
                ),
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union["torch.Tensor", tuple["torch.Tensor", Any]]:
        # Store input_ids and labels BEFORE parent modifies inputs
        input_ids_for_tracking = None
        labels_for_tracking = None
        if self._train_accuracy_tracker is not None:
            input_ids_for_tracking = inputs.get("input_ids")
            labels_for_tracking = inputs.get("labels")

        # Compute loss using parent class
        outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        if isinstance(outputs, tuple):
            loss, model_outputs = outputs
        else:
            loss = outputs
            model_outputs = None

        # Track accuracy if enabled - use memory-efficient sliced logits approach
        if self._train_accuracy_tracker is not None and model_outputs is not None:
            with torch.no_grad():
                logits = model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs.get("logits") if isinstance(model_outputs, dict) else None
                if logits is not None and input_ids_for_tracking is not None and labels_for_tracking is not None:
                    # 1. Find decision positions FIRST (no logits needed, just labels)
                    decision_positions = self._train_accuracy_tracker.handler.find_decision_positions(
                        input_ids_for_tracking, labels_for_tracking
                    )

                    # 2. IMMEDIATELY slice logits at decision positions to save memory
                    # logits: [B, S, V] → sliced_logits: [B, V]
                    batch_indices = torch.arange(logits.size(0), device=logits.device)
                    safe_positions = decision_positions.clamp(min=0)
                    sliced_logits = logits[batch_indices, safe_positions, :]  # [B, V]

                    # 3. Detach and move to CPU immediately to free GPU memory
                    sliced_logits = sliced_logits.detach().cpu()
                    decision_positions = decision_positions.detach().cpu()
                    labels_cpu = labels_for_tracking.detach().cpu()

                    # 4. Delete reference to full logits to allow GC before backward pass
                    del logits
                    if hasattr(model_outputs, "logits"):
                        model_outputs.logits = None

                    # 5. Track accuracy with sliced logits (on CPU)
                    self._train_accuracy_tracker.update_from_sliced_logits(
                        sliced_logits=sliced_logits,
                        decision_positions=decision_positions,
                        labels=labels_cpu,
                        is_training=model.training,
                    )

        if return_outputs:
            return loss, model_outputs
        return loss

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        r"""Log metrics including aggregated SFT train accuracy metrics.

        Aggregates stored batch metrics and adds them to logs before calling parent.
        """
        # Aggregate train accuracy metrics if tracker is enabled
        if self._train_accuracy_tracker is not None:
            # Determine phase based on what's in logs
            train_eval = "train" if "loss" in logs else "eval"

            if self._train_accuracy_tracker.has_metrics(train_eval):
                accuracy_metrics = self._train_accuracy_tracker.aggregate_and_reset(
                    train_eval, self.accelerator
                )
                logs.update(accuracy_metrics)

        # Call parent to actually write logs
        super().log(logs, start_time)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Prediction step with optional SFT accuracy tracking.

        When accuracy tracker is enabled and not generating:
        - Just do forward pass to get logits (no loss computation)
        - Track accuracy metrics from logits
        """
        # Fast path for accuracy tracking: just forward pass, no loss
        if self._train_accuracy_tracker is not None and not self.args.predict_with_generate:
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")

            with torch.no_grad():
                # 1. Find decision positions FIRST (no logits needed, just labels)
                decision_positions = self._train_accuracy_tracker.handler.find_decision_positions(input_ids, labels)

                # 2. Forward pass WITHOUT labels → no loss computation
                inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
                outputs = model(**inputs_no_labels)

                # 3. IMMEDIATELY slice logits at decision positions to save memory
                # logits: [B, S, V] → sliced_logits: [B, V]
                logits = outputs.logits
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                # Clamp positions to valid range (handle -1 for invalid samples)
                safe_positions = decision_positions.clamp(min=0)
                sliced_logits = logits[batch_indices, safe_positions, :]  # [B, V]

                # 4. Delete full logits to free memory
                del logits, outputs
                torch.cuda.empty_cache()

                # 5. Track accuracy with sliced logits
                self._train_accuracy_tracker.update_from_sliced_logits(
                    sliced_logits=sliced_logits,
                    decision_positions=decision_positions,
                    labels=labels,
                    is_training=False,
                )

            return None, None, None

        # Standard path: generation or no accuracy tracking
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
