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

import json
import os
from collections import defaultdict
from types import MethodType
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Trainer, TrainerCallback
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixClsModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin, TrainerControl, TrainerState, TrainingArguments
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class NoiseEMCallback(TrainerCallback):
    """Callback for EM-style training: alternates between model and noise parameter phases."""

    def __init__(self, model_epochs: float, noise_epochs: float, steps_per_epoch: int):
        self.model_steps = max(1, int(model_epochs * steps_per_epoch))
        self.noise_steps = max(1, int(noise_epochs * steps_per_epoch))
        self.cycle_steps = self.model_steps + self.noise_steps
        self._noise_params: list = None
        self._trainable_model_params: list = None
        self._current_phase: str = None

    def on_step_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model: "PreTrainedModel" = None,
        **kwargs,
    ):
        if model is None:
            return

        if self._noise_params is None:
            # Cache param references once
            self._noise_params = []
            self._trainable_model_params = []
            for name, param in model.named_parameters():
                if name in ("noise_logits.alpha", "noise_logits.beta"):
                    self._noise_params.append(param)
                elif param.requires_grad:
                    self._trainable_model_params.append(param)

        cycle_pos = state.global_step % self.cycle_steps
        phase = "noise" if cycle_pos >= self.model_steps else "model"

        if phase != self._current_phase:
            self._current_phase = phase
            if phase == "model":
                for p in self._noise_params:
                    p.requires_grad_(False)
                for p in self._trainable_model_params:
                    p.requires_grad_(True)
            else:
                for p in self._noise_params:
                    p.requires_grad_(True)
                for p in self._trainable_model_params:
                    p.requires_grad_(False)
            logger.info_rank0(f"EM phase switch at step {state.global_step}: {phase}")


class BinaryClassificationTrainer(Trainer):
    r"""Inherits Trainer to compute binary classification loss with BCE."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.add_callback(FixClsModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.noise_aware_loss and finetuning_args.noise_strategy == "em":
            # Compute steps_per_epoch lazily: max_steps / num_train_epochs is computed later,
            # so use dataset size / effective batch as approximation
            train_dataset = kwargs.get("train_dataset")
            if train_dataset is not None:
                num_gpus = max(1, int(os.environ.get("WORLD_SIZE", 1)))
                effective_batch = (
                    kwargs.get("args").per_device_train_batch_size
                    * num_gpus
                    * kwargs.get("args").gradient_accumulation_steps
                )
                steps_per_epoch = max(1, len(train_dataset) // effective_batch)
            else:
                steps_per_epoch = 1  # fallback
            self.add_callback(
                NoiseEMCallback(
                    model_epochs=finetuning_args.noise_em_model_epochs,
                    noise_epochs=finetuning_args.noise_em_noise_epochs,
                    steps_per_epoch=steps_per_epoch,
                )
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
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Compute binary classification loss using BCE, with accuracy tracking."""
        labels = inputs.pop("labels")  # Binary labels (0 or 1)
        ratings = inputs.pop("ratings", None)  # Rating values for multi-task learning
        inputs.pop("_metadata", None)  # Remove metadata before forward (not needed by model)

        # Store original binary labels for accuracy computation before smoothing
        original_labels = labels

        # Apply label smoothing if configured
        # Transforms: 0 → eps, 1 → 1-eps
        eps = self.finetuning_args.label_smoothing_eps
        if eps > 0:
            labels = labels.float() * (1 - 2 * eps) + eps

        # Build forward kwargs
        forward_kwargs = {"labels": labels}
        if self.finetuning_args.add_predict_ratings and ratings is not None:
            forward_kwargs["ratings"] = ratings
            forward_kwargs["rating_loss_weight"] = self.finetuning_args.rating_loss_weight

        # Forward pass - model returns {"loss": loss, "logits": logits, ...}
        outputs = model(**inputs, **forward_kwargs)
        loss = outputs["loss"]
        logits = outputs["logits"]

        # Compute and store accuracy for logging (use original binary labels)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            accuracy = (preds == original_labels.long()).float().mean()

        # Store metrics for aggregation in log() method (keep on device for gather)
        train_eval = "train" if model.training else "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        self._stored_metrics[train_eval][f"{prefix}accuracy"].append(accuracy.detach())
        self._stored_metrics[train_eval][f"{prefix}cls_loss"].append(loss.detach())

        # Store component losses when using multi-task learning
        if self.finetuning_args.add_predict_ratings:
            if outputs.get("loss_bce") is not None:
                self._stored_metrics[train_eval][f"{prefix}loss_bce"].append(outputs["loss_bce"].detach())
            if outputs.get("loss_rating") is not None:
                self._stored_metrics[train_eval][f"{prefix}loss_rating"].append(outputs["loss_rating"].detach())

        # Store noise transition matrix params for logging
        if self.finetuning_args.noise_aware_loss:
            if outputs.get("noise_alpha") is not None:
                self._stored_metrics[train_eval][f"{prefix}noise_alpha"].append(outputs["noise_alpha"])
            if outputs.get("noise_beta") is not None:
                self._stored_metrics[train_eval][f"{prefix}noise_beta"].append(outputs["noise_beta"])

        if return_outputs:
            return loss, (loss, logits, labels)
        return loss

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        r"""Log metrics including aggregated training accuracy.

        Aggregates stored batch metrics and adds them to logs before calling parent.
        Handles both training ("loss" in logs) and evaluation ("eval_loss" in logs).
        """
        # Determine phase based on what's in logs
        train_eval = "train" if "loss" in logs else "eval"

        # Aggregate stored metrics for this phase
        if train_eval in self._stored_metrics:
            for key, values in self._stored_metrics[train_eval].items():
                if len(values) > 0:
                    # Convert list of tensors to single averaged value
                    metric_tensor = torch.stack(values)
                    # Reduce across devices if using distributed training
                    if self.accelerator is not None and self.accelerator.num_processes > 1:
                        metric_tensor = self.accelerator.gather(metric_tensor)
                    # Move to CPU and compute mean
                    logs[key] = metric_tensor.cpu().mean().item()

            # Clear stored metrics after logging
            self._stored_metrics[train_eval].clear()

        # Call parent to actually write logs
        super().log(logs, start_time)

    @override
    def prediction_step(
        self,
        model: "PreTrainedModel",
        inputs: dict[str, "torch.Tensor"],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Override to capture both decision logits and rating logits."""
        labels = inputs.pop("labels")
        inputs.pop("ratings", None)  # Don't need ratings for prediction
        inputs.pop("_metadata", None)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            rating_logits = outputs.get("rating_logits")

            # Stack decision and rating logits if rating head exists
            if rating_logits is not None:
                # Shape: [batch, 2] where [:, 0] = decision, [:, 1] = rating
                logits = torch.stack([logits, rating_logits], dim=-1)

        if prediction_loss_only:
            return (None, None, None)

        return (None, logits, labels)

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save model predictions to `predictions_dir` (or `output_dir` if not set)."""
        if not self.is_world_process_zero():
            return

        # Use predictions_dir if set, otherwise fall back to output_dir
        predictions_dir = self.finetuning_args.predictions_dir or self.args.output_dir
        os.makedirs(predictions_dir, exist_ok=True)
        output_prediction_file = os.path.join(predictions_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        predictions = predict_results.predictions
        labels = predict_results.label_ids

        # Check if predictions include rating logits (shape: [N, 2])
        has_rating = predictions.ndim == 2 and predictions.shape[1] == 2
        if has_rating:
            decision_logits = predictions[:, 0]
            rating_logits = predictions[:, 1]
        else:
            decision_logits = predictions
            rating_logits = None

        # Get metadata from eval_dataset if available (predictions are in order)
        metadata_list = None
        if hasattr(self, "eval_dataset") and self.eval_dataset is not None:
            if "_metadata" in self.eval_dataset.column_names:
                metadata_list = self.eval_dataset["_metadata"]

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: list[str] = []
            for i, (logit, label) in enumerate(zip(decision_logits, labels)):
                prob = sigmoid(float(logit))
                pred = 1 if prob > 0.5 else 0
                result = {
                    "logit": round(float(logit), 4),
                    "prob": round(prob, 4),
                    "pred": pred,
                    "label": int(label)
                }
                # Add predicted rating if available (sigmoid applied)
                if rating_logits is not None:
                    rating_pred = sigmoid(float(rating_logits[i]))
                    result["rating_logit"] = round(float(rating_logits[i]), 4)
                    result["rating_pred"] = round(rating_pred, 4)
                # Add metadata if available
                if metadata_list is not None and i < len(metadata_list):
                    result["_metadata"] = metadata_list[i]
                res.append(json.dumps(result))

            writer.write("\n".join(res))
