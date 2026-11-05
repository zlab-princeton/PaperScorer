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
Extensible interface for computing classification accuracy during SFT training.

Supports different output formats (e.g., \boxed{Accept}, Y/N) via subclassing.
Computes three probabilities: p(accept), p(reject), p(everything_else).
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Optional

import torch

from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class BaseOutputFormatHandler(ABC):
    """Abstract base class for handling different output formats in SFT accuracy computation.

    Subclass this to support new output formats (e.g., \boxed{Accept}, Y/N, etc.).
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        positive_token: str = "Accept",
        negative_token: str = "Reject",
    ) -> None:
        self.tokenizer = tokenizer
        self.positive_token = positive_token
        self.negative_token = negative_token

        # Get token IDs for positive/negative tokens
        self.positive_token_id = self._get_token_id(positive_token)
        self.negative_token_id = self._get_token_id(negative_token)

    def _get_token_id(self, token: str) -> int:
        """Get the token ID for a given token string."""
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Token '{token}' encodes to {len(token_ids)} tokens ({token_ids}), "
                f"but expected exactly 1 token. Please use a single-token word."
            )
        return token_ids[0]

    @abstractmethod
    def find_decision_positions(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Find positions where the model makes its classification decision.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len], with IGNORE_INDEX for non-target tokens

        Returns:
            Tensor of decision positions [batch_size], or -1 for samples without valid positions
        """
        pass

    def get_ground_truth(
        self, labels: torch.Tensor, decision_positions: torch.Tensor
    ) -> torch.Tensor:
        """Extract ground truth labels at decision positions + 1.

        In causal LM, logits[i] predicts the token at position i+1.
        So decision_positions points to where we read logits, and
        ground truth is at decision_positions + 1 in labels.

        Args:
            labels: Label token IDs [batch_size, seq_len]
            decision_positions: Positions where logits are read [batch_size]

        Returns:
            Ground truth token IDs at decision_positions + 1 [batch_size]
        """
        batch_size = labels.size(0)
        seq_len = labels.size(1)
        batch_indices = torch.arange(batch_size, device=labels.device)

        # Ground truth is at position + 1 (logits[i] predicts labels[i+1])
        # Clamp to valid range (handle -1 and boundary cases)
        gt_positions = (decision_positions + 1).clamp(min=0, max=seq_len - 1)
        ground_truth = labels[batch_indices, gt_positions]

        return ground_truth

    def compute_probabilities(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute classification probabilities at decision positions.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len]

        Returns:
            Dictionary with:
                - p_positive: Probability of positive class [valid_batch_size]
                - p_negative: Probability of negative class [valid_batch_size]
                - p_other: Probability of other tokens [valid_batch_size]
                - predictions: Predicted class (1=positive, 0=negative) [valid_batch_size]
                - ground_truth: Ground truth class (1=positive, 0=negative) [valid_batch_size]
                - valid_mask: Boolean mask for valid samples [batch_size]
        """
        batch_size = logits.size(0)
        device = logits.device

        # Find decision positions
        decision_positions = self.find_decision_positions(input_ids, labels)

        # Create mask for valid positions (not -1)
        valid_mask = decision_positions >= 0

        if not valid_mask.any():
            # No valid samples in this batch
            return {
                "p_positive": torch.tensor([], device=device),
                "p_negative": torch.tensor([], device=device),
                "p_other": torch.tensor([], device=device),
                "predictions": torch.tensor([], device=device, dtype=torch.long),
                "ground_truth": torch.tensor([], device=device, dtype=torch.long),
                "valid_mask": valid_mask,
            }

        # Get valid indices
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_positions = decision_positions[valid_mask]

        # Extract logits at decision positions for valid samples
        # logits shape: [batch_size, seq_len, vocab_size]
        decision_logits = logits[valid_indices, valid_positions, :]  # [valid_batch_size, vocab_size]

        # Apply softmax to get probabilities
        probs = torch.softmax(decision_logits, dim=-1)  # [valid_batch_size, vocab_size]

        # Extract probabilities for positive/negative tokens
        p_positive = probs[:, self.positive_token_id]
        p_negative = probs[:, self.negative_token_id]
        p_other = (1.0 - p_positive - p_negative).clamp(min=0.0)

        # Get predictions using 3-class argmax (Accept vs Reject vs Other)
        # Stack: [batch, 3] where index 0=reject, 1=accept, 2=other
        prob_stack = torch.stack([p_negative, p_positive, p_other], dim=-1)
        predicted_class = prob_stack.argmax(dim=-1)  # 0=reject, 1=accept, 2=other

        # predictions: 1 if Accept wins, 0 otherwise
        predictions = (predicted_class == 1).long()

        # Get ground truth
        gt_tokens = self.get_ground_truth(labels, decision_positions)
        valid_gt_tokens = gt_tokens[valid_mask]

        # Convert ground truth tokens to binary labels
        ground_truth = (valid_gt_tokens == self.positive_token_id).long()

        # Compute p_correct: probability assigned to the correct class
        # If ground_truth=1 (Accept), use p_positive; if ground_truth=0 (Reject), use p_negative
        p_correct = torch.where(ground_truth == 1, p_positive, p_negative)

        return {
            "p_positive": p_positive,
            "p_negative": p_negative,
            "p_other": p_other,
            "p_correct": p_correct,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "valid_mask": valid_mask,
        }


class BoxedFormatHandler(BaseOutputFormatHandler):
    """Handler for \boxed{Accept/Reject} format.

    Finds the '{' token in labels and looks at the next position for Accept/Reject.
    """

    # Token ID for '{' - typically 90 for most tokenizers, but we'll look it up
    BRACE_TOKEN = "{"

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        positive_token: str = "Accept",
        negative_token: str = "Reject",
    ) -> None:
        super().__init__(tokenizer, positive_token, negative_token)

        # Get token ID for '{'
        self.brace_token_id = self._get_token_id(self.BRACE_TOKEN)

    def find_decision_positions(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Find positions where we read logits to predict Accept/Reject.

        In causal LM, logits[i] predicts token at position i+1.
        So we return the position of '{', and logits there predict Accept/Reject.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len]

        Returns:
            Decision positions [batch_size] (position of '{'), -1 for invalid samples
        """
        batch_size, seq_len = labels.size()
        device = labels.device

        # Initialize all positions as invalid (-1)
        decision_positions = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Find '{' token in labels (where labels != IGNORE_INDEX)
        valid_labels_mask = labels != IGNORE_INDEX
        brace_mask = (labels == self.brace_token_id) & valid_labels_mask

        for i in range(batch_size):
            # Find indices where '{' appears in this sample
            brace_indices = brace_mask[i].nonzero(as_tuple=True)[0]

            if len(brace_indices) > 0:
                # Take the last '{' (in case there are multiple)
                brace_pos = brace_indices[-1].item()

                # Decision position is '{' - logits here predict Accept/Reject at brace_pos+1
                # Make sure brace_pos+1 is within bounds and has valid label
                if brace_pos + 1 < seq_len and valid_labels_mask[i, brace_pos + 1]:
                    decision_positions[i] = brace_pos

        return decision_positions


class YesNoFormatHandler(BaseOutputFormatHandler):
    """Handler for Y/N format.

    Gets logits at the last position where labels != IGNORE_INDEX.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        positive_token: str = "Y",
        negative_token: str = "N",
    ) -> None:
        super().__init__(tokenizer, positive_token, negative_token)

    def find_decision_positions(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Find position where logits predict the Y/N token.

        In causal LM, logits[i] predicts token at position i+1.
        So we return position-1 of the last valid label (Y/N token).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len]

        Returns:
            Decision positions [batch_size] (one before Y/N token), -1 for invalid samples
        """
        batch_size, seq_len = labels.size()
        device = labels.device

        # Initialize all positions as invalid (-1)
        decision_positions = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Find last valid label position for each sample
        valid_labels_mask = labels != IGNORE_INDEX

        for i in range(batch_size):
            valid_indices = valid_labels_mask[i].nonzero(as_tuple=True)[0]

            if len(valid_indices) > 0:
                # Take the last valid position (where Y/N appears)
                last_pos = valid_indices[-1].item()
                # Decision position is one before - logits here predict Y/N
                if last_pos > 0:
                    decision_positions[i] = last_pos - 1

        return decision_positions


class TrainAccuracyTracker:
    """Accumulates per-batch metrics and aggregates on log.

    Follows the BinaryClassificationTrainer pattern for metric tracking.
    """

    def __init__(self, handler: BaseOutputFormatHandler) -> None:
        self.handler = handler
        self._stored_metrics: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Per-sample predictions accumulated during eval: [[idx, p_accept, p_reject, gt], ...]
        self._per_sample_predictions: dict[str, list] = defaultdict(list)
        self._sample_counter: dict[str, int] = defaultdict(int)

    def update(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        is_training: bool,
    ) -> None:
        """Update metrics with batch results.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len]
            is_training: Whether model is in training mode
        """
        # Compute probabilities and predictions
        results = self.handler.compute_probabilities(logits, input_ids, labels)

        # Skip if no valid samples
        if len(results["predictions"]) == 0:
            return

        phase = "train" if is_training else "eval"
        prefix = "" if is_training else "eval_"

        # Compute metrics for this batch
        correct = (results["predictions"] == results["ground_truth"]).float()
        accuracy = correct.mean()

        p_positive_mean = results["p_positive"].mean()
        p_negative_mean = results["p_negative"].mean()
        p_other_mean = results["p_other"].mean()
        p_correct_mean = results["p_correct"].mean()

        pred_positive_rate = results["predictions"].float().mean()
        gt_positive_rate = results["ground_truth"].float().mean()

        # Store metrics
        self._stored_metrics[phase][f"{prefix}sft_accuracy"].append(accuracy.detach())
        self._stored_metrics[phase][f"{prefix}sft_p_accept_mean"].append(p_positive_mean.detach())
        self._stored_metrics[phase][f"{prefix}sft_p_reject_mean"].append(p_negative_mean.detach())
        self._stored_metrics[phase][f"{prefix}sft_p_other_mean"].append(p_other_mean.detach())
        self._stored_metrics[phase][f"{prefix}sft_p_correct_mean"].append(p_correct_mean.detach())
        self._stored_metrics[phase][f"{prefix}sft_pred_positive_rate"].append(pred_positive_rate.detach())
        self._stored_metrics[phase][f"{prefix}sft_gt_positive_rate"].append(gt_positive_rate.detach())

        # Conditional probability metrics: p(token | ground_truth_class)
        gt_accept_mask = results["ground_truth"] == 1
        gt_reject_mask = results["ground_truth"] == 0
        if gt_accept_mask.any():
            self._stored_metrics[phase][f"{prefix}sft_p_accept_gtaccept_mean"].append(results["p_positive"][gt_accept_mask].mean().detach())
            self._stored_metrics[phase][f"{prefix}sft_p_reject_gtaccept_mean"].append(results["p_negative"][gt_accept_mask].mean().detach())
        if gt_reject_mask.any():
            self._stored_metrics[phase][f"{prefix}sft_p_accept_gtreject_mean"].append(results["p_positive"][gt_reject_mask].mean().detach())
            self._stored_metrics[phase][f"{prefix}sft_p_reject_gtreject_mean"].append(results["p_negative"][gt_reject_mask].mean().detach())

    def update_from_sliced_logits(
        self,
        sliced_logits: torch.Tensor,
        decision_positions: torch.Tensor,
        labels: torch.Tensor,
        is_training: bool,
    ) -> None:
        """Update metrics from pre-sliced logits (memory-efficient version).

        Args:
            sliced_logits: Logits at decision positions [batch_size, vocab_size]
            decision_positions: Position indices [batch_size], -1 for invalid
            labels: Label token IDs [batch_size, seq_len]
            is_training: Whether model is in training mode
        """
        device = sliced_logits.device
        batch_size = sliced_logits.size(0)

        # Create mask for valid positions (not -1)
        valid_mask = decision_positions >= 0

        if not valid_mask.any():
            return

        # Get valid sliced logits
        valid_logits = sliced_logits[valid_mask]  # [valid_batch_size, vocab_size]

        # Apply softmax to get probabilities
        probs = torch.softmax(valid_logits, dim=-1)

        # Extract probabilities for positive/negative tokens
        p_positive = probs[:, self.handler.positive_token_id]
        p_negative = probs[:, self.handler.negative_token_id]
        p_other = (1.0 - p_positive - p_negative).clamp(min=0.0)

        # Get predictions using 3-class argmax
        prob_stack = torch.stack([p_negative, p_positive, p_other], dim=-1)
        predicted_class = prob_stack.argmax(dim=-1)
        predictions = (predicted_class == 1).long()

        # Get ground truth from labels at decision_positions + 1
        gt_positions = (decision_positions + 1).clamp(min=0, max=labels.size(1) - 1)
        batch_indices = torch.arange(batch_size, device=device)
        gt_tokens = labels[batch_indices, gt_positions]
        valid_gt_tokens = gt_tokens[valid_mask]
        ground_truth = (valid_gt_tokens == self.handler.positive_token_id).long()

        # Compute p_correct
        p_correct = torch.where(ground_truth == 1, p_positive, p_negative)

        # Compute and store metrics
        phase = "train" if is_training else "eval"
        prefix = "" if is_training else "eval_"

        correct = (predictions == ground_truth).float()
        self._stored_metrics[phase][f"{prefix}sft_accuracy"].append(correct.mean().detach())
        self._stored_metrics[phase][f"{prefix}sft_p_accept_mean"].append(p_positive.mean().detach())
        self._stored_metrics[phase][f"{prefix}sft_p_reject_mean"].append(p_negative.mean().detach())
        self._stored_metrics[phase][f"{prefix}sft_p_other_mean"].append(p_other.mean().detach())
        self._stored_metrics[phase][f"{prefix}sft_p_correct_mean"].append(p_correct.mean().detach())
        self._stored_metrics[phase][f"{prefix}sft_pred_positive_rate"].append(predictions.float().mean().detach())
        self._stored_metrics[phase][f"{prefix}sft_gt_positive_rate"].append(ground_truth.float().mean().detach())

        # Conditional probability metrics: p(token | ground_truth_class)
        gt_accept_mask = ground_truth == 1
        gt_reject_mask = ground_truth == 0
        if gt_accept_mask.any():
            self._stored_metrics[phase][f"{prefix}sft_p_accept_gtaccept_mean"].append(p_positive[gt_accept_mask].mean().detach())
            self._stored_metrics[phase][f"{prefix}sft_p_reject_gtaccept_mean"].append(p_negative[gt_accept_mask].mean().detach())
        if gt_reject_mask.any():
            self._stored_metrics[phase][f"{prefix}sft_p_accept_gtreject_mean"].append(p_positive[gt_reject_mask].mean().detach())
            self._stored_metrics[phase][f"{prefix}sft_p_reject_gtreject_mean"].append(p_negative[gt_reject_mask].mean().detach())

        # Per-sample predictions (eval only to avoid unbounded growth during training)
        if not is_training:
            p_pos_list = p_positive.detach().cpu().tolist()
            p_neg_list = p_negative.detach().cpu().tolist()
            gt_list = ground_truth.detach().cpu().tolist()
            counter = self._sample_counter[phase]
            for i in range(len(p_pos_list)):
                self._per_sample_predictions[phase].append(
                    [counter, p_pos_list[i], p_neg_list[i], gt_list[i]]
                )
                counter += 1
            self._sample_counter[phase] = counter

    def has_metrics(self, phase: str) -> bool:
        """Check if there are stored metrics for the given phase."""
        return phase in self._stored_metrics and len(self._stored_metrics[phase]) > 0

    def aggregate_and_reset(
        self, phase: str, accelerator: Optional["torch.distributed.Accelerator"] = None
    ) -> dict[str, float]:
        """Aggregate stored metrics and reset for next logging period.

        Args:
            phase: "train" or "eval"
            accelerator: Optional accelerator for distributed training

        Returns:
            Dictionary of aggregated metric values
        """
        metrics = {}

        if phase not in self._stored_metrics:
            return metrics

        for key, values in self._stored_metrics[phase].items():
            if len(values) > 0:
                # Stack all batch values
                metric_tensor = torch.stack(values)

                # Reduce across devices if using distributed training
                if accelerator is not None and accelerator.num_processes > 1:
                    metric_tensor = accelerator.gather(metric_tensor)

                # Move to CPU and compute mean
                metrics[key] = metric_tensor.cpu().mean().item()

        # Include per-sample predictions if any were collected
        if phase in self._per_sample_predictions and self._per_sample_predictions[phase]:
            prefix = "" if phase == "train" else "eval_"
            metrics[f"{prefix}sft_predictions"] = self._per_sample_predictions[phase]

        # Clear stored metrics after aggregation
        self._stored_metrics[phase].clear()
        self._per_sample_predictions[phase] = []
        self._sample_counter[phase] = 0

        return metrics


def create_output_format_handler(
    format_type: Literal["boxed", "yesno"],
    tokenizer: "PreTrainedTokenizer",
    positive_token: str = "Accept",
    negative_token: str = "Reject",
) -> BaseOutputFormatHandler:
    """Factory function to create output format handlers.

    Args:
        format_type: Type of format handler ("boxed" or "yesno")
        tokenizer: Tokenizer for encoding tokens
        positive_token: Token for positive class
        negative_token: Token for negative class

    Returns:
        Configured output format handler
    """
    if format_type == "boxed":
        return BoxedFormatHandler(tokenizer, positive_token, negative_token)
    elif format_type == "yesno":
        return YesNoFormatHandler(tokenizer, positive_token, negative_token)
    else:
        raise ValueError(f"Unknown format type: {format_type}. Supported: 'boxed', 'yesno'")
