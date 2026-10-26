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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeBinaryMetrics:
    r"""Compute binary classification metrics (accuracy, precision, recall, F1) and support `batch_eval_metrics`."""

    def _dump(self) -> dict[str, float] | None:
        result = None
        if hasattr(self, "score_dict"):
            # Compute aggregate metrics
            result = {}
            total = len(self.score_dict["preds"])
            if total > 0:
                preds = np.array(self.score_dict["preds"])
                labels = np.array(self.score_dict["labels"])

                # Accuracy
                result["accuracy"] = float(np.mean(preds == labels))

                # True positives, false positives, false negatives
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))

                # Precision
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                result["precision"] = float(precision)

                # Recall
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                result["recall"] = float(recall)

                # F1
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                result["f1"] = float(f1)

        self.score_dict = {"preds": [], "labels": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> dict[str, float] | None:
        # Get predictions and labels from eval_preds
        # predictions can be either:
        #   - 1D array [N] for CLS-only (decision logits)
        #   - 2D array [N, 2] for CLS+Rating (decision logits, rating logits)
        preds_array = numpify(eval_preds.predictions)
        labels_array = numpify(eval_preds.label_ids)

        # Extract decision logits based on shape
        if preds_array.ndim == 2:
            # CLS+Rating: extract decision logits from first column
            logits = preds_array[:, 0]
        else:
            # CLS-only: use predictions directly
            logits = preds_array

        labels = labels_array

        # Apply sigmoid and threshold
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)

        if not logits.shape:
            # Scalar case
            self.score_dict["preds"].append(int(preds))
            self.score_dict["labels"].append(int(labels))
        else:
            # Batch case
            for i in range(len(logits)):
                self.score_dict["preds"].append(int(preds[i]))
                self.score_dict["labels"].append(int(labels[i]))

        if compute_result:
            return self._dump()
