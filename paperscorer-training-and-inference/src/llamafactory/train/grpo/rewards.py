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

import re
from typing import Callable, Optional


# Registry for reward functions
REWARD_REGISTRY: dict[str, Callable] = {}


def register_reward(name: str):
    """Decorator to register a reward function.

    Usage:
        @register_reward("my_reward")
        def my_reward_func(completions, ground_truth=None, **kwargs):
            return [1.0 if correct else 0.0 for ...]
    """

    def decorator(func: Callable) -> Callable:
        REWARD_REGISTRY[name] = func
        return func

    return decorator


def get_reward_func(name: str) -> Callable:
    """Get a reward function by name from the registry."""
    if name not in REWARD_REGISTRY:
        available = list(REWARD_REGISTRY.keys())
        raise ValueError(f"Unknown reward function: '{name}'. Available: {available}")
    return REWARD_REGISTRY[name]


def get_reward_funcs(names: str, weights: Optional[str] = None) -> tuple[list[Callable], Optional[list[float]]]:
    """Parse comma-separated reward function names and optional weights.

    Args:
        names: Comma-separated reward function names (e.g., "binary,format")
        weights: Optional comma-separated weights (e.g., "0.7,0.3")

    Returns:
        Tuple of (list of reward functions, optional list of weights)
    """
    reward_names = [n.strip() for n in names.split(",")]
    reward_funcs = [get_reward_func(name) for name in reward_names]

    reward_weights = None
    if weights:
        reward_weights = [float(w.strip()) for w in weights.split(",")]
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of weights ({len(reward_weights)}) must match "
                f"number of reward functions ({len(reward_funcs)})"
            )

    return reward_funcs, reward_weights


def extract_boxed_answer(text) -> Optional[str]:
    r"""Extract answer from \boxed{...} format.

    Args:
        text: The text to extract from (string or list of message dicts)

    Returns:
        The extracted answer or None if not found
    """
    # Handle case where TRL passes a list of message dicts
    # e.g., [{'role': 'assistant', 'content': '...'}]
    if isinstance(text, list):
        if text and isinstance(text[0], dict):
            # Extract content from assistant messages
            text = " ".join(msg.get("content", "") for msg in text if msg.get("content"))
        else:
            # List of strings/tokens - join them
            text = "".join(str(t) for t in text)

    if not isinstance(text, str):
        text = str(text)

    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


# =============================================================================
# Built-in reward functions by indicator type
# =============================================================================


@register_reward("binary")
def binary_reward_func(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    r"""Binary classification reward: 1.0 if correct, 0.0 otherwise.

    Extracts prediction from \boxed{...} and compares to ground truth.
    Expects ground_truth to be "0"/"1" or "Accept"/"Reject".

    Args:
        completions: List of model completions
        ground_truth: List of ground truth labels

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        pred = extract_boxed_answer(completion)
        if pred is None:
            rewards.append(0.0)
            continue
        # Normalize both to compare
        pred_norm = pred.lower().strip()
        gt_norm = str(gt).lower().strip()
        rewards.append(1.0 if pred_norm == gt_norm else 0.0)
    return rewards


@register_reward("multiclass")
def multiclass_reward_func(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    r"""Multiclass classification reward: 1.0 if correct, 0.0 otherwise.

    Extracts prediction from \boxed{...} and compares to ground truth.
    Expects ground_truth to be class labels (e.g., "0", "1", "2", "3").

    Args:
        completions: List of model completions
        ground_truth: List of ground truth class labels

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        pred = extract_boxed_answer(completion)
        if pred is None:
            rewards.append(0.0)
            continue
        pred_norm = pred.lower().strip()
        gt_norm = str(gt).lower().strip()
        rewards.append(1.0 if pred_norm == gt_norm else 0.0)
    return rewards


@register_reward("citation")
def citation_reward_func(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    r"""Citation prediction reward: normalized negative MAE.

    Extracts prediction from \boxed{...} and computes MAE against ground truth.
    Reward = 1.0 - (MAE / MAX_CITATION), clamped to [0, 1].

    Args:
        completions: List of model completions
        ground_truth: List of ground truth citation counts

    Returns:
        List of rewards (higher is better, 1.0 = perfect prediction)
    """
    MAX_CITATION = 1000.0
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        pred = extract_boxed_answer(completion)
        if pred is None:
            rewards.append(0.0)
            continue
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            mae = abs(pred_val - gt_val)
            # Convert MAE to reward: 0 error = 1.0, MAX_CITATION error = 0.0
            reward = max(0.0, 1.0 - mae / MAX_CITATION)
            rewards.append(reward)
        except ValueError:
            rewards.append(0.0)
    return rewards
