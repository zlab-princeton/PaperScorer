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

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers.utils import cached_file

from ...extras import logging
from ...extras.constants import CLS_HEAD_SAFE_WEIGHTS_NAME, CLS_HEAD_WEIGHTS_NAME


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


class ModelForBinaryClassification(nn.Module):
    """FSDP2-safe wrapper for binary classification using last token hidden state.

    Compatible with LLaMA-Factory's training infrastructure.
    All parameters created in __init__ for FSDP2 compatibility.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        add_rating_head: bool = False,
        noise_aware: bool = False,
        noise_alpha_init: float = 0.506,
        noise_beta_init: float = 0.15,
    ):
        super().__init__()
        self.backbone = backbone
        self.add_rating_head = add_rating_head
        self.noise_aware = noise_aware

        # Shared feature layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        # Decision head
        self.decision_head = nn.Linear(hidden_size, 1)
        # Optional rating head
        if add_rating_head:
            self.rating_head = nn.Linear(hidden_size, 1)

        # Noise transition matrix parameters (learnable, stored as unconstrained logits)
        # Wrapped in a submodule so FSDP2 sees dotted names (noise_logits.alpha, noise_logits.beta)
        if noise_aware:
            noise_mod = nn.Module()
            noise_mod.alpha = nn.Parameter(
                torch.tensor([math.log(noise_alpha_init / (1.0 - noise_alpha_init))])
            )
            noise_mod.beta = nn.Parameter(
                torch.tensor([math.log(noise_beta_init / (1.0 - noise_beta_init))])
            )
            self.noise_logits = noise_mod

        # For compatibility with save/load patterns
        self._keys_to_ignore_on_save = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ratings: Optional[torch.Tensor] = None,
        rating_loss_weight: float = 0.1,
        **kwargs
    ):
        # Forward through backbone - pass all kwargs for multimodal support
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Get hidden states - handle both CausalLM outputs and model outputs
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        else:
            hidden_states = outputs.last_hidden_state  # [B, T, D]

        # Get last non-padding token per sequence
        if attention_mask is not None:
            idx = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                idx
            ]
        else:
            pooled = hidden_states[:, -1, :]

        # Shared features
        features = self.shared_layer(pooled)

        # Decision logits
        decision_logits = self.decision_head(features).squeeze(-1)  # [B]

        # Rating logits (optional)
        rating_logits = None
        if self.add_rating_head:
            rating_logits = self.rating_head(features).squeeze(-1)

        # Compute losses
        loss = loss_bce = loss_rating = None
        noise_alpha = noise_beta = None
        if labels is not None:
            if self.noise_aware:
                # Noise-aware BCE in log-space for numerical stability.
                # p_noisy_accept = (1-α)·σ(z) + β·σ(-z)
                # p_noisy_reject = α·σ(z) + (1-β)·σ(-z)
                # loss = -[y·log(p_noisy_accept) + (1-y)·log(p_noisy_reject)].mean()
                alpha = torch.sigmoid(self.noise_logits.alpha)
                beta = torch.sigmoid(self.noise_logits.beta)

                log_sig_z = nn.functional.logsigmoid(decision_logits.float())
                log_sig_neg_z = nn.functional.logsigmoid(-decision_logits.float())

                log_p_noisy = torch.logaddexp(
                    torch.log(1.0 - alpha) + log_sig_z,
                    torch.log(beta) + log_sig_neg_z,
                )
                log_1mp_noisy = torch.logaddexp(
                    torch.log(alpha) + log_sig_z,
                    torch.log(1.0 - beta) + log_sig_neg_z,
                )

                loss_bce = -(labels.float() * log_p_noisy + (1.0 - labels.float()) * log_1mp_noisy).mean()
                noise_alpha = alpha.detach()
                noise_beta = beta.detach()
            else:
                loss_bce = nn.functional.binary_cross_entropy_with_logits(
                    decision_logits.float(), labels.float()
                )
            loss = loss_bce

            if self.add_rating_head and ratings is not None:
                # Mask invalid ratings (sentinel -1.0)
                valid_mask = ratings >= 0
                if valid_mask.any():
                    # Use raw logits during training (no sigmoid) - more efficient
                    # SmoothL1 with beta=0.1: quadratic for errors <10%, linear for larger
                    loss_rating = nn.functional.smooth_l1_loss(
                        rating_logits[valid_mask], ratings[valid_mask], beta=0.1
                    )
                    loss = loss_bce + rating_loss_weight * loss_rating

        return {
            "loss": loss,
            "logits": decision_logits,
            "rating_logits": rating_logits,
            "loss_bce": loss_bce,
            "loss_rating": loss_rating,
            "noise_alpha": noise_alpha,
            "noise_beta": noise_beta,
        }

    # Forward properties/methods to backbone for compatibility
    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens):
        return self.backbone.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, *args, **kwargs):
        """Delegate to backbone for saving."""
        return self.backbone.save_pretrained(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set training mode for both backbone and classifier."""
        super().train(mode)
        self.backbone.train(mode)
        return self

    def eval(self):
        """Set eval mode for both backbone and classifier."""
        super().eval()
        self.backbone.eval()
        return self


def load_cls_head_params(path_or_repo_id: str, model_args: "ModelArguments") -> Optional[dict[str, torch.Tensor]]:
    """Load classifier head parameters from checkpoint.

    Returns: dict with keys for shared_layer and decision_head (or old classifier.* keys).
    Handles backward compatibility by mapping old key names to new structure.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id, "cache_dir": model_args.cache_dir, "token": model_args.hf_hub_token}
    err_text = ""
    params = None

    try:
        from safetensors import safe_open

        cls_file = cached_file(filename=CLS_HEAD_SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(cls_file, framework="pt", device="cpu") as f:
            params = {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    if params is None:
        try:
            cls_file = cached_file(filename=CLS_HEAD_WEIGHTS_NAME, **kwargs)
            params = torch.load(cls_file, map_location="cpu", weights_only=True)
        except Exception as err:
            err_text = str(err)

    if params is None:
        logger.info_rank0(f"Provided path ({path_or_repo_id}) does not contain classifier head weights: {err_text}.")
        logger.info_rank0("Ignore the above message if you are training from scratch.")
        return None

    # Handle old checkpoint format (classifier.0.*, classifier.2.*)
    # Map to new format (shared_layer.0.*, decision_head.*)
    if "classifier.0.weight" in params:
        params["shared_layer.0.weight"] = params.pop("classifier.0.weight")
        params["shared_layer.0.bias"] = params.pop("classifier.0.bias")
        params["decision_head.weight"] = params.pop("classifier.2.weight")
        params["decision_head.bias"] = params.pop("classifier.2.bias")

    return params
