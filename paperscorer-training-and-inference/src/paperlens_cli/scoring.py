"""LF-tokenized scoring core for ``paperlens serve``.

This is the *only* code in the deployment that touches the model. It
reuses every ``llamafactory.data`` call that ``scripts/vllm_infer.py``
makes so the persistent server is byte-identical to the offline batch
on the same hardware. (Cross-hardware drift is documented in
``tools/paperlens-arxiv-server/README.md`` -- ~0.02 mean, 0.21 max,
Pearson 0.994 across GPU classes.)

Per request (``score(rows)``):

  1. Acquire a lock (data_args is mutable shared state).
  2. Materialize the rows as a temp sharegpt dataset under TemporaryDirectory.
  3. Swap ``data_args.dataset_dir`` / ``data_args.dataset`` to point at it.
  4. Call ``llamafactory.data.get_dataset(template_obj, ..., "ppo", ...)``
     -- this is the same call vllm_infer.py uses.
  5. Build per-row ``{prompt_token_ids, multi_modal_data}`` exactly like
     vllm_infer.py:200-253.
  6. ``llm.generate(vllm_inputs, sampling_params)``.
  7. Extract ``logp_accept[5]`` / ``logp_reject[5]`` and compute the
     2-token softmax. Return per-row dicts.

The lock cost is acceptable -- one `vllm.generate` per request batches
all rows together inside vLLM, so concurrency-within-request is fine and
the only thing serialized is the dataset-swap dance.
"""
from __future__ import annotations

import json
import logging
import math
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional


log = logging.getLogger(__name__)


DECISION_TOKEN_IDX = 5

# Tags shared by every paperlens sharegpt dataset.
_SHAREGPT_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system",
}


def _softmax2(logp_a: float, logp_b: float) -> float:
    """Stable 2-token softmax."""
    m = max(logp_a, logp_b)
    ea = math.exp(logp_a - m)
    eb = math.exp(logp_b - m)
    return ea / (ea + eb)


class Scorer:
    """Persistent LF-tokenized scorer.

    One model + tokenizer + template_obj loaded at __init__. ``score(rows)``
    can be called many times; each call materializes a temp dataset, runs
    inference, and returns per-row {p_accept, logp_accept, logp_reject, pred}.
    """

    def __init__(self, cfg) -> None:
        # cfg is an OmegaConf DictConfig (or dict-like) per configs/serve.yaml.
        self.cfg = cfg
        self.ckpt_path: str = str(cfg.model.ckpt_path)
        self.template: str = str(cfg.model.template)
        self.cutoff_len: int = int(cfg.model.cutoff_len)
        self.max_new_tokens: int = int(cfg.scoring.max_new_tokens)
        self.image_max_pixels: int = int(cfg.scoring.image_max_pixels)
        self.image_min_pixels: int = int(cfg.scoring.image_min_pixels)
        self.positive_token: str = str(cfg.scoring.positive_token)
        self.negative_token: str = str(cfg.scoring.negative_token)
        self.decision_token_idx: int = int(cfg.scoring.get("decision_token_idx", DECISION_TOKEN_IDX))
        self.compute_arch: str = str(cfg.get("compute_arch", "unknown"))

        # Heavy imports here so the other CLI subcommands (setup-model,
        # setup-data) don't need vllm or llamafactory installed.
        from transformers import Seq2SeqTrainingArguments
        from llamafactory.data import get_template_and_fix_tokenizer
        from llamafactory.hparams import get_infer_args
        from llamafactory.model import load_tokenizer
        from vllm import LLM, SamplingParams

        log.info(
            f"[scorer] init ckpt={self.ckpt_path} template={self.template} "
            f"cutoff_len={self.cutoff_len} max_new_tokens={self.max_new_tokens}"
        )

        # Build args. Dataset args are placeholders here; we mutate them
        # per-request inside ``_score_unlocked``.
        self.model_args, self.data_args, _, self.generating_args = get_infer_args(
            dict(
                model_name_or_path=self.ckpt_path,
                adapter_name_or_path=None,
                dataset="alpaca_en_demo",       # placeholder; overridden per request
                dataset_dir="data",             # placeholder; overridden per request
                template=self.template,
                cutoff_len=self.cutoff_len,
                max_samples=None,
                preprocessing_num_workers=4,
                default_system=None,
                enable_thinking=bool(cfg.model.get("enable_thinking", False)),
                vllm_config="{}",
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.0,
            )
        )
        self.training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
        self.tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = self.tokenizer_module["tokenizer"]
        self.template_obj = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.template_obj.mm_plugin.expand_mm_tokens = False    # required for vLLM generate

        engine_args: dict[str, Any] = {
            "model": self.model_args.model_name_or_path,
            "trust_remote_code": True,
            "dtype": self.model_args.infer_dtype,
            "max_model_len": int(cfg.vllm.get(
                "max_model_len", self.cutoff_len + self.max_new_tokens + 4
            )),
            "tensor_parallel_size": int(cfg.vllm.get("tensor_parallel_size", 1)),
            "pipeline_parallel_size": int(cfg.vllm.get("pipeline_parallel_size", 1)),
            "disable_log_stats": True,
            "gpu_memory_utilization": float(cfg.vllm.get("gpu_memory_utilization", 0.85)),
            "enable_lora": False,
        }
        if self.template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
            engine_args["limit_mm_per_prompt"] = {"image": 400, "video": 2, "audio": 2}
        log.info(f"[scorer] vLLM engine_args: {engine_args}")
        self.llm = LLM(**engine_args)

        # Resolve decision token IDs once.
        pos_ids = self.tokenizer.encode(self.positive_token, add_special_tokens=False)
        neg_ids = self.tokenizer.encode(self.negative_token, add_special_tokens=False)
        assert len(pos_ids) == 1, f"'{self.positive_token}' tokenizes to {len(pos_ids)} tokens"
        assert len(neg_ids) == 1, f"'{self.negative_token}' tokenizes to {len(neg_ids)} tokens"
        self.pos_token_id = pos_ids[0]
        self.neg_token_id = neg_ids[0]
        log.info(
            f"[scorer] decision tokens: {self.positive_token}={self.pos_token_id}, "
            f"{self.negative_token}={self.neg_token_id} (idx={self.decision_token_idx})"
        )

        self.sampling_params = SamplingParams(
            repetition_penalty=1.0,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            stop_token_ids=self.template_obj.get_stop_token_ids(self.tokenizer),
            max_tokens=self.max_new_tokens,
            skip_special_tokens=True,
            seed=None,
            logprobs=5,
        )

        # Serialize requests because data_args is mutated per call.
        self._lock = threading.Lock()
        self.calls_served = 0

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def score(self, sharegpt_rows: list[dict]) -> list[dict]:
        """Score N sharegpt rows.

        Each input row must have at minimum a ``conversations`` field (the
        sharegpt list). ``images`` and ``_metadata`` are passed through
        verbatim; the model only sees what LF's get_dataset extracts.

        Returns N dicts with keys: ``p_accept`` (float in [0,1]),
        ``logp_accept``, ``logp_reject``, ``pred`` (the greedy decode).
        """
        if not sharegpt_rows:
            return []
        with self._lock:
            with tempfile.TemporaryDirectory(prefix="paperlens_serve_") as td:
                results = self._score_unlocked(sharegpt_rows, Path(td))
            self.calls_served += 1
        return results

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _materialize_temp_dataset(self, rows: list[dict], td: Path) -> str:
        """Write rows to <td>/score_dataset/data.json and register them
        in <td>/dataset_info.json. Returns the dataset name to load.
        """
        ds_name = "score_dataset"
        ds_dir = td / ds_name
        ds_dir.mkdir()
        (ds_dir / "data.json").write_text(json.dumps(rows))

        has_images = any(r.get("images") for r in rows)
        columns: dict[str, str] = {"messages": "conversations"}
        if has_images:
            columns["images"] = "images"
        info = {ds_name: {
            "file_name": f"{ds_name}/data.json",
            "formatting": "sharegpt",
            "columns": columns,
            "tags": _SHAREGPT_TAGS,
        }}
        (td / "dataset_info.json").write_text(json.dumps(info))
        return ds_name

    def _score_unlocked(self, sharegpt_rows: list[dict], td: Path) -> list[dict]:
        from llamafactory.data import get_dataset

        ds_name = self._materialize_temp_dataset(sharegpt_rows, td)

        # Swap dataset args to point at the temp dir, then restore. This
        # is the *only* way to feed ad-hoc rows through get_dataset() while
        # preserving its exact tokenization/multi-modal pipeline.
        orig_dir = self.data_args.dataset_dir
        orig_ds = self.data_args.dataset
        try:
            self.data_args.dataset_dir = str(td)
            self.data_args.dataset = [ds_name]
            ds_module = get_dataset(
                self.template_obj, self.model_args, self.data_args,
                self.training_args, "ppo", **self.tokenizer_module,
            )
            train_dataset = ds_module["train_dataset"]
        finally:
            self.data_args.dataset_dir = orig_dir
            self.data_args.dataset = orig_ds

        # Build vllm_inputs (mirror scripts/vllm_infer.py:199-253). Use the
        # batch view to keep memory bounded for larger requests.
        n = len(train_dataset["input_ids"])
        vllm_inputs: list[dict[str, Any]] = []
        for j in range(n):
            mmd: Optional[dict[str, Any]] = None
            img_j = train_dataset["images"][j] if "images" in train_dataset.column_names else None
            if img_j is not None:
                imgs = self.template_obj.mm_plugin._regularize_images(
                    img_j,
                    image_max_pixels=self.image_max_pixels,
                    image_min_pixels=self.image_min_pixels,
                )["images"]
                mmd = {"image": imgs}
            vllm_inputs.append({
                "prompt_token_ids": train_dataset["input_ids"][j],
                "multi_modal_data": mmd,
            })

        outputs = self.llm.generate(vllm_inputs, self.sampling_params)

        results: list[dict] = []
        for out in outputs:
            o = out.outputs[0]
            pred = o.text
            per_step = o.logprobs or []
            if len(per_step) <= self.decision_token_idx or per_step[self.decision_token_idx] is None:
                results.append({
                    "p_accept": 0.5, "logp_accept": None, "logp_reject": None, "pred": pred,
                })
                continue
            step = per_step[self.decision_token_idx]
            la_o = step.get(self.pos_token_id)
            lr_o = step.get(self.neg_token_id)
            la = float(la_o.logprob) if hasattr(la_o, "logprob") else (float(la_o) if la_o is not None else -50.0)
            lr = float(lr_o.logprob) if hasattr(lr_o, "logprob") else (float(lr_o) if lr_o is not None else -50.0)
            results.append({
                "p_accept": _softmax2(la, lr),
                "logp_accept": la,
                "logp_reject": lr,
                "pred": pred,
            })
        return results

    def health(self) -> dict:
        """Lightweight introspection for /health."""
        return {
            "status": "ok",
            "ckpt_path": self.ckpt_path,
            "template": self.template,
            "cutoff_len": self.cutoff_len,
            "max_new_tokens": self.max_new_tokens,
            "image_max_pixels": self.image_max_pixels,
            "image_min_pixels": self.image_min_pixels,
            "positive_token_id": self.pos_token_id,
            "negative_token_id": self.neg_token_id,
            "decision_token_idx": self.decision_token_idx,
            "compute_arch": self.compute_arch,
            "calls_served": self.calls_served,
        }
