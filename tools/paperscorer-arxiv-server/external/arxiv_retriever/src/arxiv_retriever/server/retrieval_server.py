import json
import time
import warnings
import logging
import difflib
import threading
from typing import List, Optional, Union
import argparse

import faiss
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def should_exclude_by_title(result_title: str, exclude_title: str, threshold: float = 0.98) -> bool:
    """
    Check if a result should be excluded due to title similarity.

    Uses difflib.SequenceMatcher for fuzzy string matching.

    Args:
        result_title: Title from the search result.
        exclude_title: Title to exclude (the paper being evaluated).
        threshold: Similarity threshold (default 0.98 = 98% match).

    Returns:
        True if the titles are too similar and should be excluded.
    """
    if not exclude_title or not result_title:
        return False
    similarity = difflib.SequenceMatcher(
        None,
        result_title.lower().strip(),
        exclude_title.lower().strip()
    ).ratio()
    return similarity > threshold

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retrieval_server")


# Model-to-pooling mapping for auto-detection
MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
    "qwen": "last_token",  # Qwen3 uses last token pooling
}

# Global DataFrame for author search (loaded if --arxiv_metadata_path provided)
arxiv_df = None
# id → row-position mapping into arxiv_df. Built at load time so the
# /retrieve handler can resolve doc['id'] (from retriever.corpus) to a
# specific arxiv_df row without relying on faiss-index positions matching
# arxiv_df row positions. This decoupling lets subset indexes (per_venue
# etc.) work correctly while preserving exact behavior for full-corpus
# runs where corpus and arxiv_df are already aligned.
arxiv_id_to_pos = None


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    # model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Embed sleep/wake: page the embedding model between GPU and CPU on demand.
# Used to free GPU memory during the SkyRL trainer's training phase, mirroring
# vLLM's sleep_mode pattern.
# ---------------------------------------------------------------------------
_embed_lock = threading.Lock()
# Module-level state ("awake"|"asleep") + the device the embed model lives on.
_embed_state: str = "awake"


def _move_embed_to(device: str) -> str:
    """Move the embed model attached to the global retriever to ``device``.

    Returns the new state string. Caller MUST hold ``_embed_lock``.
    """
    global _embed_state
    enc = getattr(retriever, "encoder", None) if "retriever" in globals() else None
    if enc is None or not hasattr(enc, "model"):
        # vllm-embed path or unloaded retriever — nothing to toggle.
        return _embed_state
    target = "cuda" if device == "cuda" else "cpu"
    current = next(enc.model.parameters()).device.type
    if current != target:
        enc.model.to(target)
        if target == "cpu":
            torch.cuda.empty_cache()
    _embed_state = "awake" if target == "cuda" else "asleep"
    logger.info(f"embed model moved to {target}; state={_embed_state}")
    return _embed_state


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last_token":
        # Qwen3-style last token pooling
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, task_description=None):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.task_description = task_description or "Given a web search query, retrieve relevant passages that answer the query"

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        _enc_t0 = time.perf_counter()
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        # Qwen3 query instruction
        if "qwen" in self.model_name.lower():
            if is_query:
                query_list = [f"Instruct: {self.task_description}\nQuery:{query}" for query in query_list]
            # Note: Documents don't need instruction prefix for Qwen3

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        # Honor whatever device the model currently lives on — supports the
        # sleep/wake pattern where the model is paged to CPU between training
        # steps and back to GPU before generation.
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            # Some models (e.g., Qwen3) don't have pooler_output
            pooler_output = getattr(output, 'pooler_output', None)
            query_emb = pooling(
                pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        _enc_ms = (time.perf_counter() - _enc_t0) * 1000.0
        logger.info(f"STAGE encode_ms={_enc_ms:.1f} batch={len(query_list)}")

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self._search_lock = threading.Lock()

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool, return_indices: bool = False):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False, return_indices: bool = False):
        # Lockless: faiss IndexFlatIP / IndexIVFFlat support concurrent
        # search() calls, and the encoder forward releases the GIL.
        return self._search(query, num, return_score, return_indices)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False, return_indices: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            empty_indices = [] if return_indices else None
            if return_indices and return_score:
                return [], [], []
            if return_score:
                return [], []
            return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:num]

        # Pyserini stores the input jsonl's `id` field as the lucene docid;
        # for our arxiv corpus that's the arxiv_id (e.g. "1909.10351").
        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "arxiv_id": hit.docid,
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for hit, content in zip(hits, all_contents)
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])
            for r, h in zip(results, hits):
                r.setdefault("arxiv_id", h.docid)

        # For BM25 the "indices" we expose are the arxiv_id strings (= docids).
        # The /retrieve handler detects retrieval_method == "bm25" and skips
        # the arxiv_df.iloc[idx] lookup since the docs already include
        # arxiv_id + contents, so these "indices" are pass-through metadata.
        if return_indices and return_score:
            return results, scores, [h.docid for h in hits]
        if return_score:
            return results, scores
        return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_gpus = getattr(config, 'faiss_gpus', None)
            if faiss_gpus:
                # Shard onto a specific subset of CUDA devices, leaving others
                # free for the embed model and the trainer's vLLM/FSDP.
                print(f"Sharding faiss index across CUDA devices {faiss_gpus} (useFloat16=True, shard=True)")
                self.index = faiss.index_cpu_to_gpus_list(self.index, co=co, gpus=faiss_gpus)
            else:
                print("Sharding faiss index across all visible CUDA devices (useFloat16=True, shard=True)")
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.use_vllm_embed = getattr(config, 'use_vllm_embed', False)

        if self.use_vllm_embed:
            from vllm import LLM
            print(f"Loading vLLM embed model: {config.retrieval_model_path} "
                  f"(TP={getattr(config, 'vllm_tensor_parallel_size', 1)}, "
                  f"gpu_mem={getattr(config, 'vllm_embed_gpu_memory_utilization', 0.05)})")
            self.embed_model = LLM(
                model=config.retrieval_model_path,
                task="embed",
                tensor_parallel_size=getattr(config, 'vllm_tensor_parallel_size', 1),
                gpu_memory_utilization=getattr(config, 'vllm_embed_gpu_memory_utilization', 0.05),
                enforce_eager=True,
            )
        else:
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=config.retrieval_model_path,
                pooling_method=config.retrieval_pooling_method,
                max_length=config.retrieval_query_max_length,
                use_fp16=config.retrieval_use_fp16,
                task_description=config.task_description,
            )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _encode_query(self, query):
        if self.use_vllm_embed:
            if isinstance(query, str):
                query = [query]
            outputs = self.embed_model.embed(query)
            embs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            return embs
        else:
            return self.encoder.encode(query)

    def _search(self, query: str, num: int = None, return_score: bool = False, return_indices: bool = False):
        if num is None:
            num = self.topk
        query_emb = self._encode_query(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_indices:
            return results, scores, idxs
        elif return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self._encode_query(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]
            results.extend(batch_results)
            scores.extend(batch_scores)
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        faiss_gpus: Optional[List[int]] = None,  # If set, shard onto these CUDA indices
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        task_description: str = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.faiss_gpus = faiss_gpus
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.task_description = task_description
        self.use_vllm_embed = False
        self.vllm_tensor_parallel_size = 1
        self.vllm_embed_gpu_memory_utilization = 0.05


class QueryRequest(BaseModel):
    # Either a single query string or a list of query strings. When a list is
    # supplied (and no date/title filters are set), the server batches encoder
    # forward + faiss search and returns one result-list per query.
    query: Optional[Union[str, List[str]]] = None
    last_name_author_search: Optional[str] = None  # Comma-separated last names
    topk: Optional[int] = None
    return_scores: bool = False
    lower_bound_datetime: Optional[str] = None  # ISO format: "2020-01-01"
    upper_bound_datetime: Optional[str] = None  # ISO format: "2023-12-31"
    exclude_title: Optional[str] = None  # Title to exclude (paper being evaluated)


app = FastAPI()


@app.get("/state")
def state_endpoint():
    """Returns whether the embed model is awake (on GPU) or asleep (on CPU)."""
    enc = getattr(retriever, "encoder", None) if "retriever" in globals() else None
    if enc is None or not hasattr(enc, "model"):
        return {"state": "n/a", "device": "n/a", "note": "no Encoder (bm25 or vllm_embed path)"}
    device = next(enc.model.parameters()).device.type
    return {"state": _embed_state, "device": device}


@app.post("/sleep")
def sleep_endpoint():
    """Page the embed model from GPU → CPU. Idempotent."""
    with _embed_lock:
        new_state = _move_embed_to("cpu")
    return {"state": new_state}


@app.post("/wake")
def wake_endpoint():
    """Page the embed model from CPU → GPU. Idempotent."""
    with _embed_lock:
        new_state = _move_embed_to("cuda")
    return {"state": new_state}


def build_date_mask(df, lower_bound, upper_bound):
    """Build a boolean mask for date filtering."""
    mask = pd.Series([True] * len(df))
    if lower_bound:
        mask = mask & (df['upload_date'] >= pd.to_datetime(lower_bound))
    if upper_bound:
        mask = mask & (df['upload_date'] <= pd.to_datetime(upper_bound))
    return mask


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts a single query and performs retrieval.
    Input format:
    {
      "query": "What is Python?",
      "topk": 3,
      "return_scores": true,
      "lower_bound_datetime": "2020-01-01",
      "upper_bound_datetime": "2023-12-31"
    }
    Or for author search:
    {
      "last_name_author_search": "smith, jones",
      "topk": 5
    }
    """
    # Log incoming request
    logger.info(f"REQUEST: query='{request.query}', author_search='{request.last_name_author_search}', topk={request.topk}, date_filter=[{request.lower_bound_datetime}, {request.upper_bound_datetime}]")

    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    has_date_filter = request.lower_bound_datetime or request.upper_bound_datetime

    # Author search mode
    if request.last_name_author_search:
        if arxiv_df is None:
            logger.warning("RESPONSE: Author search not available - no metadata loaded")
            return {"error": "Author search not available. Server not started with --arxiv_metadata_path"}

        # Parse comma-separated last names
        last_names = [name.strip().lower() for name in request.last_name_author_search.split(',')]

        # Build filter: all names must be present
        mask = pd.Series([True] * len(arxiv_df))
        for name in last_names:
            mask = mask & arxiv_df['authors_lower'].str.contains(name, na=False)

        # Add date filter if specified
        if has_date_filter:
            mask = mask & build_date_mask(arxiv_df, request.lower_bound_datetime, request.upper_bound_datetime)

        # Add title exclusion filter if specified
        if request.exclude_title:
            mask = mask & ~arxiv_df['title'].apply(
                lambda t: should_exclude_by_title(t, request.exclude_title)
            )

        # Get matching rows (include title for filtering)
        matching_rows = arxiv_df.loc[mask].head(request.topk)
        results = [{"arxiv_id": row['id'], "contents": row['content'], "title": row['title']} for _, row in matching_rows.iterrows()]

        if request.exclude_title:
            logger.info(f"RESPONSE [author_search]: {len(results)} results for authors '{request.last_name_author_search}' (excluding title similar to '{request.exclude_title[:50]}...')")
        else:
            logger.info(f"RESPONSE [author_search]: {len(results)} results for authors '{request.last_name_author_search}'")
        if results:
            logger.debug(f"  First result preview: {results[0]['contents'][:200]}...")
        return {"result": [results]}

    # Regular embedding search
    if not request.query:
        logger.warning("RESPONSE: Error - no query or author_search provided")
        return {"error": "Either 'query' or 'last_name_author_search' must be provided"}

    # Normalize to List[str] so we have one code path. Single-query callers
    # (query: "...") and multi-query callers (query: ["q1","q2","q3"]) flow
    # through the same logic.
    if isinstance(request.query, str):
        queries = [request.query] if request.query.strip() else []
    else:
        queries = [q.strip() for q in (request.query or []) if isinstance(q, str) and q.strip()]
    if not queries:
        return {"error": "Empty query"}

    # Determine how many results to fetch
    needs_post_filtering = has_date_filter or request.exclude_title
    if needs_post_filtering:
        if arxiv_df is None:
            logger.warning("RESPONSE: Error - date/title filtering requires metadata")
            return {"error": "Date/title filtering requires --arxiv_metadata_path"}
        fetch_num = request.topk * 20  # Fetch 20x for post-filtering
    else:
        fetch_num = request.topk

    # Auto-wake the embed model if a query arrives while it's asleep.
    # Fast-path is lockless; the trainer's wake/sleep cycle is sequenced
    # around generation so the asleep-while-querying race is essentially
    # never hit at steady state.
    if _embed_state == "asleep":
        with _embed_lock:
            if _embed_state == "asleep":
                logger.info("auto-wake: query arrived while embed model on CPU")
                _move_embed_to("cuda")

    # ─── Batched fast-path: multi-query with no post-filter ─────────────
    # Single batched encoder forward + single batched faiss matmul. Avoids
    # N sequential roundtrips for the SFT-style <ssearch>q1\nq2\n...</ssearch>.
    # Falls back to per-query loop when post-filter (date / title) is needed
    # because batch_search() doesn't return indices.
    if len(queries) > 1 and not needs_post_filtering:
        _t0 = time.perf_counter()
        results_list, scores_list = retriever.batch_search(
            query_list=queries, num=fetch_num, return_score=True,
        )
        _batch_ms = (time.perf_counter() - _t0) * 1000.0
        resp = []
        for docs, scores in zip(results_list, scores_list):
            if request.return_scores:
                resp.append([{"document": d, "score": float(s)} for d, s in zip(docs, scores)])
            else:
                resp.append(docs[:request.topk])
        logger.info(f"STAGE batch_search_ms={_batch_ms:.1f} batch={len(queries)}")
        logger.info(f"RESPONSE [batch_semantic_search]: {len(queries)} queries; results per query={[len(r) for r in resp]}")
        return {"result": resp}

    # ─── Per-query path: single query OR multi-query with filters ──────
    resp = []
    excluded_total = 0
    lower_dt = pd.to_datetime(request.lower_bound_datetime) if request.lower_bound_datetime else None
    upper_dt = pd.to_datetime(request.upper_bound_datetime) if request.upper_bound_datetime else None

    is_bm25 = getattr(retriever, "retrieval_method", "") == "bm25"

    for q in queries:
        _search_t0 = time.perf_counter()
        if is_bm25:
            # BM25 path: lucene returns docs with arxiv_id + contents already,
            # and date/title filters are not supported (would need metadata
            # lookup not done by pyserini). For our LitSearch eval we don't
            # need post-filter, so this branch is sufficient.
            results, scores = retriever.search(
                query=q, num=request.topk, return_score=True, return_indices=False,
            )
            idxs = []  # unused in BM25 path
        else:
            results, scores, idxs = retriever.search(
                query=q, num=fetch_num, return_score=True, return_indices=True,
            )
        _search_ms = (time.perf_counter() - _search_t0) * 1000.0

        # Post-filter (date / title) + content lookup. Resolve each retrieved
        # doc to its arxiv_df row by id (via arxiv_id_to_pos), not by faiss
        # position. This makes the lookup correct for both full-corpus and
        # subset indexes; for the full-corpus case it yields the same row
        # as the legacy iloc[idx] for all unique ids.
        # Skip for BM25 (already has arxiv_id + contents from lucene).
        if not is_bm25 and arxiv_df is not None:
            filtered_results = []
            filtered_scores = []
            for score, doc in zip(scores, results):
                arxiv_id = doc['id']
                pos = arxiv_id_to_pos.get(arxiv_id) if arxiv_id_to_pos is not None else None
                if pos is not None:
                    row = arxiv_df.iloc[pos]
                    if has_date_filter:
                        doc_date = row['upload_date']
                        if lower_dt and doc_date < lower_dt:
                            continue
                        if upper_dt and doc_date > upper_dt:
                            continue
                    if request.exclude_title and should_exclude_by_title(row['title'], request.exclude_title):
                        excluded_total += 1
                        continue
                    filtered_results.append({"arxiv_id": arxiv_id, "contents": row['content'], "title": row['title']})
                    filtered_scores.append(score)
                else:
                    # arxiv_id not in arxiv_df (rare: corpus has papers absent
                    # from the metadata snapshot). Fall back to corpus contents
                    # and skip filters that need metadata.
                    if has_date_filter or request.exclude_title:
                        continue
                    filtered_results.append({"arxiv_id": arxiv_id, "contents": doc.get('contents', ''), "title": ''})
                    filtered_scores.append(score)
                if len(filtered_results) >= request.topk:
                    break
            results = filtered_results
            scores = filtered_scores
        elif not is_bm25:
            # No arxiv_df loaded — surface arxiv_id from corpus directly.
            results = [{"arxiv_id": doc['id'], "contents": doc.get('contents', ''), "title": ''}
                       for doc in results[:request.topk]]
            scores = list(scores[:request.topk])

        if request.return_scores:
            resp.append([{"document": d, "score": float(s)} for d, s in zip(results, scores)])
        else:
            resp.append(results[:request.topk])

        logger.info(f"STAGE search_ms={_search_ms:.1f} q='{q[:60]}' method={'bm25' if is_bm25 else 'dense'}")

    if excluded_total:
        logger.info(f"Excluded {excluded_total} results due to title similarity with '{(request.exclude_title or '')[:50]}...'")

    logger.info(f"RESPONSE [semantic_search]: {len(queries)} quer{'y' if len(queries)==1 else 'ies'}; results per query={[len(r) for r in resp]}")

    return {"result": resp}


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Launch the local faiss retriever (OmegaConf config).")
    parser.add_argument("--config", required=True, type=str, help="Path to OmegaConf YAML config.")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    print("Loaded retrieval config:")
    print(OmegaConf.to_yaml(cfg))

    retriever_name = cfg.retriever.name
    pooling_method = cfg.retriever.get("pooling_method", None)
    if pooling_method is None:
        pooling_method = "mean"
        for k, v in MODEL2POOLING.items():
            if k in retriever_name.lower():
                pooling_method = v
                break
    print(f"Using pooling method: {pooling_method}")

    # `server.faiss_gpus`: optional list of CUDA indices to shard the faiss
    # index over (e.g. [1,2,3] to keep cuda:0 free for the embed model). If
    # absent, faiss spreads across all visible GPUs (the legacy default).
    faiss_gpus_cfg = cfg.server.get("faiss_gpus", None)
    if faiss_gpus_cfg is not None:
        # Hydra/OmegaConf list -> python list of ints
        faiss_gpus_cfg = [int(x) for x in faiss_gpus_cfg]

    config = Config(
        retrieval_method=retriever_name,
        index_path=cfg.index.path,
        corpus_path=cfg.index.corpus_path,
        retrieval_topk=cfg.server.topk,
        faiss_gpu=cfg.server.faiss_gpu,
        faiss_gpus=faiss_gpus_cfg,
        retrieval_model_path=cfg.retriever.model_path,
        retrieval_pooling_method=pooling_method,
        retrieval_query_max_length=cfg.retriever.get("query_max_length", 256),
        retrieval_use_fp16=cfg.retriever.get("use_fp16", True),
        retrieval_batch_size=cfg.retriever.get("batch_size", 512),
        task_description=cfg.retriever.get("task_description", None),
    )
    config.use_vllm_embed = cfg.retriever.get("use_vllm_embed", False)
    config.vllm_tensor_parallel_size = cfg.retriever.get("tensor_parallel_size", 1)
    config.vllm_embed_gpu_memory_utilization = cfg.retriever.get("embed_gpu_memory_utilization", 0.05)

    retriever = get_retriever(config)

    arxiv_metadata_path = cfg.index.get("arxiv_metadata_path", None)
    if arxiv_metadata_path:
        print("Loading arxiv metadata for author search...")
        arxiv_df = pd.read_json(arxiv_metadata_path, lines=True)
        arxiv_df['upload_date'] = pd.to_datetime(arxiv_df['versions'].map(lambda x: x[0]['created']))

        def truncate_abstract(abstract, max_chars=150):
            if pd.isna(abstract):
                return ""
            abstract = str(abstract).strip()
            if len(abstract) > max_chars:
                return abstract[:max_chars] + "..."
            return abstract

        arxiv_df['content'] = (
            '"' + arxiv_df['title'].str.replace('\n', ' ').str.strip() + '"\n' +
            'Authors:' + arxiv_df['authors'].str.replace('\n', ' ').str.strip() + '\n\n' +
            'Abstract:' + arxiv_df['abstract'] + '\n\n' +
            'DOI:' + arxiv_df['doi'].fillna('N/A') + '\n' +
            'Publish Date:' + arxiv_df['upload_date'].dt.strftime('%Y-%m-%d')
        )
        arxiv_df['authors_lower'] = arxiv_df['authors'].str.lower()
        print(f"Loaded {len(arxiv_df)} arxiv papers for author/date search")

        # Build arxiv_id → row-position map for id-based lookup in /retrieve.
        # For the full-corpus case (corpus and arxiv_df row-aligned), this
        # yields the same row as the legacy iloc[idx] path for all unique
        # ids; only ~26 duplicate-id papers resolve to first occurrence
        # rather than whichever-faiss-returned. For subset indexes (e.g.
        # per_venue), this is the correct lookup since faiss positions
        # don't align with arxiv_df row positions.
        _id_pos = pd.Series(range(len(arxiv_df)), index=arxiv_df['id'])
        arxiv_id_to_pos = _id_pos[~_id_pos.index.duplicated(keep='first')].to_dict()
        print(f"Built id→position map for {len(arxiv_id_to_pos)} unique arxiv_ids")

    # Default boot state: embed model parked on CPU so we don't compete with
    # the SkyRL trainer's vLLM warmup. Trainer (or any /retrieve) will wake
    # it on first need. Override with `server.start_awake=true` to keep the
    # legacy always-on-GPU behaviour.
    if not cfg.server.get("start_awake", False):
        with _embed_lock:
            _move_embed_to("cpu")
        print(f"Embed model parked on CPU at boot (state=asleep). POST /wake to load onto GPU.")

    port = cfg.server.get("port", 8000)
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
