"""
Online RAG orchestrator: 4-stage pipeline as specified by the project brief.

  Stage 1 — Query Expansion        (YandexGPT Lite, separate sync call)
  Stage 2 — Vector search           (Qdrant + filters at HNSW level)
  Stage 3 — Answer generation       (YandexGPT Pro, separate sync call)
  Stage 4 — Final response shaping  (text + array of source review_ids)

Reranking is intentionally absent (per spec). Precision is improved with two
post-retrieval knobs that are NOT reranking:
  * oversampling — fetch `top_k * factor` candidates, then truncate;
  * score threshold — drop hits below a min similarity.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rag_pipline.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_OVERSAMPLE_FACTOR,
    DEFAULT_TOP_K,
)
from rag_pipline.pipeline.embedder import E5Embedder
from rag_pipline.pipeline.indexer import QdrantStore
from rag_pipline.pipeline.retriever import build_qdrant_filter, format_hit

from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_expansion_user_prompt,
)
from .yandex_provider import YandexLLM

log = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Everything produced by one RAG turn — ready to render or export."""
    user_query: str
    expanded_query: str
    answer: str
    hits: list[dict] = field(default_factory=list)
    source_review_ids: list[str] = field(default_factory=list)
    filters: dict | None = None
    top_k: int = DEFAULT_TOP_K
    collection: str = ""
    timings: dict = field(default_factory=dict)
    transport: dict = field(default_factory=dict)
    error: str | None = None
    expansion_failed: bool = False
    expansion_skipped: bool = False
    min_score: float = DEFAULT_MIN_SCORE
    oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR
    candidates_fetched: int = 0   
    candidates_kept: int = 0      

    def to_export_dict(self) -> dict:
        return {
            "user_query": self.user_query,
            "expanded_query": self.expanded_query,
            "answer": self.answer,
            "filters": self.filters,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "oversample_factor": self.oversample_factor,
            "candidates_fetched": self.candidates_fetched,
            "candidates_kept": self.candidates_kept,
            "collection": self.collection,
            "source_review_ids": self.source_review_ids,
            "hits": self.hits,
            "timings": self.timings,
            "transport": self.transport,
            "expansion_failed": self.expansion_failed,
            "expansion_skipped": self.expansion_skipped,
            "error": self.error,
        }


class RAGOrchestrator:
    def __init__(
        self,
        *,
        store: QdrantStore,
        embedder: E5Embedder,
        lite_llm: YandexLLM,
        pro_llm: YandexLLM,
    ):
        self.store = store
        self.embedder = embedder
        self.lite = lite_llm
        self.pro = pro_llm


    def expand_query(self, query: str) -> tuple[str, bool]:
        """Stage 1. Returns (expanded_query, failed_flag)."""
        try:
            expanded = self.lite.generate(
                QUERY_EXPANSION_SYSTEM_PROMPT,
                build_expansion_user_prompt(query),
            ).strip()
            if not expanded:
                return query, True
            return expanded, False
        except Exception as exc:
            log.warning("Query expansion failed (%s) — falling back to raw query", exc)
            return query, True

    def retrieve(
        self,
        expanded_query: str,
        *,
        collection: str,
        top_k: int,
        filters: dict | None,
        oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> tuple[list[dict], int]:
        """
        Stage 2. Embed query (with `query: ` prefix), fetch oversampled candidate
        set, drop low-score hits, truncate to top_k.

        Returns (final_hits, n_candidates_before_filter).
        """
        qvec = self.embedder.embed_query(expanded_query)
        qfilter = build_qdrant_filter(filters)
        fetch_k = max(int(round(top_k * max(oversample_factor, 1.0))), top_k)
        raw_hits = self.store.search(collection, qvec, top_k=fetch_k, qdrant_filter=qfilter)
        candidates = [format_hit(h) for h in raw_hits]
        n_candidates = len(candidates)

        if min_score > 0:
            candidates = [h for h in candidates if (h.get("score") or 0.0) >= min_score]

        final_hits = candidates[:top_k]
        return final_hits, n_candidates

    def generate_answer(
        self,
        *,
        user_query: str,
        expanded_query: str,
        hits: list[dict],
        history: list[dict] | None,
        filters: dict | None,
    ) -> str:
        """Stage 3."""
        user_prompt = build_answer_user_prompt(
            query=user_query,
            expanded_query=expanded_query,
            hits=hits,
            history=history,
            filters=filters,
        )
        return self.pro.generate(ANSWER_SYSTEM_PROMPT, user_prompt).strip()


    def run(
        self,
        user_query: str,
        *,
        collection: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict | None = None,
        history: list[dict] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        oversample_factor: float = DEFAULT_OVERSAMPLE_FACTOR,
        skip_expansion: bool = False,
    ) -> RAGResult:
        timings: dict = {}
        result = RAGResult(
            user_query=user_query,
            expanded_query=user_query,
            answer="",
            collection=collection,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
            oversample_factor=oversample_factor,
            transport={"lite": self.lite.transport, "pro": self.pro.transport},
        )

        if skip_expansion:
            result.expansion_skipped = True
            result.expanded_query = user_query
            timings["expansion_sec"] = 0.0
        else:
            t0 = time.perf_counter()
            expanded, failed = self.expand_query(user_query)
            timings["expansion_sec"] = time.perf_counter() - t0
            result.expanded_query = expanded
            result.expansion_failed = failed

        try:
            t0 = time.perf_counter()
            hits, n_candidates = self.retrieve(
                result.expanded_query,
                collection=collection,
                top_k=top_k,
                filters=filters,
                oversample_factor=oversample_factor,
                min_score=min_score,
            )
            timings["retrieval_sec"] = time.perf_counter() - t0
        except Exception as exc:
            log.exception("Retrieval failed")
            result.error = f"Retrieval failed: {exc}"
            result.timings = timings
            return result
        result.hits = hits
        result.candidates_fetched = n_candidates
        result.candidates_kept = len(hits)
        result.source_review_ids = [h["review_id"] for h in hits if h.get("review_id")]

        try:
            t0 = time.perf_counter()
            answer = self.generate_answer(
                user_query=user_query,
                expanded_query=result.expanded_query,
                hits=hits,
                history=history,
                filters=filters,
            )
            timings["generation_sec"] = time.perf_counter() - t0
        except Exception as exc:
            log.exception("Answer generation failed")
            result.error = f"Answer generation failed: {exc}"
            result.timings = timings
            return result
        result.answer = answer

        timings["total_sec"] = sum(timings.values())
        result.timings = timings
        return result
