"""
Sentence embeddings via sentence-transformers + intfloat/multilingual-e5-large.

Important
---------
e5-* models REQUIRE prefixes:
    "passage: <text>"  — when indexing documents
    "query: <text>"    — when embedding a search query

Mixing them up silently kills recall — keep the two methods separate.
"""
from __future__ import annotations

import gc
import logging
import time

import numpy as np

from rag_pipline.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
)

log = logging.getLogger(__name__)


class E5Embedder:
    """Lifecycle-managed wrapper for multilingual-e5-large."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = "auto",
        max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH,
    ):
        self.model_name = model_name
        self._device_arg = device
        self.max_seq_length = max_seq_length
        self._model = None
        self.dim = EMBEDDING_DIM


    def _resolve_device(self) -> str:
        import torch
        if self._device_arg == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._device_arg

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer
        device = self._resolve_device()
        log.info("Loading embedder %s on %s", self.model_name, device)
        self._model = SentenceTransformer(self.model_name, device=device, trust_remote_code=True)
        self._model.max_seq_length = self.max_seq_length

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


    def _encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool,
        progress_callback,
    ) -> np.ndarray:
        if self._model is None:
            self.load()

        if progress_callback is None:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=EMBEDDING_NORMALIZE,
                convert_to_numpy=True,
            )

        out_chunks: list[np.ndarray] = []
        total = len(texts)
        for i in range(0, total, batch_size):
            chunk = texts[i:i + batch_size]
            vecs = self._model.encode(
                chunk,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=EMBEDDING_NORMALIZE,
                convert_to_numpy=True,
            )
            out_chunks.append(vecs)
            try:
                progress_callback(min(i + batch_size, total), total)
            except Exception:
                pass
        return np.vstack(out_chunks) if out_chunks else np.zeros((0, self.dim), dtype=np.float32)

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
        progress_callback=None,
    ) -> np.ndarray:
        prefixed = [f"{EMBEDDING_PASSAGE_PREFIX}{t}" for t in texts]
        t0 = time.perf_counter()
        vecs = self._encode(prefixed, batch_size, show_progress, progress_callback)
        elapsed = time.perf_counter() - t0
        log.info("Embedded %d passages in %.1fs (%.1f items/s)", len(texts), elapsed, (len(texts) / elapsed) if elapsed else 0)
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        if self._model is None:
            self.load()
        prefixed = f"{EMBEDDING_QUERY_PREFIX}{text}"
        vec = self._model.encode(
            [prefixed],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=EMBEDDING_NORMALIZE,
            convert_to_numpy=True,
        )[0]
        return vec.astype(np.float32)


    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()
