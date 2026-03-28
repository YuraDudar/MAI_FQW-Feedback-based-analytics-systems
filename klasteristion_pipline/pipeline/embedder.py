"""
Embedding Generator with strict VRAM lifecycle management.

State machine:
  IDLE → load_model() → LOADED → encode() → ... → unload() → IDLE

After encoding the full matrix is held in CPU RAM / disk; the GPU model
is explicitly deleted to make room for subsequent stages.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_CACHE_DIR
from pipeline.vram_utils import clear_gpu, log_memory_state

log = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate sentence embeddings with automatic VRAM management."""

    def __init__(self, model_key: str = DEFAULT_EMBEDDING_MODEL):
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from {list(EMBEDDING_MODELS)}")
        self.model_key = model_key
        self.cfg = EMBEDDING_MODELS[model_key]
        self._model = None


    def load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        log_memory_state("before load")
        log.info("Loading embedding model: %s (%s) …", self.model_key, self.cfg["name"])
        self._model = SentenceTransformer(
            self.cfg["name"],
            trust_remote_code=True,
        )
        self._model.max_seq_length = self.cfg["max_seq_length"]
        log_memory_state("after load")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        clear_gpu()
        log_memory_state("after unload")


    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode *texts* into an (N, D) float32 matrix.

        Uses on-disk cache keyed by (model_key, sha256(sorted_texts)) to avoid
        recomputation across experiment runs.
        """
        if use_cache:
            cache_path = self._cache_path(texts)
            if cache_path.exists():
                log.info("Loading cached embeddings from %s", cache_path.name)
                return np.load(cache_path)

        if self._model is None:
            self.load_model()

        batch_size = batch_size or self.cfg["batch_size"]
        prefix = self.cfg.get("prefix")
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]

        log.info("Encoding %d texts  (batch_size=%d, dim=%d) …", len(texts), batch_size, self.cfg["dim"])
        t0 = time.perf_counter()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        elapsed = time.perf_counter() - t0
        log.info("Encoding done in %.1f s  (%.1f texts/s)", elapsed, len(texts) / elapsed)

        if use_cache:
            np.save(cache_path, embeddings)
            log.info("Cached embeddings → %s", cache_path.name)

        return embeddings


    def _cache_path(self, texts: list[str]) -> Path:
        h = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        return EMBEDDINGS_CACHE_DIR / f"emb_{self.model_key}_{len(texts)}_{h}.npy"

    def clear_cache(self) -> None:
        for f in EMBEDDINGS_CACHE_DIR.glob(f"emb_{self.model_key}_*.npy"):
            f.unlink()
            log.info("Deleted cache file: %s", f.name)


    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, *exc):
        self.unload()
