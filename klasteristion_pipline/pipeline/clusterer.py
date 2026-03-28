"""
BERTopic-based clustering with UMAP → HDBSCAN → c-TF-IDF pipeline.

Operates entirely on CPU (State 2 in the VRAM state machine) because
embeddings are pre-computed on GPU and passed as numpy arrays.

Falls back to KMeans when HDBSCAN produces too few clusters for the
target business requirement (5-8 clusters for 200-600 reviews).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import (
    DEFAULT_UMAP_PARAMS,
    DEFAULT_HDBSCAN_PARAMS,
    UMAP_VIS_PARAMS,
    VECTORIZER_PARAMS,
    ALL_STOP_WORDS,
    TOP_N_WORDS,
    MAX_TOPICS,
    get_target_topics,
)

log = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Container for all outputs of a single clustering run."""
    pool_name: str
    topics: list[int] = field(default_factory=list)
    probs: np.ndarray | None = None
    topic_info: pd.DataFrame | None = None
    topic_model: object = None  
    embeddings: np.ndarray | None = None
    umap_2d: np.ndarray | None = None
    umap_3d: np.ndarray | None = None
    texts: list[str] = field(default_factory=list)
    n_clusters: int = 0
    noise_count: int = 0
    noise_pct: float = 0.0
    elapsed_sec: float = 0.0
    clustering_method: str = "hdbscan"


class ReviewClusterer:
    """
    Wraps BERTopic with custom UMAP / HDBSCAN / Vectorizer.

    Usage::

        clusterer = ReviewClusterer(umap_params={...}, hdbscan_params={...})
        result = clusterer.fit(texts, embeddings, pool_name="negative")
    """

    def __init__(
        self,
        umap_params: dict | None = None,
        hdbscan_params: dict | None = None,
        vectorizer_params: dict | None = None,
        target_topics: int | None = None,
        embedding_model_name: str | None = None,
        representation: str = "default_ctfidf",
        min_clusters: int | None = None,
        build_visualizations: bool = True,
    ):
        self.umap_params = {**DEFAULT_UMAP_PARAMS, **(umap_params or {})}
        self.hdbscan_params = {**DEFAULT_HDBSCAN_PARAMS, **(hdbscan_params or {})}
        self.vectorizer_params = {**VECTORIZER_PARAMS, **(vectorizer_params or {})}
        self.target_topics = target_topics
        self.embedding_model_name = embedding_model_name
        self.representation = representation  
        self.min_clusters = min_clusters
        self.build_visualizations = build_visualizations

    def fit(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        pool_name: str = "",
    ) -> ClusteringResult:
        from umap import UMAP

        n = len(texts)
        log.info("[%s] Clustering %d documents …", pool_name, n)
        t0 = time.perf_counter()

        target = self.target_topics or get_target_topics(n)
        target = min(target, MAX_TOPICS)
        min_acceptable = self.min_clusters or max(3, target // 2)

        topic_model, topics, probs = self._fit_hdbscan(texts, embeddings, pool_name)
        n_real = len(set(topics)) - (1 if -1 in topics else 0)
        method = "hdbscan"

        if n_real < min_acceptable:
            log.info(
                "[%s] HDBSCAN found %d clusters (need >= %d), switching to KMeans(k=%d)",
                pool_name, n_real, min_acceptable, target,
            )
            topic_model, topics, probs = self._fit_kmeans(texts, embeddings, target)
            method = "kmeans"

        current = len(set(topics)) - (1 if -1 in topics else 0)
        if current > MAX_TOPICS:
            log.info("[%s] Reducing topics %d → %d", pool_name, current, MAX_TOPICS)
            topic_model.reduce_topics(texts, nr_topics=MAX_TOPICS)
            topics = topic_model.topics_

        topic_info = topic_model.get_topic_info()
        elapsed = time.perf_counter() - t0

        noise_count = sum(1 for t in topics if t == -1)
        real_topics = set(topics) - {-1}

        result = ClusteringResult(
            pool_name=pool_name,
            topics=topics,
            probs=probs if isinstance(probs, np.ndarray) else None,
            topic_info=topic_info,
            topic_model=topic_model,
            embeddings=embeddings,
            texts=texts,
            n_clusters=len(real_topics),
            noise_count=noise_count,
            noise_pct=noise_count / n * 100 if n else 0,
            elapsed_sec=elapsed,
            clustering_method=method,
        )

        if self.build_visualizations:
            result.umap_2d = self._reduce_for_vis(embeddings, n_components=2)
            result.umap_3d = self._reduce_for_vis(embeddings, n_components=3)

        self._print_summary(result)
        return result


    def _build_representation_model(self):
        """Build a BERTopic representation_model based on self.representation."""
        if self.representation == "keybert_inspired":
            from bertopic.representation import KeyBERTInspired
            return KeyBERTInspired(top_n_words=TOP_N_WORDS)
        if self.representation == "mmr_diversity":
            from bertopic.representation import MaximalMarginalRelevance
            return MaximalMarginalRelevance(diversity=0.3, top_n_words=TOP_N_WORDS)
        return None  

    def _fit_hdbscan(self, texts, embeddings, pool_name):
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        hdbscan_params = self._adapt_hdbscan(len(texts))
        umap_model = UMAP(**self.umap_params)
        hdbscan_model = HDBSCAN(**hdbscan_params)
        vectorizer = self._make_vectorizer(with_stop_words=True)
        representation_model = self._build_representation_model()

        embedding_model = self._load_embedding_model()

        topic_model = BERTopic(
            language="multilingual",
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            top_n_words=TOP_N_WORDS,
            nr_topics=None,
            verbose=False,
            calculate_probabilities=True,
        )

        topics, probs = self._safe_fit(topic_model, texts, embeddings)
        return topic_model, topics, probs

    def _fit_kmeans(self, texts, embeddings, n_clusters):
        from bertopic import BERTopic
        from sklearn.cluster import KMeans
        from umap import UMAP

        umap_model = UMAP(**self.umap_params)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        vectorizer = self._make_vectorizer(with_stop_words=True)
        representation_model = self._build_representation_model()

        embedding_model = self._load_embedding_model()

        topic_model = BERTopic(
            language="multilingual",
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=kmeans,
            vectorizer_model=vectorizer,
            representation_model=representation_model,
            top_n_words=TOP_N_WORDS,
            nr_topics=None,
            verbose=False,
            calculate_probabilities=False,
        )

        topics, probs = self._safe_fit(topic_model, texts, embeddings)
        return topic_model, topics, probs

    def _safe_fit(self, topic_model, texts, embeddings):
        """Fit with fallback: if stop words prune all terms, retry without them."""
        try:
            return topic_model.fit_transform(texts, embeddings=embeddings)
        except ValueError as exc:
            if "empty vocabulary" in str(exc) or "no terms remain" in str(exc):
                log.warning("Vectorizer failed with stop words, retrying without them")
                topic_model.vectorizer_model = self._make_vectorizer(with_stop_words=False)
                return topic_model.fit_transform(texts, embeddings=embeddings)
            raise

    def _make_vectorizer(self, with_stop_words: bool = True):
        from sklearn.feature_extraction.text import CountVectorizer

        kw = dict(self.vectorizer_params)
        if with_stop_words:
            kw["stop_words"] = ALL_STOP_WORDS
        return CountVectorizer(**kw)

    def _load_embedding_model(self):
        if not self.embedding_model_name:
            return None
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.embedding_model_name, trust_remote_code=True)


    def _adapt_hdbscan(self, n: int) -> dict:
        """Scale min_cluster_size relative to dataset size."""
        params = dict(self.hdbscan_params)
        mcs = params["min_cluster_size"]
        if n < 50:
            mcs = max(3, min(mcs, 5))
        elif n < 200:
            mcs = max(3, min(mcs, 8))
        params["min_cluster_size"] = mcs
        return params

    @staticmethod
    def _reduce_for_vis(embeddings: np.ndarray, n_components: int) -> np.ndarray:
        """UMAP reduction specifically for visualisation (2-D or 3-D)."""
        from umap import UMAP

        params = dict(UMAP_VIS_PARAMS)
        params["n_components"] = n_components
        reducer = UMAP(**params)
        return reducer.fit_transform(embeddings)

    @staticmethod
    def _print_summary(r: ClusteringResult) -> None:
        print(f"\n{'=' * 60}")
        print(f"  CLUSTERING RESULT — {r.pool_name.upper()}")
        print(f"{'=' * 60}")
        print(f"  Documents  : {len(r.texts):,}")
        print(f"  Clusters   : {r.n_clusters}")
        print(f"  Noise (–1) : {r.noise_count} ({r.noise_pct:.1f}%)")
        print(f"  Method     : {r.clustering_method}")
        print(f"  Time       : {r.elapsed_sec:.1f} s")

        if r.topic_info is not None and len(r.topic_info):
            rows = []
            for _, row in r.topic_info.iterrows():
                tid = row.get("Topic", "?")
                if tid == -1:
                    continue
                count = row.get("Count", 0)
                name = row.get("Name", "")
                rows.append({"topic": tid, "count": count, "keywords": name})
            if rows:
                print()
                print(tabulate(rows, headers="keys", tablefmt="simple"))

        print(f"{'=' * 60}\n")
