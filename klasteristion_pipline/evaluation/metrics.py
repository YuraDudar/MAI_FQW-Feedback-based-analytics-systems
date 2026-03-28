"""
Clustering quality metrics: internal, external, and topic coherence.

All functions accept numpy arrays and integer label vectors.
Cluster label −1 (noise) is excluded from metric computation.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

log = logging.getLogger(__name__)


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: list[int] | np.ndarray,
    texts: list[str] | None = None,
    topic_model: Any = None,
    pool_name: str = "",
) -> dict:
    """
    Compute every available metric and return a flat dict.

    Metrics
    -------
    * silhouette_score         — cohesion vs separation (−1 … 1, ↑)
    * davies_bouldin_index     — intra/inter cluster ratio (0 … ∞, ↓)
    * calinski_harabasz_index  — ratio of between/within variance (↑)
    * noise_pct                — % documents in cluster −1
    * n_clusters               — number of real clusters (excl. noise)
    * cluster_sizes            — {topic_id: count}
    * topic_coherence_cv       — semantic coherence of top words (↑)
    """
    labels = np.asarray(labels)
    mask = labels != -1
    unique_real = set(labels[mask].tolist())

    result: dict = {
        "pool": pool_name,
        "n_docs": len(labels),
        "n_clusters": len(unique_real),
        "noise_count": int((~mask).sum()),
        "noise_pct": float((~mask).sum()) / len(labels) * 100 if len(labels) else 0.0,
    }

    sizes = []
    for tid in sorted(unique_real):
        sizes.append(int((labels == tid).sum()))
    result["cluster_sizes"] = dict(zip(sorted(unique_real), sizes))
    if sizes:
        result["cluster_size_mean"] = float(np.mean(sizes))
        result["cluster_size_std"] = float(np.std(sizes))
        result["cluster_size_min"] = int(np.min(sizes))
        result["cluster_size_max"] = int(np.max(sizes))

    if mask.sum() > 1 and len(unique_real) >= 2:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_samples,
            silhouette_score,
        )

        emb_clean = embeddings[mask]
        lab_clean = labels[mask]

        result["silhouette_score"] = float(silhouette_score(emb_clean, lab_clean, metric="cosine"))
        result["davies_bouldin_index"] = float(davies_bouldin_score(emb_clean, lab_clean))
        result["calinski_harabasz_index"] = float(calinski_harabasz_score(emb_clean, lab_clean))

        sil_per_sample = silhouette_samples(emb_clean, lab_clean, metric="cosine")
        per_cluster_sil = {}
        for tid in sorted(unique_real):
            idx = lab_clean == tid
            per_cluster_sil[int(tid)] = float(np.mean(sil_per_sample[idx]))
        result["silhouette_per_cluster"] = per_cluster_sil
    else:
        result["silhouette_score"] = None
        result["davies_bouldin_index"] = None
        result["calinski_harabasz_index"] = None
        result["silhouette_per_cluster"] = {}

    if texts is not None and topic_model is not None:
        try:
            result["topic_coherence_cv"] = _topic_coherence(texts, topic_model, measure="c_v")
            result["topic_coherence_npmi"] = _topic_coherence(texts, topic_model, measure="c_npmi")
        except Exception as e:
            log.warning("Topic coherence computation failed: %s", e)
            result["topic_coherence_cv"] = None
            result["topic_coherence_npmi"] = None
    else:
        result["topic_coherence_cv"] = None
        result["topic_coherence_npmi"] = None

    return result


def _topic_coherence(texts: list[str], topic_model, measure: str = "c_v") -> float:
    """Compute topic coherence using gensim CoherenceModel."""
    import re
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    _tok_re = re.compile(r"[а-яёa-z]{2,}")
    tokenized = [_tok_re.findall(t.lower()) for t in texts]
    tokenized = [t for t in tokenized if t]
    if len(tokenized) < 5:
        return 0.0

    dictionary = Dictionary(tokenized)

    topic_words = []
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        raw_words = [w for w, _ in topic_model.get_topic(tid)]
        single_tokens = []
        for w in raw_words:
            for token in _tok_re.findall(w.lower()):
                if token not in single_tokens:
                    single_tokens.append(token)
        valid = [t for t in single_tokens if t in dictionary.token2id]
        if valid:
            topic_words.append(valid[:10])

    if not topic_words:
        return 0.0

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence=measure,
    )
    return float(cm.get_coherence())


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    pool = metrics.get("pool", "")
    print(f"\n{'=' * 60}")
    print(f"  METRICS — {pool.upper()}" if pool else "  METRICS")
    print(f"{'=' * 60}")

    rows = [
        ["Documents", metrics["n_docs"]],
        ["Clusters", metrics["n_clusters"]],
        ["Noise", f"{metrics['noise_count']} ({metrics['noise_pct']:.1f}%)"],
    ]

    for key, label in [
        ("silhouette_score", "Silhouette ↑"),
        ("davies_bouldin_index", "Davies–Bouldin ↓"),
        ("calinski_harabasz_index", "Calinski–Harabasz ↑"),
        ("topic_coherence_cv", "Coherence Cv ↑"),
        ("topic_coherence_npmi", "Coherence NPMI ↑"),
    ]:
        val = metrics.get(key)
        rows.append([label, f"{val:.4f}" if val is not None else "—"])

    if metrics.get("cluster_sizes"):
        rows.append(["Cluster size (mean±std)",
                      f"{metrics.get('cluster_size_mean', 0):.1f} ± {metrics.get('cluster_size_std', 0):.1f}"])
        rows.append(["Cluster size (min/max)",
                      f"{metrics.get('cluster_size_min', 0)} / {metrics.get('cluster_size_max', 0)}"])

    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    sil = metrics.get("silhouette_per_cluster", {})
    if sil:
        print("\n  Silhouette per cluster:")
        for tid, val in sorted(sil.items()):
            bar = "█" * max(1, int(val * 40)) if val > 0 else "░" * max(1, int(abs(val) * 40))
            print(f"    Topic {tid:>3}: {val:+.3f}  {bar}")

    print(f"{'=' * 60}\n")


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Convert a list of metrics dicts to a flat DataFrame for comparison."""
    flat = []
    for m in metrics_list:
        row = {k: v for k, v in m.items()
               if k not in ("cluster_sizes", "silhouette_per_cluster")}
        flat.append(row)
    return pd.DataFrame(flat)
