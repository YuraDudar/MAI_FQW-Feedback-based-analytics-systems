"""
Compact positive-pool refinement experiment.

Goals:
1) Search a compact grid with additional HDBSCAN `min_samples`.
2) Compare bge-m3 vs e5-large on the same compact grid.
3) Run KeyBERT representation on the best config per model.
4) Produce final per-pool presets for the next full pipeline stage.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tabulate import tabulate

from config import REPORTS_DIR, EMBEDDING_MODELS
from experiments.exp2_umap_hdbscan import _composite_score
from experiments.exp3_topic_coherence import _build_topic_model
from pipeline.data_loader import load_reviews
from pipeline.preprocessor import preprocess, heuristic_split, pool_texts
from pipeline.embedder import EmbeddingGenerator
from pipeline.clusterer import ReviewClusterer
from evaluation.metrics import compute_all_metrics, metrics_to_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("exp2_positive_refine")

MODELS = ["bge-m3", "e5-large"]
UMAP_N_NEIGHBORS = [10, 15, 20]
UMAP_N_COMPONENTS = [5, 10]
HDBSCAN_MIN_CLUSTER_SIZE = [5, 8, 12]
HDBSCAN_MIN_SAMPLES = [1, 2, 3]
TARGET_CLUSTERS = 8


def _cluster_balance_penalty(row: pd.Series) -> float:
    """Penalize giant dominant clusters."""
    n_docs = max(float(row.get("n_docs", 1)), 1.0)
    max_cluster = float(row.get("cluster_size_max", 0))
    dominance = max_cluster / n_docs
    if dominance <= 0.45:
        return 0.0
    if dominance >= 0.80:
        return -0.20
    frac = (dominance - 0.45) / (0.80 - 0.45)
    return -0.20 * frac


def _refine_score(row: pd.Series) -> float:
    """
    Composite score aligned with desired 7-10 positive clusters.
    """
    base = _composite_score(row, target_clusters=TARGET_CLUSTERS)
    n_clust = int(row.get("n_clusters", 0))
    in_band = 0.08 if 7 <= n_clust <= 10 else -0.08
    balance = _cluster_balance_penalty(row)
    return float(base + in_band + balance)


def run(csv_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_reviews(csv_path)
    df = preprocess(df)
    split = heuristic_split(df)
    texts = pool_texts(split.positive_pool)
    if len(texts) < 10:
        raise RuntimeError("Positive pool is too small for refinement.")

    grid = list(itertools.product(
        UMAP_N_NEIGHBORS,
        UMAP_N_COMPONENTS,
        HDBSCAN_MIN_CLUSTER_SIZE,
        HDBSCAN_MIN_SAMPLES,
    ))
    total = len(grid) * len(MODELS)

    print("\n" + "=" * 76)
    print("  POSITIVE REFINE — COMPACT GRID SEARCH")
    print("=" * 76)
    print(f"Models: {MODELS}")
    print(f"n_neighbors: {UMAP_N_NEIGHBORS}")
    print(f"n_components: {UMAP_N_COMPONENTS}")
    print(f"min_cluster_size: {HDBSCAN_MIN_CLUSTER_SIZE}")
    print(f"min_samples: {HDBSCAN_MIN_SAMPLES}")
    print(f"Total runs: {total}")
    print("=" * 76)

    all_metrics: list[dict] = []
    run_idx = 0

    for model_key in MODELS:
        emb_gen = EmbeddingGenerator(model_key)
        try:
            embeddings = emb_gen.encode(texts, use_cache=True)
        finally:
            emb_gen.unload()

        for nn, nc, mcs, ms in grid:
            run_idx += 1
            tag = f"{model_key}_nn{nn}_nc{nc}_mcs{mcs}_ms{ms}"
            log.info("[%d/%d] %s", run_idx, total, tag)
            t0 = time.perf_counter()

            clusterer = ReviewClusterer(
                umap_params={"n_neighbors": nn, "n_components": nc},
                hdbscan_params={"min_cluster_size": mcs, "min_samples": ms},
                min_clusters=1,
                build_visualizations=False,
            )
            result = clusterer.fit(texts, embeddings, pool_name=f"positive_{tag}")

            m = compute_all_metrics(
                embeddings=embeddings,
                labels=result.topics,
                texts=None,
                topic_model=None,
                pool_name=tag,
            )
            m["model"] = model_key
            m["n_neighbors"] = nn
            m["n_components"] = nc
            m["min_cluster_size"] = mcs
            m["min_samples"] = ms
            m["label"] = tag
            m["time_sec"] = time.perf_counter() - t0
            m["clustering_method"] = result.clustering_method
            all_metrics.append(m)

    fast_df = metrics_to_dataframe(all_metrics)
    fast_df["refine_score"] = fast_df.apply(_refine_score, axis=1)

    selected_rows = []
    for model_key in MODELS:
        subset = fast_df[fast_df["model"] == model_key].copy()
        subset = subset[subset["noise_pct"] <= 35] if "noise_pct" in subset.columns else subset
        if len(subset) == 0:
            subset = fast_df[fast_df["model"] == model_key].copy()
        best = subset.loc[subset["refine_score"].idxmax()]
        selected_rows.append(best)
    selected_df = pd.DataFrame(selected_rows)

    keybert_metrics: list[dict] = []
    for _, row in selected_df.iterrows():
        model_key = str(row["model"])
        nn = int(row["n_neighbors"])
        nc = int(row["n_components"])
        mcs = int(row["min_cluster_size"])
        ms = int(row["min_samples"])

        emb_gen = EmbeddingGenerator(model_key)
        try:
            embeddings = emb_gen.encode(texts, use_cache=True)
        finally:
            emb_gen.unload()

        model_hf_name = EMBEDDING_MODELS[model_key]["name"]
        topic_model, topics, _ = _build_topic_model(
            texts=texts,
            embeddings=embeddings,
            representation="keybert",
            umap_params={"n_neighbors": nn, "n_components": nc, "min_dist": 0.0, "metric": "cosine", "random_state": 42, "low_memory": True},
            hdbscan_params={"min_cluster_size": mcs, "min_samples": ms, "metric": "euclidean", "cluster_selection_method": "eom", "prediction_data": True},
            target_topics=TARGET_CLUSTERS,
            model_hf_name=model_hf_name,
        )
        mm = compute_all_metrics(
            embeddings=embeddings,
            labels=topics,
            texts=texts,
            topic_model=topic_model,
            pool_name=f"{model_key}_keybert",
        )
        mm["model"] = model_key
        mm["representation"] = "keybert_inspired"
        mm["n_neighbors"] = nn
        mm["n_components"] = nc
        mm["min_cluster_size"] = mcs
        mm["min_samples"] = ms
        keybert_metrics.append(mm)

    keybert_df = metrics_to_dataframe(keybert_metrics)

    fast_path = REPORTS_DIR / "exp2_positive_refine_fast.csv"
    keybert_path = REPORTS_DIR / "exp2_positive_refine_keybert_compare.csv"
    preset_path = REPORTS_DIR / "pipeline_presets_candidate.json"
    fast_df.to_csv(fast_path, index=False)
    keybert_df.to_csv(keybert_path, index=False)

    best_keybert = keybert_df.loc[keybert_df["topic_coherence_cv"].idxmax()].to_dict()
    presets = {
        "negative": {
            "model": "e5-large",
            "representation": "default_ctfidf",
            "umap": {"n_neighbors": 5, "n_components": 10},
            "hdbscan": {"min_cluster_size": 12, "min_samples": 2},
            "source": "exp2_negative_grid_results + exp3_negative_results",
        },
        "positive": {
            "model": best_keybert["model"],
            "representation": "keybert_inspired",
            "umap": {
                "n_neighbors": int(best_keybert["n_neighbors"]),
                "n_components": int(best_keybert["n_components"]),
            },
            "hdbscan": {
                "min_cluster_size": int(best_keybert["min_cluster_size"]),
                "min_samples": int(best_keybert["min_samples"]),
            },
            "source": "exp2_positive_refine + keybert comparison",
        },
    }
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print("  POSITIVE REFINE — TOP CONFIGS PER MODEL")
    print("=" * 76)
    show_cols = [
        "model", "n_neighbors", "n_components", "min_cluster_size", "min_samples",
        "n_clusters", "noise_pct", "silhouette_score", "cluster_size_max", "refine_score",
    ]
    print(tabulate(selected_df[show_cols], headers="keys", tablefmt="grid", floatfmt=".4f"))

    print("\n" + "=" * 76)
    print("  POSITIVE REFINE — KEYBERT COMPARISON")
    print("=" * 76)
    show_cols2 = [
        "model", "n_neighbors", "n_components", "min_cluster_size", "min_samples",
        "n_clusters", "noise_pct", "silhouette_score", "topic_coherence_cv", "topic_coherence_npmi",
    ]
    print(tabulate(keybert_df[show_cols2], headers="keys", tablefmt="grid", floatfmt=".4f"))

    print(f"\nSaved: {fast_path}")
    print(f"Saved: {keybert_path}")
    print(f"Saved: {preset_path}")

    return fast_df, keybert_df


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run(csv_arg)
