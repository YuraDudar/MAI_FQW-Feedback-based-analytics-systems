"""
Experiment 2 — UMAP & HDBSCAN Grid Search
==========================================
Find the best combination of UMAP ``n_neighbors``, ``n_components``,
and HDBSCAN ``min_cluster_size`` using the winning embedding model
from Experiment 1.

Best configuration is selected by a composite score that balances:
  - Silhouette score (cluster separation)
  - Closeness to target cluster count (business requirement)
  - Low noise percentage
"""
from __future__ import annotations

import itertools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import REPORTS_DIR, get_target_topics
from pipeline.data_loader import load_reviews
from pipeline.preprocessor import preprocess, heuristic_split, pool_texts
from pipeline.embedder import EmbeddingGenerator
from pipeline.clusterer import ReviewClusterer
from evaluation.metrics import compute_all_metrics, metrics_to_dataframe
from evaluation.visualizer import PipelineVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("exp2")

UMAP_N_NEIGHBORS = [5, 10, 15, 25]
UMAP_N_COMPONENTS = [3, 5, 10]
HDBSCAN_MIN_CLUSTER_SIZE = [3, 5, 8, 12]


def _composite_score(row: pd.Series, target_clusters: int) -> float:
    """
    Composite metric that balances cluster quality with business requirements.

    Components (each normalised to ~[0, 1]):
      - silhouette:     higher is better           (weight 0.25)
      - cluster_match:  closer to target is better  (weight 0.40)
      - low_noise:      lower noise is better       (weight 0.20)
      - enough_clusters: penalise configs with < 3 real clusters (weight 0.15)
    """
    sil = float(row.get("silhouette_score", 0))
    noise = float(row.get("noise_pct", 0))
    n_clust = int(row.get("n_clusters", 1))

    sil_norm = (sil + 1) / 2
    cluster_dist = abs(n_clust - target_clusters) / max(target_clusters, 1)
    cluster_match = max(0.0, 1.0 - cluster_dist)
    noise_score = max(0.0, 1.0 - noise / 50.0)
    enough = 1.0 if n_clust >= 3 else 0.2

    return 0.25 * sil_norm + 0.40 * cluster_match + 0.20 * noise_score + 0.15 * enough


def run(
    csv_path: str | None = None,
    pool: str = "negative",
    model_key: str = "bge-m3",
) -> pd.DataFrame:
    """Run grid search for UMAP + HDBSCAN params."""
    df = load_reviews(csv_path)
    df = preprocess(df)
    split = heuristic_split(df)

    entries = split.negative_pool if pool == "negative" else split.positive_pool
    texts = pool_texts(entries)

    if len(texts) < 10:
        log.error("Pool '%s' has only %d texts — aborting.", pool, len(texts))
        return pd.DataFrame()

    target_clusters = get_target_topics(len(texts))

    emb_gen = EmbeddingGenerator(model_key)
    try:
        embeddings = emb_gen.encode(texts, use_cache=True)
    finally:
        emb_gen.unload()

    grid = list(itertools.product(UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS, HDBSCAN_MIN_CLUSTER_SIZE))
    all_metrics: list[dict] = []
    total = len(grid)

    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT 2 — GRID SEARCH ({total} combinations)")
    print(f"  n_neighbors={UMAP_N_NEIGHBORS}")
    print(f"  n_components={UMAP_N_COMPONENTS}")
    print(f"  min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}")
    print(f"  target_clusters={target_clusters}")
    print(f"{'=' * 60}")

    viz = PipelineVisualizer()

    for i, (nn, nc, mcs) in enumerate(grid, 1):
        tag = f"nn{nn}_nc{nc}_mcs{mcs}"
        log.info("[%d/%d] n_neighbors=%d  n_components=%d  min_cluster_size=%d", i, total, nn, nc, mcs)

        t0 = time.perf_counter()

        clusterer = ReviewClusterer(
            umap_params={"n_neighbors": nn, "n_components": nc},
            hdbscan_params={"min_cluster_size": mcs},
            min_clusters=1,
        )

        try:
            result = clusterer.fit(texts, embeddings, pool_name=f"{pool}_{tag}")
        except Exception as e:
            log.warning("Combination %s failed: %s", tag, e)
            continue

        m = compute_all_metrics(
            embeddings=embeddings,
            labels=result.topics,
            texts=texts,
            topic_model=result.topic_model,
            pool_name=tag,
        )
        m["n_neighbors"] = nn
        m["n_components"] = nc
        m["min_cluster_size"] = mcs
        m["label"] = tag
        m["time_sec"] = time.perf_counter() - t0
        m["clustering_method"] = result.clustering_method
        all_metrics.append(m)

    results_df = metrics_to_dataframe(all_metrics)

    print("\n" + "=" * 80)
    print("  EXPERIMENT 2 — FULL RESULTS")
    print("=" * 80)
    display_cols = [
        "n_neighbors", "n_components", "min_cluster_size", "n_clusters", "noise_pct",
        "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index",
    ]
    show = [c for c in display_cols if c in results_df.columns]
    print(tabulate(results_df[show], headers="keys", tablefmt="grid", floatfmt=".4f"))

    for metric_col, title in [
        ("silhouette_score", "Silhouette Score"),
        ("noise_pct", "Noise %"),
        ("calinski_harabasz_index", "Calinski\u2013Harabasz"),
        ("n_clusters", "Number of Clusters"),
    ]:
        if metric_col not in results_df.columns:
            continue
        try:
            fig = viz.plot_experiment_heatmap(
                results_df, x_col="n_neighbors", y_col="min_cluster_size",
                value_col=metric_col, title=f"Exp 2 \u2014 {title}",
            )
            viz.save_figures({f"exp2_{pool}_{metric_col}_heatmap": fig})
        except Exception as e:
            log.warning("Heatmap for %s failed: %s", metric_col, e)

    results_df["composite_score"] = results_df.apply(
        _composite_score, axis=1, target_clusters=target_clusters,
    )

    viable = results_df[results_df["noise_pct"] <= 35].copy() if "noise_pct" in results_df.columns else results_df.copy()
    if len(viable) == 0:
        viable = results_df.copy()

    if "composite_score" in viable.columns and viable["composite_score"].notna().any():
        best_idx = viable["composite_score"].idxmax()
        best = viable.loc[best_idx]
        nn_best = int(best["n_neighbors"])
        nc_best = int(best["n_components"])
        mcs_best = int(best["min_cluster_size"])

        print(f"\n  \u2605 Best config (composite): "
              f"n_neighbors={nn_best}, "
              f"n_components={nc_best}, "
              f"min_cluster_size={mcs_best}  \u2014  "
              f"Score={best['composite_score']:.4f}, "
              f"Sil={best['silhouette_score']:.4f}, "
              f"Noise={best['noise_pct']:.1f}%, "
              f"Clusters={int(best['n_clusters'])}")

        clusterer_best = ReviewClusterer(
            umap_params={"n_neighbors": nn_best, "n_components": nc_best},
            hdbscan_params={"min_cluster_size": mcs_best},
            min_clusters=1,
        )
        result_best = clusterer_best.fit(texts, embeddings, pool_name=f"{pool}_best")

        prefix = f"exp2_{pool}_best_"
        figs = {
            "umap2d": viz.plot_umap_2d(result_best.umap_2d, result_best.topics,
                                        title=f"Best config: nn={nn_best}, nc={nc_best}, mcs={mcs_best}"),
            "cluster_sizes": viz.plot_cluster_sizes(result_best.topic_info,
                                                     title="Best \u2014 Cluster sizes"),
            "silhouette": viz.plot_silhouette_per_cluster(embeddings, result_best.topics,
                                                          title="Best \u2014 Silhouette"),
            "top_words": viz.plot_top_words(result_best.topic_model,
                                             title="Best \u2014 Top words"),
            "wordclouds": viz.plot_word_clouds(result_best.topic_model,
                                                title="Best \u2014 Word Clouds"),
        }
        viz.save_figures(figs, prefix=prefix)
        viz.save_bertopic_plots(result_best.topic_model, texts, embeddings, prefix=prefix)

    report_path = REPORTS_DIR / f"exp2_{pool}_grid_results.csv"
    results_df.to_csv(report_path, index=False)
    log.info("Saved report: %s", report_path)

    return results_df


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pool_arg = sys.argv[2] if len(sys.argv) > 2 else "negative"
    model_arg = sys.argv[3] if len(sys.argv) > 3 else "bge-m3"
    run(csv_arg, pool_arg, model_arg)
