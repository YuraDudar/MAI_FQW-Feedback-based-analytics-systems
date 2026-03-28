"""
Experiment 1 — Embedding Space Quality
=======================================
Compare embedding models while keeping UMAP and HDBSCAN parameters frozen.

Frozen params
    UMAP:    n_neighbors=15, n_components=5
    HDBSCAN: min_cluster_size=10, min_samples=5

Varied variable
    Embedding model: bge-m3 | multilingual-e5-large | rubert-tiny2

Target metrics
    Silhouette Score ↑, Davies–Bouldin Index ↓, Calinski–Harabasz Index ↑,
    Noise %, Topic Coherence Cv
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tabulate import tabulate

from config import EMBEDDING_MODELS, REPORTS_DIR
from pipeline.data_loader import load_reviews
from pipeline.preprocessor import preprocess, heuristic_split, pool_texts
from pipeline.embedder import EmbeddingGenerator
from pipeline.clusterer import ReviewClusterer
from evaluation.metrics import compute_all_metrics, print_metrics, metrics_to_dataframe
from evaluation.visualizer import PipelineVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("exp1")


def run(csv_path: str | None = None, pool: str = "negative") -> pd.DataFrame:
    """
    Run Experiment 1 on a single pool (negative or positive).

    Returns a DataFrame of metrics for each embedding model.
    """
    df = load_reviews(csv_path)
    df = preprocess(df)
    split = heuristic_split(df)

    entries = split.negative_pool if pool == "negative" else split.positive_pool
    texts = pool_texts(entries)
    ratings = [e.rating for e in entries]

    if len(texts) < 10:
        log.error("Pool '%s' has only %d texts — too few for clustering.", pool, len(texts))
        return pd.DataFrame()

    all_metrics: list[dict] = []
    viz = PipelineVisualizer()

    for model_key in EMBEDDING_MODELS:
        print(f"\n{'#' * 60}")
        print(f"  MODEL: {model_key}  ({EMBEDDING_MODELS[model_key]['name']})")
        print(f"{'#' * 60}")

        t0 = time.perf_counter()

        emb_gen = EmbeddingGenerator(model_key)
        try:
            embeddings = emb_gen.encode(texts, use_cache=True)
        finally:
            emb_gen.unload()

        clusterer = ReviewClusterer()
        result = clusterer.fit(texts, embeddings, pool_name=f"{pool}_{model_key}")

        m = compute_all_metrics(
            embeddings=embeddings,
            labels=result.topics,
            texts=texts,
            topic_model=result.topic_model,
            pool_name=f"{pool}_{model_key}",
        )
        m["model"] = model_key
        m["label"] = model_key
        m["total_time_sec"] = time.perf_counter() - t0
        all_metrics.append(m)

        print_metrics(m)

        prefix = f"exp1_{pool}_{model_key}_"
        figs = {
            "umap2d": viz.plot_umap_2d(result.umap_2d, result.topics,
                                        title=f"UMAP 2-D — {model_key} [{pool}]"),
            "cluster_sizes": viz.plot_cluster_sizes(result.topic_info,
                                                     title=f"Cluster sizes — {model_key}"),
            "silhouette": viz.plot_silhouette_per_cluster(embeddings, result.topics,
                                                          title=f"Silhouette — {model_key}"),
            "top_words": viz.plot_top_words(result.topic_model,
                                             title=f"Top words — {model_key}"),
        }
        viz.save_figures(figs, prefix=prefix)

        try:
            fig3d = viz.plot_umap_3d_interactive(result.umap_3d, result.topics, texts,
                                                  title=f"UMAP 3-D — {model_key}")
            viz.save_plotly({"umap3d": fig3d}, prefix=prefix)
        except Exception as e:
            log.warning("3-D plot failed: %s", e)

        viz.save_bertopic_plots(result.topic_model, texts, embeddings, prefix=prefix)

    results_df = metrics_to_dataframe(all_metrics)
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1 — SUMMARY")
    print("=" * 80)
    display_cols = [
        "model", "n_clusters", "noise_pct",
        "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index",
        "topic_coherence_cv", "total_time_sec",
    ]
    show = [c for c in display_cols if c in results_df.columns]
    print(tabulate(results_df[show], headers="keys", tablefmt="grid", floatfmt=".4f"))

    fig_cmp = viz.plot_metrics_comparison(all_metrics, label_key="model",
                                          title="Exp 1 — Embedding model comparison")
    viz.save_figures({"exp1_comparison": fig_cmp})

    report_path = REPORTS_DIR / f"exp1_{pool}_results.csv"
    results_df.to_csv(report_path, index=False)
    log.info("Saved report: %s", report_path)

    if "silhouette_score" in results_df.columns:
        best = results_df.loc[results_df["silhouette_score"].idxmax()]
        print(f"\n  ★ Best model by Silhouette: {best['model']}  "
              f"(Sil={best['silhouette_score']:.4f}, "
              f"DB={best.get('davies_bouldin_index', '?'):.4f})")

    return results_df


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pool_arg = sys.argv[2] if len(sys.argv) > 2 else "negative"
    run(csv_arg, pool_arg)
