"""
Usage
-----
    # Auto-detect latest CSV, use defaults:
    python run_pipeline.py

    # Specify CSV, model, pool:
    python run_pipeline.py  path/to/reviews.csv  --model bge-m3  --pool both

    # Override UMAP / HDBSCAN:
    python run_pipeline.py  --nn 15  --mcs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from config import (
    DEFAULT_UMAP_PARAMS,
    DEFAULT_HDBSCAN_PARAMS,
    EMBEDDING_MODELS,
    REPORTS_DIR,
    PLOTS_DIR,
    TOPIC_NAMING_MODELS,
    DEFAULT_TOPIC_NAMING_MODEL,
)
from pipeline.data_loader import load_reviews
from pipeline.preprocessor import (
    preprocess,
    heuristic_split,
    pool_texts,
    PoolEntry,
)
from pipeline.embedder import EmbeddingGenerator
from pipeline.clusterer import ReviewClusterer, ClusteringResult
from evaluation.metrics import compute_all_metrics, print_metrics
from evaluation.visualizer import PipelineVisualizer
from topic_naming import TopicTitleGenerator, build_topic_payloads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline")

PRESET_CANDIDATE_PATH = REPORTS_DIR / "pipeline_presets_candidate.json"
DEFAULT_PRESETS = {
    "negative": {
        "model": "e5-large",
        "representation": "default_ctfidf",
        "umap": {"n_neighbors": 5, "n_components": 10},
        "hdbscan": {"min_cluster_size": 12, "min_samples": 2},
    },
    "positive": {
        "model": "bge-m3",
        "representation": "keybert_inspired",
        "umap": {"n_neighbors": 15, "n_components": 10},
        "hdbscan": {"min_cluster_size": 12, "min_samples": 3},
    },
}


def load_best_presets() -> dict:
    if PRESET_CANDIDATE_PATH.exists():
        try:
            with open(PRESET_CANDIDATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            log.warning("Failed to read preset candidate file: %s", exc)
    return DEFAULT_PRESETS


def run_single_pool(
    entries: list[PoolEntry],
    pool_name: str,
    model_key: str,
    umap_params: dict,
    hdbscan_params: dict,
    representation: str,
    viz: PipelineVisualizer,
    target_topics: int | None = None,
    build_visualizations: bool = True,
    topic_naming_model_key: str | None = None,
    yandex_api_key: str | None = None,
    yandex_catalog_id: str | None = None,
) -> dict | None:
    """Process a single pool (positive or negative) end-to-end."""
    texts = pool_texts(entries)
    ratings = [e.rating for e in entries]

    if len(texts) < 10:
        log.warning("Pool '%s' has only %d entries — skipping.", pool_name, len(texts))
        return None

    # ── State 1: Embeddings (GPU) ────────────────────────────
    emb_gen = EmbeddingGenerator(model_key)
    try:
        embeddings = emb_gen.encode(texts, use_cache=True)
    finally:
        emb_gen.unload()

    # ── State 2: UMAP + HDBSCAN + c-TF-IDF (CPU) ────────────
    clusterer = ReviewClusterer(
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        target_topics=target_topics,
        embedding_model_name=EMBEDDING_MODELS.get(model_key, {}).get("name"),
        representation=representation,
        build_visualizations=build_visualizations,
    )
    result: ClusteringResult = clusterer.fit(texts, embeddings, pool_name=pool_name)

    # ── Metrics ──────────────────────────────────────────────
    metrics = compute_all_metrics(
        embeddings=embeddings,
        labels=result.topics,
        texts=texts,
        topic_model=result.topic_model,
        pool_name=pool_name,
    )
    print_metrics(metrics)

    # ── Visualisations ───────────────────────────────────────
    if build_visualizations:
        prefix = f"pipeline_{pool_name}_"
        figs = {
            "umap2d": viz.plot_umap_2d(
                result.umap_2d, result.topics,
                title=f"UMAP 2-D — {pool_name}", texts=texts,
            ),
            "cluster_sizes": viz.plot_cluster_sizes(
                result.topic_info, title=f"Cluster sizes — {pool_name}",
            ),
            "silhouette": viz.plot_silhouette_per_cluster(
                embeddings, result.topics, title=f"Silhouette — {pool_name}",
            ),
            "top_words": viz.plot_top_words(
                result.topic_model, title=f"Top words — {pool_name}",
            ),
            "wordclouds": viz.plot_word_clouds(
                result.topic_model, title=f"Word Clouds — {pool_name}",
            ),
        }

        if ratings:
            figs["rating_dist"] = viz.plot_rating_per_cluster(
                result.topics, ratings, title=f"Rating per cluster — {pool_name}",
            )

        viz.save_figures(figs, prefix=prefix)

        try:
            fig3d = viz.plot_umap_3d_interactive(
                result.umap_3d, result.topics, texts,
                title=f"UMAP 3-D — {pool_name}",
            )
            viz.save_plotly({"umap3d": fig3d}, prefix=prefix)
        except Exception as e:
            log.warning("3-D plot skipped: %s", e)

        viz.save_bertopic_plots(result.topic_model, texts, embeddings, prefix=prefix)

    # ── Build output structure ───────────────────────────────
    topics_summary = []
    topic_documents: dict[int, list[dict]] = {}
    topic_model = result.topic_model
    for tid in sorted(topic_model.get_topics()):
        if tid == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(tid)]
        count = sum(1 for t in result.topics if t == tid)
        try:
            rep_docs = topic_model.get_representative_docs(tid)[:5]
        except Exception:
            rep_docs = [texts[i] for i in range(len(texts)) if result.topics[i] == tid][:5]

        topics_summary.append({
            "topic_id": tid,
            "count": count,
            "top_words": words[:10],
            "keyword_label": " | ".join(words[:3]),
            "representative_docs": rep_docs,
        })
        docs = []
        for i, lbl in enumerate(result.topics):
            if lbl != tid:
                continue
            docs.append(
                {
                    "text": texts[i],
                    "rating": ratings[i] if i < len(ratings) else None,
                    "review_id": entries[i].review_id if i < len(entries) else None,
                    "source_fields": entries[i].source_fields if i < len(entries) else None,
                }
            )
        topic_documents[int(tid)] = docs

    # ── Topic naming (LLM) ────────────────────────────────────
    topic_titles: dict[int, dict] = {}
    if topic_naming_model_key:
        payloads = build_topic_payloads(topics_summary, topic_documents, sample_size=10)
        title_generator = TopicTitleGenerator(
            model_key=topic_naming_model_key,
            yandex_api_key=yandex_api_key,
            yandex_catalog_id=yandex_catalog_id,
        )
        try:
            topic_titles = title_generator.generate_for_pool(pool_name, payloads)
        except Exception as exc:
            log.warning("Topic naming failed for pool '%s': %s", pool_name, exc)
            topic_titles = {}

        for topic in topics_summary:
            tid = int(topic["topic_id"])
            naming = topic_titles.get(tid, {})
            topic["generated_title"] = naming.get("title")
            topic["title_reason"] = naming.get("reason")

    return {
        "pool_name": pool_name,
        "n_docs": len(texts),
        "metrics": metrics,
        "topics": topics_summary,
        "topic_documents": topic_documents,
        "umap_2d": result.umap_2d.tolist() if result.umap_2d is not None else None,
        "umap_3d": result.umap_3d.tolist() if result.umap_3d is not None else None,
        "labels": result.topics,
        "texts": texts,
        "ratings": ratings,
        "embeddings": embeddings,
        "topic_model": result.topic_model,
        "elapsed_sec": result.elapsed_sec,
        "topic_titles": topic_titles,
    }


def main():
    parser = argparse.ArgumentParser(description="Review Clustering Pipeline")
    parser.add_argument("csv", nargs="?", default=None, help="Path to CSV file")
    parser.add_argument("--interface", choices=["cli", "streamlit"], default="cli", help="Run as CLI or Streamlit app")
    parser.add_argument("--model", default=None, help="Embedding model key")
    parser.add_argument("--representation", default=None, choices=["default_ctfidf", "keybert_inspired", "mmr_diversity"], help="Topic representation mode")
    parser.add_argument("--pool", default="both", choices=["negative", "positive", "both"])
    parser.add_argument("--nn", type=int, default=None, help="UMAP n_neighbors override")
    parser.add_argument("--nc", type=int, default=None, help="UMAP n_components override")
    parser.add_argument("--mcs", type=int, default=None, help="HDBSCAN min_cluster_size override")
    parser.add_argument("--ms", type=int, default=None, help="HDBSCAN min_samples override")
    parser.add_argument("--topics", type=int, default=None, help="Force target topic count")
    parser.add_argument("--preset", choices=["negative", "positive"], default=None, help="Use preset params for a single pool")
    parser.add_argument("--use-best-presets", action="store_true", help="Use best presets from experiment reports")
    parser.add_argument(
        "--topic-naming-model",
        default=DEFAULT_TOPIC_NAMING_MODEL,
        choices=list(TOPIC_NAMING_MODELS.keys()),
        help="Model key for topic name generation",
    )
    parser.add_argument("--yandex-api-key", default=None, help="Yandex API key (or env YANDEX_API_KEY)")
    parser.add_argument("--yandex-catalog-id", default=None, help="Yandex catalog id (or env YANDEX_CATALOG_ID)")
    parser.add_argument("--disable-topic-naming", action="store_true", help="Disable topic name generation")
    args = parser.parse_args()

    if args.interface == "streamlit":
        script = Path(__file__).resolve().parent / "streamlit_app.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(script)]
        if args.csv:
            cmd += ["--", "--csv", str(args.csv)]
        subprocess.run(cmd, check=False)
        return

    # ── Load & preprocess ────────────────────────────────────
    t_total = time.perf_counter()
    df = load_reviews(args.csv)
    df = preprocess(df)
    split = heuristic_split(df)

    viz = PipelineVisualizer()
    outputs = {}

    pools = []
    if args.pool in ("negative", "both"):
        pools.append(("negative", split.negative_pool))
    if args.pool in ("positive", "both"):
        pools.append(("positive", split.positive_pool))

    presets = load_best_presets()

    for pool_name, entries in pools:
        preset_name = args.preset if args.preset else (pool_name if args.use_best_presets else None)
        preset = presets.get(preset_name, {}) if preset_name else {}
        preset_umap = preset.get("umap", {})
        preset_hdb = preset.get("hdbscan", {})

        model_key = args.model or preset.get("model") or "bge-m3"
        representation = args.representation or preset.get("representation") or "default_ctfidf"

        umap_params = dict(DEFAULT_UMAP_PARAMS)
        hdbscan_params = dict(DEFAULT_HDBSCAN_PARAMS)
        umap_params.update(preset_umap)
        hdbscan_params.update(preset_hdb)
        if args.nn:
            umap_params["n_neighbors"] = args.nn
        if args.nc:
            umap_params["n_components"] = args.nc
        if args.mcs:
            hdbscan_params["min_cluster_size"] = args.mcs
        if args.ms:
            hdbscan_params["min_samples"] = args.ms

        result = run_single_pool(
            entries, pool_name, model_key,
            umap_params, hdbscan_params,
            representation=representation,
            viz=viz,
            target_topics=args.topics,
            build_visualizations=True,
            topic_naming_model_key=None if args.disable_topic_naming else TOPIC_NAMING_MODELS[args.topic_naming_model],
            yandex_api_key=args.yandex_api_key,
            yandex_catalog_id=args.yandex_catalog_id,
        )
        if result:
            result["model"] = model_key
            result["representation"] = representation
            result["umap_params"] = umap_params
            result["hdbscan_params"] = hdbscan_params
            outputs[pool_name] = result

    # ── Final report ─────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f} s total")
    print(f"{'=' * 70}")

    for pool_name, out in outputs.items():
        print(f"\n  [{pool_name.upper()}]  {out['n_docs']} documents → {len(out['topics'])} topics")
        for t in out["topics"]:
            title = t.get("generated_title") or t["keyword_label"]
            print(f"    • {title}:  {t['count']} reviews")

    # ── Save JSON report ─────────────────────────────────────
    report = {
        "elapsed_sec": elapsed,
        "pools": {},
    }
    for pool_name, out in outputs.items():
        pool_data = {
            "n_docs": out["n_docs"],
            "model": out.get("model"),
            "representation": out.get("representation"),
            "umap_params": out.get("umap_params"),
            "hdbscan_params": out.get("hdbscan_params"),
            "metrics": {k: v for k, v in out["metrics"].items()
                        if not isinstance(v, (dict, list, np.ndarray))},
            "topics": out["topics"],
            "topic_documents": out["topic_documents"],
        }
        report["pools"][pool_name] = pool_data

    report_path = REPORTS_DIR / "pipeline_result.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    print(f"  Plots saved:  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
