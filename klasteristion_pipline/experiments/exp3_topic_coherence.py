"""
Experiment 3 — Topic Coherence & Representation Quality
========================================================
Evaluate final topic quality using the best embedding model and UMAP/HDBSCAN
parameters obtained from Experiments 1 & 2.

Varied variables
    BERTopic representation approaches:
      - Default c-TF-IDF
      - c-TF-IDF + KeyBERTInspired
      - c-TF-IDF + MaximalMarginalRelevance

Target metrics
    Topic Coherence Cv ↑, NPMI ↑, human-readable top-words quality.
    Generates representative documents per cluster for later LLM naming.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import (
    DEFAULT_UMAP_PARAMS,
    DEFAULT_HDBSCAN_PARAMS,
    EMBEDDING_MODELS,
    VECTORIZER_PARAMS,
    ALL_STOP_WORDS,
    TOP_N_WORDS,
    MAX_TOPICS,
    REPORTS_DIR,
    get_target_topics,
)
from pipeline.data_loader import load_reviews
from pipeline.preprocessor import preprocess, heuristic_split, pool_texts, pool_to_dataframe
from pipeline.embedder import EmbeddingGenerator
from pipeline.vram_utils import clear_gpu
from evaluation.metrics import compute_all_metrics, print_metrics, metrics_to_dataframe
from evaluation.visualizer import PipelineVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("exp3")


def _build_topic_model(
    texts: list[str],
    embeddings: np.ndarray,
    representation: str,
    umap_params: dict,
    hdbscan_params: dict,
    target_topics: int,
    model_hf_name: str | None = None,
) -> tuple:
    """Build and fit a BERTopic model with the given representation strategy."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(**umap_params)
    hdbscan_model = HDBSCAN(**hdbscan_params)
    vectorizer = CountVectorizer(stop_words=ALL_STOP_WORDS, **VECTORIZER_PARAMS)

    rep_model = None
    embedding_model = None

    if representation == "keybert":
        rep_model = KeyBERTInspired(top_n_words=TOP_N_WORDS)
    elif representation == "mmr":
        rep_model = MaximalMarginalRelevance(diversity=0.3, top_n_words=TOP_N_WORDS)

    if rep_model is not None and model_hf_name:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(model_hf_name, trust_remote_code=True)

    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=rep_model,
        top_n_words=TOP_N_WORDS,
        nr_topics=None,
        verbose=False,
        calculate_probabilities=True,
    )

    try:
        topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "no terms remain" in str(exc):
            topic_model.vectorizer_model = CountVectorizer(**VECTORIZER_PARAMS)
            topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
        else:
            raise

    n_real = len(set(topics)) - (1 if -1 in topics else 0)
    min_acceptable = max(3, target_topics // 2)

    if n_real < min_acceptable:
        from sklearn.cluster import KMeans
        log.info("HDBSCAN found %d clusters (need >= %d), falling back to KMeans(k=%d)",
                 n_real, min_acceptable, target_topics)
        kmeans = KMeans(n_clusters=target_topics, random_state=42, n_init=10)
        vectorizer2 = CountVectorizer(stop_words=ALL_STOP_WORDS, **VECTORIZER_PARAMS)
        topic_model = BERTopic(
            language="multilingual",
            embedding_model=embedding_model,
            umap_model=UMAP(**umap_params),
            hdbscan_model=kmeans,
            vectorizer_model=vectorizer2,
            representation_model=rep_model,
            top_n_words=TOP_N_WORDS,
            nr_topics=None,
            verbose=False,
            calculate_probabilities=False,
        )
        try:
            topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
        except ValueError as exc2:
            if "empty vocabulary" in str(exc2) or "no terms remain" in str(exc2):
                topic_model.vectorizer_model = CountVectorizer(**VECTORIZER_PARAMS)
                topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
            else:
                raise

    current = len(set(topics)) - (1 if -1 in topics else 0)
    if current > MAX_TOPICS:
        topic_model.reduce_topics(texts, nr_topics=MAX_TOPICS)
        topics = topic_model.topics_

    return topic_model, topics, probs


def _extract_representative_docs(
    topic_model,
    texts: list[str],
    n_per_topic: int = 5,
) -> dict:
    """Extract most representative documents per topic for LLM naming."""
    result = {}
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        try:
            rep_docs = topic_model.get_representative_docs(tid)
            result[tid] = rep_docs[:n_per_topic]
        except Exception:
            idx = [i for i, t in enumerate(topic_model.topics_) if t == tid]
            result[tid] = [texts[i] for i in idx[:n_per_topic]]
    return result


def run(
    csv_path: str | None = None,
    pool: str = "negative",
    model_key: str = "bge-m3",
    umap_params: dict | None = None,
    hdbscan_params: dict | None = None,
) -> pd.DataFrame:
    """Run Experiment 3 comparing representation strategies."""
    umap_params = umap_params or dict(DEFAULT_UMAP_PARAMS)
    hdbscan_params = hdbscan_params or dict(DEFAULT_HDBSCAN_PARAMS)

    df = load_reviews(csv_path)
    df = preprocess(df)
    split = heuristic_split(df)

    entries = split.negative_pool if pool == "negative" else split.positive_pool
    texts = pool_texts(entries)

    if len(texts) < 10:
        log.error("Pool '%s' has only %d texts — aborting.", pool, len(texts))
        return pd.DataFrame()

    emb_gen = EmbeddingGenerator(model_key)
    try:
        embeddings = emb_gen.encode(texts, use_cache=True)
    finally:
        emb_gen.unload()

    target = get_target_topics(len(texts))

    representations = {
        "default_ctfidf": "default",
        "keybert_inspired": "keybert",
        "mmr_diversity": "mmr",
    }

    all_metrics: list[dict] = []
    all_rep_docs: dict = {}
    viz = PipelineVisualizer()

    for label, rep_key in representations.items():
        print(f"\n{'#' * 60}")
        print(f"  REPRESENTATION: {label}")
        print(f"{'#' * 60}")

        t0 = time.perf_counter()
        model_hf_name = EMBEDDING_MODELS[model_key]["name"]
        topic_model, topics, probs = _build_topic_model(
            texts, embeddings, rep_key, umap_params, hdbscan_params, target,
            model_hf_name=model_hf_name,
        )
        elapsed = time.perf_counter() - t0

        m = compute_all_metrics(
            embeddings=embeddings,
            labels=topics,
            texts=texts,
            topic_model=topic_model,
            pool_name=label,
        )
        m["representation"] = label
        m["label"] = label
        m["time_sec"] = elapsed
        all_metrics.append(m)
        print_metrics(m)

        print(f"\n  Top words per topic ({label}):")
        for tid in sorted(topic_model.get_topics()):
            if tid == -1:
                continue
            words = [w for w, _ in topic_model.get_topic(tid)]
            count = sum(1 for t in topics if t == tid)
            print(f"    Topic {tid} (n={count}): {', '.join(words[:8])}")

        rep_docs = _extract_representative_docs(topic_model, texts)
        all_rep_docs[label] = rep_docs

        prefix = f"exp3_{pool}_{label}_"
        from umap import UMAP as UMAPVis
        umap_2d = UMAPVis(n_components=2, n_neighbors=15, min_dist=0.1,
                           metric="cosine", random_state=42).fit_transform(embeddings)

        figs = {
            "umap2d": viz.plot_umap_2d(umap_2d, topics,
                                        title=f"UMAP 2-D — {label} [{pool}]"),
            "cluster_sizes": viz.plot_cluster_sizes(topic_model.get_topic_info(),
                                                     title=f"Cluster sizes — {label}"),
            "top_words": viz.plot_top_words(topic_model, title=f"Top words — {label}"),
            "wordclouds": viz.plot_word_clouds(topic_model, title=f"Word Clouds — {label}"),
        }
        viz.save_figures(figs, prefix=prefix)
        viz.save_bertopic_plots(topic_model, texts, embeddings, prefix=prefix)

    results_df = metrics_to_dataframe(all_metrics)

    print("\n" + "=" * 80)
    print("  EXPERIMENT 3 — SUMMARY")
    print("=" * 80)
    display_cols = [
        "representation", "n_clusters", "noise_pct",
        "silhouette_score", "topic_coherence_cv", "topic_coherence_npmi",
    ]
    show = [c for c in display_cols if c in results_df.columns]
    print(tabulate(results_df[show], headers="keys", tablefmt="grid", floatfmt=".4f"))

    fig_cmp = viz.plot_metrics_comparison(all_metrics, label_key="representation",
                                          title="Exp 3 — Representation comparison")
    viz.save_figures({"exp3_comparison": fig_cmp})

    llm_input_path = REPORTS_DIR / f"exp3_{pool}_llm_input.json"
    llm_data = {}
    for rep_label, docs_by_topic in all_rep_docs.items():
        llm_data[rep_label] = {
            str(tid): docs for tid, docs in docs_by_topic.items()
        }
    with open(llm_input_path, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=2)
    log.info("Saved LLM naming input: %s", llm_input_path)

    pool_df = pool_to_dataframe(entries)
    best_rep = "default_ctfidf"
    if "topic_coherence_cv" in results_df.columns and results_df["topic_coherence_cv"].notna().any():
        best_rep = results_df.loc[results_df["topic_coherence_cv"].idxmax(), "representation"]

    print(f"\n  ★ Best representation by Coherence Cv: {best_rep}")

    report_path = REPORTS_DIR / f"exp3_{pool}_results.csv"
    results_df.to_csv(report_path, index=False)
    log.info("Saved report: %s", report_path)

    return results_df


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    pool_arg = sys.argv[2] if len(sys.argv) > 2 else "negative"
    model_arg = sys.argv[3] if len(sys.argv) > 3 else "bge-m3"
    run(csv_arg, pool_arg, model_arg)
