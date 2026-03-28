"""
Streamlit UI for the Review Clustering Pipeline.

Launch:
    python run_pipeline.py --interface streamlit
    # or directly:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# -- project imports ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CSV_DIR,
    DEFAULT_UMAP_PARAMS,
    DEFAULT_HDBSCAN_PARAMS,
    EMBEDDING_MODELS,
    REPORTS_DIR,
    ALL_STOP_WORDS,
    TOPIC_NAMING_MODELS,
    DEFAULT_TOPIC_NAMING_MODEL,
)
from pipeline.data_loader import load_reviews
from pipeline.preprocessor import (
    preprocess,
    heuristic_split,
    pool_texts,
)
from pipeline.embedder import EmbeddingGenerator
from pipeline.clusterer import ReviewClusterer
from evaluation.metrics import compute_all_metrics
from topic_naming import TopicTitleGenerator, build_topic_payloads
from topic_naming.prompts import SYSTEM_PROMPT

# Reduce Streamlit bare-mode warning noise in terminal output
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# -- cached data loading (prevents re-execution on every Streamlit rerun) -----

@st.cache_data(show_spinner=False)
def _load_raw(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str)


@st.cache_data(show_spinner=False)
def _load_and_preprocess(csv_path: str):
    df = load_reviews(csv_path, print_stats=False)
    df = preprocess(df)
    return df


@st.cache_data(show_spinner=False)
def _heuristic_split_cached(csv_path: str):
    """Cached split keyed by csv_path; avoids re-running on every rerun."""
    df = _load_and_preprocess(csv_path)
    return heuristic_split(df)


# -- page config --------------------------------------------------------------
st.set_page_config(
    page_title="Review Clustering Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- preset loading -----------------------------------------------------------
PRESET_PATH = REPORTS_DIR / "pipeline_presets_candidate.json"
HARDCODED_PRESETS = {
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

def _load_presets() -> dict:
    if PRESET_PATH.exists():
        try:
            with open(PRESET_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return HARDCODED_PRESETS


def _preset_for(pool: str) -> dict:
    return _load_presets().get(pool, HARDCODED_PRESETS.get(pool, {}))


# =============================================================================
#  SIDEBAR: file picker + hyperparameters
# =============================================================================
def sidebar_config():
    """Render sidebar and return config dict."""
    st.sidebar.header("📁 Dataset")

    csv_files = sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    csv_names = [f.name for f in csv_files]
    if not csv_names:
        st.sidebar.error(f"No CSV files in {CSV_DIR}")
        st.stop()

    chosen_csv = st.sidebar.selectbox("CSV file", csv_names, index=0)
    csv_path = CSV_DIR / chosen_csv

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Pipeline")

    pools_to_run = st.sidebar.multiselect(
        "Pools to cluster",
        ["negative", "positive"],
        default=["negative", "positive"],
    )

    use_best = st.sidebar.checkbox("Use best presets from experiments", value=True)

    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Topic Naming")
    naming_model_key = st.sidebar.selectbox(
        "Topic naming model",
        options=list(TOPIC_NAMING_MODELS.keys()),
        index=list(TOPIC_NAMING_MODELS.keys()).index(DEFAULT_TOPIC_NAMING_MODEL),
        help="Default uses Yandex API, local models run on your machine",
    )
    yandex_catalog_id = st.sidebar.text_input(
        "Yandex catalog id",
        value="",
        help="Needed only for yandex-* naming models",
    )
    yandex_api_key = st.sidebar.text_input(
        "Yandex API key",
        value="",
        type="password",
        help="Optional here; safer to set env YANDEX_API_KEY",
    )

    configs: dict[str, dict] = {}
    for pool in pools_to_run:
        preset = _preset_for(pool)
        p_umap = preset.get("umap", {})
        p_hdb = preset.get("hdbscan", {})

        with st.sidebar.expander(f"{'🔴' if pool == 'negative' else '🟢'} {pool.capitalize()} pool", expanded=not use_best):
            st.caption("Embedding model")
            model = st.selectbox(
                "Model",
                list(EMBEDDING_MODELS.keys()),
                index=list(EMBEDDING_MODELS.keys()).index(preset.get("model", "bge-m3")),
                key=f"model_{pool}",
                help="Sentence-transformer model for semantic embeddings",
            )

            st.caption("Topic representation")
            rep_options = ["default_ctfidf", "keybert_inspired", "mmr_diversity"]
            representation = st.selectbox(
                "Representation",
                rep_options,
                index=rep_options.index(preset.get("representation", "default_ctfidf")),
                key=f"rep_{pool}",
                help="default_ctfidf = standard c-TF-IDF keywords; "
                     "keybert_inspired = uses embedding similarity for keywords; "
                     "mmr_diversity = Maximal Marginal Relevance for diverse keywords",
            )

            st.caption("UMAP (dimensionality reduction)")
            nn = st.slider(
                "n_neighbors",
                min_value=3, max_value=50,
                value=int(p_umap.get("n_neighbors", DEFAULT_UMAP_PARAMS["n_neighbors"])) if use_best else DEFAULT_UMAP_PARAMS["n_neighbors"],
                key=f"nn_{pool}",
                help="Controls local vs global structure preservation. Lower = more local detail",
            )
            nc = st.slider(
                "n_components",
                min_value=2, max_value=50,
                value=int(p_umap.get("n_components", DEFAULT_UMAP_PARAMS["n_components"])) if use_best else DEFAULT_UMAP_PARAMS["n_components"],
                key=f"nc_{pool}",
                help="Target dimensionality after UMAP. Higher = more information preserved",
            )

            st.caption("HDBSCAN (clustering)")
            mcs = st.slider(
                "min_cluster_size",
                min_value=2, max_value=50,
                value=int(p_hdb.get("min_cluster_size", DEFAULT_HDBSCAN_PARAMS["min_cluster_size"])) if use_best else DEFAULT_HDBSCAN_PARAMS["min_cluster_size"],
                key=f"mcs_{pool}",
                help="Minimum number of reviews to form a cluster. Higher = fewer, larger clusters",
            )
            ms = st.slider(
                "min_samples",
                min_value=1, max_value=20,
                value=int(p_hdb.get("min_samples", DEFAULT_HDBSCAN_PARAMS["min_samples"])) if use_best else DEFAULT_HDBSCAN_PARAMS["min_samples"],
                key=f"ms_{pool}",
                help="How conservative clustering is. Higher = more noise, denser clusters",
            )

            target_topics = st.number_input(
                "Target topics (0 = auto)",
                min_value=0, max_value=30,
                value=0,
                key=f"tt_{pool}",
                help="Force a specific number of topics. 0 = adaptive based on dataset size",
            )

        configs[pool] = {
            "model": model,
            "representation": representation,
            "umap": {
                **DEFAULT_UMAP_PARAMS,
                "n_neighbors": nn,
                "n_components": nc,
            },
            "hdbscan": {
                **DEFAULT_HDBSCAN_PARAMS,
                "min_cluster_size": mcs,
                "min_samples": ms,
            },
            "target_topics": target_topics if target_topics > 0 else None,
        }

    if st.sidebar.button("🔄 Reset to best presets"):
        st.cache_data.clear()
        st.rerun()

    return {
        "csv_path": csv_path,
        "pools": pools_to_run,
        "configs": configs,
        "topic_naming_model_key": naming_model_key,
        "yandex_catalog_id": yandex_catalog_id.strip() or None,
        "yandex_api_key": yandex_api_key.strip() or None,
    }


# =============================================================================
#  TAB 1: Dataset overview
# =============================================================================
def tab_dataset(df_raw: pd.DataFrame, df: pd.DataFrame, split):
    st.header("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reviews (raw)", f"{len(df_raw):,}")
    c2.metric("After dedup + filter", f"{len(df):,}")
    c3.metric("Negative pool", f"{len(split.negative_pool):,}")
    c4.metric("Positive pool", f"{len(split.positive_pool):,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Rating Distribution")
        if "rating" in df.columns:
            rating_counts = df["rating"].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index.astype(str),
                y=rating_counts.values,
                labels={"x": "Rating", "y": "Count"},
                color=rating_counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_rating.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_rating, width="stretch")

    with col_right:
        st.subheader("Review Length Distribution")
        if "combined_text" in df.columns:
            lengths = df["combined_text"].str.len()
            fig_len = px.histogram(
                lengths, nbins=50,
                labels={"value": "Characters", "count": "Reviews"},
                color_discrete_sequence=["#636EFA"],
            )
            fig_len.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_len, width="stretch")

    # field fill rates
    st.subheader("Field Coverage")
    fields = ["advantages", "disadvantages", "comment", "tags"]
    rows = []
    for f in fields:
        if f in df.columns:
            filled = (df[f].astype(str).str.strip() != "").sum()
            rows.append({"Field": f, "Filled": filled, "Fill %": f"{filled/len(df)*100:.1f}%",
                         "Avg length (chars)": int(df.loc[df[f].astype(str).str.strip() != "", f].str.len().mean()) if filled else 0})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # date range
    date_col = None
    for candidate in ["date", "created_at", "review_date", "createdDate"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col:
        st.subheader("Reviews Over Time")
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            by_month = dates.dt.to_period("M").value_counts().sort_index()
            fig_time = px.bar(
                x=by_month.index.astype(str), y=by_month.values,
                labels={"x": "Month", "y": "Reviews"},
            )
            fig_time.update_layout(height=300)
            st.plotly_chart(fig_time, width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Date range", f"{dates.min().date()} — {dates.max().date()}")
            c2.metric("Span", f"{(dates.max() - dates.min()).days} days")

    # Missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        st.dataframe(
            pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                           "%": (missing.values / len(df) * 100).round(1)}),
            width="stretch", hide_index=True,
        )
    else:
        st.success("No missing values after preprocessing")

    # Word cloud of all text
    st.subheader("Word Cloud (all reviews)")
    all_text = " ".join(df["combined_text"].dropna().tolist())
    if all_text.strip():
        wc = WordCloud(
            width=1200, height=400, background_color="white",
            max_words=150, colormap="viridis",
            stopwords=set(ALL_STOP_WORDS),
        ).generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(14, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc, width="stretch")
        plt.close(fig_wc)


# =============================================================================
#  TAB 2: Pipeline config summary (read-only view of what will run)
# =============================================================================
def tab_pipeline_info(configs: dict, topic_naming_model_key: str):
    st.header("🔧 Pipeline Architecture")

    st.markdown("""
**Pipeline steps:**
1. **Data loading** — CSV parsing, deduplication, type casting
2. **Preprocessing** — HTML/URL/emoji removal, smart concatenation with structural prefixes
3. **Heuristic split** — Reviews split into Negative (1-3★ + disadvantages from 4-5★) and Positive (advantages+comments from 4-5★) pools
4. **Embedding** — Sentence-transformer model encodes each review into a dense vector (GPU)
5. **UMAP** — Dimensionality reduction preserving local/global topology (CPU)
6. **HDBSCAN** — Density-based clustering with automatic noise detection (KMeans fallback if too few clusters)
7. **c-TF-IDF / KeyBERT / MMR** — Extract representative keywords per cluster
8. **Metrics** — Silhouette, Davies-Bouldin, Calinski-Harabasz, Topic Coherence, etc.
    """)

    st.markdown("---")
    st.subheader("Current Configuration")
    st.markdown(f"**Topic naming model:** `{topic_naming_model_key}`")

    for pool_name, cfg in configs.items():
        emoji = "🔴" if pool_name == "negative" else "🟢"
        with st.expander(f"{emoji} {pool_name.capitalize()} pool", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Embedding model:** `{cfg['model']}` ({EMBEDDING_MODELS[cfg['model']]['name']})")
                st.markdown(f"**Representation:** `{cfg['representation']}`")
                st.markdown(f"**Target topics:** `{cfg.get('target_topics') or 'auto'}`")
            with c2:
                st.markdown("**UMAP params:**")
                st.json({k: v for k, v in cfg["umap"].items() if k in ("n_neighbors", "n_components", "min_dist", "metric")})
                st.markdown("**HDBSCAN params:**")
                st.json({k: v for k, v in cfg["hdbscan"].items() if k in ("min_cluster_size", "min_samples", "metric", "cluster_selection_method")})


# =============================================================================
#  Helper: run clustering for one pool (cached)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def _run_pool_cached(
    _texts: tuple,
    _ratings: tuple,
    _review_ids: tuple,
    _source_fields: tuple,
    pool_name: str,
    model_key: str,
    representation: str,
    umap_params_json: str,
    hdbscan_params_json: str,
    target_topics: int | None,
    topic_naming_model: str | None,
    yandex_api_key: str | None,
    yandex_catalog_id: str | None,
):
    """Run clustering; inputs are hashable for caching."""
    texts = list(_texts)
    ratings = list(_ratings)
    review_ids = list(_review_ids)
    source_fields = list(_source_fields)
    umap_params = json.loads(umap_params_json)
    hdbscan_params = json.loads(hdbscan_params_json)

    t0 = time.perf_counter()

    # Embeddings
    t_embed = time.perf_counter()
    emb_gen = EmbeddingGenerator(model_key)
    try:
        embeddings = emb_gen.encode(texts, use_cache=True, show_progress=False)
    finally:
        emb_gen.unload()
    embed_sec = time.perf_counter() - t_embed

    # Clustering
    t_cluster = time.perf_counter()
    clusterer = ReviewClusterer(
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        target_topics=target_topics,
        embedding_model_name=EMBEDDING_MODELS.get(model_key, {}).get("name"),
        representation=representation,
        build_visualizations=True,
    )
    result = clusterer.fit(texts, embeddings, pool_name=pool_name)
    cluster_sec = time.perf_counter() - t_cluster

    # Metrics
    metrics = compute_all_metrics(
        embeddings=embeddings, labels=result.topics,
        texts=texts, topic_model=result.topic_model,
        pool_name=pool_name,
    )

    # Prepare topic summaries
    topics_summary = []
    topic_documents: dict[int, list[dict]] = {}
    tm = result.topic_model

    for tid in sorted(tm.get_topics()):
        if tid == -1:
            continue
        words = [w for w, _ in tm.get_topic(tid)]
        count = sum(1 for t in result.topics if t == tid)
        try:
            rep_docs = tm.get_representative_docs(tid)[:5]
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
            if lbl == tid:
                docs.append({
                    "text": texts[i],
                    "rating": ratings[i] if i < len(ratings) else None,
                    "review_id": review_ids[i] if i < len(review_ids) else None,
                    "source_fields": source_fields[i] if i < len(source_fields) else None,
                })
        topic_documents[int(tid)] = docs

    topic_titles: dict[int, dict] = {}
    naming_sec = 0.0
    if topic_naming_model:
        payloads = build_topic_payloads(topics_summary, topic_documents, sample_size=10)
        try:
            t_name = time.perf_counter()
            generator = TopicTitleGenerator(
                model_key=TOPIC_NAMING_MODELS[topic_naming_model],
                yandex_api_key=yandex_api_key,
                yandex_catalog_id=yandex_catalog_id,
            )
            topic_titles = generator.generate_for_pool(pool_name, payloads)
            naming_sec = time.perf_counter() - t_name
        except Exception as exc:
            topic_titles = {}
            topic_titles[-1] = {"title": None, "reason": f"Topic naming failed: {exc}"}

        for topic in topics_summary:
            tid = int(topic["topic_id"])
            naming = topic_titles.get(tid, {})
            topic["generated_title"] = naming.get("title")
            topic["title_reason"] = naming.get("reason")

    elapsed = time.perf_counter() - t0

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
        "elapsed_sec": elapsed,
        "model": model_key,
        "representation": representation,
        "umap_params": umap_params,
        "hdbscan_params": hdbscan_params,
        "clustering_method": result.clustering_method,
        "topic_titles": topic_titles,
        "timings": {
            "embeddings_sec": embed_sec,
            "clustering_sec": cluster_sec,
            "naming_sec": naming_sec,
            "total_sec": elapsed,
        },
    }


# =============================================================================
#  TAB 3: Clustering results
# =============================================================================
def tab_results(pool_data: dict):
    pool = pool_data["pool_name"]
    emoji = "🔴" if pool == "negative" else "🟢"

    st.header(f"{emoji} {pool.capitalize()} Clusters")

    m = pool_data["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", f"{pool_data['n_docs']:,}")
    c2.metric("Clusters", m.get("n_clusters", 0))
    c3.metric("Noise %", f"{m.get('noise_pct', 0):.1f}%")
    c4.metric("Method", pool_data.get("clustering_method", "?"))
    c5.metric("Time", f"{pool_data.get('elapsed_sec', 0):.1f}s")
    timings = pool_data.get("timings", {})
    if timings:
        st.caption(
            "Этапы времени: "
            f"embeddings={timings.get('embeddings_sec', 0):.1f}s, "
            f"clustering={timings.get('clustering_sec', 0):.1f}s, "
            f"naming={timings.get('naming_sec', 0):.1f}s"
        )

    naming_error = (pool_data.get("topic_titles") or {}).get(-1, {}).get("reason")
    if naming_error:
        st.warning(f"Генерация заголовков не удалась: {naming_error}")

    st.markdown("---")

    # Cluster overview table
    topics = pool_data["topics"]
    overview_rows = []
    for t in topics:
        overview_rows.append({
            "Topic": t["topic_id"],
            "Generated title": t.get("generated_title") or "—",
            "Title status": t.get("title_reason") or "—",
            "Reviews": t["count"],
            "Keywords": t["keyword_label"],
            "Top words": ", ".join(t["top_words"][:6]),
        })
    st.dataframe(pd.DataFrame(overview_rows), width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Explore Clusters")

    for t in topics:
        tid = t["topic_id"]
        docs = pool_data["topic_documents"].get(tid, [])
        sampled_docs = docs if len(docs) <= 10 else random.sample(docs, 10)
        with st.expander(
            f"Topic {tid}: {(t.get('generated_title') or t['keyword_label'])}  ({t['count']} reviews)",
            expanded=False,
        ):
            st.markdown(f"**Сгенерированный заголовок:** {t.get('generated_title') or '—'}")
            if t.get("title_reason"):
                st.caption(f"Статус генерации: {t.get('title_reason')}")
            st.markdown(f"**Ключевая фраза:** {t['keyword_label']}")
            st.markdown(f"**Top words:** {', '.join(t['top_words'])}")
            st.markdown("**Representative docs:**")
            for rd in t["representative_docs"]:
                st.info(rd[:500])

            st.markdown(f"**Случайные 10 отзывов ({len(sampled_docs)}):**")
            if sampled_docs:
                docs_df = pd.DataFrame(sampled_docs)
                st.dataframe(docs_df, width="stretch", hide_index=True, height=300)


# =============================================================================
#  TAB 4: Visualizations
# =============================================================================
def tab_visualizations(pool_data: dict):
    pool = pool_data["pool_name"]
    st.header(f"📊 Visualizations — {pool.capitalize()}")

    labels = np.array(pool_data["labels"])
    texts = pool_data["texts"]
    ratings = pool_data["ratings"]
    topics = pool_data["topics"]

    # ── UMAP 2D ─────────────────────────────────────
    st.subheader("UMAP 2D Scatter")
    umap_2d = pool_data.get("umap_2d")
    if umap_2d is not None:
        umap_2d = np.array(umap_2d)
        df_scatter = pd.DataFrame({
            "x": umap_2d[:, 0], "y": umap_2d[:, 1],
            "topic": [str(l) for l in labels],
            "text": [t[:120] for t in texts],
        })
        fig2d = px.scatter(
            df_scatter, x="x", y="y", color="topic",
            hover_data=["text"], opacity=0.7,
            title=f"UMAP 2D — {pool}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2d.update_traces(marker_size=5)
        fig2d.update_layout(height=600)
        st.plotly_chart(fig2d, width="stretch")
    else:
        st.warning("UMAP 2D not available")

    # ── UMAP 3D ─────────────────────────────────────
    st.subheader("UMAP 3D Scatter")
    umap_3d = pool_data.get("umap_3d")
    if umap_3d is not None:
        umap_3d = np.array(umap_3d)
        df_3d = pd.DataFrame({
            "x": umap_3d[:, 0], "y": umap_3d[:, 1], "z": umap_3d[:, 2],
            "topic": [str(l) for l in labels],
            "text": [t[:120] for t in texts],
        })
        fig3d = px.scatter_3d(
            df_3d, x="x", y="y", z="z", color="topic",
            hover_data=["text"], opacity=0.7,
            title=f"UMAP 3D — {pool}",
        )
        fig3d.update_traces(marker_size=3)
        fig3d.update_layout(height=700)
        st.plotly_chart(fig3d, width="stretch")
    else:
        st.warning("UMAP 3D not available")

    col1, col2 = st.columns(2)

    # ── Cluster sizes ──────────────────────────────
    with col1:
        st.subheader("Cluster Sizes")
        sizes = [(t["topic_id"], t["count"]) for t in topics]
        sizes_df = pd.DataFrame(sizes, columns=["Topic", "Count"]).sort_values("Count")
        fig_bar = px.bar(
            sizes_df, y="Topic", x="Count", orientation="h",
            color="Topic", color_discrete_sequence=px.colors.qualitative.Set2,
            text="Count",
        )
        fig_bar.update_layout(showlegend=False, height=max(300, len(sizes) * 40))
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, width="stretch")

    # ── Rating distribution per cluster ─────────────
    with col2:
        st.subheader("Rating per Cluster")
        if ratings:
            mask = labels != -1
            df_rat = pd.DataFrame({"topic": labels[mask], "rating": np.array(ratings)[mask]})
            fig_rat = px.histogram(
                df_rat, x="rating", color="topic", barmode="group",
                nbins=5, color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_rat.update_layout(height=400)
            st.plotly_chart(fig_rat, width="stretch")

    # ── Word clouds per topic ──────────────────────
    st.subheader("Word Clouds per Topic")
    n_topics = len(topics)
    cols_per_row = min(4, n_topics)
    for row_start in range(0, n_topics, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= n_topics:
                break
            t = topics[idx]
            freq = {w: max(0.01, 1.0 - i * 0.08) for i, w in enumerate(t["top_words"])}
            if freq:
                wc = WordCloud(
                    width=400, height=300, background_color="white",
                    max_words=30, colormap="viridis",
                ).generate_from_frequencies(freq)
                fig_wc, ax_wc = plt.subplots(figsize=(5, 3.5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.set_title(f"Topic {t['topic_id']}", fontsize=10)
                ax_wc.axis("off")
                col.pyplot(fig_wc, width="stretch")
                plt.close(fig_wc)

    # ── Cluster size pie ────────────────────────────
    st.subheader("Cluster Proportions")
    noise_cnt = int(pool_data["metrics"].get("noise_count", 0))
    pie_labels = [f"Topic {t['topic_id']}" for t in topics]
    pie_values = [t["count"] for t in topics]
    if noise_cnt > 0:
        pie_labels.append("Noise")
        pie_values.append(noise_cnt)
    fig_pie = px.pie(values=pie_values, names=pie_labels, hole=0.3,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, width="stretch")


# =============================================================================
#  TAB 5: Metrics
# =============================================================================
def tab_metrics(pool_data: dict):
    pool = pool_data["pool_name"]
    m = pool_data["metrics"]
    st.header(f"📏 Metrics — {pool.capitalize()}")

    # Main metrics as cards
    cols = st.columns(4)
    metric_cards = [
        ("Silhouette Score ↑", m.get("silhouette_score"), "Range: -1 to 1. Higher = better separation"),
        ("Davies-Bouldin ↓", m.get("davies_bouldin_index"), "Lower = better. Ratio of within-cluster to between-cluster distance"),
        ("Calinski-Harabasz ↑", m.get("calinski_harabasz_index"), "Higher = denser, well-separated clusters"),
        ("Topic Coherence Cv ↑", m.get("topic_coherence_cv"), "Semantic coherence of top words per topic"),
    ]
    for i, (name, val, help_text) in enumerate(metric_cards):
        with cols[i]:
            display = f"{val:.4f}" if val is not None else "N/A"
            st.metric(name, display, help=help_text)

    st.markdown("---")

    cols2 = st.columns(4)
    extra_metrics = [
        ("NPMI Coherence ↑", m.get("topic_coherence_npmi")),
        ("Noise %", m.get("noise_pct")),
        ("Clusters", m.get("n_clusters")),
        ("Documents", m.get("n_docs")),
    ]
    for i, (name, val) in enumerate(extra_metrics):
        with cols2[i]:
            display = f"{val:.4f}" if isinstance(val, float) else str(val) if val is not None else "N/A"
            st.metric(name, display)

    st.markdown("---")

    # Cluster size stats
    st.subheader("Cluster Size Statistics")
    size_stats = {
        "Mean": m.get("cluster_size_mean"),
        "Std": m.get("cluster_size_std"),
        "Min": m.get("cluster_size_min"),
        "Max": m.get("cluster_size_max"),
    }
    st.json({k: round(v, 2) if isinstance(v, float) else v for k, v in size_stats.items() if v is not None})

    # Silhouette per cluster
    sil_per = m.get("silhouette_per_cluster", {})
    if sil_per:
        st.subheader("Silhouette Score per Cluster")
        sil_df = pd.DataFrame([
            {"Topic": f"Topic {k}", "Silhouette": v}
            for k, v in sorted(sil_per.items(), key=lambda x: int(x[0]))
        ])
        fig_sil = px.bar(
            sil_df, x="Topic", y="Silhouette",
            color="Silhouette",
            color_continuous_scale="RdYlGn",
            range_color=[-0.5, 1.0],
        )
        fig_sil.update_layout(height=350)
        st.plotly_chart(fig_sil, width="stretch")

    # Topic diversity
    st.subheader("Topic Diversity")
    topics = pool_data["topics"]
    all_words = set()
    total_words = 0
    for t in topics:
        for w in t["top_words"]:
            all_words.add(w)
            total_words += 1
    diversity = len(all_words) / max(total_words, 1)
    st.metric("Topic Diversity", f"{diversity:.3f}",
              help="Ratio of unique words to total top words across all topics. Higher = more diverse topics")


# =============================================================================
#  TAB 6: LLM Export
# =============================================================================
def tab_export(all_results: dict):
    st.header("📤 Export for LLM / Analytics")

    st.markdown("""
Export clustering results in LLM-friendly JSON format for:
- Automated cluster naming via LLM
- Further analysis in external tools
- Report generation
    """)

    for pool_name, data in all_results.items():
        emoji = "🔴" if pool_name == "negative" else "🟢"
        st.subheader(f"{emoji} {pool_name.capitalize()} Pool")

        # LLM naming input
        llm_input = {
            "task": "Generate concise business-oriented names for customer review clusters",
            "pool": pool_name,
            "n_clusters": data["metrics"]["n_clusters"],
            "topics": [],
        }
        for t in data["topics"]:
            llm_input["topics"].append({
                "topic_id": t["topic_id"],
                "review_count": t["count"],
                "top_keywords": t["top_words"],
                "representative_reviews": t["representative_docs"][:3],
            })

        llm_json = json.dumps(llm_input, ensure_ascii=False, indent=2)

        st.download_button(
            f"⬇️ LLM input — {pool_name} (topic naming)",
            data=llm_json,
            file_name=f"llm_input_{pool_name}.json",
            mime="application/json",
            key=f"llm_{pool_name}",
        )

        with st.expander("Preview LLM input JSON"):
            st.code(llm_json, language="json")

        # Full export
        full_export = {
            "pool": pool_name,
            "model": data.get("model"),
            "representation": data.get("representation"),
            "umap_params": data.get("umap_params"),
            "hdbscan_params": data.get("hdbscan_params"),
            "metrics": {k: v for k, v in data["metrics"].items()
                        if not isinstance(v, (np.ndarray,))},
            "topics": data["topics"],
            "topic_documents": {str(k): v for k, v in data["topic_documents"].items()},
        }
        full_json = json.dumps(full_export, ensure_ascii=False, indent=2, default=str)

        st.download_button(
            f"⬇️ Full report — {pool_name}",
            data=full_json,
            file_name=f"full_report_{pool_name}.json",
            mime="application/json",
            key=f"full_{pool_name}",
        )

        # CSV export of clusters
        rows = []
        for t in data["topics"]:
            for doc in data["topic_documents"].get(t["topic_id"], []):
                rows.append({
                    "topic_id": t["topic_id"],
                    "keyword_label": t["keyword_label"],
                    "text": doc["text"],
                    "rating": doc.get("rating"),
                    "review_id": doc.get("review_id"),
                })
        if rows:
            csv_df = pd.DataFrame(rows)
            csv_data = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Clusters CSV — {pool_name}",
                data=csv_data,
                file_name=f"clusters_{pool_name}.csv",
                mime="text/csv",
                key=f"csv_{pool_name}",
            )

        # Metrics CSV
        metrics_row = {k: v for k, v in data["metrics"].items()
                       if not isinstance(v, (dict, list, np.ndarray))}
        metrics_csv = pd.DataFrame([metrics_row]).to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Metrics CSV — {pool_name}",
            data=metrics_csv,
            file_name=f"metrics_{pool_name}.csv",
            mime="text/csv",
            key=f"mcsv_{pool_name}",
        )

    st.markdown("---")
    # Combined export
    if len(all_results) > 1:
        combined = {
            "pools": {},
        }
        for pn, data in all_results.items():
            combined["pools"][pn] = {
                "metrics": {k: v for k, v in data["metrics"].items()
                            if not isinstance(v, (np.ndarray,))},
                "topics": data["topics"],
                "model": data.get("model"),
                "representation": data.get("representation"),
            }
        combined_json = json.dumps(combined, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            "⬇️ Combined report (all pools)",
            data=combined_json,
            file_name="combined_report.json",
            mime="application/json",
        )


def tab_naming_sandbox(
    *,
    default_model_key: str,
    default_catalog_id: str | None,
    default_api_key: str | None,
):
    st.header("🧪 Naming Sandbox")
    st.caption("Ручной тест генерации заголовков для топиков")

    model_key = st.selectbox(
        "Модель генерации заголовков",
        options=list(TOPIC_NAMING_MODELS.keys()),
        index=list(TOPIC_NAMING_MODELS.keys()).index(default_model_key),
        key="sandbox_model",
    )
    pool_name = st.text_input("Название пула (контекст)", value="negative", key="sandbox_pool")
    yandex_catalog_id = st.text_input(
        "Yandex catalog id",
        value=default_catalog_id or "",
        key="sandbox_catalog",
    )
    yandex_api_key = st.text_input(
        "Yandex API key",
        value=default_api_key or "",
        type="password",
        key="sandbox_api",
    )

    system_prompt = st.text_area(
        "Промпт (system)",
        value=SYSTEM_PROMPT,
        height=220,
        key="sandbox_system_prompt",
    )
    extra_instruction = st.text_area(
        "Дополнительная инструкция (опционально)",
        value="",
        height=80,
        key="sandbox_extra_instruction",
    )

    n_topics = int(
        st.number_input("Количество топиков", min_value=1, max_value=30, value=3, key="sandbox_n_topics")
    )

    topic_payloads = []
    for i in range(n_topics):
        with st.expander(f"Топик #{i}", expanded=(i == 0)):
            topic_id = int(st.number_input("Topic ID", value=i, key=f"sandbox_tid_{i}"))
            top_words_raw = st.text_input(
                "Top words (через запятую)",
                value="упаковка, коробка, помята, скотч",
                key=f"sandbox_words_{i}",
            )
            keyword_label = st.text_input(
                "Keyword label",
                value="упаковка | коробка | помята",
                key=f"sandbox_label_{i}",
            )
            n_reviews = int(
                st.number_input("Количество отзывов для контекста", min_value=1, max_value=50, value=10, key=f"sandbox_n_reviews_{i}")
            )
            reviews_text = st.text_area(
                "Отзывы (по одному на строку)",
                value=(
                    "Упаковка пришла мятая\n"
                    "Коробка порвана, но товар целый\n"
                    "Скотч плохо держит\n"
                    "На подарок не подойдет"
                ),
                height=140,
                key=f"sandbox_reviews_{i}",
            )
            reviews = [r.strip() for r in reviews_text.splitlines() if r.strip()]
            topic_payloads.append(
                {
                    "topic_id": topic_id,
                    "keywords": [w.strip() for w in top_words_raw.split(",") if w.strip()],
                    "keyword_label": keyword_label,
                    "sample_reviews": reviews[:n_reviews],
                }
            )

    if st.button("🚀 Сгенерировать заголовки", type="primary", width="stretch", key="sandbox_run"):
        try:
            with st.spinner("Генерация заголовков..."):
                generator = TopicTitleGenerator(
                    model_key=TOPIC_NAMING_MODELS[model_key],
                    yandex_api_key=yandex_api_key or None,
                    yandex_catalog_id=yandex_catalog_id or None,
                )
                composed_prompt = system_prompt
                if extra_instruction.strip():
                    composed_prompt += "\n\nДоп.инструкция:\n" + extra_instruction.strip()
                output = generator.generate_for_pool(
                    pool_name=pool_name,
                    topic_payloads=topic_payloads,
                    system_prompt=composed_prompt,
                )
        except Exception as exc:
            st.error("Ошибка генерации заголовков")
            st.exception(exc)
            return

        rows = []
        for payload in topic_payloads:
            tid = int(payload["topic_id"])
            item = output.get(tid, {})
            rows.append(
                {
                    "topic_id": tid,
                    "generated_title": item.get("title", "—"),
                    "reason": item.get("reason", ""),
                    "keywords": ", ".join(payload["keywords"][:10]),
                    "n_reviews": len(payload["sample_reviews"]),
                }
            )
        st.subheader("Результат генерации")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# =============================================================================
#  MAIN
# =============================================================================
def main():
    st.title("🔬 Review Clustering Pipeline")
    st.caption("Interactive demo — clustering customer reviews with BERTopic + UMAP + HDBSCAN")

    cfg = sidebar_config()
    csv_path = cfg["csv_path"]
    pools_to_run = cfg["pools"]
    configs = cfg["configs"]
    topic_naming_model_key = cfg["topic_naming_model_key"]
    yandex_catalog_id = cfg["yandex_catalog_id"]
    yandex_api_key = cfg["yandex_api_key"]

    # Load data (all cached — no re-execution on reruns)
    with st.spinner("Loading and preprocessing data..."):
        df_raw = _load_raw(str(csv_path))
        df = _load_and_preprocess(str(csv_path))
        split = _heuristic_split_cached(str(csv_path))

    # Create tabs
    tab_names = ["📋 Dataset", "🔧 Pipeline", "▶️ Run & Results", "📊 Visualizations", "📏 Metrics", "📤 Export", "🧪 Naming Sandbox"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_dataset(df_raw, df, split)

    with tabs[1]:
        tab_pipeline_info(configs, topic_naming_model_key)

    # Run button
    with tabs[2]:
        st.header("▶️ Run Pipeline")

        if not pools_to_run:
            st.warning("Select at least one pool in the sidebar")
            st.stop()

        run_btn = st.button("🚀 Run Clustering Pipeline", type="primary", width="stretch")

        if run_btn:
            all_results = {}
            progress = st.progress(0, text="Подготовка запуска...")
            for idx, pool_name in enumerate(pools_to_run, start=1):
                pool_cfg = configs[pool_name]
                entries = split.negative_pool if pool_name == "negative" else split.positive_pool

                if len(entries) < 10:
                    st.warning(f"Pool '{pool_name}' has only {len(entries)} entries — skipping")
                    continue

                texts = tuple(e.text for e in entries)
                ratings = tuple(e.rating for e in entries)
                review_ids = tuple(e.review_id for e in entries)
                source_fields = tuple(e.source_fields for e in entries)

                progress.progress(int((idx - 1) / max(len(pools_to_run), 1) * 100), text=f"Обработка пула {pool_name}...")
                with st.spinner(f"Clustering {pool_name} pool ({len(entries)} docs)..."):
                    result = _run_pool_cached(
                        texts, ratings, review_ids, source_fields,
                        pool_name=pool_name,
                        model_key=pool_cfg["model"],
                        representation=pool_cfg["representation"],
                        umap_params_json=json.dumps(pool_cfg["umap"]),
                        hdbscan_params_json=json.dumps(pool_cfg["hdbscan"]),
                        target_topics=pool_cfg.get("target_topics"),
                            topic_naming_model=topic_naming_model_key,
                            yandex_api_key=yandex_api_key,
                            yandex_catalog_id=yandex_catalog_id,
                    )
                    all_results[pool_name] = result
            progress.progress(100, text="Готово")

            st.session_state["results"] = all_results

        all_results = st.session_state.get("results", {})

        if all_results:
            st.success(f"Pipeline complete! Processed {len(all_results)} pool(s)")
            for pool_name, data in all_results.items():
                tab_results(data)

    with tabs[3]:
        results = st.session_state.get("results", {})
        if not results:
            st.info("Run the pipeline first (tab 'Run & Results')")
        else:
            pool_select = st.selectbox("Select pool", list(results.keys()), key="viz_pool")
            if pool_select:
                tab_visualizations(results[pool_select])

    with tabs[4]:
        results = st.session_state.get("results", {})
        if not results:
            st.info("Run the pipeline first")
        else:
            pool_select = st.selectbox("Select pool", list(results.keys()), key="met_pool")
            if pool_select:
                tab_metrics(results[pool_select])

    with tabs[5]:
        results = st.session_state.get("results", {})
        if not results:
            st.info("Run the pipeline first")
        else:
            tab_export(results)

    with tabs[6]:
        tab_naming_sandbox(
            default_model_key=topic_naming_model_key,
            default_catalog_id=yandex_catalog_id,
            default_api_key=yandex_api_key,
        )


if __name__ == "__main__":
    main()
