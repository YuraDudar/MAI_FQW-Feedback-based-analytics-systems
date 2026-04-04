"""
Tab 4 — Index analytics & retrieval inspector.

Two halves:
  A) Distributions across the indexed collection (sentiment, gender, ratings,
     dates, length). Pulls every payload via Qdrant scroll.
  B) Retrieval inspector for the LAST chat answer: similarity histogram,
     score-vs-rating scatter, sentiment/gender breakdown of retrieved docs,
     and a “filters reduction funnel” showing how candidate pool shrinks.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_pipline.pipeline.indexer import QdrantStore, read_manifest
from rag_pipline.pipeline.retriever import build_qdrant_filter
from rag_pipline.ui.state import get_store


def _payload_dataframe(store: QdrantStore, collection: str) -> pd.DataFrame:
    points = store.scroll_all(collection, limit=20000)
    rows = []
    for p in points:
        pl = p.payload or {}
        rows.append({
            "review_id": pl.get("review_id"),
            "rating": pl.get("rating"),
            "sentiment_label": pl.get("sentiment_label"),
            "reviewer_gender": pl.get("reviewer_gender"),
            "created_date": pl.get("created_date"),
            "text_len": len(pl.get("text") or ""),
        })
    df = pd.DataFrame(rows)
    if "created_date" in df.columns:
        df["created_dt"] = pd.to_datetime(df["created_date"], errors="coerce")
    return df


def _filters_funnel(store: QdrantStore, collection: str, filters: dict | None) -> dict:
    """Show how each filter subset narrows the candidate pool."""
    info = store.collection_info(collection)
    total = info.get("points_count") or 0
    out = {"all": int(total or 0)}
    if not filters:
        return out

    parts = {
        "rating": {k: filters[k] for k in ("rating_min", "rating_max") if k in filters},
        "date": {k: filters[k] for k in ("date_from", "date_to") if k in filters},
        "sentiment": {"sentiment_labels": filters["sentiment_labels"]} if filters.get("sentiment_labels") else {},
        "gender": {"genders": filters["genders"]} if filters.get("genders") else {},
    }
    for name, sub in parts.items():
        if not sub:
            continue
        qf = build_qdrant_filter(sub)
        try:
            cnt = store.client.count(collection_name=collection, count_filter=qf, exact=True).count
        except Exception:
            cnt = -1
        out[f"only {name}"] = int(cnt)

    qf_all = build_qdrant_filter(filters)
    try:
        out["all filters combined"] = int(
            store.client.count(collection_name=collection, count_filter=qf_all, exact=True).count
        )
    except Exception:
        out["all filters combined"] = -1
    return out


def render() -> None:
    st.header("📊 Аналитика индекса")
    st.caption("Распределения по проиндексированной коллекции + инспектор последнего RAG-запроса.")

    store: QdrantStore = get_store()
    collections = [c for c in store.list_collections() if c.startswith("reviews_")]
    if not collections:
        st.info("Нет коллекций. Сначала запусти индексацию.")
        return

    active = st.session_state.get("active_collection")
    default_idx = collections.index(active) if active in collections else 0
    chosen = st.selectbox("Коллекция", collections, index=default_idx, key="analytics_coll_select")

    manifest = read_manifest(chosen) or {}
    if manifest:
        cols = st.columns(4)
        cols[0].metric("Записей", manifest.get("n_records", "?"))
        cols[1].metric("Время индексации, s", manifest.get("elapsed_sec", "?"))
        cols[2].metric("Sentiment", manifest.get("sentiment_model", "—").split("/")[-1])
        cols[3].metric("Embedder", manifest.get("embedding_model", "—").split("/")[-1])

    with st.spinner("Чтение payload…"):
        df = _payload_dataframe(store, chosen)

    if df.empty:
        st.warning("Коллекция пуста.")
        return

    st.subheader("Распределения по коллекции")
    a, b = st.columns(2)
    with a:
        sent_counts = df["sentiment_label"].value_counts()
        fig = px.pie(
            values=sent_counts.values, names=sent_counts.index,
            title="Sentiment", hole=0.4,
            color=sent_counts.index,
            color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"},
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with b:
        gen_counts = df["reviewer_gender"].value_counts()
        fig = px.pie(
            values=gen_counts.values, names=gen_counts.index,
            title="Gender", hole=0.4,
            color=gen_counts.index,
            color_discrete_map={"male": "#3498DB", "female": "#E91E63", "unknown": "#7F8C8D"},
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2)
    with c:
        rating_counts = df["rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index.astype(str), y=rating_counts.values,
            labels={"x": "Звёзды", "y": "Отзывы"},
            color=rating_counts.index.astype(str),
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, height=320, title="Рейтинги")
        st.plotly_chart(fig, width="stretch")
    with d:
        fig = px.histogram(
            df, x="text_len", nbins=40,
            title="Длина текста (символы)",
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Кросс-распределения")
    e, f = st.columns(2)
    with e:
        ct = pd.crosstab(df["rating"], df["sentiment_label"])
        fig = px.imshow(
            ct, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "sentiment", "y": "rating", "color": "count"},
            title="Rating × Sentiment",
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    with f:
        ct = pd.crosstab(df["reviewer_gender"], df["sentiment_label"])
        fig = px.imshow(
            ct, text_auto=True, aspect="auto",
            color_continuous_scale="Purples",
            labels={"x": "sentiment", "y": "gender", "color": "count"},
            title="Gender × Sentiment",
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

    if "created_dt" in df.columns and df["created_dt"].notna().any():
        st.subheader("Отзывы по времени")
        ts = df.dropna(subset=["created_dt"]).copy()
        ts["month"] = ts["created_dt"].dt.to_period("M").astype(str)
        ts_count = ts.groupby(["month", "sentiment_label"]).size().reset_index(name="count")
        fig = px.bar(
            ts_count, x="month", y="count", color="sentiment_label",
            color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"},
            title="Отзывы по месяцам (стек по sentiment)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("🔬 Инспектор последнего RAG-запроса")

    last = st.session_state.get("last_rag_result")
    if not last:
        st.info("Сделай хотя бы один запрос во вкладке «Чат RAG».")
        return

    cols = st.columns(4)
    cols[0].metric("Запрос top-K", last.get("top_k"))
    cols[1].metric("Найдено отзывов", len(last.get("hits") or []))
    cols[2].metric("Коллекция", last.get("collection", "—"))
    cols[3].metric("Total, s", f"{(last.get('timings') or {}).get('total_sec', 0):.2f}")

    st.markdown("**Запрос:** " + (last.get("user_query") or ""))
    st.caption("Расширенный: " + (last.get("expanded_query") or ""))

    hits = last.get("hits") or []
    if hits:
        hits_df = pd.DataFrame(hits)
        a, b = st.columns(2)
        with a:
            fig = px.histogram(
                hits_df, x="score", nbins=15,
                title="Распределение similarity-скоров",
                color_discrete_sequence=["#FFA15A"],
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.scatter(
                hits_df, x="rating", y="score",
                color="sentiment_label",
                size_max=14,
                hover_data=["review_id", "reviewer_gender"],
                color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"},
                title="Score × Rating",
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")

        c, d = st.columns(2)
        with c:
            sent = hits_df["sentiment_label"].value_counts()
            fig = px.pie(values=sent.values, names=sent.index,
                         title="Sentiment в найденных", hole=0.4,
                         color=sent.index,
                         color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"})
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")
        with d:
            gen = hits_df["reviewer_gender"].value_counts()
            fig = px.pie(values=gen.values, names=gen.index,
                         title="Gender в найденных", hole=0.4,
                         color=gen.index,
                         color_discrete_map={"male": "#3498DB", "female": "#E91E63", "unknown": "#7F8C8D"})
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")

    st.markdown("**Воронка фильтров (точное число точек, попадающих под условия):**")
    funnel = _filters_funnel(store, last.get("collection") or chosen, last.get("filters"))
    funnel_df = pd.DataFrame({"stage": list(funnel.keys()), "points": list(funnel.values())})
    fig = px.bar(
        funnel_df, x="stage", y="points", text="points",
        color="stage", color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, width="stretch")
