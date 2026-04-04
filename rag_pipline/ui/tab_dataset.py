"""
Tab 1 — Dataset overview.

Pick a CSV from review_parser/results, see what's inside, choose a product
(input_sku) for indexing.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_pipline.config import (
    CREATED_DATE_FIELD,
    CSV_DIR,
    PRODUCT_ID_FIELD,
    RATING_FIELD,
    REVIEWER_NAME_FIELD,
    TEXT_FIELDS,
)
from rag_pipline.pipeline.data_loader import (
    list_csv_files,
    list_products,
    load_reviews,
)


@st.cache_data(show_spinner=False)
def _cached_load(csv_path: str) -> pd.DataFrame:
    return load_reviews(csv_path)


def render() -> None:
    st.header("📋 Датасет")
    st.caption("Выбери CSV-файл с отзывами и продукт для индексации.")

    # ── CSV picker ──────────────────────────────────────────
    csv_files = list_csv_files(CSV_DIR)
    if not csv_files:
        st.error(
            f"Не найдено CSV-файлов в `{CSV_DIR}`. "
            "Запусти парсер review_parser, чтобы появились отзывы."
        )
        return

    names = [p.name for p in csv_files]
    current = st.session_state.get("selected_csv")
    default_idx = 0
    if current and current.name in names:
        default_idx = names.index(current.name)

    chosen_name = st.selectbox(
        "CSV файл (из review_parser/results)",
        names,
        index=default_idx,
        key="dataset_csv_select",
    )
    chosen_path = next(p for p in csv_files if p.name == chosen_name)
    st.session_state["selected_csv"] = chosen_path

    df = _cached_load(str(chosen_path))

    # ── Top-line metrics ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего отзывов", f"{len(df):,}")
    if RATING_FIELD in df.columns:
        rseries = pd.to_numeric(df[RATING_FIELD], errors="coerce").dropna()
        c2.metric("Средний рейтинг", f"{rseries.mean():.2f}" if len(rseries) else "—")
    if PRODUCT_ID_FIELD in df.columns:
        c3.metric("Уникальных товаров (input_sku)", df[PRODUCT_ID_FIELD].nunique())
    if CREATED_DATE_FIELD in df.columns:
        dates = pd.to_datetime(df[CREATED_DATE_FIELD], errors="coerce").dropna()
        if len(dates):
            c4.metric("Период", f"{dates.min().date()} — {dates.max().date()}")
        else:
            c4.metric("Период", "—")

    st.markdown("---")

    # ── Product picker ──────────────────────────────────────
    st.subheader("🛍 Выбор товара для индексации")
    products = list_products(df)
    if not products:
        st.warning("В CSV нет колонки input_sku — индексация может работать на всём файле, но коллекция получит общий ID.")
    else:
        labels = [f"{p['product_id']} ({p['reviews']} отзывов)" for p in products]
        idx_default = 0
        cur_pid = st.session_state.get("selected_product_id")
        if cur_pid is not None:
            for i, p in enumerate(products):
                if p["product_id"] == cur_pid:
                    idx_default = i
                    break
        chosen_label = st.selectbox("input_sku", labels, index=idx_default, key="dataset_product_select")
        chosen_idx = labels.index(chosen_label)
        st.session_state["selected_product_id"] = products[chosen_idx]["product_id"]

    st.markdown("---")

    # ── Charts ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Распределение рейтингов")
        if RATING_FIELD in df.columns:
            r = pd.to_numeric(df[RATING_FIELD], errors="coerce").dropna().astype(int)
            counts = r.value_counts().sort_index()
            fig = px.bar(
                x=counts.index.astype(str),
                y=counts.values,
                labels={"x": "Звёзды", "y": "Кол-во"},
                color=counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Длина комбинированного текста (символы)")
        text_cols_present = [c for c in TEXT_FIELDS if c in df.columns]
        if text_cols_present:
            combined_lengths = (
                df[text_cols_present]
                .fillna("").astype(str)
                .agg(lambda r: sum(len(s) for s in r), axis=1)
            )
            fig = px.histogram(
                combined_lengths, nbins=40,
                labels={"value": "Символы", "count": "Отзывы"},
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, width="stretch")

    # ── Time series ────────────────────────────────────────
    if CREATED_DATE_FIELD in df.columns:
        st.subheader("Отзывы по времени")
        dates = pd.to_datetime(df[CREATED_DATE_FIELD], errors="coerce").dropna()
        if len(dates):
            by_month = dates.dt.to_period("M").value_counts().sort_index()
            fig = px.bar(
                x=by_month.index.astype(str), y=by_month.values,
                labels={"x": "Месяц", "y": "Отзывы"},
                color_discrete_sequence=["#19D3F3"],
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

    # ── Field coverage ─────────────────────────────────────
    st.subheader("Заполненность полей")
    rows = []
    for f in TEXT_FIELDS + [REVIEWER_NAME_FIELD]:
        if f in df.columns:
            filled = (df[f].astype(str).str.strip().str.lower().isin(["", "nan", "none"]) == False).sum()
            rows.append({
                "Поле": f,
                "Заполнено": int(filled),
                "Заполнение, %": f"{filled / max(len(df), 1) * 100:.1f}%",
                "Средняя длина": int(df.loc[df[f].astype(str).str.strip() != "", f].str.len().mean()) if filled else 0,
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── Sample preview ─────────────────────────────────────
    with st.expander("👀 Превью первых 20 строк CSV"):
        preview_cols = [c for c in [
            "review_id", PRODUCT_ID_FIELD, RATING_FIELD, REVIEWER_NAME_FIELD,
            "advantages", "disadvantages", "comment", CREATED_DATE_FIELD,
        ] if c in df.columns]
        st.dataframe(df[preview_cols].head(20), width="stretch")
