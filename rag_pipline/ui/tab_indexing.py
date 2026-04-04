"""
Tab 2 — Indexing (offline preprocessing + Qdrant upload).

Pipeline order (sequential to keep VRAM peak low on RTX 3070 8 GB):
  1. Load CSV + filter by product_id + build combined_text
  2. Sentiment classification — load model → predict → unload
  3. Gender heuristic via pymorphy3 (CPU only)
  4. Embeddings — load e5-large → encode → unload
  5. Qdrant: recreate collection with payload indices, upsert points
  6. Persist a manifest JSON
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_pipline.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    SENTIMENT_MODELS,
)
from rag_pipline.pipeline.data_loader import (
    load_reviews,
    prepare_records,
    records_summary,
)
from rag_pipline.pipeline.embedder import E5Embedder
from rag_pipline.pipeline.gender import GenderDetector, gender_distribution
from rag_pipline.pipeline.indexer import (
    QdrantStore,
    build_manifest,
    collection_name_for,
    make_point_id,
    write_manifest,
)
from rag_pipline.pipeline.sentiment import SentimentAnalyzer, label_distribution
from rag_pipline.ui.state import clear_caches, get_store, push_log


def _model_choices() -> list[str]:
    return list(SENTIMENT_MODELS.keys())


def render() -> None:
    st.header("⚙️ Индексация в Qdrant")
    st.caption(
        "Препроцессинг (sentiment + gender) + эмбеддинги + загрузка в Qdrant. "
        "Этапы запускаются последовательно — VRAM освобождается между шагами."
    )

    csv_path: Path | None = st.session_state.get("selected_csv")
    product_id = st.session_state.get("selected_product_id")

    if csv_path is None:
        st.warning("Сначала выбери CSV-файл во вкладке «Датасет».")
        return

    st.markdown(f"**CSV:** `{csv_path.name}`  &nbsp;·&nbsp;  **product_id:** `{product_id or 'все'}`")

    # ── Settings panel ─────────────────────────────────────
    with st.expander("🛠 Настройки индексации", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            sentiment_key = st.selectbox(
                "Модель sentiment",
                _model_choices(),
                index=_model_choices().index(st.session_state.get("sentiment_model_key", _model_choices()[0])),
                help="\n".join(f"{k}: {v['description']}" for k, v in SENTIMENT_MODELS.items()),
            )
            st.session_state["sentiment_model_key"] = sentiment_key

        with col2:
            embed_device = st.selectbox(
                "Устройство для эмбеддингов",
                ["auto", "cuda", "cpu"],
                index=["auto", "cuda", "cpu"].index(st.session_state.get("embedder_device", "auto")),
            )
            st.session_state["embedder_device"] = embed_device

        with col3:
            sent_device = st.selectbox(
                "Устройство для sentiment",
                ["auto", "cuda", "cpu"],
                index=["auto", "cuda", "cpu"].index(st.session_state.get("sentiment_device", "auto")),
            )
            st.session_state["sentiment_device"] = sent_device

        col4, col5 = st.columns(2)
        with col4:
            embed_bs = st.number_input(
                "Batch size эмбеддера",
                min_value=2, max_value=64,
                value=EMBEDDING_BATCH_SIZE,
                step=2,
                help="На RTX 3070 8GB безопасно 4-8 для e5-large.",
            )
        with col5:
            recreate = st.checkbox(
                "Пересоздать коллекцию (drop + create)",
                value=True,
                help="Если выключено — коллекция доиндексируется (upsert по UUID5 от review_id).",
            )

    # ── Records preview ────────────────────────────────────
    df = load_reviews(str(csv_path))
    records = prepare_records(df, product_id=product_id)
    summary = records_summary(records)

    if summary["total"] == 0:
        st.error("После фильтрации не осталось ни одного отзыва (проверь min_chars и product_id).")
        return

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Готово к индексации", f"{summary['total']:,}")
    cs2.metric("Средняя длина текста", f"{summary['avg_text_len']:.0f}")
    cs3.metric("С датой", f"{summary['with_date']}/{summary['total']}")
    cs4.metric("С именем автора", f"{summary['with_name']}/{summary['total']}")

    # ── Run button ─────────────────────────────────────────
    st.markdown("---")
    can_run = summary["total"] > 0 and product_id is not None
    if not can_run:
        st.info("Выбери product_id во вкладке «Датасет», чтобы запустить индексацию.")

    run_clicked = st.button(
        "🚀 Запустить индексацию",
        type="primary",
        disabled=not can_run,
        width="stretch",
    )

    if run_clicked:
        _run_indexing(
            records=records,
            product_id=int(product_id),
            csv_name=csv_path.name,
            sentiment_key=sentiment_key,
            sentiment_device=sent_device,
            embed_device=embed_device,
            embed_batch_size=int(embed_bs),
            recreate=recreate,
        )

    # ── Last manifest (collapsible) ────────────────────────
    last_manifest = st.session_state.get("last_index_manifest")
    if last_manifest:
        st.markdown("---")
        st.subheader("📦 Последний манифест индексации")
        cols = st.columns(4)
        cols[0].metric("Коллекция", last_manifest["collection"])
        cols[1].metric("Записей", last_manifest["n_records"])
        cols[2].metric("Время, сек", last_manifest["elapsed_sec"])
        cols[3].metric("Sentiment модель", last_manifest["sentiment_model"])

        c1, c2 = st.columns(2)
        sent_dist = last_manifest.get("sentiment_distribution") or {}
        if sent_dist:
            with c1:
                fig = px.pie(
                    values=list(sent_dist.values()),
                    names=list(sent_dist.keys()),
                    title="Sentiment",
                    color_discrete_map={"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"},
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, width="stretch")
        gen_dist = last_manifest.get("gender_distribution") or {}
        if gen_dist:
            with c2:
                fig = px.pie(
                    values=list(gen_dist.values()),
                    names=list(gen_dist.keys()),
                    title="Gender",
                    color_discrete_map={"male": "#3498DB", "female": "#E91E63", "unknown": "#7F8C8D"},
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, width="stretch")

        with st.expander("Полный JSON манифеста"):
            st.json(last_manifest)


# ── Indexing engine (sequential, VRAM-safe) ─────────────────────

def _run_indexing(
    *,
    records: list,
    product_id: int,
    csv_name: str,
    sentiment_key: str,
    sentiment_device: str,
    embed_device: str,
    embed_batch_size: int,
    recreate: bool,
) -> None:
    st.session_state["indexing_log"] = []
    container = st.container()
    overall_progress = container.progress(0.0, text="Подготовка…")
    log_box = container.empty()
    detail_progress = container.empty()


    clear_caches()

    def _refresh_log():
        log_box.code("\n".join(st.session_state["indexing_log"]) or "(пусто)", language="text")

    def _stage_progress(stage_name: str, frac: float):
        overall_progress.progress(min(max(frac, 0.0), 1.0), text=stage_name)

    t_total = time.perf_counter()

    push_log(f"[1/5] Подготовлено {len(records)} записей.")
    _refresh_log()
    _stage_progress("Этап 1/5 — записи готовы", 0.05)

    texts = [r.combined_text for r in records]
    names = [r.reviewer_name for r in records]

    _stage_progress("Этап 2/5 — Sentiment", 0.1)
    push_log(f"[2/5] Загрузка sentiment-модели '{sentiment_key}' ({SENTIMENT_MODELS[sentiment_key]['name']})…")
    _refresh_log()

    sent_progress = detail_progress.progress(0.0, text="Sentiment 0%")

    def _sent_cb(done, total):
        sent_progress.progress(done / max(total, 1), text=f"Sentiment {done}/{total}")

    analyzer = SentimentAnalyzer(model_key=sentiment_key, device=sentiment_device)
    try:
        analyzer.load()
        labels = analyzer.predict(texts, progress_callback=_sent_cb)
    finally:
        analyzer.unload()
    detail_progress.empty()

    for r, lbl in zip(records, labels):
        r.sentiment_label = lbl
    sent_dist = label_distribution(labels)
    push_log(f"[2/5] Sentiment готов: {sent_dist}")
    _refresh_log()

    _stage_progress("Этап 3/5 — Gender", 0.4)
    push_log("[3/5] Определение пола авторов через pymorphy3…")
    _refresh_log()

    gender_progress = detail_progress.progress(0.0, text="Gender 0%")

    def _gen_cb(done, total):
        gender_progress.progress(done / max(total, 1), text=f"Gender {done}/{total}")

    detector = GenderDetector()
    genders = detector.detect_many(names, progress_callback=_gen_cb)
    detail_progress.empty()

    for r, g in zip(records, genders):
        r.reviewer_gender = g
    gen_dist = gender_distribution(genders)
    push_log(f"[3/5] Gender готов: {gen_dist}")
    _refresh_log()

    _stage_progress("Этап 4/5 — Embeddings", 0.55)
    push_log(f"[4/5] Эмбеддинги e5-large на {embed_device}…")
    _refresh_log()

    emb_progress = detail_progress.progress(0.0, text="Embeddings 0%")

    def _emb_cb(done, total):
        emb_progress.progress(done / max(total, 1), text=f"Embeddings {done}/{total}")

    embedder = E5Embedder(device=embed_device)
    try:
        embedder.load()
        vectors = embedder.embed_passages(
            texts,
            batch_size=embed_batch_size,
            show_progress=False,
            progress_callback=_emb_cb,
        )
    finally:
        embedder.unload()
        clear_caches()
    detail_progress.empty()
    push_log(f"[4/5] Эмбеддинги готовы: shape={vectors.shape}")
    _refresh_log()

    _stage_progress("Этап 5/5 — Qdrant", 0.85)
    coll_name = collection_name_for(product_id)
    store: QdrantStore = get_store()
    if recreate or not store.collection_exists(coll_name):
        push_log(f"[5/5] Создание коллекции {coll_name}…")
        _refresh_log()
        store.recreate_collection(coll_name, vector_size=vectors.shape[1])

    push_log(f"[5/5] Загрузка {len(records)} точек в Qdrant…")
    _refresh_log()

    qdrant_progress = detail_progress.progress(0.0, text="Upsert 0%")

    def _qd_cb(done, total):
        qdrant_progress.progress(done / max(total, 1), text=f"Upsert {done}/{total}")

    ids = [make_point_id(r.review_id) for r in records]
    payloads = [r.to_payload() for r in records]
    store.upsert_points(coll_name, ids, vectors, payloads, batch_size=64, progress_callback=_qd_cb)
    detail_progress.empty()

    elapsed = time.perf_counter() - t_total
    rating_dist = {s: sum(1 for r in records if r.rating == s) for s in range(1, 6)}
    manifest = build_manifest(
        collection_name=coll_name,
        product_id=product_id,
        n_records=len(records),
        embedding_model=EMBEDDING_MODEL_NAME,
        sentiment_model=SENTIMENT_MODELS[sentiment_key]["name"],
        csv_file=csv_name,
        sentiment_dist=sent_dist,
        gender_dist=gen_dist,
        rating_dist=rating_dist,
        elapsed_sec=elapsed,
    )
    write_manifest(coll_name, manifest)
    st.session_state["last_index_manifest"] = manifest
    st.session_state["active_collection"] = coll_name

    _stage_progress("Готово", 1.0)
    push_log(f"[OK] Индексация завершена за {elapsed:.1f}s. Коллекция: {coll_name}")
    _refresh_log()
    st.success(f"✅ Готово! Коллекция `{coll_name}` ({len(records)} точек) — переходи во вкладку «Чат RAG».")
