"""
Tab 3 — RAG Chat: dialog with the indexed reviews.

Layout
------
- Top row: collection picker, top-K, filters
- Center: chat history (st.chat_message)
- Bottom: st.chat_input
- Each assistant turn renders the answer + an expandable "Источники" with
  cards for every retrieved review, plus a debug expander showing the
  expanded query, filters, timings and transport (sdk/rest).
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime

import pandas as pd
import streamlit as st

from rag_pipline.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_OVERSAMPLE_FACTOR,
    DEFAULT_TOP_K,
    MAX_OVERSAMPLE_FACTOR,
    MAX_TOP_K,
    MIN_OVERSAMPLE_FACTOR,
    MIN_SCORE_CEIL,
    MIN_SCORE_FLOOR,
    MIN_TOP_K,
)
from rag_pipline.pipeline.indexer import QdrantStore, read_manifest
from rag_pipline.rag.orchestrator import RAGOrchestrator
from rag_pipline.rag.yandex_provider import YandexConfigError, YandexLLM
from rag_pipline.ui.state import get_embedder, get_store, reset_chat


def _list_indexed_collections() -> list[str]:
    store: QdrantStore = get_store()
    return [c for c in store.list_collections() if c.startswith("reviews_")]


def _filters_panel() -> dict:
    """Render filter inputs in an expander, return a filters dict."""
    with st.expander("🎯 Фильтры (применяются на уровне HNSW-индекса Qdrant)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Рейтинг (звёзды)**")
            rmin, rmax = st.slider(
                "Диапазон",
                min_value=1, max_value=5,
                value=(1, 5),
                key="filter_rating",
            )

            st.markdown("**Sentiment**")
            sentiments = st.multiselect(
                "Метки",
                options=["positive", "negative", "neutral"],
                default=[],
                key="filter_sentiment",
                help="Пусто = не фильтровать.",
            )

        with col2:
            st.markdown("**Дата отзыва**")
            use_date = st.checkbox("Включить фильтр по дате", value=False, key="filter_date_enabled")
            if use_date:
                today = date.today()
                df_default = date(today.year - 1, today.month, today.day)
                date_from, date_to = st.date_input(
                    "Период",
                    value=(df_default, today),
                    key="filter_date_range",
                )
            else:
                date_from = date_to = None

            st.markdown("**Пол автора**")
            genders = st.multiselect(
                "Пол",
                options=["male", "female", "unknown"],
                default=[],
                key="filter_gender",
                help="Пусто = не фильтровать.",
            )

    filters: dict = {}
    if (rmin, rmax) != (1, 5):
        filters["rating_min"] = int(rmin)
        filters["rating_max"] = int(rmax)
    if sentiments:
        filters["sentiment_labels"] = sentiments
    if genders:
        filters["genders"] = genders
    if date_from and date_to:
        filters["date_from"] = datetime.combine(date_from, dtime.min).isoformat()
        filters["date_to"] = datetime.combine(date_to, dtime.max).isoformat()
    return filters


def _render_assistant_message(idx: int, msg: dict) -> None:
    """Render an assistant turn (answer + sources + debug)."""
    with st.chat_message("assistant"):
        result = msg.get("result")
        if msg.get("content"):
            st.markdown(msg["content"])

        if not result:
            return

        hits = result.get("hits") or []
        if hits:
            with st.expander(f"📚 Источники ({len(hits)})", expanded=False):
                for i, h in enumerate(hits, start=1):
                    score = h.get("score") or 0.0
                    rid = h.get("review_id", "?")
                    rating = h.get("rating", "?")
                    sent = h.get("sentiment_label", "?")
                    gender = h.get("reviewer_gender", "?")
                    date_s = h.get("created_date") or "—"
                    text = (h.get("text") or "").strip()
                    badge_color = {
                        "positive": "🟢", "negative": "🔴", "neutral": "⚪",
                    }.get(sent, "⚪")
                    st.markdown(
                        f"**{i}. {rating}★ {badge_color} `{sent}` · 👤 {gender} · 🗓 {date_s} · "
                        f"sim={score:.3f}**  \n"
                        f"`review_id: {rid}`"
                    )
                    st.write(text[:1200] + ("…" if len(text) > 1200 else ""))
                    st.markdown("---")
        else:
            st.info("По указанным фильтрам не нашлось ни одного отзыва.")

        with st.expander("🔍 Детали запроса", expanded=False):
            st.markdown("**Расширенный запрос (Lite):**")
            if result.get("expansion_skipped"):
                st.code("(пропущено — поиск по исходному запросу)", language="text")
            else:
                st.code(result.get("expanded_query") or "—", language="text")

            st.markdown("**Фильтры:**")
            st.json(result.get("filters") or {})

            timings = result.get("timings") or {}
            transport = result.get("transport") or {}
            cols = st.columns(4)
            cols[0].metric("expansion, s", f"{timings.get('expansion_sec', 0):.2f}")
            cols[1].metric("retrieval, s", f"{timings.get('retrieval_sec', 0):.2f}")
            cols[2].metric("generation, s", f"{timings.get('generation_sec', 0):.2f}")
            cols[3].metric("total, s", f"{timings.get('total_sec', 0):.2f}")

            rcols = st.columns(4)
            rcols[0].metric("Кандидатов из Qdrant", result.get("candidates_fetched", "—"))
            rcols[1].metric("После score-фильтра", result.get("candidates_kept", "—"))
            rcols[2].metric("Мин. similarity", f"{(result.get('min_score') or 0):.2f}")
            rcols[3].metric("Oversample ×", f"{(result.get('oversample_factor') or 0):.1f}")

            st.caption(
                f"Lite transport: `{transport.get('lite', '?')}` · "
                f"Pro transport: `{transport.get('pro', '?')}` · "
                f"top-K: {result.get('top_k')} · "
                f"collection: `{result.get('collection')}`"
            )

            if result.get("expansion_failed"):
                st.warning("Query expansion не сработала — поиск выполнен по исходному запросу.")
            if result.get("error"):
                st.error(result["error"])


def render() -> None:
    st.header("💬 RAG Chat")
    st.caption("4-этапный пайплайн: Lite expand → Qdrant search → Pro answer → формирование ответа.")

    collections = _list_indexed_collections()
    if not collections:
        st.warning("Нет проиндексированных коллекций. Запусти индексацию во вкладке «Индексация».")
        return

    active = st.session_state.get("active_collection")
    default_idx = collections.index(active) if active in collections else 0
    chosen = st.selectbox("Коллекция (товар)", collections, index=default_idx, key="chat_collection_select")
    st.session_state["active_collection"] = chosen

    manifest = read_manifest(chosen) or {}
    if manifest:
        cols = st.columns(4)
        cols[0].metric("Записей", manifest.get("n_records", "?"))
        cols[1].metric("CSV", manifest.get("csv_file", "—"), help=manifest.get("csv_file"))
        cols[2].metric("Embedder", "e5-large")
        cols[3].metric("Sentiment модель", manifest.get("sentiment_model", "—").split("/")[-1])

    knob_a, knob_b, knob_c, knob_d = st.columns([1, 1, 1, 2])
    with knob_a:
        top_k = st.number_input(
            "top-K",
            min_value=MIN_TOP_K, max_value=MAX_TOP_K,
            value=int(st.session_state.get("rag_top_k", DEFAULT_TOP_K)),
            step=1,
            key="chat_top_k",
            help=(
                "Сколько отзывов в итоге передать в Pro. "
                "С учётом oversample фактически из Qdrant фетчится top_k × oversample кандидатов."
            ),
        )
        st.session_state["rag_top_k"] = int(top_k)
    with knob_b:
        min_score = st.number_input(
            "Мин. similarity",
            min_value=float(MIN_SCORE_FLOOR), max_value=float(MIN_SCORE_CEIL),
            value=float(st.session_state.get("rag_min_score", DEFAULT_MIN_SCORE)),
            step=0.01,
            format="%.2f",
            key="chat_min_score",
            help=(
                "Отбрасываем отзывы с similarity ниже порога. "
                "Для e5-large normalised cosine: >0.85 — сильно релевантно, "
                "0.80-0.85 — связано по теме, <0.78 — обычно шум. "
                "Поставь 0.0 чтобы отключить."
            ),
        )
        st.session_state["rag_min_score"] = float(min_score)
    with knob_c:
        oversample = st.number_input(
            "Oversample ×",
            min_value=float(MIN_OVERSAMPLE_FACTOR), max_value=float(MAX_OVERSAMPLE_FACTOR),
            value=float(st.session_state.get("rag_oversample", DEFAULT_OVERSAMPLE_FACTOR)),
            step=0.5,
            format="%.1f",
            key="chat_oversample",
            help=(
                "Кратность кандидатов: при top_k=10 и факторе 2.0 из Qdrant "
                "забираем 20 хитов, после score-фильтра обрезаем до 10. "
                "Помогает, когда первые N хитов состоят из шума с проходным скором."
            ),
        )
        st.session_state["rag_oversample"] = float(oversample)
    with knob_d:
        filters = _filters_panel()
        st.session_state["rag_filters"] = filters

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        if st.button("🧹 Очистить чат", width="stretch"):
            reset_chat()
            st.rerun()
    with c2:
        skip_expansion = st.toggle(
            "Без expansion",
            value=bool(st.session_state.get("rag_skip_expansion", False)),
            key="chat_skip_expansion",
            help=(
                "Пропустить вызов Lite — поиск пойдёт по исходному запросу. "
                "Полезно для запросов с редкими/конкретными терминами, "
                "когда расширение только размывает."
            ),
        )
        st.session_state["rag_skip_expansion"] = bool(skip_expansion)
    with c3:
        st.caption(
            "Каждый ответ строится с учётом последних реплик диалога; "
            "поиск в Qdrant — по текущему вопросу."
        )

    history = st.session_state.setdefault("chat_history", [])
    for i, m in enumerate(history):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            _render_assistant_message(i, m)

    user_query = st.chat_input("Задай вопрос об отзывах товара…")
    if not user_query:
        return

    folder_id = (st.session_state.get("yandex_folder_id") or "").strip()
    api_key = (st.session_state.get("yandex_api_key") or "").strip()
    if not folder_id or not api_key:
        st.error(
            "Не задан Yandex API key или folder_id. "
            "Введи их во вкладке «Настройки» (или экспортируй YANDEX_API_KEY / YANDEX_CATALOG_ID)."
        )
        return

    history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        lite = YandexLLM(folder_id=folder_id, api_key=api_key, kind="lite",
                         use_sdk=st.session_state.get("yandex_use_sdk", True))
        pro = YandexLLM(folder_id=folder_id, api_key=api_key, kind="pro",
                        use_sdk=st.session_state.get("yandex_use_sdk", True))
    except YandexConfigError as exc:
        st.error(f"Yandex config error: {exc}")
        return

    embedder = get_embedder(st.session_state.get("embedder_device", "auto"))
    store = get_store()
    orch = RAGOrchestrator(store=store, embedder=embedder, lite_llm=lite, pro_llm=pro)

    history_for_prompt = [
        {"role": h["role"], "content": h["content"]}
        for h in history[:-1]
        if h.get("content")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Расширение запроса → поиск → генерация…"):
            result = orch.run(
                user_query,
                collection=chosen,
                top_k=int(top_k),
                filters=filters or None,
                history=history_for_prompt,
                min_score=float(min_score),
                oversample_factor=float(oversample),
                skip_expansion=bool(skip_expansion),
            )

        if result.error:
            st.error(result.error)
            history.pop()  
            return

        st.markdown(result.answer)

        if result.hits:
            with st.expander(f"📚 Источники ({len(result.hits)})", expanded=False):
                for i, h in enumerate(result.hits, start=1):
                    score = h.get("score") or 0.0
                    rid = h.get("review_id", "?")
                    rating = h.get("rating", "?")
                    sent = h.get("sentiment_label", "?")
                    gender = h.get("reviewer_gender", "?")
                    date_s = h.get("created_date") or "—"
                    text = (h.get("text") or "").strip()
                    badge = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent, "⚪")
                    st.markdown(
                        f"**{i}. {rating}★ {badge} `{sent}` · 👤 {gender} · 🗓 {date_s} · sim={score:.3f}**  \n"
                        f"`review_id: {rid}`"
                    )
                    st.write(text[:1200] + ("…" if len(text) > 1200 else ""))
                    st.markdown("---")
        else:
            st.info("По указанным фильтрам не нашлось ни одного отзыва.")

        with st.expander("🔍 Детали запроса", expanded=False):
            st.markdown("**Расширенный запрос (Lite):**")
            if result.expansion_skipped:
                st.code("(пропущено — поиск по исходному запросу)", language="text")
            else:
                st.code(result.expanded_query, language="text")
            st.markdown("**Фильтры:**")
            st.json(result.filters or {})

            tcols = st.columns(4)
            tcols[0].metric("expansion, s", f"{result.timings.get('expansion_sec', 0):.2f}")
            tcols[1].metric("retrieval, s", f"{result.timings.get('retrieval_sec', 0):.2f}")
            tcols[2].metric("generation, s", f"{result.timings.get('generation_sec', 0):.2f}")
            tcols[3].metric("total, s", f"{result.timings.get('total_sec', 0):.2f}")

            rcols = st.columns(4)
            rcols[0].metric("Кандидатов из Qdrant", result.candidates_fetched)
            rcols[1].metric("После score-фильтра", result.candidates_kept)
            rcols[2].metric("Мин. similarity", f"{result.min_score:.2f}")
            rcols[3].metric("Oversample ×", f"{result.oversample_factor:.1f}")

            st.caption(
                f"Lite: `{result.transport.get('lite', '?')}` · Pro: `{result.transport.get('pro', '?')}` · "
                f"top-K: {result.top_k} · collection: `{result.collection}`"
            )
            if result.expansion_failed:
                st.warning("Query expansion не сработала — поиск выполнен по исходному запросу.")
            if result.candidates_fetched > 0 and result.candidates_kept == 0:
                st.warning(
                    f"Все {result.candidates_fetched} кандидатов отсеяны по min_score={result.min_score:.2f}. "
                    "Снизь порог или ослабь фильтры."
                )

    history.append({
        "role": "assistant",
        "content": result.answer,
        "result": result.to_export_dict(),
    })
    st.session_state["last_rag_result"] = result.to_export_dict()
    st.session_state["last_run_id"] = st.session_state.get("last_run_id", 0) + 1
