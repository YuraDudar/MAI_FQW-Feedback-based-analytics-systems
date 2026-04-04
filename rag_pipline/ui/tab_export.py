"""
Tab 5 — Export.

LLM-friendly downloads, each in its own button:
  - Полный диалог (chat history JSON)
  - Релевантные отзывы из последнего ответа (JSON + CSV)
  - Запрос к LLM (system + user prompts that were sent to Pro/Lite — JSON)
  - Активные фильтры (JSON)
  - Полный последний RAG-result (JSON)
  - Манифест индекса (JSON)
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from rag_pipline.pipeline.indexer import read_manifest
from rag_pipline.rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_expansion_user_prompt,
)


def _ts_suffix() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _download_json(label: str, data: dict | list, filename: str, key: str) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    st.download_button(
        label, data=payload, file_name=filename, mime="application/json", key=key,
        width="stretch",
    )


def _download_csv(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    payload = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label, data=payload, file_name=filename, mime="text/csv", key=key,
        width="stretch",
    )


def render() -> None:
    st.header("📤 Экспорт")
    st.caption("Каждый кусок — отдельным файлом, в LLM-friendly виде.")

    history = st.session_state.get("chat_history") or []
    last = st.session_state.get("last_rag_result")
    active_collection = st.session_state.get("active_collection")
    filters = st.session_state.get("rag_filters") or {}

    suffix = _ts_suffix()

    st.subheader("Диалог")
    if not history:
        st.info("История пуста.")
    else:
        clean_history = []
        for m in history:
            entry = {"role": m["role"], "content": m["content"]}
            if m.get("result"):
                entry["meta"] = {
                    "expanded_query": m["result"].get("expanded_query"),
                    "filters": m["result"].get("filters"),
                    "top_k": m["result"].get("top_k"),
                    "collection": m["result"].get("collection"),
                    "source_review_ids": m["result"].get("source_review_ids"),
                    "timings": m["result"].get("timings"),
                }
            clean_history.append(entry)
        _download_json(
            "⬇️ Диалог (JSON)",
            {"messages": clean_history, "exported_at": datetime.utcnow().isoformat() + "Z"},
            f"rag_chat_{suffix}.json",
            key="dl_chat",
        )

    st.subheader("Релевантные отзывы (последний ответ)")
    if not last:
        st.info("Сделай хотя бы один запрос в чате.")
    else:
        hits = last.get("hits") or []
        if hits:
            col1, col2 = st.columns(2)
            with col1:
                _download_json(
                    "⬇️ Hits (JSON)",
                    {
                        "user_query": last.get("user_query"),
                        "collection": last.get("collection"),
                        "filters": last.get("filters"),
                        "top_k": last.get("top_k"),
                        "hits": hits,
                    },
                    f"rag_hits_{suffix}.json",
                    key="dl_hits_json",
                )
            with col2:
                df = pd.DataFrame(hits)
                _download_csv(
                    "⬇️ Hits (CSV)",
                    df,
                    f"rag_hits_{suffix}.csv",
                    key="dl_hits_csv",
                )

    st.subheader("Запрос к LLM (system + user)")
    if not last:
        st.info("Нет последнего запроса.")
    else:
        expansion_user = build_expansion_user_prompt(last.get("user_query") or "")
        history_for_prompt = []
        for m in history:
            if m["role"] == "user" and m["content"] == last.get("user_query"):
                break
            if m.get("content"):
                history_for_prompt.append({"role": m["role"], "content": m["content"]})
        answer_user = build_answer_user_prompt(
            query=last.get("user_query") or "",
            expanded_query=last.get("expanded_query") or "",
            hits=last.get("hits") or [],
            history=history_for_prompt,
            filters=last.get("filters"),
        )

        prompts_payload = {
            "stage_1_query_expansion": {
                "model": "yandexgpt-lite",
                "system_prompt": QUERY_EXPANSION_SYSTEM_PROMPT,
                "user_prompt": expansion_user,
                "raw_user_query": last.get("user_query"),
                "expanded_query_returned": last.get("expanded_query"),
            },
            "stage_3_answer_generation": {
                "model": "yandexgpt",
                "system_prompt": ANSWER_SYSTEM_PROMPT,
                "user_prompt": answer_user,
                "answer_returned": last.get("answer"),
            },
            "transport": last.get("transport"),
        }
        _download_json(
            "⬇️ Промпты к LLM (JSON)",
            prompts_payload,
            f"rag_prompts_{suffix}.json",
            key="dl_prompts",
        )

        with st.expander("📄 Превью промптов", expanded=False):
            st.markdown("**Stage 1 — Lite system:**")
            st.code(QUERY_EXPANSION_SYSTEM_PROMPT, language="text")
            st.markdown("**Stage 1 — Lite user:**")
            st.code(expansion_user, language="text")
            st.markdown("**Stage 3 — Pro system:**")
            st.code(ANSWER_SYSTEM_PROMPT, language="text")
            st.markdown("**Stage 3 — Pro user:**")
            st.code(answer_user[:6000] + ("…" if len(answer_user) > 6000 else ""), language="text")

    st.subheader("Фильтры")
    _download_json(
        "⬇️ Фильтры (JSON)",
        {
            "filters": filters,
            "top_k": st.session_state.get("rag_top_k"),
            "collection": active_collection,
        },
        f"rag_filters_{suffix}.json",
        key="dl_filters",
    )

    st.subheader("Полный последний RAG-result")
    if last:
        _download_json(
            "⬇️ RAG-result (JSON)",
            last,
            f"rag_result_{suffix}.json",
            key="dl_result",
        )

    st.subheader("Манифест коллекции")
    if active_collection:
        manifest = read_manifest(active_collection)
        if manifest:
            _download_json(
                "⬇️ Манифест (JSON)",
                manifest,
                f"manifest_{active_collection}_{suffix}.json",
                key="dl_manifest",
            )
            with st.expander("Манифест"):
                st.json(manifest)
        else:
            st.info("Манифест не найден.")
    else:
        st.info("Не выбрана активная коллекция.")
