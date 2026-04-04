"""
Tab 6 — Settings.

Yandex credentials (folder_id, API key), SDK toggle, GPU device defaults,
collection management (delete), cache controls.
"""
from __future__ import annotations

import os

import streamlit as st

from rag_pipline.config import (
    DEFAULT_YANDEX_API_KEY,
    DEFAULT_YANDEX_FOLDER_ID,
    EMBEDDING_MODEL_NAME,
    QDRANT_STORAGE_DIR,
    SENTIMENT_MODELS,
)
from rag_pipline.pipeline.indexer import QdrantStore, _meta_path
from rag_pipline.ui.state import clear_caches, get_store


def render() -> None:
    st.header("🛠 Настройки")

    st.subheader("🔑 Yandex Cloud — статус")
    st.caption("Учётные данные вводятся в боковой панели слева — они применяются ко всем вкладкам.")

    folder_id = st.session_state.get("yandex_folder_id", "")
    api_key = st.session_state.get("yandex_api_key", "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Folder ID", folder_id or "—")
    col2.metric("API Key", "✅ задан" if api_key else "❌ не задан")
    col3.metric("SDK", "вкл" if st.session_state.get("yandex_use_sdk", True) else "REST")

    if folder_id and api_key:
        st.success("✅ Учётные данные заполнены — RAG-чат разблокирован.")
    else:
        st.warning("⚠️ Заполни folder_id и API key в боковой панели слева, чтобы пользоваться чатом.")

    st.markdown("---")

    st.subheader("🎛 Устройства по умолчанию")
    col_a, col_b = st.columns(2)
    with col_a:
        emb_dev = st.selectbox(
            "Embedder",
            ["auto", "cuda", "cpu"],
            index=["auto", "cuda", "cpu"].index(st.session_state.get("embedder_device", "auto")),
            key="settings_emb_device",
        )
        st.session_state["embedder_device"] = emb_dev
    with col_b:
        sent_dev = st.selectbox(
            "Sentiment",
            ["auto", "cuda", "cpu"],
            index=["auto", "cuda", "cpu"].index(st.session_state.get("sentiment_device", "auto")),
            key="settings_sent_device",
        )
        st.session_state["sentiment_device"] = sent_dev

    st.markdown("---")
    st.subheader("🩺 Диагностика")

    diag = _gather_diagnostics()
    cols = st.columns(3)
    cols[0].metric("CUDA", "доступна" if diag["cuda_available"] else "недоступна")
    cols[1].metric("VRAM, GB", diag["vram_gb"] if diag["cuda_available"] else "—")
    cols[2].metric("torch", diag["torch_version"])

    cols2 = st.columns(3)
    cols2[0].metric("yandex-cloud-ml-sdk", "ok" if diag["yandex_sdk_ok"] else "нет")
    cols2[1].metric("qdrant-client", diag["qdrant_version"])
    cols2[2].metric("pymorphy3", "ok" if diag["pymorphy3_ok"] else "нет")

    with st.expander("Дополнительно"):
        st.json({
            "embedding_model": EMBEDDING_MODEL_NAME,
            "sentiment_models": list(SENTIMENT_MODELS.keys()),
            "qdrant_storage": str(QDRANT_STORAGE_DIR),
            "yandex_api_key_env_set": bool(os.getenv("YANDEX_API_KEY")),
            "yandex_folder_env_set": bool(os.getenv("YANDEX_CATALOG_ID") or os.getenv("YANDEX_FOLDER_ID")),
        })

    st.markdown("---")
    st.subheader("🗂 Управление коллекциями Qdrant")

    store: QdrantStore = get_store()
    collections = [c for c in store.list_collections() if c.startswith("reviews_")]
    if not collections:
        st.info("Коллекций пока нет.")
    else:
        st.caption(f"Всего коллекций: {len(collections)}.")
        for c in collections:
            row = st.container()
            cc1, cc2, cc3 = row.columns([4, 2, 1])
            info = store.collection_info(c)
            cc1.markdown(f"**`{c}`** — точек: {info.get('points_count', '?')}")
            if cc2.button("Очистить (drop)", key=f"drop_{c}"):
                store.delete_collection(c)
                st.success(f"Коллекция `{c}` удалена.")
                st.rerun()
            if cc3.button("📄", key=f"info_{c}", help="Показать манифест"):
                meta = _meta_path(c)
                if meta.exists():
                    st.json(meta.read_text(encoding="utf-8"))
                else:
                    st.info("Манифест не найден.")

    st.markdown("---")
    st.subheader("♻️ Кэш")

    cc1, cc2 = st.columns(2)
    if cc1.button("Сбросить кэш ресурсов (модели/Qdrant клиент)", width="stretch"):
        clear_caches()
        st.success("Кэш ресурсов сброшен.")
    if cc2.button("Сбросить data-кэш Streamlit (CSV-загрузки и т.п.)", width="stretch"):
        st.cache_data.clear()
        st.success("Data-кэш сброшен.")



def _gather_diagnostics() -> dict:
    out = {
        "cuda_available": False,
        "vram_gb": "—",
        "torch_version": "—",
        "yandex_sdk_ok": False,
        "qdrant_version": "—",
        "pymorphy3_ok": False,
    }
    try:
        import torch
        out["torch_version"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out["vram_gb"] = f"{props.total_memory / 1024**3:.1f}"
    except Exception:
        pass
    try:
        import yandex_cloud_ml_sdk  # noqa: F401
        out["yandex_sdk_ok"] = True
    except Exception:
        pass
    try:
        import qdrant_client
        out["qdrant_version"] = getattr(qdrant_client, "__version__", "ok")
    except Exception:
        pass
    try:
        import pymorphy3  # noqa: F401
        out["pymorphy3_ok"] = True
    except Exception:
        pass
    return out
