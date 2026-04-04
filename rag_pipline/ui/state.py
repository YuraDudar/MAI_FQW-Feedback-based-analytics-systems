"""
Shared Streamlit session-state helpers + cached resource builders.

We deliberately keep the QdrantStore as a singleton on st.session_state — embedded
mode allows only one process holding the storage path open, so every tab that
needs it goes through `get_store()`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from rag_pipline.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_OVERSAMPLE_FACTOR,
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_YANDEX_API_KEY,
    DEFAULT_YANDEX_FOLDER_ID,
    QDRANT_STORAGE_DIR,
)
from rag_pipline.pipeline.embedder import E5Embedder
from rag_pipline.pipeline.indexer import QdrantStore

log = logging.getLogger(__name__)



def init_session_state() -> None:
    """Populate session_state with defaults on first run of each session."""
    defaults: dict[str, Any] = {
        "yandex_api_key": DEFAULT_YANDEX_API_KEY,
        "yandex_folder_id": DEFAULT_YANDEX_FOLDER_ID,
        "yandex_use_sdk": True,

        "selected_csv": None,            
        "selected_product_id": None,     

        "sentiment_model_key": DEFAULT_SENTIMENT_MODEL,
        "indexing_running": False,
        "indexing_log": [],              
        "last_index_manifest": None,

        "chat_history": [],              
        "active_collection": None,       
        "rag_top_k": DEFAULT_TOP_K,
        "rag_min_score": DEFAULT_MIN_SCORE,
        "rag_oversample": DEFAULT_OVERSAMPLE_FACTOR,
        "rag_skip_expansion": False,
        "rag_filters": {},
        "last_rag_result": None,
        "last_run_id": 0,

        "embedder_device": "auto",
        "sentiment_device": "auto",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)



@st.cache_resource(show_spinner=False)
def get_store(storage_path: str = str(QDRANT_STORAGE_DIR)) -> QdrantStore:
    """Singleton QdrantStore (embedded mode requires exclusive access)."""
    return QdrantStore(storage_path=Path(storage_path))


@st.cache_resource(show_spinner=False)
def get_embedder(device: str = "auto") -> E5Embedder:
    """Cached E5 embedder. Reuses GPU memory across queries."""
    e = E5Embedder(device=device)
    e.load()
    return e


def reset_chat() -> None:
    st.session_state["chat_history"] = []
    st.session_state["last_rag_result"] = None


def clear_caches() -> None:
    """Drop cached resources (useful when changing device or storage path)."""
    st.cache_resource.clear()



def push_log(line: str) -> None:
    log_list = st.session_state.setdefault("indexing_log", [])
    log_list.append(line)


def get_log_text() -> str:
    return "\n".join(st.session_state.get("indexing_log", []))
