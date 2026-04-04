"""
RAG Streamlit demo — entry point.

Run:
    streamlit run rag_pipline/streamlit_app.py
or:
    python -m streamlit run rag_pipline/streamlit_app.py

Tabs:
    📋 Датасет        — выбор CSV, статистика, выбор товара
    ⚙️ Индексация     — sentiment + gender + embeddings → Qdrant
    💬 Чат RAG        — диалог с фильтрами и историей
    📊 Аналитика      — распределения коллекции + инспектор последнего запроса
    📤 Экспорт        — LLM-friendly выгрузки
    🛠 Настройки      — Yandex creds, устройства, кэш, управление коллекциями
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import streamlit as st

from rag_pipline.ui import tab_analytics, tab_chat, tab_dataset, tab_export, tab_indexing, tab_settings
from rag_pipline.ui.state import init_session_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)


def _sidebar() -> None:
    st.sidebar.title("🔬 RAG Reviews")
    st.sidebar.caption("MVP: 4-этапный online RAG над отзывами маркетплейса.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Yandex Cloud")

    folder_id = st.sidebar.text_input(
        "Folder ID (catalog id)",
        value=st.session_state.get("yandex_folder_id", ""),
        key="sidebar_folder_id",
        help="Идентификатор каталога Yandex Cloud (b1g… или aje3…). "
             "Можно задать переменной YANDEX_CATALOG_ID.",
    )
    st.session_state["yandex_folder_id"] = folder_id.strip()

    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.get("yandex_api_key", ""),
        type="password",
        key="sidebar_api_key",
        help="API key Yandex Cloud для Foundation Models. "
             "Можно задать переменной YANDEX_API_KEY.",
    )
    st.session_state["yandex_api_key"] = api_key.strip()

    use_sdk = st.sidebar.checkbox(
        "Использовать yandex-cloud-ml-sdk",
        value=bool(st.session_state.get("yandex_use_sdk", True)),
        key="sidebar_use_sdk",
        help="Если SDK недоступен — автоматически используется REST.",
    )
    st.session_state["yandex_use_sdk"] = bool(use_sdk)

    if folder_id and api_key:
        st.sidebar.success("✅ Креды заполнены")
    else:
        st.sidebar.warning("⚠️ Заполни folder_id и API key для чата")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Статус**")
    coll = st.session_state.get("active_collection") or "—"
    st.sidebar.markdown(f"Активная коллекция: `{coll}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Архитектура запроса**")
    st.sidebar.markdown(
        "1. Lite — расширение запроса\n"
        "2. Qdrant — векторный поиск с фильтрами\n"
        "3. Pro — генерация ответа\n"
        "4. Формирование финального ответа"
    )

    st.sidebar.markdown("**Стек**")
    st.sidebar.markdown(
        "- Qdrant (embedded local)\n"
        "- multilingual-e5-large (1024-d)\n"
        "- YandexGPT Lite + Pro\n"
        "- pymorphy3 (gender)\n"
        "- ruBERT-sentiment"
    )


def main() -> None:
    st.set_page_config(
        page_title="RAG Reviews",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    _sidebar()

    st.title("🔬 RAG Reviews — диалоговая аналитика отзывов")
    st.caption(
        "Прототип: фильтруемая RAG-аналитика по отзывам с маркетплейса. "
        "Pipeline: индексация (sentiment + gender + e5-large → Qdrant) → "
        "online RAG (Lite expand → search → Pro answer)."
    )

    tabs = st.tabs([
        "📋 Датасет",
        "⚙️ Индексация",
        "💬 Чат RAG",
        "📊 Аналитика",
        "📤 Экспорт",
        "🛠 Настройки",
    ])

    with tabs[0]:
        tab_dataset.render()
    with tabs[1]:
        tab_indexing.render()
    with tabs[2]:
        tab_chat.render()
    with tabs[3]:
        tab_analytics.render()
    with tabs[4]:
        tab_export.render()
    with tabs[5]:
        tab_settings.render()


if __name__ == "__main__":
    main()
