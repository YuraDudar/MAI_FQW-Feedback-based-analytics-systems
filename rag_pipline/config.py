"""
Central configuration for the RAG pipeline.

Layout:
    rag_pipline/
        config.py            ← this file
        streamlit_app.py     ← UI entry-point
        pipeline/            ← offline preprocessing + indexing
        rag/                 ← online query-time RAG orchestration
        ui/                  ← Streamlit tab modules
        results/             ← Qdrant storage, exports, logs (gitignored)
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CSV_DIR = PROJECT_ROOT / "review_parser" / "results"

RESULTS_DIR = BASE_DIR / "results"
QDRANT_STORAGE_DIR = RESULTS_DIR / "qdrant_storage"
EXPORTS_DIR = RESULTS_DIR / "exports"
LOGS_DIR = RESULTS_DIR / "logs"
INDEX_META_DIR = RESULTS_DIR / "index_meta"   

for _d in (RESULTS_DIR, QDRANT_STORAGE_DIR, EXPORTS_DIR, LOGS_DIR, INDEX_META_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Embedding model  ──
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_BATCH_SIZE = 8           
EMBEDDING_NORMALIZE = True         


# ── Sentiment models ──────────────────────────────────────────
SENTIMENT_MODELS: dict[str, dict] = {
    "rubert-base": {
        "name": "blanchefort/rubert-base-cased-sentiment",
        "description": "RuBERT-base, 3 класса (positive/negative/neutral). Точнее, ~700 MB.",
        "max_seq_length": 512,
        "batch_size": 32,
    },
    "rubert-tiny2": {
        "name": "seara/rubert-tiny2-russian-sentiment",
        "description": "RuBERT-tiny2, очень лёгкая (~120 MB), быстрая, чуть менее точная.",
        "max_seq_length": 512,
        "batch_size": 64,
    },
}
DEFAULT_SENTIMENT_MODEL = "rubert-base"
SENTIMENT_LABELS = ("positive", "negative", "neutral")


# ── Qdrant ────────────────────────────────────────────────────
QDRANT_COLLECTION_PREFIX = "reviews_"
QDRANT_PAYLOAD_INDEX_FIELDS = {
    "rating": "integer",
    "sentiment_label": "keyword",
    "reviewer_gender": "keyword",
    "created_date": "datetime",
    "product_id": "integer",
}


# ── YandexGPT (via yandex-cloud-ml-sdk) ───────────────────────
YANDEX_LITE_MODEL = "yandexgpt-lite"   
YANDEX_PRO_MODEL = "yandexgpt"         
YANDEX_MODEL_VERSION = "latest"

YANDEX_LITE_TEMPERATURE = 0.3
YANDEX_LITE_MAX_TOKENS = 400
YANDEX_PRO_TEMPERATURE = 0.4
YANDEX_PRO_MAX_TOKENS = 1500


# ── RAG defaults ──────────────────────────────────────────────
DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 100


DEFAULT_MIN_SCORE = 0.78
MIN_SCORE_FLOOR = 0.0
MIN_SCORE_CEIL = 0.99


DEFAULT_OVERSAMPLE_FACTOR = 2.0
MIN_OVERSAMPLE_FACTOR = 1.0
MAX_OVERSAMPLE_FACTOR = 5.0

CONVERSATION_HISTORY_LIMIT = 6   


# ── Text fields & preprocessing ──────────────────────────────
TEXT_FIELDS = ["advantages", "disadvantages", "comment"]
MIN_TEXT_CHARS = 15
TEXT_TRUNCATE_FOR_CONTEXT = 800  

REVIEW_ID_FIELD = "review_id"
PRODUCT_ID_FIELD = "input_sku"          
VARIANT_ID_FIELD = "nm_id"              
RATING_FIELD = "rating"
CREATED_DATE_FIELD = "created_date"
REVIEWER_NAME_FIELD = "reviewer_name"


# ── Defaults from environment ────────────────────────────────
DEFAULT_YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
DEFAULT_YANDEX_FOLDER_ID = os.getenv("YANDEX_CATALOG_ID", "") or os.getenv("YANDEX_FOLDER_ID", "")
