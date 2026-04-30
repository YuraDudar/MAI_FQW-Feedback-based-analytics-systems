from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# ── Приложение ────────────────────────────────────────────────────────────────
APP_TITLE = "Feedback Analytics"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# ── JWT ───────────────────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_SUPER_SECRET_KEY_32CHARS!!")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_TTL_MINUTES = 15
JWT_REFRESH_TOKEN_TTL_DAYS = 7

# ── Backend DB (PostgreSQL) ───────────────────────────────────────────────────
BACKEND_DB_HOST = os.getenv("BACKEND_DB_HOST", "postgres_backend")
BACKEND_DB_PORT = int(os.getenv("BACKEND_DB_PORT", "5432"))
BACKEND_DB_NAME = os.getenv("BACKEND_DB_NAME", "backend_db")
BACKEND_DB_USER = os.getenv("BACKEND_DB_USER", "backend_user")
BACKEND_DB_PASSWORD = os.getenv("BACKEND_DB_PASSWORD", "backend_pass")
BACKEND_DB_URL = (
    f"postgresql+asyncpg://{BACKEND_DB_USER}:{BACKEND_DB_PASSWORD}"
    f"@{BACKEND_DB_HOST}:{BACKEND_DB_PORT}/{BACKEND_DB_NAME}"
)
BACKEND_DB_URL_SYNC = (
    f"postgresql://{BACKEND_DB_USER}:{BACKEND_DB_PASSWORD}"
    f"@{BACKEND_DB_HOST}:{BACKEND_DB_PORT}/{BACKEND_DB_NAME}"
)
BACKEND_DB_POOL_SIZE = int(os.getenv("BACKEND_DB_POOL_SIZE", "10"))
BACKEND_DB_MAX_OVERFLOW = int(os.getenv("BACKEND_DB_MAX_OVERFLOW", "20"))

# ── ML DB (PostgreSQL) ────────────────────────────────────────────────────────
ML_DB_HOST = os.getenv("ML_DB_HOST", "postgres_ml")
ML_DB_PORT = int(os.getenv("ML_DB_PORT", "5432"))
ML_DB_NAME = os.getenv("ML_DB_NAME", "ml_db")
ML_DB_USER = os.getenv("ML_DB_USER", "ml_user")
ML_DB_PASSWORD = os.getenv("ML_DB_PASSWORD", "ml_pass")
ML_DB_URL = (
    f"postgresql+asyncpg://{ML_DB_USER}:{ML_DB_PASSWORD}"
    f"@{ML_DB_HOST}:{ML_DB_PORT}/{ML_DB_NAME}"
)
ML_DB_URL_SYNC = (
    f"postgresql://{ML_DB_USER}:{ML_DB_PASSWORD}"
    f"@{ML_DB_HOST}:{ML_DB_PORT}/{ML_DB_NAME}"
)
ML_DB_POOL_SIZE = int(os.getenv("ML_DB_POOL_SIZE", "10"))
ML_DB_MAX_OVERFLOW = int(os.getenv("ML_DB_MAX_OVERFLOW", "20"))

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
REDIS_DASHBOARD_TTL = 1800
REDIS_PRODUCT_STATUS_TTL = 60
REDIS_LOCK_TTL = 300

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC_PARSE_JOBS = "parse_jobs"
KAFKA_TOPIC_CLUSTER_JOBS = "cluster_jobs"
KAFKA_TOPIC_ANALYSIS_DONE = "analysis_done"
KAFKA_TOPIC_AUTO_REPLY_JOBS = "auto_reply_jobs"
KAFKA_TOPIC_PARSE_JOBS_DLT = "parse_jobs.DLT"
KAFKA_TOPIC_CLUSTER_JOBS_DLT = "cluster_jobs.DLT"
KAFKA_TOPIC_AUTO_REPLY_DLT = "auto_reply_jobs.DLT"
KAFKA_CONSUMER_GROUP_BACKEND = "backend-group"
KAFKA_CONSUMER_GROUP_ML = "ml-service-group"
KAFKA_ACKS = "all"
KAFKA_ENABLE_IDEMPOTENCE = True
KAFKA_MAX_IN_FLIGHT = 1
KAFKA_RETRIES = 5
KAFKA_RETRY_BACKOFF_MS = 500

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_COLLECTION_PREFIX = "reviews_"
QDRANT_VECTOR_DIM = 1024
QDRANT_DISTANCE = "Cosine"
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 100
QDRANT_PAYLOAD_INDEX_FIELDS: dict[str, str] = {
    "review_id": "keyword",
    "product_id": "integer",
    "created_date": "datetime",
    "rating": "integer",
    "sentiment_label": "keyword",
    "reviewer_gender": "keyword",
}

# ── Yandex GPT ────────────────────────────────────────────────────────────────
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "") or os.getenv("YANDEX_CATALOG_ID", "")
YANDEX_LITE_MODEL = "yandexgpt-lite"
YANDEX_PRO_MODEL = "yandexgpt"
YANDEX_MODEL_VERSION = "latest"
YANDEX_LITE_TEMPERATURE = 0.3
YANDEX_LITE_MAX_TOKENS = 400
YANDEX_PRO_TEMPERATURE = 0.4
YANDEX_PRO_MAX_TOKENS = 1500

# ── Эмбеддинг-модели ──────────────────────────────────────────────────────────
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dim": 1024,
        "max_seq_length": 8192,
        "batch_size": 32,
        "prefix": None,
    },
    "e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_seq_length": 512,
        "batch_size": 24,
        "prefix": "passage: ",
    },
    "rubert-tiny2": {
        "name": "cointegrated/rubert-tiny2",
        "dim": 312,
        "max_seq_length": 2048,
        "batch_size": 64,
        "prefix": None,
    },
}
DEFAULT_EMBEDDING_MODEL = "bge-m3"
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_NORMALIZE = True

# ── Модели нейминга тем ──────────────────────────────────────────────────────
TOPIC_NAMING_MODELS: dict[str, str] = {
    "yandex-lite": "yandex:yandexgpt-lite/latest",
    "yandex-pro": "yandex:yandexgpt/latest",
    "local-qwen25-3b": "local:Qwen/Qwen2.5-3B-Instruct",
    "local-saiga-7b": "local:IlyaGusev/saiga_mistral_7b",
}
DEFAULT_TOPIC_NAMING_MODEL = "yandex-lite"

# ── Sentiment модели ──────────────────────────────────────────────────────────
SENTIMENT_MODELS: dict[str, dict] = {
    "rubert-base": {
        "name": "blanchefort/rubert-base-cased-sentiment",
        "max_seq_length": 512,
        "batch_size": 32,
    },
    "rubert-tiny2": {
        "name": "seara/rubert-tiny2-russian-sentiment",
        "max_seq_length": 512,
        "batch_size": 64,
    },
}
DEFAULT_SENTIMENT_MODEL = "rubert-base"

# ── UMAP параметры ────────────────────────────────────────────────────────────
UMAP_PARAMS: dict[str, Any] = {
    "n_neighbors": 5,
    "n_components": 10,
    "min_dist": 0.0,
    "metric": "cosine",
    "random_state": 42,
    "low_memory": True,
}
UMAP_VIS_PARAMS: dict[str, Any] = {
    "n_neighbors": 15,
    "n_components": 2,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": 42,
}

# ── HDBSCAN параметры ─────────────────────────────────────────────────────────
HDBSCAN_PARAMS: dict[str, Any] = {
    "min_cluster_size": 12,
    "min_samples": 2,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "prediction_data": True,
}

# ── Кластеризация ─────────────────────────────────────────────────────────────
MAX_TOPICS = 15
MIN_TEXT_CHARS = 15
NEGATIVE_RATING_THRESHOLD = 3

def get_target_topics(n_docs: int) -> int:
    if n_docs <= 50:
        return 3
    if n_docs <= 200:
        return 5
    if n_docs <= 600:
        return 8
    return 12

# ── BERTopic / Vectorizer ─────────────────────────────────────────────────────
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]{2,}")
TOP_N_WORDS = 10

VECTORIZER_PARAMS: dict[str, Any] = {
    "min_df": 1,
    "ngram_range": (1, 2),
}

# ── Стоп-слова ────────────────────────────────────────────────────────────────
RUSSIAN_STOP_WORDS: list[str] = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со",
    "как", "а", "то", "все", "она", "так", "его", "но", "да",
    "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
    "ее", "мне", "было", "вот", "от", "меня", "еще", "нет",
    "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут",
    "где", "есть", "надо", "ней", "для", "мы", "тебя", "их",
    "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот",
    "того", "потому", "этого", "какой", "совсем", "ним", "здесь",
    "этом", "один", "почти", "мой", "тем", "чтобы", "нее",
    "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед",
    "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им",
    "более", "всегда", "конечно", "всю", "между", "достоинства",
    "недостатки", "комментарий",
]
MARKETPLACE_STOP_WORDS: list[str] = [
    "wb", "wildberries", "вайлдберриз", "вб", "валдберис", "озон", "ozon",
    "товар", "товара", "товаром", "товару", "заказ", "заказа", "заказала",
    "доставка", "доставки", "доставили", "пришел", "пришла", "получила",
    "получил", "купила", "купил", "продавец", "магазин", "рекомендую",
    "спасибо", "фото", "отзыв", "отзывы", "звезда", "звезды",
]
ENGLISH_STOP_WORDS: list[str] = [
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "not",
    "it", "to", "in", "for", "was", "are", "as", "with", "be", "have",
    "do", "if", "will", "about", "so", "good", "great", "nice", "very",
]
ALL_STOP_WORDS: list[str] = list(set(
    RUSSIAN_STOP_WORDS + MARKETPLACE_STOP_WORDS + ENGLISH_STOP_WORDS
))

# ── RAG параметры ─────────────────────────────────────────────────────────────
RAG_DEFAULT_TOP_K = 10
RAG_MIN_TOP_K = 5
RAG_MAX_TOP_K = 40
RAG_DEFAULT_MIN_SCORE = 0.78
RAG_DEFAULT_OVERSAMPLE_FACTOR = 2.0
RAG_CONVERSATION_HISTORY_LIMIT = 6
TEXT_TRUNCATE_FOR_CONTEXT = 800

# ── WB Parser ─────────────────────────────────────────────────────────────────
WB_FEEDBACKS_HOSTS: list[str] = [
    "https://feedbacks1.wb.ru",
    "https://feedbacks2.wb.ru",
]
WB_FEEDBACKS_PATH = "/feedbacks/v1/{imt_id}"
WB_PAGE_SIZE = 30
WB_BASKET_DOMAINS: list[str] = ["wbbasket.ru", "wb.ru"]
WB_BASKET_CARD_PATH = "/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
WB_BASKET_THRESHOLDS: list[tuple[int, str]] = [
    (143, "01"), (287, "02"), (431, "03"), (719, "04"),
    (1007, "05"), (1061, "06"), (1115, "07"), (1169, "08"),
    (1313, "09"), (1601, "10"), (1655, "11"), (1919, "12"),
    (2045, "13"), (2189, "14"), (2405, "15"), (2621, "16"),
    (2837, "17"), (3053, "18"), (3269, "19"), (3485, "20"),
    (3701, "21"), (3917, "22"), (4133, "23"), (4349, "24"),
    (4565, "25"), (4781, "26"), (4997, "27"), (5213, "28"),
    (5429, "29"), (5645, "30"),
]
WB_BASKET_FALLBACK = "31"
WB_REQUEST_DELAY_MIN = 1.0
WB_REQUEST_DELAY_MAX = 3.5
WB_REQUEST_TIMEOUT = 30
WB_MAX_RETRIES = 3
WB_RETRY_DELAY_BASE = 5.0
WB_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

# ── Backend сервис ────────────────────────────────────────────────────────────
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_WORKERS = int(os.getenv("BACKEND_WORKERS", "4"))
BACKEND_CORS_ORIGINS: list[str] = os.getenv(
    "BACKEND_CORS_ORIGINS", "http://localhost,http://localhost:3000,http://localhost:80"
).split(",")
BACKEND_API_PREFIX = "/api/v1"

# ── ML сервис ─────────────────────────────────────────────────────────────────
ML_SERVICE_HOST = os.getenv("ML_SERVICE_HOST", "0.0.0.0")
ML_SERVICE_PORT = int(os.getenv("ML_SERVICE_PORT", "8001"))
ML_SERVICE_WORKERS = int(os.getenv("ML_SERVICE_WORKERS", "2"))
ML_SERVICE_INTERNAL_URL = os.getenv("ML_SERVICE_INTERNAL_URL", "http://ml_service:8001")

# ── Эмбеддинг кэш ────────────────────────────────────────────────────────────
EMBEDDINGS_CACHE_DIR = Path(os.getenv("EMBEDDINGS_CACHE_DIR", "/app/cache/embeddings"))

# ── Логирование ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Nginx ─────────────────────────────────────────────────────────────────────
NGINX_RATE_LIMIT_RPS = 10
NGINX_RATE_LIMIT_BURST = 20

# ── Инсайты ───────────────────────────────────────────────────────────────────
INSIGHTS_CRON_HOUR = int(os.getenv("INSIGHTS_CRON_HOUR", "3"))
INSIGHTS_CRON_MINUTE = int(os.getenv("INSIGHTS_CRON_MINUTE", "0"))
