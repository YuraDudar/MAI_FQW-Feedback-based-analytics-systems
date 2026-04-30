# Система автоматической кластеризации и анализа отзывов покупателей

| Студент                | Группа      | Тип работы |
| ---------------------- | ----------- | ---------- |
| Дударь Юрий Мохсенович | М8О-409Б-22 | ВКР (Дипломная работа) |

> Система трансформирует неструктурированный массив текстовых отзывов с маркетплейсов Wildberries и Ozon в тематические кластеры, сентимент-метрики и LLM-аналитику — без ручной разметки и без дата-сайентиста в штате.

---

## Оглавление

- [Архитектура](#архитектура)
- [Пайплайн кластеризации](#пайплайн-кластеризации)
- [RAG-система](#rag-система)
- [Автоответчик](#автоответчик)
- [Базы данных](#базы-данных)
- [Kafka топики](#kafka-топики)
- [Структура проекта](#структура-проекта)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация](#конфигурация)
- [Метрики качества](#метрики-качества)

---

## Архитектура

10 Docker-контейнеров, два изолированных сервиса, асинхронная коммуникация через Kafka.

[![Arhitektura1.jpg](https://i.postimg.cc/zDjWcVkz/Arhitektura1.jpg)](https://postimg.cc/SjnJXQ53)

### Контейнеры

| Контейнер | Тип | Технология | Назначение |
|---|---|---|---|
| `backend` | сервис | Python · FastAPI | REST API, аутентификация, парсинг WB, LLM-диалоги |
| `ml_service` | сервис | Python · FastAPI | Кластеризация, RAG, сентимент, автоответы |
| `frontend` | сервис | React · Vite | SPA: дашборд, граф кластеров, чат, выгрузка CSV |
| `postgres_backend` | инфра | PostgreSQL 15 | backend_db: пользователи, товары, задачи, отзывы |
| `postgres_ml` | инфра | PostgreSQL 15 | ml_db: кластеры, сентимент, инсайты |
| `redis` | инфра | Redis 7 | Кэш дашбордов (TTL 30 мин), distributed locks |
| `kafka` | инфра | Apache Kafka 3 | Асинхронная передача задач |
| `zookeeper` | инфра | ZooKeeper | Координация Kafka |
| `qdrant` | инфра | Qdrant 1.x | Векторная БД для RAG (cosine, 1024d) |
| `nginx` | инфра | NGINX 1.25 | Gateway, TLS, rate limiting, статика |

---

## Пайплайн кластеризации

```
CSV отзывов
    │
    ▼
Препроцессинг
    ├── удаление дублей и мусора
    ├── smart concat: «Достоинства: ... Недостатки: ... Комментарий: ...»
    ├── отсев текстов < 15 символов
    └── эвристический сплит на пулы:
        ├── negative: rating ≤ 3  +  disadvantages из 4–5★
        └── positive: advantages + comment из 4–5★
    │
    ▼
Эмбеддинги (SentenceTransformer, GPU, батчинг, кэш .npy)
    │
    ▼
UMAP  n_neighbors=5 · n_components=10 · min_dist=0.0 · metric=cosine
    │
    ▼
HDBSCAN  min_cluster_size=12 · min_samples=2 · EOM
    │
    ├── n_real ≥ min_acceptable → OK
    └── n_real < min_acceptable → KMeans fallback (target topics)
    │
    ▼
BERTopic (language=multilingual, кириллический токенизатор)
    ├── c-TF-IDF (default)
    ├── KeyBERT-inspired
    └── MMR diversity
    │
    ▼
LLM-нейминг → YandexGPT Lite / Pro
    │
    ▼
Сохранение: Clusters + ReviewClusterMapping (ml_db) + Qdrant upsert
```

### Модели эмбеддингов

| Модель | Dim | Max tokens | Batch | Префикс |
|---|---|---|---|---|
| `BAAI/bge-m3` | 1024 | 8192 | 32 | — |
| `intfloat/multilingual-e5-large` | 1024 | 512 | 24 | `passage:` / `query:` |
| `cointegrated/rubert-tiny2` | 312 | 2048 | 64 | — |


KMeans-fallback включается если `n_real < max(3, target // 2)`.


---

## RAG-система

Три последовательных этапа на каждый запрос пользователя:

```
User Query
    │
    ▼ Stage 1 — Query Expansion
YandexGPT Lite API  →  expanded_query (синонимы + ключевые слова)
    │
    ▼ Stage 2 — Retrieval
e5-large embed(query:) → Qdrant HNSW + native filters → top-K отзывов
    │
    ▼ Stage 3 — Generation
YandexGPT Pro  →  answer + review_ids источников
    │
    ▼
Frontend: ответ LLM + карточки отзывов-источников
```

### Фильтры (применяются на уровне HNSW-индекса Qdrant)

| Фильтр | Поле | Тип |
|---|---|---|
| Временное окно | `created_date` | date range |
| Рейтинг | `rating` | int range (1–5) |
| Тональность | `sentiment_label` | enum (positive/negative) |
| Гендер автора | `reviewer_gender` | enum (male/female/unknown) |

Гендер определяется через `pymorphy3` по полю `reviewer_name` (эвристика по грамматическому роду).

### Qdrant коллекция

```
Название:   reviews_{product_id}
Метрика:    Cosine
Размерность: 1024
Payload:    review_id · product_id · created_date · rating · sentiment_label · reviewer_gender
```

### Параметры запроса

| Параметр | Значение |
|---|---|
| top-K | выбирается пользователем |
| Inference time | ~44 сек |
| Токенов/запрос | ~1785 |
| Min similarity | 0.78 |

---

## Автоответчик

Асинхронный пайплайн генерации ответов продавца на отзывы:

```
Kafka: auto_reply_jobs
    │
    ▼
Проверка на спам (excluded_from_rating=true → пропуск)
    │
    ▼
Обогащение контекста
    ├── sentiment + gender ← ReviewSentiment (ml_db)
    │
    ▼
Дифференциация промпта:
    ├── rating ≤ 3 → эмпатичный тон + предложение решения
    └── rating ≥ 4 → благодарственный тон + акцент на достоинствах
    │
    ▼
YandexGPT Pro → generated_text
    │
    ▼
AutoReplyLog (status: generated)
    │
    ▼
Human-in-the-loop: менеджер одобряет → статус sent / failed
```


---

## Kafka топики

| Топик | Producer | Consumer | Payload |
|---|---|---|---|
| `parse_jobs` | backend | ml_service | `{product_id, platform, source_product_id, job_id}` |
| `cluster_jobs` | ml_service | ml_service | `{product_id, job_id, review_count}` |
| `analysis_done` | ml_service | backend | `{product_id, job_id}` |
| `auto_reply_jobs` | backend | ml_service | `{review_id, product_id, job_id}` |

**Гарантии:** `acks=all`, идемпотентный продьюсер, dead-letter topic при ошибках, offset коммит после успешной обработки.

### Полный асинхронный флоу

```
POST /api/v1/products
    → backend сохраняет в Products
    → publish parse_jobs
    → 202 Accepted 

ml_service: parse_jobs
    → парсинг WB (HTTP) / Ozon (chromedriver)
    → INSERT RawReviews → publish cluster_jobs

ml_service: cluster_jobs
    → полный пайплайн кластеризации
    → INSERT Clusters, ReviewClusterMapping, ReviewNLP, ReviewSentiment
    → upsert Qdrant → publish analysis_done

backend: analysis_done
    → UPDATE AnalysisJobs.status = completed
    → DEL cache:dashboard:{product_id}
    → следующий запрос дашборда → cache miss → SELECT pg → SET cache
```

---

## Структура проекта

```
.
├── infrastructure/
│   ├── config.py                  # все параметры системы
│   ├── docker-compose.yml         # контейнеры
│   ├── .env                       # секреты
│   ├── nginx/nginx.conf
│   ├── postgres/
│   │   ├── init_backend.sql       # backend_db схема
│   │   └── init_ml.sql            # ml_db схема
│   ├── redis/redis.conf
│   ├── kafka/init-topics.sh
│   └── qdrant/config.yaml
│
├── backend/
│   ├── main.py
│   ├── core/                      # database, redis, kafka, security
│   ├── models/                    # db_models, schemas
│   ├── services/                  # auth, product, job, review, llm
│   ├── api/                       # auth, products, jobs, reviews, rag, admin, export
│   ├── parsers/wb_parser.py       
│   └── consumers/analysis_done.py
│
├── ml_service/
│   ├── main.py
│   ├── core/                      # database, kafka, model_registry
│   ├── models/                    # db_models (ml_db)
│   ├── services/
│   │   ├── clustering/            # preprocessor, embedder, clusterer, naming
│   │   ├── sentiment/             # rubert-base-cased-sentiment + pymorphy3
│   │   ├── rag/                   # orchestrator, retriever, generator
│   │   ├── auto_reply/            # prompt builder, generator
│   │   └── insights/              # daily aggregator
│   ├── api/                       # rag, internal
│   └── consumers/                 # parse_jobs, cluster_jobs, auto_reply_jobs
│
├── frontend/
│   └── src/
│       ├── pages/                 # Dashboard, Cluster, Chat, Reviews, Export, Profile, Admin
│       ├── components/Layout/     # боковая навигация
│       ├── api/                   # auth, products, llm клиенты
│       ├── stores/                # authStore, appStore 
│       └── types/                 # TypeScript типы
│
└── tests/
    ├── test_backend/              # test_auth, test_products, test_jobs, test_reviews
    └── test_ml_service/           # test_sentiment, test_clustering, test_rag
```

---

## Быстрый старт

### Требования

- Docker ≥ 24.0, Docker Compose ≥ 2.20
- GPU с поддержкой CUDA (рекомендуется ≥ 8 GB VRAM для инференса e5-large)
- Python ≥ 3.10 (для автономных модулей)

### Запуск всей системы

```bash
git clone <repo-url>
cd review-analytics

# Заполнить секреты
cp infrastructure/.env.example infrastructure/.env
# отредактировать: YANDEX_API_KEY, YANDEX_FOLDER_ID, JWT_SECRET, ...

# Запустить всё
cd infrastructure
docker compose up -d

# Проверить статус
docker compose ps
```

Система поднимется на:
- Frontend: `http://localhost`
- Backend API: `http://localhost/api/v1`
- API docs (Swagger): `http://localhost/api/v1/docs`


---

## Конфигурация

Все параметры вынесены в `infrastructure/config.py`. Секреты — через `.env`.

### Ключевые параметры ML

```python
# Модель эмбеддингов
DEFAULT_EMBEDDING_MODEL = "e5-large"   # bge-m3 | e5-large | rubert-tiny2

# UMAP
UMAP_PARAMS = {
    "n_neighbors": 5,
    "n_components": 10,
    "min_dist": 0.0,
    "metric": "cosine",
}

# HDBSCAN
HDBSCAN_PARAMS = {
    "min_cluster_size": 12,
    "min_samples": 2,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

# BERTopic
TOP_N_WORDS = 10
MAX_TOPICS = 15
MIN_TEXT_CHARS = 15

# Representation
REPRESENTATION_MODEL = "default_ctfidf"  # keybert_inspired | mmr_diversity
```

### Ключевые параметры Redis

```python
CACHE_DASHBOARD_TTL = 1800   # 30 мин
CACHE_JOB_STATUS_TTL = 60    # 1 мин
REDIS_EVICTION_POLICY = "allkeys-lru"
```

### Переменные окружения (.env)

```env
# YandexGPT
YANDEX_API_KEY=...
YANDEX_FOLDER_ID=...
YANDEX_LITE_MODEL=yandexgpt-lite/latest
YANDEX_PRO_MODEL=yandexgpt/latest

# JWT
JWT_SECRET=...
JWT_ACCESS_TTL=900      # 15 мин
JWT_REFRESH_TTL=604800  # 7 дней

# PostgreSQL
BACKEND_DB_URL=postgresql://...
ML_DB_URL=postgresql://...

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_ACKS=all
```

---


## Технологический стек

| Слой | Технологии |
|---|---|
| **Backend** | Python 3.11 · FastAPI · NGINX · JWT |
| **Frontend** | React · Vite · TypeScript · Zustand · Streamlit|
| **ML / NLP** | BERTopic · HDBSCAN · KMeans · UMAP · sentence-transformers · pymorphy3 · Gensim |
| **Эмбеддинги** | BAAI/bge-m3 · intfloat/multilingual-e5-large · rubert-tiny2 |
| **Сентимент** | rubert-base-cased-sentiment |
| **LLM** | YandexGPT Lite · YandexGPT Pro · Qwen2.5-3B · Saiga Mistral 7B |
| **Парсинг** | requests · undetected-chromedriver · Selenium · BeautifulSoup · lxml |
| **БД** | PostgreSQL 15 (×2) · Redis 7 · Qdrant |
| **Очереди** | Apache Kafka 3 · ZooKeeper |
| **Инфра** | Docker · Docker Compose |

---
