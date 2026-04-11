import logging
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/rag_pipline")

from infrastructure.config import (
    YANDEX_API_KEY, YANDEX_FOLDER_ID,
    YANDEX_LITE_MODEL, YANDEX_PRO_MODEL, YANDEX_MODEL_VERSION,
    YANDEX_LITE_TEMPERATURE, YANDEX_LITE_MAX_TOKENS,
    YANDEX_PRO_TEMPERATURE, YANDEX_PRO_MAX_TOKENS,
    RAG_DEFAULT_TOP_K, RAG_DEFAULT_MIN_SCORE, RAG_DEFAULT_OVERSAMPLE_FACTOR,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_PREFIX,
)

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is not None:
            return self._orchestrator

        from rag_pipline.pipeline.embedder import E5Embedder
        from rag_pipline.pipeline.indexer import QdrantStore
        from rag_pipline.rag.orchestrator import RAGOrchestrator
        from rag_pipline.rag.yandex_provider import YandexLLM

        store = QdrantStore(host=QDRANT_HOST, port=QDRANT_PORT)
        embedder = E5Embedder()
        lite_llm = YandexLLM(
            api_key=YANDEX_API_KEY,
            folder_id=YANDEX_FOLDER_ID,
            model=YANDEX_LITE_MODEL,
            version=YANDEX_MODEL_VERSION,
            temperature=YANDEX_LITE_TEMPERATURE,
            max_tokens=YANDEX_LITE_MAX_TOKENS,
        )
        pro_llm = YandexLLM(
            api_key=YANDEX_API_KEY,
            folder_id=YANDEX_FOLDER_ID,
            model=YANDEX_PRO_MODEL,
            version=YANDEX_MODEL_VERSION,
            temperature=YANDEX_PRO_TEMPERATURE,
            max_tokens=YANDEX_PRO_MAX_TOKENS,
        )
        self._orchestrator = RAGOrchestrator(
            store=store,
            embedder=embedder,
            lite_llm=lite_llm,
            pro_llm=pro_llm,
        )
        return self._orchestrator

    async def query(
        self,
        query: str,
        product_id: int,
        top_k: int = RAG_DEFAULT_TOP_K,
        filters: dict | None = None,
    ) -> dict:
        import asyncio
        collection = f"{QDRANT_COLLECTION_PREFIX}{product_id}"
        orchestrator = self._get_orchestrator()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.run(
                query,
                collection=collection,
                top_k=top_k,
                filters=_map_filters(filters),
                min_score=RAG_DEFAULT_MIN_SCORE,
                oversample_factor=RAG_DEFAULT_OVERSAMPLE_FACTOR,
            ),
        )
        return {
            "answer": result.answer,
            "sources": result.source_review_ids,
            "expanded_query": result.expanded_query,
            "timings": result.timings,
        }


def _map_filters(filters: dict | None) -> dict | None:
    if not filters:
        return None
    mapped = {}
    if filters.get("date_from"):
        mapped["created_date_from"] = filters["date_from"]
    if filters.get("date_to"):
        mapped["created_date_to"] = filters["date_to"]
    if filters.get("stars_min"):
        mapped["rating_min"] = filters["stars_min"]
    if filters.get("stars_max"):
        mapped["rating_max"] = filters["stars_max"]
    if filters.get("sentiment"):
        mapped["sentiment_label"] = filters["sentiment"]
    if filters.get("gender"):
        mapped["reviewer_gender"] = filters["gender"]
    return mapped or None


rag_service = RAGService()
