import logging

import sys
sys.path.insert(0, "/app")
from infrastructure.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    EMBEDDING_MODELS,
    SENTIMENT_MODELS,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.embedder = None
        self.sentiment_model = None
        self.morph = None

    async def load_models(self):
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self):
        logger.info("Загрузка эмбеддинг-модели...")
        try:
            sys.path.insert(0, "/app/rag_pipline")
            from rag_pipline.pipeline.embedder import E5Embedder
            self.embedder = E5Embedder()
            logger.info("Эмбеддинг-модель загружена")
        except Exception as exc:
            logger.error("Ошибка загрузки эмбеддинг-модели: %s", exc)

        logger.info("Загрузка модели сентимента...")
        try:
            from transformers import pipeline as hf_pipeline
            cfg = SENTIMENT_MODELS[DEFAULT_SENTIMENT_MODEL]
            self.sentiment_model = hf_pipeline(
                "text-classification",
                model=cfg["name"],
                tokenizer=cfg["name"],
                max_length=cfg["max_seq_length"],
                truncation=True,
                device=-1,
            )
            logger.info("Модель сентимента загружена")
        except Exception as exc:
            logger.error("Ошибка загрузки модели сентимента: %s", exc)

        logger.info("Загрузка pymorphy3...")
        try:
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
            logger.info("pymorphy3 загружен")
        except Exception as exc:
            logger.error("Ошибка загрузки pymorphy3: %s", exc)


model_registry = ModelRegistry()
