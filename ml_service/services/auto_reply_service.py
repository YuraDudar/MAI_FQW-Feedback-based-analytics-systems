import logging
import sys

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, "/app")
from infrastructure.config import (
    YANDEX_API_KEY, YANDEX_FOLDER_ID,
    YANDEX_PRO_MODEL, YANDEX_MODEL_VERSION,
    YANDEX_PRO_TEMPERATURE, YANDEX_PRO_MAX_TOKENS,
)
from models.db_models import AutoReplyLog, ReviewSentiment, ReviewClusterMapping, Cluster

logger = logging.getLogger(__name__)

_NEGATIVE_PROMPT = """Ты — вежливый представитель поддержки магазина.
Покупатель оставил негативный отзыв. Напиши эмпатичный, дружелюбный ответ.
Признай проблему, предложи решение или компенсацию. Не более 3-4 предложений.
Отзыв покупателя: {review_text}
Кластер проблем: {cluster_label}"""

_POSITIVE_PROMPT = """Ты — вежливый представитель поддержки магазина.
Покупатель оставил положительный отзыв. Напиши благодарственный ответ.
Подчеркни ценность отзыва. Не более 2-3 предложений.
Отзыв покупателя: {review_text}"""


class AutoReplyService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            sys.path.insert(0, "/app/rag_pipline")
            from rag_pipline.rag.yandex_provider import YandexLLM
            self._llm = YandexLLM(
                api_key=YANDEX_API_KEY,
                folder_id=YANDEX_FOLDER_ID,
                model=YANDEX_PRO_MODEL,
                version=YANDEX_MODEL_VERSION,
                temperature=YANDEX_PRO_TEMPERATURE,
                max_tokens=YANDEX_PRO_MAX_TOKENS,
            )
        return self._llm

    async def generate_reply(
        self,
        review_id: str,
        product_id: int,
        job_id: int,
        review_data: dict,
    ) -> str | None:
        if review_data.get("excluded_from_rating"):
            logger.info("Пропуск спам-отзыва review_id=%s", review_id)
            return None

        sent_result = await self.db.execute(
            select(ReviewSentiment).where(ReviewSentiment.review_id == review_id)
        )
        sentiment = sent_result.scalar_one_or_none()

        cluster_label = ""
        if sentiment:
            mapping_result = await self.db.execute(
                select(ReviewClusterMapping).where(ReviewClusterMapping.review_id == review_id)
            )
            mapping = mapping_result.scalar_one_or_none()
            if mapping:
                cluster_result = await self.db.execute(
                    select(Cluster).where(Cluster.cluster_id == mapping.cluster_id)
                )
                cluster = cluster_result.scalar_one_or_none()
                if cluster:
                    cluster_label = cluster.llm_label or ""

        review_text = " ".join(filter(None, [
            review_data.get("advantages", ""),
            review_data.get("disadvantages", ""),
            review_data.get("comment", ""),
        ]))

        rating = review_data.get("rating", 3)
        is_negative = (rating <= 3) or (
            sentiment and sentiment.sentiment_label.value == "negative"
        )

        if is_negative:
            prompt_text = _NEGATIVE_PROMPT.format(
                review_text=review_text[:500],
                cluster_label=cluster_label,
            )
        else:
            prompt_text = _POSITIVE_PROMPT.format(review_text=review_text[:500])

        try:
            llm = self._get_llm()
            import asyncio
            loop = asyncio.get_event_loop()
            generated = await loop.run_in_executor(
                None,
                lambda: llm.generate("Ты — помощник магазина.", prompt_text),
            )
        except Exception as exc:
            logger.error("Ошибка генерации автоответа: %s", exc)
            stmt = insert(AutoReplyLog).values(
                review_id=review_id,
                product_id=product_id,
                job_id=job_id,
                generated_text="",
                status="failed",
                error_message=str(exc),
            ).on_conflict_do_update(
                index_elements=["review_id"],
                set_={"status": "failed", "error_message": str(exc)},
            )
            await self.db.execute(stmt)
            await self.db.flush()
            return None

        stmt = insert(AutoReplyLog).values(
            review_id=review_id,
            product_id=product_id,
            job_id=job_id,
            generated_text=generated,
            status="generated",
        ).on_conflict_do_update(
            index_elements=["review_id"],
            set_={"generated_text": generated, "status": "generated"},
        )
        await self.db.execute(stmt)
        await self.db.flush()
        logger.info("Автоответ сгенерирован для review_id=%s", review_id)
        return generated
