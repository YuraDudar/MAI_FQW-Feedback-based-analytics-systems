import logging
import sys
from datetime import date, datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/rag_pipline")

from infrastructure.config import (
    YANDEX_API_KEY, YANDEX_FOLDER_ID,
    YANDEX_PRO_MODEL, YANDEX_MODEL_VERSION,
    YANDEX_PRO_TEMPERATURE, YANDEX_PRO_MAX_TOKENS,
)
from models.db_models import ProductDailyInsights, ReviewSentiment, Cluster

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """На основе следующей статистики по отзывам товара подготовь аналитическое резюме (3-5 предложений).
Выдели главные проблемы и достоинства. Дай рекомендацию продавцу.

Статистика:
- Всего отзывов: {total}
- Средний рейтинг: {avg_rating:.1f}
- Положительных: {positive}
- Отрицательных: {negative}
- Нейтральных: {neutral}
- Spam rate: {spam_rate:.1%}

Главные проблемы: {top_problems}
Главные достоинства: {top_positives}"""


class InsightsService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
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

    async def compute_daily_insights(self, product_id: int):
        today = date.today()

        stats_result = await self.db.execute(
            select(
                func.count(ReviewSentiment.review_id).label("total"),
                func.avg(ReviewSentiment.sentiment_score).label("avg_sentiment"),
            ).where(ReviewSentiment.product_id == product_id)
        )
        stats = stats_result.one()
        total = stats.total or 0

        positive_count = await self._count_by_sentiment(product_id, "positive")
        negative_count = await self._count_by_sentiment(product_id, "negative")
        neutral_count = await self._count_by_sentiment(product_id, "neutral")

        spam_result = await self.db.execute(
            select(func.count()).select_from(ReviewSentiment)
            .where(
                ReviewSentiment.product_id == product_id,
                ReviewSentiment.sentiment_score < 0.3,
            )
        )
        low_quality = spam_result.scalar_one() or 0
        spam_rate = low_quality / total if total > 0 else 0.0

        health_score = self._calc_health_score(
            positive_count, negative_count, neutral_count, spam_rate
        )

        top_problems = await self._get_top_clusters(product_id, "negative")
        top_positives = await self._get_top_clusters(product_id, "positive")

        prompt = _SUMMARY_PROMPT.format(
            total=total,
            avg_rating=0.0,
            positive=positive_count,
            negative=negative_count,
            neutral=neutral_count,
            spam_rate=spam_rate,
            top_problems=", ".join([c["label"] for c in top_problems[:3]]),
            top_positives=", ".join([c["label"] for c in top_positives[:3]]),
        )

        llm_summary = ""
        try:
            llm = self._get_llm()
            import asyncio
            loop = asyncio.get_event_loop()
            llm_summary = await loop.run_in_executor(
                None,
                lambda: llm.generate("Ты — аналитик маркетплейса.", prompt),
            )
        except Exception as exc:
            logger.error("Ошибка генерации инсайтов: %s", exc)

        stmt = insert(ProductDailyInsights).values(
            product_id=product_id,
            analysis_date=today,
            health_score=health_score,
            spam_rate=spam_rate,
            avg_rating=None,
            total_reviews=total,
            positive_count=positive_count,
            negative_count=negative_count,
            llm_summary=llm_summary,
            top_problems=top_problems,
            top_positives=top_positives,
        ).on_conflict_do_update(
            index_elements=["product_id", "analysis_date"],
            set_={
                "health_score": health_score,
                "total_reviews": total,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "llm_summary": llm_summary,
            },
        )
        await self.db.execute(stmt)
        await self.db.flush()
        logger.info("Инсайты обновлены для product_id=%s", product_id)

    async def _count_by_sentiment(self, product_id: int, label: str) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(ReviewSentiment)
            .where(ReviewSentiment.product_id == product_id, ReviewSentiment.sentiment_label == label)
        )
        return result.scalar_one() or 0

    async def _get_top_clusters(self, product_id: int, sentiment: str) -> list[dict]:
        result = await self.db.execute(
            select(Cluster)
            .where(Cluster.product_id == product_id, Cluster.sentiment_category == sentiment)
            .order_by(Cluster.review_count.desc())
            .limit(5)
        )
        clusters = result.scalars().all()
        return [
            {"label": c.llm_label or f"Кластер {c.cluster_id}", "count": c.review_count}
            for c in clusters
        ]

    def _calc_health_score(self, pos: int, neg: int, neu: int, spam_rate: float) -> float:
        total = pos + neg + neu
        if total == 0:
            return 5.0
        ratio = pos / total
        score = ratio * 10.0 - spam_rate * 2.0
        return max(0.0, min(10.0, score))
