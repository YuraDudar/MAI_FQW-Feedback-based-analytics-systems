import logging

from core.database import AsyncSessionLocal

import sys
sys.path.insert(0, "/app")
from infrastructure.config import KAFKA_TOPIC_ANALYSIS_DONE, BACKEND_DB_URL

logger = logging.getLogger(__name__)


async def handle_cluster_job(message: dict):
    product_id = message.get("product_id")
    job_id = message.get("job_id")
    review_count = message.get("review_count", 0)

    logger.info("Задача кластеризации: product_id=%s job_id=%s count=%s", product_id, job_id, review_count)

    if review_count == 0:
        logger.warning("Нет отзывов для кластеризации product_id=%s", product_id)
        return

    reviews = await _load_reviews_from_backend(product_id)
    if not reviews:
        logger.warning("Не загружены отзывы из backend_db для product_id=%s", product_id)
        return

    async with AsyncSessionLocal() as db:
        from services.clustering_service import ClusteringService
        svc = ClusteringService(db)
        try:
            await svc.run_full_pipeline(product_id, job_id, reviews)
            await db.commit()
        except Exception as exc:
            logger.exception("Ошибка кластеризации product_id=%s: %s", product_id, exc)
            await db.rollback()
            return

    from core.kafka_client import publish
    await publish(
        KAFKA_TOPIC_ANALYSIS_DONE,
        {"product_id": product_id, "job_id": job_id},
        key=str(product_id),
    )
    logger.info("Опубликовано analysis_done для product_id=%s", product_id)


async def _load_reviews_from_backend(product_id: int) -> list[dict]:
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import text

    engine = create_async_engine(BACKEND_DB_URL, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT review_id, product_id, rating, advantages, disadvantages, comment,
                       reviewer_name, created_date, excluded_from_rating
                FROM raw_reviews
                WHERE product_id = :product_id
                ORDER BY created_date DESC
            """),
            {"product_id": product_id},
        )
        rows = result.mappings().all()

    await engine.dispose()
    return [dict(r) for r in rows]
