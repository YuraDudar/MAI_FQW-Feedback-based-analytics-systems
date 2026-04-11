import logging

from core.database import AsyncSessionLocal

import sys
sys.path.insert(0, "/app")
from infrastructure.config import BACKEND_DB_URL

logger = logging.getLogger(__name__)


async def handle_auto_reply_job(message: dict):
    review_id = message.get("review_id")
    product_id = message.get("product_id")
    job_id = message.get("job_id")

    logger.info("Автоответ: review_id=%s product_id=%s", review_id, product_id)

    review_data = await _load_review(review_id)
    if not review_data:
        logger.warning("Отзыв не найден: review_id=%s", review_id)
        return

    async with AsyncSessionLocal() as db:
        from services.auto_reply_service import AutoReplyService
        svc = AutoReplyService(db)
        try:
            await svc.generate_reply(review_id, product_id, job_id, review_data)
            await db.commit()
        except Exception as exc:
            logger.exception("Ошибка генерации автоответа: %s", exc)
            await db.rollback()


async def _load_review(review_id: str) -> dict | None:
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import text

    engine = create_async_engine(BACKEND_DB_URL, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT review_id, product_id, rating, advantages, disadvantages,
                       comment, excluded_from_rating
                FROM raw_reviews WHERE review_id = :review_id
            """),
            {"review_id": review_id},
        )
        row = result.mappings().one_or_none()

    await engine.dispose()
    return dict(row) if row else None
