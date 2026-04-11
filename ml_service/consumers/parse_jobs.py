import logging
import sys

sys.path.insert(0, "/app")

from core.database import AsyncSessionLocal

import sys
sys.path.insert(0, "/app")
from infrastructure.config import KAFKA_TOPIC_CLUSTER_JOBS

logger = logging.getLogger(__name__)


async def handle_parse_job(message: dict):
    product_id = message.get("product_id")
    platform = message.get("platform")
    source_product_id = message.get("source_product_id")
    job_id = message.get("job_id")
    max_reviews = message.get("max_reviews")

    logger.info(
        "Задача парсинга: platform=%s product_id=%s job_id=%s",
        platform, product_id, job_id,
    )

    reviews = []
    if platform == "wildberries":
        reviews = await _parse_wb(source_product_id, product_id, job_id, max_reviews)
    elif platform == "ozon":
        reviews = await _parse_ozon(source_product_id, product_id, job_id, max_reviews)

    if not reviews:
        logger.warning("Не получено отзывов для product_id=%s", product_id)
        return

    await _save_reviews_to_backend_db(reviews)

    async with AsyncSessionLocal() as db:
        from services.sentiment_service import SentimentService
        svc = SentimentService(db)
        await svc.process_reviews(reviews, product_id)
        await db.commit()

    from core.kafka_client import publish
    await publish(
        KAFKA_TOPIC_CLUSTER_JOBS,
        {"product_id": product_id, "job_id": job_id, "review_count": len(reviews)},
        key=str(product_id),
    )
    logger.info("Опубликована задача кластеризации для product_id=%s", product_id)


async def _parse_wb(nm_id: str, product_id: int, job_id: int, max_reviews: int | None) -> list[dict]:
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync():
        sys.path.insert(0, "/app/review_parser")
        from review_parser.wb_parser_util.core.fetcher import WBFetcher
        from review_parser.wb_parser_util.core.extractor import ReviewExtractor
        import random, time

        fetcher = WBFetcher()
        extractor = ReviewExtractor()
        imt_id = fetcher.fetch_imt_id(nm_id) or nm_id

        all_reviews = []
        skip = 0
        take = 30
        while True:
            page = fetcher.fetch_reviews_page(imt_id, nm_id, take=take, skip=skip)
            if not page:
                break
            feedbacks = page.get("feedbacks") or []
            if not feedbacks:
                break
            for raw in feedbacks:
                extracted = extractor.extract(raw)
                if extracted:
                    r = extracted if isinstance(extracted, dict) else vars(extracted)
                    all_reviews.append({
                        "review_id": str(r.get("id") or r.get("review_id", f"wb_{nm_id}_{skip}")),
                        "product_id": product_id,
                        "parsing_job_id": job_id,
                        "input_sku": nm_id,
                        "platform": "wildberries",
                        "nm_id": int(nm_id) if nm_id else None,
                        "reviewer_name": r.get("userName") or r.get("reviewer_name"),
                        "rating": r.get("productValuation") or r.get("rating"),
                        "advantages": r.get("pros") or r.get("advantages"),
                        "disadvantages": r.get("cons") or r.get("disadvantages"),
                        "comment": r.get("text") or r.get("comment"),
                        "created_date": str(r.get("createdDate") or r.get("created_date", "")),
                        "excluded_from_rating": r.get("isRejected", False),
                        "votes_plus": r.get("votesPlus", 0),
                        "votes_minus": r.get("votesMinus", 0),
                        "has_video": r.get("hasVideo", False),
                    })
            skip += len(feedbacks)
            if len(feedbacks) < take:
                break
            if max_reviews and len(all_reviews) >= max_reviews:
                break

        fetcher.close()
        return all_reviews[:max_reviews] if max_reviews else all_reviews

    return await loop.run_in_executor(None, _sync)


async def _parse_ozon(product_url: str, product_id: int, job_id: int, max_reviews: int | None) -> list[dict]:
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync():
        sys.path.insert(0, "/app/review_parser")
        from review_parser.ozon_parser_util.core.fetcher import OzonFetcher
        from review_parser.ozon_parser_util.core.extractor import OzonExtractor

        fetcher = OzonFetcher()
        extractor = OzonExtractor()
        raw_reviews = fetcher.fetch_all(product_url)
        result = []
        for raw in raw_reviews:
            r = extractor.extract(raw)
            if r:
                result.append({
                    "review_id": str(r.get("id", "")),
                    "product_id": product_id,
                    "parsing_job_id": job_id,
                    "input_sku": product_url,
                    "platform": "ozon",
                    "reviewer_name": r.get("author"),
                    "rating": r.get("rating"),
                    "advantages": r.get("pros"),
                    "disadvantages": r.get("cons"),
                    "comment": r.get("text"),
                    "created_date": str(r.get("created_at", "")),
                    "excluded_from_rating": False,
                    "votes_plus": 0,
                    "votes_minus": 0,
                    "has_video": False,
                })
        return result[:max_reviews] if max_reviews else result

    return await loop.run_in_executor(None, _sync)


async def _save_reviews_to_backend_db(reviews: list[dict]):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    import sys
    sys.path.insert(0, "/app")
    from infrastructure.config import BACKEND_DB_URL

    engine = create_async_engine(BACKEND_DB_URL, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        for r in reviews:
            from sqlalchemy import text
            try:
                await session.execute(
                    text("""
                        INSERT INTO raw_reviews (
                            review_id, product_id, parsing_job_id, input_sku, platform,
                            nm_id, reviewer_name, rating, advantages, disadvantages, comment,
                            created_date, excluded_from_rating, votes_plus, votes_minus, has_video
                        ) VALUES (
                            :review_id, :product_id, :parsing_job_id, :input_sku, :platform,
                            :nm_id, :reviewer_name, :rating, :advantages, :disadvantages, :comment,
                            :created_date, :excluded_from_rating, :votes_plus, :votes_minus, :has_video
                        ) ON CONFLICT (review_id) DO NOTHING
                    """),
                    {
                        "review_id": r.get("review_id"),
                        "product_id": r.get("product_id"),
                        "parsing_job_id": r.get("parsing_job_id"),
                        "input_sku": r.get("input_sku"),
                        "platform": r.get("platform"),
                        "nm_id": r.get("nm_id"),
                        "reviewer_name": r.get("reviewer_name"),
                        "rating": r.get("rating"),
                        "advantages": r.get("advantages"),
                        "disadvantages": r.get("disadvantages"),
                        "comment": r.get("comment"),
                        "created_date": r.get("created_date"),
                        "excluded_from_rating": r.get("excluded_from_rating", False),
                        "votes_plus": r.get("votes_plus", 0),
                        "votes_minus": r.get("votes_minus", 0),
                        "has_video": r.get("has_video", False),
                    },
                )
            except Exception as exc:
                logger.warning("Ошибка вставки отзыва %s: %s", r.get("review_id"), exc)
        await session.commit()

    await engine.dispose()
