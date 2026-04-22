import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from models.db_models import AnalysisJob, JobStatus, JobType, Product, PlatformType
from models.schemas import JobCreate
from core.kafka_client import KafkaProducerClient
from core.redis_client import RedisPool

import sys
sys.path.insert(0, "/app")
from infrastructure.config import (
    KAFKA_TOPIC_PARSE_JOBS,
    KAFKA_TOPIC_AUTO_REPLY_JOBS,
)

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, db: AsyncSession, kafka: KafkaProducerClient, redis: RedisPool):
        self.db = db
        self.kafka = kafka
        self.redis = redis

    async def create_parse_job(self, product_id: int, user_id: int, max_reviews: int | None = None) -> AnalysisJob:
        product = await self._get_product(product_id, user_id)
        job = AnalysisJob(
            product_id=product_id,
            user_id=user_id,
            job_type=JobType.parsing,
            status=JobStatus.pending,
            parameters={"max_reviews": max_reviews} if max_reviews else {},
        )
        self.db.add(job)
        await self.db.flush()
        await self.db.refresh(job)

        await self.kafka.send(
            KAFKA_TOPIC_PARSE_JOBS,
            {
                "product_id": product_id,
                "platform": product.source.platform.value,
                "source_product_id": product.source_product_id,
                "job_id": job.job_id,
                "max_reviews": max_reviews,
            },
            key=str(product_id),
        )
        await self.redis.set_product_status(product_id, "parsing")
        logger.info("Задача парсинга создана: job_id=%s product_id=%s", job.job_id, product_id)
        return job

    async def create_auto_reply_job(self, product_id: int, review_ids: list[str], user_id: int) -> AnalysisJob:
        job = AnalysisJob(
            product_id=product_id,
            user_id=user_id,
            job_type=JobType.auto_reply,
            status=JobStatus.pending,
            parameters={"review_ids": review_ids},
        )
        self.db.add(job)
        await self.db.flush()
        await self.db.refresh(job)

        for review_id in review_ids:
            await self.kafka.send(
                KAFKA_TOPIC_AUTO_REPLY_JOBS,
                {"review_id": review_id, "product_id": product_id, "job_id": job.job_id},
                key=str(product_id),
            )
        return job

    async def get_job(self, job_id: int, user_id: int) -> AnalysisJob:
        result = await self.db.execute(
            select(AnalysisJob).where(
                AnalysisJob.job_id == job_id,
                AnalysisJob.user_id == user_id,
            )
        )
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Задача не найдена")
        return job

    async def list_jobs(self, user_id: int, product_id: int | None = None) -> list[AnalysisJob]:
        q = select(AnalysisJob).where(AnalysisJob.user_id == user_id)
        if product_id:
            q = q.where(AnalysisJob.product_id == product_id)
        q = q.order_by(AnalysisJob.created_at.desc()).limit(100)
        result = await self.db.execute(q)
        return list(result.scalars().all())

    async def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        error: str | None = None,
        results_summary: dict | None = None,
    ):
        values: dict = {"status": status}
        if status == JobStatus.running:
            values["start_time"] = datetime.now(timezone.utc)
        if status in (JobStatus.completed, JobStatus.failed):
            values["end_time"] = datetime.now(timezone.utc)
        if error:
            values["error_message"] = error
        if results_summary:
            values["results_summary"] = results_summary
        await self.db.execute(
            update(AnalysisJob).where(AnalysisJob.job_id == job_id).values(**values)
        )

    async def _get_product(self, product_id: int, user_id: int) -> Product:
        from sqlalchemy.orm import joinedload
        result = await self.db.execute(
            select(Product)
            .options(joinedload(Product.source))
            .where(Product.product_id == product_id, Product.user_id == user_id)
        )
        product = result.scalar_one_or_none()
        if not product:
            raise HTTPException(status_code=404, detail="Товар не найден")
        return product
