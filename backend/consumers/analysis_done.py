import logging

from core.database import AsyncSessionLocal
from core.redis_client import redis_pool
from models.db_models import AnalysisJob, JobStatus
from sqlalchemy import update

logger = logging.getLogger(__name__)


async def handle_analysis_done(message: dict):
    product_id = message.get("product_id")
    job_id = message.get("job_id")
    logger.info("Получено analysis_done: product_id=%s job_id=%s", product_id, job_id)

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                update(AnalysisJob)
                .where(AnalysisJob.job_id == job_id)
                .values(status=JobStatus.completed)
            )
            await session.commit()
        except Exception as exc:
            logger.error("Ошибка обновления статуса задачи %s: %s", job_id, exc)
            await session.rollback()

    if product_id:
        await redis_pool.invalidate_dashboard(product_id)
        await redis_pool.set_product_status(product_id, "completed")
        logger.info("Кэш дашборда инвалидирован для product_id=%s", product_id)
