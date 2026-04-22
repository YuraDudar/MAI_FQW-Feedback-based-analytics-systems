from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from core.kafka_client import get_kafka_producer, KafkaProducerClient
from core.redis_client import get_redis, RedisPool
from models.schemas import ParseJobRequest, ClusterJobRequest, AutoReplyJobRequest, JobResponse
from services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["Задачи"])


@router.post("/parse", response_model=JobResponse, status_code=202)
async def start_parse(
    data: ParseJobRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    kafka: KafkaProducerClient = Depends(get_kafka_producer),
    redis: RedisPool = Depends(get_redis),
):
    svc = JobService(db, kafka, redis)
    return await svc.create_parse_job(data.product_id, user_id, data.max_reviews)


@router.post("/auto-reply", response_model=JobResponse, status_code=202)
async def start_auto_reply(
    data: AutoReplyJobRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    kafka: KafkaProducerClient = Depends(get_kafka_producer),
    redis: RedisPool = Depends(get_redis),
):
    svc = JobService(db, kafka, redis)
    return await svc.create_auto_reply_job(data.product_id, data.review_ids, user_id)


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    product_id: int | None = None,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    kafka: KafkaProducerClient = Depends(get_kafka_producer),
    redis: RedisPool = Depends(get_redis),
):
    svc = JobService(db, kafka, redis)
    return await svc.list_jobs(user_id, product_id)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    kafka: KafkaProducerClient = Depends(get_kafka_producer),
    redis: RedisPool = Depends(get_redis),
):
    svc = JobService(db, kafka, redis)
    return await svc.get_job(job_id, user_id)
