from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from core.redis_client import get_redis, RedisPool
from models.schemas import ReviewsListResponse, ReviewResponse, DashboardResponse
from services.review_service import ReviewService

router = APIRouter(prefix="/reviews", tags=["Отзывы"])


@router.get("/{product_id}/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    product_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    redis: RedisPool = Depends(get_redis),
):
    cached = await redis.get_dashboard(product_id)
    if cached:
        return cached
    svc = ReviewService(db)
    data = await svc.get_dashboard_data(product_id, user_id)
    await redis.set_dashboard(product_id, data)
    return data


@router.get("/{product_id}", response_model=ReviewsListResponse)
async def list_reviews(
    product_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    rating_min: int | None = Query(default=None, ge=1, le=5),
    rating_max: int | None = Query(default=None, ge=1, le=5),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = ReviewService(db)
    return await svc.get_reviews(
        product_id, user_id, page, page_size,
        rating_min, rating_max, date_from, date_to,
    )


@router.get("/item/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = ReviewService(db)
    return await svc.get_review(review_id, user_id)
