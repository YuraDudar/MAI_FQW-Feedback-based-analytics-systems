import logging
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from models.db_models import RawReview, Product
from models.schemas import ReviewsListResponse

logger = logging.getLogger(__name__)


class ReviewService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_reviews(
        self,
        product_id: int,
        user_id: int,
        page: int = 1,
        page_size: int = 50,
        rating_min: int | None = None,
        rating_max: int | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> ReviewsListResponse:
        product = await self._check_access(product_id, user_id)

        filters = [RawReview.product_id == product_id]
        if rating_min:
            filters.append(RawReview.rating >= rating_min)
        if rating_max:
            filters.append(RawReview.rating <= rating_max)
        if date_from:
            filters.append(RawReview.created_date >= date_from)
        if date_to:
            filters.append(RawReview.created_date <= date_to)

        count_result = await self.db.execute(
            select(func.count()).select_from(RawReview).where(and_(*filters))
        )
        total = count_result.scalar_one()

        offset = (page - 1) * page_size
        result = await self.db.execute(
            select(RawReview)
            .where(and_(*filters))
            .order_by(RawReview.created_date.desc())
            .offset(offset)
            .limit(page_size)
        )
        reviews = list(result.scalars().all())

        return ReviewsListResponse(
            items=reviews,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def get_review(self, review_id: str, user_id: int) -> RawReview:
        result = await self.db.execute(
            select(RawReview)
            .join(Product)
            .where(RawReview.review_id == review_id, Product.user_id == user_id)
        )
        review = result.scalar_one_or_none()
        if not review:
            raise HTTPException(status_code=404, detail="Отзыв не найден")
        return review

    async def get_dashboard_data(self, product_id: int, user_id: int) -> dict:
        await self._check_access(product_id, user_id)

        stats = await self.db.execute(
            select(
                func.count(RawReview.review_id).label("total"),
                func.avg(RawReview.rating).label("avg_rating"),
            ).where(RawReview.product_id == product_id)
        )
        row = stats.one()

        return {
            "product_id": product_id,
            "total_reviews": row.total or 0,
            "avg_rating": float(row.avg_rating) if row.avg_rating else None,
        }

    async def _check_access(self, product_id: int, user_id: int) -> Product:
        result = await self.db.execute(
            select(Product).where(Product.product_id == product_id, Product.user_id == user_id)
        )
        product = result.scalar_one_or_none()
        if not product:
            raise HTTPException(status_code=404, detail="Товар не найден")
        return product
