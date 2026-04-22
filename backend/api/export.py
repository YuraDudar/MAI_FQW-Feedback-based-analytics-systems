import csv
import io
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from models.db_models import RawReview, Product

router = APIRouter(prefix="/export", tags=["Экспорт"])


@router.get("/reviews/{product_id}/csv")
async def export_reviews_csv(
    product_id: int,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    rating_min: int | None = Query(default=None, ge=1, le=5),
    rating_max: int | None = Query(default=None, ge=1, le=5),
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    product = await db.execute(
        select(Product).where(Product.product_id == product_id, Product.user_id == user_id)
    )
    product = product.scalar_one_or_none()
    if not product:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Товар не найден")

    filters = [RawReview.product_id == product_id]
    if date_from:
        filters.append(RawReview.created_date >= date_from)
    if date_to:
        filters.append(RawReview.created_date <= date_to)
    if rating_min:
        filters.append(RawReview.rating >= rating_min)
    if rating_max:
        filters.append(RawReview.rating <= rating_max)

    result = await db.execute(
        select(RawReview).where(and_(*filters)).order_by(RawReview.created_date.desc())
    )
    reviews = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "review_id", "rating", "advantages", "disadvantages", "comment",
        "reviewer_name", "reviewer_country", "created_date",
        "votes_plus", "votes_minus", "has_video", "excluded_from_rating",
    ])
    for r in reviews:
        writer.writerow([
            r.review_id, r.rating, r.advantages, r.disadvantages, r.comment,
            r.reviewer_name, r.reviewer_country, r.created_date,
            r.votes_plus, r.votes_minus, r.has_video, r.excluded_from_rating,
        ])

    output.seek(0)
    filename = f"reviews_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
