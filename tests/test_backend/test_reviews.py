import pytest
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.services.review_service import ReviewService
from backend.models.db_models import RawReview, PlatformType


async def _insert_product(db: AsyncSession, user_id: int, sku: str = "test_sku") -> int:
    await db.execute(text("""
        INSERT OR IGNORE INTO data_sources (source_id, name, platform, site_url)
        VALUES (1, 'Wildberries', 'wildberries', 'https://www.wildberries.ru')
    """))
    await db.execute(text("""
        INSERT INTO products (name, source_product_id, source_id, user_id)
        VALUES ('Test Product', :sku, 1, :user_id)
    """), {"sku": sku, "user_id": user_id})
    await db.flush()
    result = await db.execute(text(
        "SELECT product_id FROM products WHERE user_id = :uid ORDER BY product_id DESC LIMIT 1"
    ), {"uid": user_id})
    return result.scalar_one()


async def _insert_user(db: AsyncSession, suffix: str = "rv") -> int:
    from backend.core.security import hash_password
    await db.execute(text("""
        INSERT INTO users (username, email, password_hash, role)
        VALUES (:u, :e, :h, 'analyst')
    """), {"u": f"user_{suffix}", "e": f"{suffix}@t.com", "h": hash_password("pass")})
    await db.flush()
    result = await db.execute(text(
        "SELECT user_id FROM users WHERE username = :u"
    ), {"u": f"user_{suffix}"})
    return result.scalar_one()


async def _insert_review(db: AsyncSession, product_id: int, review_id: str, rating: int = 5):
    review = RawReview(
        review_id=review_id,
        product_id=product_id,
        platform=PlatformType.wildberries,
        rating=rating,
        advantages="Хорошо",
        comment="Отличный товар",
        created_date=datetime.now(timezone.utc),
    )
    db.add(review)
    await db.flush()


@pytest.mark.asyncio
async def test_get_reviews_pagination(db_session: AsyncSession):
    user_id = await _insert_user(db_session, "rv1")
    product_id = await _insert_product(db_session, user_id, "rv_sku1")
    for i in range(5):
        await _insert_review(db_session, product_id, f"rev_{product_id}_{i}", rating=i + 1)

    svc = ReviewService(db_session)
    result = await svc.get_reviews(product_id, user_id, page=1, page_size=3)
    assert result.total == 5
    assert len(result.items) == 3
    assert result.page == 1


@pytest.mark.asyncio
async def test_get_reviews_rating_filter(db_session: AsyncSession):
    user_id = await _insert_user(db_session, "rv2")
    product_id = await _insert_product(db_session, user_id, "rv_sku2")
    for i in range(5):
        await _insert_review(db_session, product_id, f"rfilt_{product_id}_{i}", rating=i + 1)

    svc = ReviewService(db_session)
    result = await svc.get_reviews(product_id, user_id, rating_min=4)
    assert all(r.rating >= 4 for r in result.items)


@pytest.mark.asyncio
async def test_get_review_single(db_session: AsyncSession):
    user_id = await _insert_user(db_session, "rv3")
    product_id = await _insert_product(db_session, user_id, "rv_sku3")
    await _insert_review(db_session, product_id, "unique_rev_123", rating=3)

    svc = ReviewService(db_session)
    review = await svc.get_review("unique_rev_123", user_id)
    assert review.review_id == "unique_rev_123"
    assert review.rating == 3


@pytest.mark.asyncio
async def test_get_reviews_wrong_user_raises(db_session: AsyncSession):
    from fastapi import HTTPException
    user_id = await _insert_user(db_session, "rv4")
    other_id = await _insert_user(db_session, "rv4b")
    product_id = await _insert_product(db_session, user_id, "rv_sku4")

    svc = ReviewService(db_session)
    with pytest.raises(HTTPException) as exc:
        await svc.get_reviews(product_id, other_id)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_dashboard_data(db_session: AsyncSession):
    user_id = await _insert_user(db_session, "rv5")
    product_id = await _insert_product(db_session, user_id, "rv_sku5")
    for i in range(3):
        await _insert_review(db_session, product_id, f"dash_{product_id}_{i}", rating=4)

    svc = ReviewService(db_session)
    data = await svc.get_dashboard_data(product_id, user_id)
    assert data["total_reviews"] == 3
    assert data["product_id"] == product_id
