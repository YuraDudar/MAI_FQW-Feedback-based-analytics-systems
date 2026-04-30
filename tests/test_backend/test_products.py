import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.services.product_service import ProductService
from backend.services.auth_service import AuthService
from backend.models.schemas import UserCreate, ProductCreate


async def _create_test_user(db: AsyncSession, suffix: str = "prod") -> int:
    svc = AuthService(db)
    user = await svc.register(UserCreate(
        username=f"user_{suffix}",
        email=f"user_{suffix}@test.com",
        password="password123",
    ))
    return user.user_id


async def _ensure_data_source(db: AsyncSession):
    await db.execute(text("""
        INSERT OR IGNORE INTO data_sources (source_id, name, platform, site_url)
        VALUES (1, 'Wildberries', 'wildberries', 'https://www.wildberries.ru'),
               (2, 'Ozon', 'ozon', 'https://www.ozon.ru')
    """))
    await db.flush()


@pytest.mark.asyncio
async def test_create_product(db_session: AsyncSession):
    await _ensure_data_source(db_session)
    user_id = await _create_test_user(db_session, "p1")
    svc = ProductService(db_session)
    product = await svc.create(
        ProductCreate(name="Тестовый товар", source_product_id="12345678", platform="wildberries"),
        user_id,
    )
    assert product.product_id is not None
    assert product.name == "Тестовый товар"
    assert product.source_product_id == "12345678"


@pytest.mark.asyncio
async def test_create_duplicate_product_raises(db_session: AsyncSession):
    await _ensure_data_source(db_session)
    user_id = await _create_test_user(db_session, "p2")
    svc = ProductService(db_session)
    await svc.create(
        ProductCreate(name="Товар A", source_product_id="99999", platform="wildberries"),
        user_id,
    )
    with pytest.raises(HTTPException) as exc:
        await svc.create(
            ProductCreate(name="Товар A Дубль", source_product_id="99999", platform="wildberries"),
            user_id,
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_list_products(db_session: AsyncSession):
    await _ensure_data_source(db_session)
    user_id = await _create_test_user(db_session, "p3")
    svc = ProductService(db_session)
    await svc.create(ProductCreate(name="Prod 1", source_product_id="111", platform="wildberries"), user_id)
    await svc.create(ProductCreate(name="Prod 2", source_product_id="222", platform="ozon"), user_id)
    products = await svc.list_products(user_id)
    assert len(products) == 2


@pytest.mark.asyncio
async def test_get_product_wrong_user_raises(db_session: AsyncSession):
    await _ensure_data_source(db_session)
    user_id = await _create_test_user(db_session, "p4")
    other_user_id = await _create_test_user(db_session, "p4b")
    svc = ProductService(db_session)
    product = await svc.create(
        ProductCreate(name="Secret Product", source_product_id="555", platform="wildberries"),
        user_id,
    )
    with pytest.raises(HTTPException) as exc:
        await svc.get(product.product_id, other_user_id)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_product(db_session: AsyncSession):
    await _ensure_data_source(db_session)
    user_id = await _create_test_user(db_session, "p5")
    svc = ProductService(db_session)
    product = await svc.create(
        ProductCreate(name="To Delete", source_product_id="777", platform="wildberries"),
        user_id,
    )
    await svc.delete(product.product_id, user_id)
    products = await svc.list_products(user_id)
    assert all(p.product_id != product.product_id for p in products)
