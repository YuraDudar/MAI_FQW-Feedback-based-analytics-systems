from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from models.db_models import Product, DataSource, PlatformType
from models.schemas import ProductCreate


class ProductService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: ProductCreate, user_id: int) -> Product:
        source_result = await self.db.execute(
            select(DataSource).where(DataSource.platform == data.platform)
        )
        source = source_result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=400, detail=f"Источник для платформы {data.platform} не найден")

        existing = await self.db.execute(
            select(Product).where(
                Product.source_id == source.source_id,
                Product.source_product_id == data.source_product_id,
                Product.user_id == user_id,
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Товар уже добавлен")

        product = Product(
            name=data.name,
            source_product_id=data.source_product_id,
            source_id=source.source_id,
            user_id=user_id,
        )
        self.db.add(product)
        await self.db.flush()
        await self.db.refresh(product)
        return product

    async def list_products(self, user_id: int) -> list[Product]:
        result = await self.db.execute(
            select(Product).where(Product.user_id == user_id).order_by(Product.created_at.desc())
        )
        return list(result.scalars().all())

    async def get(self, product_id: int, user_id: int) -> Product:
        result = await self.db.execute(
            select(Product).where(Product.product_id == product_id, Product.user_id == user_id)
        )
        product = result.scalar_one_or_none()
        if not product:
            raise HTTPException(status_code=404, detail="Товар не найден")
        return product

    async def delete(self, product_id: int, user_id: int):
        product = await self.get(product_id, user_id)
        await self.db.delete(product)
