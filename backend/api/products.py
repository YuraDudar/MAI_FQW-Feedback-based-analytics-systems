from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from models.schemas import ProductCreate, ProductResponse
from services.product_service import ProductService

router = APIRouter(prefix="/products", tags=["Товары"])


@router.post("", response_model=ProductResponse, status_code=201)
async def create_product(
    data: ProductCreate,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = ProductService(db)
    return await svc.create(data, user_id)


@router.get("", response_model=list[ProductResponse])
async def list_products(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = ProductService(db)
    return await svc.list_products(user_id)


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = ProductService(db)
    return await svc.get(product_id, user_id)


@router.delete("/{product_id}", status_code=204)
async def delete_product(
    product_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = ProductService(db)
    await svc.delete(product_id, user_id)
