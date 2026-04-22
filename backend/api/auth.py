from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from models.schemas import UserCreate, UserLogin, TokenPair, RefreshRequest, UserResponse
from services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Авторизация"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    svc = AuthService(db)
    user = await svc.register(data)
    return user


@router.post("/login", response_model=TokenPair)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    svc = AuthService(db)
    return await svc.login(data.username, data.password)


@router.post("/refresh", response_model=TokenPair)
async def refresh(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    svc = AuthService(db)
    return await svc.refresh(data.refresh_token)


@router.get("/me", response_model=UserResponse)
async def me(user_id: int = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)):
    svc = AuthService(db)
    return await svc.get_user(user_id)
