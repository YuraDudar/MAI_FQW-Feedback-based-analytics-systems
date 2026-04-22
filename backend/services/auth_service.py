from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.db_models import User
from models.schemas import UserCreate, TokenPair
from core.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def register(self, data: UserCreate) -> User:
        existing = await self.db.execute(
            select(User).where(
                (User.username == data.username) | (User.email == data.email)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Пользователь с таким именем или email уже существует",
            )
        user = User(
            username=data.username,
            email=data.email,
            password_hash=hash_password(data.password),
        )
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def login(self, username: str, password: str) -> TokenPair:
        result = await self.db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверное имя пользователя или пароль",
            )
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Аккаунт деактивирован")
        payload = {"sub": str(user.user_id), "role": user.role.value}
        return TokenPair(
            access_token=create_access_token(payload),
            refresh_token=create_refresh_token(payload),
        )

    async def refresh(self, refresh_token: str) -> TokenPair:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Требуется refresh-токен")
        user_id = payload.get("sub")
        result = await self.db.execute(select(User).where(User.user_id == int(user_id)))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Пользователь не найден")
        new_payload = {"sub": str(user.user_id), "role": user.role.value}
        return TokenPair(
            access_token=create_access_token(new_payload),
            refresh_token=create_refresh_token(new_payload),
        )

    async def get_user(self, user_id: int) -> User:
        result = await self.db.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        return user
