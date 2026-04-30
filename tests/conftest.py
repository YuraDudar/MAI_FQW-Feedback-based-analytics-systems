import asyncio
import sys
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("BACKEND_DB_HOST", "localhost")
os.environ.setdefault("ML_DB_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_testing_only!!")
os.environ.setdefault("YANDEX_API_KEY", "test_key")
os.environ.setdefault("YANDEX_FOLDER_ID", "test_folder")


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    from backend.core.database import Base
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def auth_headers(db_session: AsyncSession) -> dict:
    from backend.services.auth_service import AuthService
    from backend.models.schemas import UserCreate
    svc = AuthService(db_session)
    await svc.register(UserCreate(username="testuser", email="test@test.com", password="password123"))
    tokens = await svc.login("testuser", "password123")
    return {"Authorization": f"Bearer {tokens.access_token}"}
