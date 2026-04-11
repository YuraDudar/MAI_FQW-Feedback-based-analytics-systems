from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

import sys
sys.path.insert(0, "/app")
from infrastructure.config import ML_DB_URL, ML_DB_POOL_SIZE, ML_DB_MAX_OVERFLOW

engine = create_async_engine(
    ML_DB_URL,
    pool_size=ML_DB_POOL_SIZE,
    max_overflow=ML_DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
