import json
import logging
from typing import Any

import redis.asyncio as aioredis

import sys
sys.path.insert(0, "/app")
from infrastructure.config import (
    REDIS_URL, REDIS_DASHBOARD_TTL, REDIS_PRODUCT_STATUS_TTL, REDIS_LOCK_TTL
)

logger = logging.getLogger(__name__)


class RedisPool:
    def __init__(self):
        self._client: aioredis.Redis | None = None

    async def connect(self):
        self._client = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        await self._client.ping()
        logger.info("Redis подключён")

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("Redis не инициализирован")
        return self._client

    async def get_dashboard(self, product_id: int) -> dict | None:
        raw = await self._client.get(f"cache:dashboard:{product_id}")
        if raw:
            return json.loads(raw)
        return None

    async def set_dashboard(self, product_id: int, data: dict):
        await self._client.setex(
            f"cache:dashboard:{product_id}",
            REDIS_DASHBOARD_TTL,
            json.dumps(data, ensure_ascii=False),
        )

    async def invalidate_dashboard(self, product_id: int):
        await self._client.delete(f"cache:dashboard:{product_id}")

    async def get_product_status(self, product_id: int) -> str | None:
        return await self._client.get(f"cache:product:{product_id}:status")

    async def set_product_status(self, product_id: int, status: str):
        await self._client.setex(
            f"cache:product:{product_id}:status",
            REDIS_PRODUCT_STATUS_TTL,
            status,
        )

    async def acquire_clustering_lock(self, product_id: int) -> bool:
        result = await self._client.set(
            f"lock:clustering:{product_id}",
            "1",
            ex=REDIS_LOCK_TTL,
            nx=True,
        )
        return result is True

    async def release_clustering_lock(self, product_id: int):
        await self._client.delete(f"lock:clustering:{product_id}")

    async def cache_generic(self, key: str, value: Any, ttl: int = 300):
        await self._client.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    async def get_generic(self, key: str) -> Any | None:
        raw = await self._client.get(key)
        return json.loads(raw) if raw else None


redis_pool = RedisPool()


async def get_redis() -> RedisPool:
    return redis_pool
