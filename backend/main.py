import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

sys.path.insert(0, "/app")

from infrastructure.config import (
    APP_TITLE, APP_VERSION, BACKEND_CORS_ORIGINS, BACKEND_API_PREFIX,
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
)
from core.database import engine, Base
from core.redis_client import redis_pool
from core.kafka_client import kafka_producer, kafka_consumer_manager
from api import auth, products, jobs, reviews, rag, llm, admin, export

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск Backend сервиса...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await redis_pool.connect()
    await kafka_producer.start()
    await kafka_consumer_manager.start()
    logger.info("Backend сервис готов.")
    yield
    logger.info("Завершение работы Backend сервиса...")
    await kafka_consumer_manager.stop()
    await kafka_producer.stop()
    await redis_pool.disconnect()
    await engine.dispose()


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=BACKEND_API_PREFIX)
app.include_router(products.router, prefix=BACKEND_API_PREFIX)
app.include_router(jobs.router, prefix=BACKEND_API_PREFIX)
app.include_router(reviews.router, prefix=BACKEND_API_PREFIX)
app.include_router(rag.router, prefix=BACKEND_API_PREFIX)
app.include_router(llm.router, prefix=BACKEND_API_PREFIX)
app.include_router(admin.router, prefix=BACKEND_API_PREFIX)
app.include_router(export.router, prefix=BACKEND_API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend"}
