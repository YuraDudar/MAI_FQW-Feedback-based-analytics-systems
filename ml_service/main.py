import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

sys.path.insert(0, "/app")

from infrastructure.config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
from core.database import engine, Base
from core.kafka_client import kafka_consumer_manager
from core.model_registry import model_registry
from api import rag as rag_api, internal

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск ML сервиса...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await model_registry.load_models()
    await kafka_consumer_manager.start()
    logger.info("ML сервис готов.")
    yield
    logger.info("Завершение работы ML сервиса...")
    await kafka_consumer_manager.stop()
    await engine.dispose()


app = FastAPI(
    title="ML Service",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_api.router, prefix="/api/v1")
app.include_router(internal.router, prefix="/internal")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ml_service"}
