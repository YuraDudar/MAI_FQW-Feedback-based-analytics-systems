from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.rag_service import rag_service

router = APIRouter(tags=["Internal"])


class InternalRAGRequest(BaseModel):
    query: str
    product_id: int
    top_k: int = 10
    filters: dict | None = None


@router.post("/rag/query")
async def internal_rag_query(data: InternalRAGRequest):
    return await rag_service.query(
        query=data.query,
        product_id=data.product_id,
        top_k=data.top_k,
        filters=data.filters,
    )


@router.get("/clusters/{product_id}")
async def get_clusters(product_id: int):
    from core.database import AsyncSessionLocal
    from models.db_models import Cluster, ReviewClusterMapping
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Cluster).where(Cluster.product_id == product_id)
        )
        clusters = result.scalars().all()
        return [
            {
                "cluster_id": c.cluster_id,
                "sentiment_category": c.sentiment_category,
                "bertopic_topic_id": c.bertopic_topic_id,
                "llm_label": c.llm_label,
                "keywords": c.keywords,
                "review_count": c.review_count,
                "avg_rating": c.avg_rating,
            }
            for c in clusters
        ]


@router.get("/sentiments/{product_id}")
async def get_sentiments(product_id: int, limit: int = 1000):
    from core.database import AsyncSessionLocal
    from models.db_models import ReviewSentiment
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ReviewSentiment)
            .where(ReviewSentiment.product_id == product_id)
            .limit(limit)
        )
        sentiments = result.scalars().all()
        return [
            {
                "review_id": s.review_id,
                "sentiment_label": s.sentiment_label.value,
                "sentiment_score": s.sentiment_score,
                "reviewer_gender": s.reviewer_gender,
            }
            for s in sentiments
        ]
