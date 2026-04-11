from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    product_id: int
    top_k: int = Field(default=10, ge=5, le=40)
    filters: dict | None = None


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    expanded_query: str
    timings: dict


@router.post("/query", response_model=RAGResponse)
async def rag_query(data: RAGRequest):
    result = await rag_service.query(
        query=data.query,
        product_id=data.product_id,
        top_k=data.top_k,
        filters=data.filters,
    )
    return result
