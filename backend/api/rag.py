from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from models.schemas import RAGQueryRequest, RAGResponse
from services.llm_service import LLMService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=RAGResponse)
async def rag_query(
    data: RAGQueryRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = LLMService(db)
    result = await svc._forward_rag(
        query=data.query,
        product_id=data.product_id,
        top_k=data.top_k,
        filters=data.filters.model_dump() if data.filters else None,
    )
    return result
