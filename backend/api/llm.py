from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_id
from models.schemas import (
    ConversationCreate, ConversationResponse,
    MessageCreate, MessageResponse,
)
from services.llm_service import LLMService

router = APIRouter(prefix="/llm", tags=["LLM Чат"])


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    data: ConversationCreate,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = LLMService(db)
    return await svc.create_conversation(user_id, data.product_id)


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = LLMService(db)
    return await svc.list_conversations(user_id)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = LLMService(db)
    return await svc.get_conversation_messages(conversation_id, user_id)


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, status_code=201)
async def send_message(
    conversation_id: int,
    data: MessageCreate,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    svc = LLMService(db)
    return await svc.send_message(conversation_id, user_id, data)
