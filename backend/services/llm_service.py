import logging

import httpx
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.db_models import LLMConversation, LLMMessage, Product
from models.schemas import RAGQueryRequest, RAGResponse, MessageCreate

import sys
sys.path.insert(0, "/app")
from infrastructure.config import ML_SERVICE_INTERNAL_URL

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(self, user_id: int, product_id: int | None = None) -> LLMConversation:
        conversation = LLMConversation(user_id=user_id, product_id=product_id)
        self.db.add(conversation)
        await self.db.flush()
        await self.db.refresh(conversation)
        return conversation

    async def list_conversations(self, user_id: int) -> list[LLMConversation]:
        result = await self.db.execute(
            select(LLMConversation)
            .where(LLMConversation.user_id == user_id)
            .order_by(LLMConversation.created_at.desc())
            .limit(50)
        )
        return list(result.scalars().all())

    async def get_conversation_messages(self, conversation_id: int, user_id: int) -> list[LLMMessage]:
        conv = await self._get_conversation(conversation_id, user_id)
        result = await self.db.execute(
            select(LLMMessage)
            .where(LLMMessage.conversation_id == conversation_id)
            .order_by(LLMMessage.created_at.asc())
        )
        return list(result.scalars().all())

    async def send_message(
        self,
        conversation_id: int,
        user_id: int,
        data: MessageCreate,
    ) -> LLMMessage:
        conv = await self._get_conversation(conversation_id, user_id)
        if not conv.product_id:
            raise HTTPException(status_code=400, detail="Для чата необходимо указать товар")

        user_msg = LLMMessage(
            conversation_id=conversation_id,
            role="user",
            content=data.content,
            filters_applied=data.filters.model_dump() if data.filters else None,
        )
        self.db.add(user_msg)
        await self.db.flush()

        rag_result = await self._forward_rag(
            query=data.content,
            product_id=conv.product_id,
            top_k=data.top_k,
            filters=data.filters.model_dump() if data.filters else None,
        )

        assistant_msg = LLMMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=rag_result["answer"],
            rag_review_ids=rag_result.get("sources", []),
            filters_applied=data.filters.model_dump() if data.filters else None,
        )
        self.db.add(assistant_msg)
        await self.db.flush()
        await self.db.refresh(assistant_msg)
        return assistant_msg

    async def _forward_rag(
        self,
        query: str,
        product_id: int,
        top_k: int,
        filters: dict | None,
    ) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    f"{ML_SERVICE_INTERNAL_URL}/internal/rag/query",
                    json={
                        "query": query,
                        "product_id": product_id,
                        "top_k": top_k,
                        "filters": filters,
                    },
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                logger.error("Ошибка RAG запроса к ml_service: %s", exc)
                raise HTTPException(status_code=502, detail="Сервис ML недоступен")

    async def _get_conversation(self, conversation_id: int, user_id: int) -> LLMConversation:
        result = await self.db.execute(
            select(LLMConversation).where(
                LLMConversation.conversation_id == conversation_id,
                LLMConversation.user_id == user_id,
            )
        )
        conv = result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Беседа не найдена")
        return conv
