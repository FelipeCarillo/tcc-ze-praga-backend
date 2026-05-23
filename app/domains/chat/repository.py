"""Repositórios pra ChatSession + ChatMessage.

Pattern alinhado com diagnoses/repository.py — async, retorna DTOs, commit
explícito em cada operação de escrita.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.chat.dto import ChatMessageDTO, ChatSessionDTO
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession


class ChatSessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, user_id: str, title: str | None = None) -> ChatSessionDTO:
        session = ChatSession(user_id=user_id, title=title)
        self._db.add(session)
        await self._db.commit()
        await self._db.refresh(session)
        return self._to_dto(session)

    async def get_by_id(self, session_id: str, user_id: str | None = None) -> ChatSessionDTO | None:
        query = select(ChatSession).where(ChatSession.id == session_id)
        if user_id is not None:
            query = query.where(ChatSession.user_id == user_id)
        result = await self._db.execute(query)
        session = result.scalar_one_or_none()
        return self._to_dto(session) if session else None

    async def get_or_create_for_user(
        self, user_id: str, session_id: str | None = None
    ) -> ChatSessionDTO:
        """Se session_id passado e pertence ao user, retorna. Senão cria nova."""
        if session_id is not None:
            existing = await self.get_by_id(session_id, user_id=user_id)
            if existing is not None:
                return existing
        return await self.create(user_id)

    async def list_for_user(self, user_id: str) -> list[ChatSessionDTO]:
        """Lista todas as sessoes do usuario ordenadas por updated_at desc."""
        result = await self._db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        )
        return [self._to_dto(s) for s in result.scalars().all()]

    async def update_summary(
        self, session_id: str, user_id: str, summary_text: str
    ) -> ChatSessionDTO | None:
        """Atualiza ``summary_text`` da sessao se pertencer ao usuario.

        Retorna o DTO atualizado ou ``None`` se a sessao nao for encontrada.
        """
        result = await self._db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            return None
        session.summary_text = summary_text
        await self._db.commit()
        await self._db.refresh(session)
        return self._to_dto(session)

    @staticmethod
    def _to_dto(s: ChatSession) -> ChatSessionDTO:
        return ChatSessionDTO(
            id=s.id,
            user_id=s.user_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            summary_text=s.summary_text,
        )


class ChatMessageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        session_id: str,
        role: str,
        content: str,
        diagnosis_id: str | None = None,
        metadata: dict | None = None,
    ) -> ChatMessageDTO:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            diagnosis_id=diagnosis_id,
            metadata_=metadata,
        )
        self._db.add(message)
        await self._db.commit()
        await self._db.refresh(message)
        return self._to_dto(message)

    async def list_by_session(self, session_id: str) -> list[ChatMessageDTO]:
        result = await self._db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        return [self._to_dto(m) for m in result.scalars().all()]

    @staticmethod
    def _to_dto(m: ChatMessage) -> ChatMessageDTO:
        return ChatMessageDTO(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            diagnosis_id=m.diagnosis_id,
            created_at=m.created_at,
            metadata=m.metadata_,
        )
