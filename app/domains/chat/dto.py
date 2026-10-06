from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ChatSessionDTO:
    id: str
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    summary_text: str | None = None


@dataclass(frozen=True)
class ChatMessageDTO:
    id: str
    session_id: str
    role: str
    content: str
    diagnosis_id: str | None
    created_at: datetime
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class SessionPreviewDTO:
    """Uma conversa na lista do historico (TCC-097).

    Alem da previa (primeira mensagem do usuario), traz o que o card de
    "Conversas" mostra: a ultima resposta do Ze, quantos laudos sairam dela, a
    foto e o talhao do laudo mais recente. ``image_key`` e' a storage key — a
    URL assinada e' resolvida no service.
    """

    session: ChatSessionDTO
    message_count: int
    preview: str | None
    last_reply: str | None = None
    diagnosis_count: int = 0
    image_key: str | None = None
    talhao_nome: str | None = None
