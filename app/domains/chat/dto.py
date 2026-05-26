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
