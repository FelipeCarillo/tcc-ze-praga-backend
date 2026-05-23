from datetime import datetime

from pydantic import BaseModel, Field


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120, description="Nome humano pra identificar")


class ApiKeyCreatedResponse(BaseModel):
    """Retornado **uma unica vez** na criacao — inclui ``key`` em plain text.

    O cliente deve guardar; nao ha como recuperar depois (so' hash no DB).
    """

    id: str
    name: str
    key: str = Field(description="API key plain text — guarde, nao podera ser exibida novamente")
    key_prefix: str
    scopes: list[str]
    created_at: datetime


class ApiKeyResponse(BaseModel):
    """Listagem — sem plain text."""

    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
