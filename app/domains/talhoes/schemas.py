from datetime import date, datetime

from pydantic import BaseModel, Field


class CreateTalhaoRequest(BaseModel):
    nome: str = Field(min_length=1, max_length=120)
    apelido: str | None = Field(default=None, max_length=120)
    hectares: float | None = Field(default=None, ge=0)
    cultura: str = Field(default="soja", max_length=60)
    data_semeadura: date | None = None


class TalhaoResponse(BaseModel):
    id: str
    nome: str
    apelido: str | None
    hectares: float | None
    cultura: str
    data_semeadura: date | None
    created_at: datetime
