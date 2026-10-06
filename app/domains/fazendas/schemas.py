from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.domains.talhoes.schemas import TalhaoResponse


class _FazendaCampos(BaseModel):
    municipio: str | None = Field(default=None, max_length=120)
    uf: str | None = Field(default=None, min_length=2, max_length=2)
    hectares: float | None = Field(default=None, ge=0)
    agronomo_nome: str | None = Field(default=None, max_length=120)
    agronomo_crea: str | None = Field(default=None, max_length=40)

    @field_validator("uf")
    @classmethod
    def _uf_maiuscula(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class CreateFazendaRequest(_FazendaCampos):
    nome: str = Field(min_length=1, max_length=120)


class UpdateFazendaRequest(_FazendaCampos):
    """PATCH parcial: só os campos enviados mudam."""

    nome: str | None = Field(default=None, min_length=1, max_length=120)


class FazendaResponse(BaseModel):
    id: str
    nome: str
    municipio: str | None
    uf: str | None
    hectares: float | None
    agronomo_nome: str | None
    agronomo_crea: str | None
    created_at: datetime
    talhoes: list[TalhaoResponse] = []
