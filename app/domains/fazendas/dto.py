from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FazendaDTO:
    id: str
    user_id: str
    nome: str
    municipio: str | None
    uf: str | None
    hectares: float | None
    agronomo_nome: str | None
    agronomo_crea: str | None
    created_at: datetime
