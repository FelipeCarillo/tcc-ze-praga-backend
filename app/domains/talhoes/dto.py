from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class TalhaoDTO:
    id: str
    user_id: str
    nome: str
    apelido: str | None
    hectares: float | None
    cultura: str
    data_semeadura: date | None
    created_at: datetime
