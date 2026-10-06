from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Top3PredictionDTO:
    rank: int
    disease_name: str
    disease_id: str
    scientific_name: str | None
    confidence: float
    severity: str | None


@dataclass(frozen=True)
class DiagnosisDTO:
    id: str
    user_id: str
    disease_name: str
    disease_id: str
    scientific_name: str | None
    confidence: float
    severity: str
    description: str | None
    model_used: str
    image_url: str | None
    image_name: str | None
    created_at: datetime
    top3: list[Top3PredictionDTO] = field(default_factory=list)
    # TCC-056 — evidencia externa persistida em diagnoses.sources (JSONB).
    sources: list[dict[str, Any]] = field(default_factory=list)
    # TCC-093 — talhao do laudo (None = "Sem talhao").
    talhao_id: str | None = None
    talhao_nome: str | None = None


@dataclass(frozen=True)
class TalhaoGroupDTO:
    """Um grupo do historico agrupado por talhao (TCC-093).

    ``talhao_id``/``talhao_nome`` None representam o grupo "Sem talhao".
    ``recent`` traz os laudos mais recentes do grupo (ja limitados);
    ``total`` conta todos.
    """

    talhao_id: str | None
    talhao_nome: str | None
    total: int
    last_at: datetime | None
    recent: list[DiagnosisDTO] = field(default_factory=list)
    # TCC-096: a fazenda do talhao (None no grupo "Sem talhao").
    fazenda_id: str | None = None
    fazenda_nome: str | None = None
