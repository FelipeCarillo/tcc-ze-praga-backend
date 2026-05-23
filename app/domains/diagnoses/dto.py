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
