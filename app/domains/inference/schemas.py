from pydantic import BaseModel, Field

from app.domains.diagnoses.schemas import Top3PredictionSchema
from app.shared.enums import SeverityEnum


class InferenceResult(BaseModel):
    """Resultado bruto da inferência — antes de virar Diagnosis persistido."""

    disease_id: str
    disease_name: str
    scientific_name: str | None = None
    severity: SeverityEnum
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    model_id: str
    image_name: str
    top3: list[Top3PredictionSchema]
