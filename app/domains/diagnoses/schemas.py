from datetime import datetime

from pydantic import BaseModel, Field

from app.shared.enums import SeverityEnum


class Top3PredictionSchema(BaseModel):
    rank: int
    disease_name: str
    disease_id: str
    scientific_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str | None = None


class CreateDiagnosisRequest(BaseModel):
    disease_name: str
    disease_id: str
    scientific_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    severity: SeverityEnum
    description: str | None = None
    model_used: str
    image_url: str | None = None
    image_name: str | None = None
    top3: list[Top3PredictionSchema] = Field(default_factory=list)


class DiagnosisResponse(BaseModel):
    id: str
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
    top3: list[Top3PredictionSchema]


class DiagnosisFilters(BaseModel):
    severity: SeverityEnum | None = None
    search: str | None = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit
