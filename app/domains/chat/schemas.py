from pydantic import BaseModel

from app.domains.diagnoses.schemas import DiagnosisResponse


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    diagnosis: DiagnosisResponse | None = None
    session_id: str | None = None


class CloseSessionResponse(BaseModel):
    session_id: str
    summary_text: str | None = None


class SemanticDiagnosisHit(BaseModel):
    """Item retornado pelo endpoint de busca semantica de diagnoses."""

    summary_text: str
    diagnosis_id: str
    disease_id: str | None = None
    disease_name: str | None = None
    crop_id: str | None = None
    confidence: float | None = None
    severity: str | None = None
    created_at: str | None = None
