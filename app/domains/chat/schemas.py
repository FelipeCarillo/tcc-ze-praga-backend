from pydantic import BaseModel

from app.domains.diagnoses.schemas import DiagnosisResponse


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    diagnosis: DiagnosisResponse | None = None
    session_id: str | None = None
