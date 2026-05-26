from pydantic import BaseModel

from app.domains.diagnoses.schemas import DiagnosisResponse


class InterruptInfo(BaseModel):
    """Payload de um interrupt pendente — exposto em ChatResponse e GET /chat/interrupts."""

    kind: str
    question: str
    response_kind: str
    options: list[str] | None = None
    asked_at: str | None = None


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    diagnosis: DiagnosisResponse | None = None
    session_id: str | None = None
    # TCC-081: quando o turno veio de áudio, devolve o texto transcrito pra UI
    # exibir o que foi falado.
    transcript: str | None = None
    # Sprint A4.5: quando o agente pausa via ask_user, este campo carrega
    # o payload do interrupt e o cliente deve renderizar o dialog +
    # disparar POST /chat/resume com a resposta.
    interrupt: InterruptInfo | None = None


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


class ResumeRequest(BaseModel):
    """Payload do POST /chat/resume — retoma uma sessao interrompida.

    O ``response`` eh repassado para ``Command(resume=response)`` no grafo,
    onde a tool ``ask_user`` o devolve para o LLM.
    """

    thread_id: str
    response: str


class PendingInterrupt(BaseModel):
    """Item do GET /chat/interrupts — uma sessao com interrupt aguardando resposta."""

    session_id: str
    interrupt: InterruptInfo
    created_at: str | None = None
