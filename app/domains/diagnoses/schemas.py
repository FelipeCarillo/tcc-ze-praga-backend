from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domains.action_plans.schemas import ActionPlanLevelResponse
from app.shared.enums import ActionPlanLevelEnum, SeverityEnum


class Top3PredictionSchema(BaseModel):
    rank: int
    disease_name: str
    disease_id: str
    scientific_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str | None = None


class DiagnosisSourceSchema(BaseModel):
    """Evidencia externa anexada ao diagnostico (TCC-056).

    Populado pelo ``gather_evidence_node`` rodando ``search_web`` (Tavily) e/ou
    ``search_scientific`` (SciELO) em paralelo ao ``compose_action_plan``.

    Args:
        type: ``"web"`` (Tavily) ou ``"scientific"`` (SciELO).
        url: link pra fonte original. Sempre presente (string vazia se ausente).
        title: titulo do artigo/pagina. String vazia se ausente.
        snippet: trecho relevante (max 500 chars). Opcional.
        doi: DOI quando aplicavel (so' ``scientific``). Opcional.
    """

    type: Literal["web", "scientific"]
    url: str
    title: str
    snippet: str | None = None
    doi: str | None = None


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
    sources: list[DiagnosisSourceSchema] = Field(default_factory=list)
    # TCC-093: talhao onde a folha foi fotografada. Talhao de outro usuario
    # (ou inexistente) e' descartado na persistencia, nunca vinculado.
    talhao_id: str | None = None


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
    sources: list[DiagnosisSourceSchema] = Field(default_factory=list)
    talhao_id: str | None = None
    talhao_nome: str | None = None


class DiagnosisActionPlanSourceSchema(BaseModel):
    """Referencia tecnica do plano de acao (Embrapa, bula, etc.)."""

    name: str
    url: str | None = None


class DiagnosisActionPlanSchema(BaseModel):
    """Plano de acao anexado a um laudo recem-analisado (TCC-105).

    ``levels`` ja vem filtrado pelo plano do usuario; ``allowed_levels`` diz
    quais niveis o plano libera, para o cliente distinguir "nivel nao existe
    para esta doenca" de "nivel existe mas o plano nao cobre".
    """

    levels: list[ActionPlanLevelResponse]
    allowed_levels: list[ActionPlanLevelEnum]
    sources: list[DiagnosisActionPlanSourceSchema] = Field(default_factory=list)


class AnalyzedDiagnosisResponse(DiagnosisResponse):
    """Item de ``POST /diagnoses/analyze``: o laudo + o plano de acao (TCC-105).

    Subclasse em vez de campo novo em ``DiagnosisResponse`` porque o plano so'
    e' calculado na analise — o historico nao o carrega. ``None`` quando a
    doenca nao tem plano cadastrado.
    """

    action_plan: DiagnosisActionPlanSchema | None = None


class SetDiagnosisTalhaoRequest(BaseModel):
    """Move um laudo para um talhao (``None`` = "Sem talhao")."""

    talhao_id: str | None = None


class TalhaoGroupResponse(BaseModel):
    """Grupo do historico por talhao (TCC-093). ``talhao_id`` None = sem talhao."""

    talhao_id: str | None
    talhao_nome: str | None
    fazenda_id: str | None = None
    fazenda_nome: str | None = None
    total: int
    last_at: datetime | None
    # Severidades do mais antigo pro mais recente entre os laudos de ``recent``
    # — a UI desenha a tendencia ("piorando") sem outra consulta.
    severity_trend: list[str]
    recent: list[DiagnosisResponse]


# Valor de ``talhao_id`` no filtro que seleciona os laudos sem talhao.
SEM_TALHAO = "sem-talhao"


class DiagnosisFilters(BaseModel):
    severity: SeverityEnum | None = None
    search: str | None = None
    # TCC-093: id de um talhao, ou ``SEM_TALHAO`` para os laudos sem vinculo.
    talhao_id: str | None = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.limit
