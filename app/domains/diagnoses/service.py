from app.core.exceptions import ForbiddenError, NotFoundError
from app.domains.diagnoses.dto import DiagnosisDTO
from app.domains.diagnoses.repository import DiagnosisRepository
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisFilters,
    DiagnosisResponse,
    DiagnosisSourceSchema,
    Top3PredictionSchema,
)
from app.shared.pagination import PaginatedResponse


class DiagnosisService:
    def __init__(self, repo: DiagnosisRepository) -> None:
        self._repo = repo

    async def create(self, user_id: str, request: CreateDiagnosisRequest) -> DiagnosisResponse:
        diagnosis = await self._repo.create(user_id, request)
        return self._to_response(diagnosis)

    async def get_by_id(self, diagnosis_id: str, user_id: str) -> DiagnosisResponse:
        diagnosis = await self._repo.find_by_id(diagnosis_id, user_id)
        if not diagnosis:
            raise NotFoundError("Diagnosis", diagnosis_id)
        if diagnosis.user_id != user_id:
            raise ForbiddenError()
        return self._to_response(diagnosis)

    async def list_for_user(
        self, user_id: str, filters: DiagnosisFilters
    ) -> PaginatedResponse[DiagnosisResponse]:
        items, total = await self._repo.find_all_by_user(user_id, filters)
        return PaginatedResponse(
            items=[self._to_response(d) for d in items],
            total=total,
            page=filters.page,
            limit=filters.limit,
        )

    async def delete(self, diagnosis_id: str, user_id: str) -> None:
        found = await self._repo.delete(diagnosis_id, user_id)
        if not found:
            raise NotFoundError("Diagnosis", diagnosis_id)

    async def clear_all(self, user_id: str) -> int:
        return await self._repo.delete_all_by_user(user_id)

    @staticmethod
    def _to_response(d: DiagnosisDTO) -> DiagnosisResponse:
        return DiagnosisResponse(
            id=d.id,
            disease_name=d.disease_name,
            disease_id=d.disease_id,
            scientific_name=d.scientific_name,
            confidence=d.confidence,
            severity=d.severity,
            description=d.description,
            model_used=d.model_used,
            image_url=d.image_url,
            image_name=d.image_name,
            created_at=d.created_at,
            top3=[
                Top3PredictionSchema(
                    rank=t.rank,
                    disease_name=t.disease_name,
                    disease_id=t.disease_id,
                    scientific_name=t.scientific_name,
                    confidence=t.confidence,
                    severity=t.severity,
                )
                for t in d.top3
            ],
            sources=[
                _safe_source(s) for s in d.sources if isinstance(s, dict)
            ],
        )


def _safe_source(raw: dict) -> DiagnosisSourceSchema:
    """Constroi DiagnosisSourceSchema tolerando campos extras/faltantes.

    JSONB do DB pode ter shape ligeiramente diferente do schema (versoes
    antigas, dados injetados manualmente). Faz fallback pra ``type=web`` e
    string vazia em campos obrigatorios ausentes.
    """
    src_type = raw.get("type")
    if src_type not in {"web", "scientific"}:
        src_type = "web"
    return DiagnosisSourceSchema(
        type=src_type,
        url=raw.get("url", "") or "",
        title=raw.get("title", "") or "",
        snippet=raw.get("snippet"),
        doi=raw.get("doi"),
    )
