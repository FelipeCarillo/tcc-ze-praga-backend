import logging
from collections.abc import Callable
from typing import Any, Literal

from app.core.exceptions import ForbiddenError, NotFoundError
from app.domains.diagnoses.dto import DiagnosisDTO, TalhaoGroupDTO
from app.domains.diagnoses.repository import DiagnosisRepository
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisFilters,
    DiagnosisResponse,
    DiagnosisSourceSchema,
    TalhaoGroupResponse,
    Top3PredictionSchema,
)
from app.shared.pagination import PaginatedResponse

logger = logging.getLogger(__name__)

# Quantos laudos recentes cada grupo do historico por talhao traz (TCC-093).
RECENT_PER_TALHAO = 3


class DiagnosisService:
    """Leitura/escrita de diagnosticos.

    ``image_url`` e' persistido como **storage key** do bucket, nao como URL:
    URL assinada expira, e guardar uma URL morta no banco seria pior que nao
    guardar nada. A conversao pra URL acontece na leitura, via
    ``image_url_resolver`` — em lote, um unico round-trip ao storage por
    resposta, inclusive na listagem paginada do historico.
    """

    def __init__(
        self,
        repo: DiagnosisRepository,
        image_url_resolver: Callable[[list[str]], dict[str, str]] | None = None,
    ) -> None:
        self._repo = repo
        self._resolve_image_urls = image_url_resolver

    async def create(
        self, user_id: str, request: CreateDiagnosisRequest, *, crop_id: str
    ) -> DiagnosisResponse:
        # TCC-093: todos os caminhos de criacao (chat, sub-grafo, REST) passam
        # por aqui. Talhao alheio ou apagado vira "Sem talhao" em vez de
        # derrubar o diagnostico, que e' o que o usuario veio buscar.
        if request.talhao_id and not await self._repo.talhao_belongs_to_user(
            request.talhao_id, user_id
        ):
            logger.warning(
                "talhao_id %s nao pertence ao usuario %s; laudo fica sem talhao",
                request.talhao_id,
                user_id,
            )
            request = request.model_copy(update={"talhao_id": None})
        diagnosis = await self._repo.create(user_id, request, crop_id=crop_id)
        return self._to_response(diagnosis, self._resolve_for([diagnosis]))

    async def get_by_id(self, diagnosis_id: str, user_id: str) -> DiagnosisResponse:
        diagnosis = await self._repo.find_by_id(diagnosis_id, user_id)
        if not diagnosis:
            raise NotFoundError("Diagnosis", diagnosis_id)
        if diagnosis.user_id != user_id:
            raise ForbiddenError()
        return self._to_response(diagnosis, self._resolve_for([diagnosis]))

    async def list_for_user(
        self, user_id: str, filters: DiagnosisFilters
    ) -> PaginatedResponse[DiagnosisResponse]:
        items, total = await self._repo.find_all_by_user(user_id, filters)
        urls = self._resolve_for(items)
        return PaginatedResponse(
            items=[self._to_response(d, urls) for d in items],
            total=total,
            page=filters.page,
            limit=filters.limit,
        )

    async def set_talhao(
        self, diagnosis_id: str, user_id: str, talhao_id: str | None
    ) -> DiagnosisResponse:
        """Move o laudo para outro talhao (None = "Sem talhao")."""
        if talhao_id is not None and not await self._repo.talhao_belongs_to_user(
            talhao_id, user_id
        ):
            raise NotFoundError("Talhao", talhao_id)
        diagnosis = await self._repo.set_talhao(diagnosis_id, user_id, talhao_id)
        if not diagnosis:
            raise NotFoundError("Diagnosis", diagnosis_id)
        return self._to_response(diagnosis, self._resolve_for([diagnosis]))

    async def group_by_talhao(
        self, user_id: str, per_group: int = RECENT_PER_TALHAO
    ) -> list[TalhaoGroupResponse]:
        """Historico agrupado por talhao, com os laudos recentes de cada um."""
        groups = await self._repo.group_by_talhao(user_id, per_group)
        # Uma assinatura de URLs so' para todas as miniaturas da tela.
        urls = self._resolve_for([d for g in groups for d in g.recent])
        return [self._to_group_response(g, urls) for g in groups]

    async def delete(self, diagnosis_id: str, user_id: str) -> None:
        found = await self._repo.delete(diagnosis_id, user_id)
        if not found:
            raise NotFoundError("Diagnosis", diagnosis_id)

    async def clear_all(self, user_id: str) -> int:
        return await self._repo.delete_all_by_user(user_id)

    def _resolve_for(self, items: list[DiagnosisDTO]) -> dict[str, str]:
        """Assina de uma vez as chaves de imagem de um conjunto de diagnosticos."""
        if self._resolve_image_urls is None:
            return {}
        keys = [
            d.image_url
            for d in items
            # Diagnosticos antigos (pre-upload) tem image_url None; se algum dia
            # alguem gravar uma URL completa ali, passa direto sem assinar.
            if d.image_url and not d.image_url.startswith("http")
        ]
        if not keys:
            return {}
        return self._resolve_image_urls(keys)

    @classmethod
    def _to_group_response(
        cls, g: TalhaoGroupDTO, image_urls: dict[str, str]
    ) -> TalhaoGroupResponse:
        return TalhaoGroupResponse(
            talhao_id=g.talhao_id,
            talhao_nome=g.talhao_nome,
            total=g.total,
            last_at=g.last_at,
            # ``recent`` vem do mais novo pro mais antigo; a tendencia le ao
            # contrario, como uma linha do tempo.
            severity_trend=[d.severity for d in reversed(g.recent)],
            recent=[cls._to_response(d, image_urls) for d in g.recent],
        )

    @staticmethod
    def _to_response(
        d: DiagnosisDTO, image_urls: dict[str, str] | None = None
    ) -> DiagnosisResponse:
        raw_image = d.image_url
        image_url = (image_urls or {}).get(raw_image or "", raw_image)
        return DiagnosisResponse(
            id=d.id,
            disease_name=d.disease_name,
            disease_id=d.disease_id,
            scientific_name=d.scientific_name,
            confidence=d.confidence,
            severity=d.severity,
            description=d.description,
            model_used=d.model_used,
            image_url=image_url,
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
            talhao_id=d.talhao_id,
            talhao_nome=d.talhao_nome,
        )


def _safe_source(raw: dict[str, Any]) -> DiagnosisSourceSchema:
    """Constroi DiagnosisSourceSchema tolerando campos extras/faltantes.

    JSONB do DB pode ter shape ligeiramente diferente do schema (versoes
    antigas, dados injetados manualmente). Faz fallback pra ``type=web`` e
    string vazia em campos obrigatorios ausentes.
    """
    raw_type = raw.get("type")
    src_type: Literal["web", "scientific"] = (
        raw_type if raw_type in {"web", "scientific"} else "web"
    )
    return DiagnosisSourceSchema(
        type=src_type,
        url=raw.get("url", "") or "",
        title=raw.get("title", "") or "",
        snippet=raw.get("snippet"),
        doi=raw.get("doi"),
    )
