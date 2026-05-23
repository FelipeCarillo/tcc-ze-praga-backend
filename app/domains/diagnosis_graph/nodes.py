"""Nodes do sub-grafo de diagnostico (TCC-040).

Cada node recebe ``state: DiagnosisState`` + services por keyword-only injetados
via ``functools.partial`` em ``graph.build_diagnosis_graph``. Isso mantem o
modulo testavel sem container DI e desacoplado do FastAPI Depends.

Em Sprint A2 o ``run_inference_node`` ainda usa o mock de ``InferenceService.predict``
(synchronous random); quando o modelo real chegar, basta trocar a chamada por
``await inference_svc.predict_batch_async(...)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.domains.chat.memory import index_diagnosis_in_store
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisSourceSchema,
    Top3PredictionSchema,
)
from app.domains.diagnosis_graph.state import DiagnosisState
from app.shared.enums import SeverityEnum

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from app.domains.action_plans.service import ActionPlanService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService

logger = logging.getLogger(__name__)


async def load_model_node(
    state: DiagnosisState,
    *,
    inference_svc: InferenceService,
) -> dict:
    """Placeholder pra inicializacao do modelo + validacao do crop_id.

    Em Sprint A2 mantemos no-op: o ``InferenceService`` ja foi instanciado
    com o catalogo certo via factory (DI). Quando o modelo real ONNX chegar,
    aqui sera o ponto onde carregamos pesos por crop_id se nao tiverem no
    cache de processo.
    """
    crop_id = state.get("crop_id")
    if not crop_id:
        # Defensive — sem crop nao tem sentido tentar inferir.
        return {"errors": [{"node": "load_model", "message": "crop_id ausente"}]}
    return {}


async def run_inference_node(
    state: DiagnosisState,
    *,
    inference_svc: InferenceService,
) -> dict:
    """Roda predict pra cada imagem do batch e populates ``predictions``.

    Em Sprint A2 ``InferenceService.predict`` continua sincrono (mock random);
    a chamada nao bloqueia significativamente. Quando o modelo real entrar,
    o service deve expor ``predict_batch_async`` e este node fara await
    em uma chamada batch.
    """
    predictions: list[dict] = []
    for image_id in state.get("image_ids", []):
        result = inference_svc.predict(
            model_id=state.get("model_id", "ensemble"),
            image_name=image_id,
            crop_id=state.get("crop_id"),
        )
        predictions.append(
            {
                "disease_id": result.disease_id,
                "disease_name": result.disease_name,
                "scientific_name": result.scientific_name,
                "severity": str(result.severity),
                "description": result.description,
                "confidence": result.confidence,
                "top3": [
                    {
                        "rank": p.rank,
                        "disease_id": p.disease_id,
                        "disease_name": p.disease_name,
                        "scientific_name": p.scientific_name,
                        "confidence": p.confidence,
                        "severity": p.severity,
                    }
                    for p in result.top3
                ],
            }
        )
    return {"predictions": predictions}


async def compose_action_plan_node(
    state: DiagnosisState,
    *,
    action_plan_svc: ActionPlanService,
) -> dict:
    """Busca plano de acao por disease detectado.

    Em Sprint A2 retorna todos os niveis. A filtragem por ``preferred_action_level``
    (vinda do ChatState) entra em Sprint A3 quando PlanFeatures existir.
    """
    plans: list[dict] = []
    for pred in state.get("predictions", []):
        disease_id = pred["disease_id"]
        try:
            plan = await action_plan_svc.get_by_disease(disease_id)
            plans.append(
                {
                    "disease_id": disease_id,
                    "levels": [
                        {"level": str(lvl.level), "actions": list(lvl.actions)}
                        for lvl in plan.levels
                    ],
                    "sources": [
                        {"name": s.name, "url": s.url} for s in plan.sources
                    ],
                }
            )
        except Exception:  # noqa: BLE001 — node nao deve quebrar o grafo
            plans.append(
                {"disease_id": disease_id, "levels": [], "sources": []}
            )
    return {"action_plans": plans}


async def persist_node(
    state: DiagnosisState,
    *,
    diagnosis_svc: DiagnosisService,
    store: BaseStore | None = None,
) -> dict:
    """Cria 1 row em ``diagnoses`` por imagem do batch e indexa no Store.

    Args:
        state: estado do sub-grafo (predictions populadas pelo node anterior).
        diagnosis_svc: service de persistencia.
        store: opcional — quando passado, cada diagnostico criado eh
            tambem indexado em ``("user", uid, "diagnoses")`` pra busca
            semantica futura via ``search_my_diagnoses``. Quando ``None``,
            so persiste no DB (usado em testes / smoke).
    """
    persisted: list[str] = []
    image_ids = state.get("image_ids", [])
    predictions = state.get("predictions", [])
    model_id = state.get("model_id", "ensemble")
    user_id = state.get("user_id", "")
    # TCC-056: evidence_per_image vem do gather_evidence_node (paralelo).
    # Index alinhado com predictions — quando ausente, persistimos lista vazia.
    evidence_per_image = state.get("evidence_per_image") or []

    for i, pred in enumerate(predictions):
        image_name = image_ids[i] if i < len(image_ids) else None
        raw_sources = (
            evidence_per_image[i]
            if i < len(evidence_per_image)
            else []
        )
        sources = _build_diagnosis_sources(raw_sources)
        body = CreateDiagnosisRequest(
            disease_name=pred["disease_name"],
            disease_id=pred["disease_id"],
            scientific_name=pred.get("scientific_name"),
            confidence=pred["confidence"],
            severity=SeverityEnum(pred["severity"]),
            description=pred.get("description"),
            model_used=model_id,
            image_url=None,
            image_name=image_name,
            top3=[
                Top3PredictionSchema(
                    rank=t["rank"],
                    disease_name=t["disease_name"],
                    disease_id=t["disease_id"],
                    scientific_name=t.get("scientific_name"),
                    confidence=t["confidence"],
                    severity=t.get("severity"),
                )
                for t in pred.get("top3", [])
            ],
            sources=sources,
        )
        diag = await diagnosis_svc.create(user_id, body)
        persisted.append(diag.id)

        # Indexa no Store (best-effort — falhas nao quebram o grafo).
        if store is not None and user_id:
            try:
                await index_diagnosis_in_store(store, user_id, diag)
            except Exception:  # noqa: BLE001 — Store offline nao bloqueia diagnostico
                logger.exception(
                    "Failed to index diagnosis %s in Store", diag.id
                )

    return {"persisted_ids": persisted}


def _build_diagnosis_sources(
    raw_sources: list[dict],
) -> list[DiagnosisSourceSchema]:
    """Converte evidencia bruta (gather_evidence) em DiagnosisSourceSchema.

    Heuristica de classificacao de ``type``:
    - Possui ``doi`` -> ``scientific`` (vem do SciELO).
    - Senao -> ``web`` (vem do Tavily).

    Tolera campos faltantes — items sem ``url`` ou ``title`` sao mantidos
    com string vazia. Items que nao sao dict sao filtrados.
    """
    out: list[DiagnosisSourceSchema] = []
    for raw in raw_sources or []:
        if not isinstance(raw, dict):
            continue
        has_doi = bool(raw.get("doi"))
        # Abstract eh o campo do SciELO; snippet do Tavily.
        snippet = raw.get("snippet") or raw.get("abstract") or None
        out.append(
            DiagnosisSourceSchema(
                type="scientific" if has_doi or "abstract" in raw else "web",
                url=raw.get("url", "") or "",
                title=raw.get("title", "") or "",
                snippet=snippet,
                doi=raw.get("doi") or None,
            )
        )
    return out
