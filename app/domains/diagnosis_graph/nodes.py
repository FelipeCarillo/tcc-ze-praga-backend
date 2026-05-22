"""Nodes do sub-grafo de diagnostico (TCC-040).

Cada node recebe ``state: DiagnosisState`` + services por keyword-only injetados
via ``functools.partial`` em ``graph.build_diagnosis_graph``. Isso mantem o
modulo testavel sem container DI e desacoplado do FastAPI Depends.

Em Sprint A2 o ``run_inference_node`` ainda usa o mock de ``InferenceService.predict``
(synchronous random); quando o modelo real chegar, basta trocar a chamada por
``await inference_svc.predict_batch_async(...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    Top3PredictionSchema,
)
from app.domains.diagnosis_graph.state import DiagnosisState
from app.shared.enums import SeverityEnum

if TYPE_CHECKING:
    from app.domains.action_plans.service import ActionPlanService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService


async def load_model_node(
    state: DiagnosisState,
    *,
    inference_svc: "InferenceService",
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
    inference_svc: "InferenceService",
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
    action_plan_svc: "ActionPlanService",
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
    diagnosis_svc: "DiagnosisService",
) -> dict:
    """Cria 1 row em ``diagnoses`` por imagem do batch."""
    persisted: list[str] = []
    image_ids = state.get("image_ids", [])
    predictions = state.get("predictions", [])
    model_id = state.get("model_id", "ensemble")
    user_id = state.get("user_id", "")

    for i, pred in enumerate(predictions):
        image_name = image_ids[i] if i < len(image_ids) else None
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
        )
        diag = await diagnosis_svc.create(user_id, body)
        persisted.append(diag.id)
    return {"persisted_ids": persisted}
