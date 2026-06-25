"""Tool ``analyze_image`` (state-aware, TCC-079).

Substitui a ``analyze_image`` legada (que recebia ``image_filename`` como arg do
LLM) por uma versão que resolve a imagem do ``state`` (InjectedState), roda a
inferência (mock CNN/ViT — integração ONNX real é o epic TCC-020) e **persiste o
Diagnosis** via ``DiagnosisService``, registrando o id em ``diagnoses_in_turn``
pra o ``ChatService`` montar o ``ChatResponse.diagnosis``.

Use SOMENTE depois que ``inspect_image`` confirmar que a imagem é uma planta.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.domains.chat.agent_state import ChatState, resolve_image
from app.domains.diagnoses.schemas import CreateDiagnosisRequest

if TYPE_CHECKING:
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService


def build_analyze_image_tool(
    inference_svc: InferenceService,
    diagnosis_svc: DiagnosisService,
) -> BaseTool:
    """Factory pra ``analyze_image`` — fecha sobre inference + diagnosis services."""

    @tool
    async def analyze_image(
        image_id: str | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[Any]:
        """Analisa a foto de folha/planta enviada e registra o diagnóstico.

        Use SOMENTE após ``inspect_image`` confirmar que é uma planta analisável.
        Se ``image_id`` for omitido, usa a primeira imagem do turno.
        """
        files = state.get("uploaded_files", []) or []
        image = resolve_image(state, image_id) if image_id else (files[0] if files else None)
        if image is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                {"error": "Nenhuma imagem disponível no turno."},
                                ensure_ascii=False,
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        model_id = state.get("selected_model") or "ensemble"
        user_id = state.get("current_user_id") or ""

        image_bytes = base64.b64decode(image.b64) if image.b64 else None
        result = inference_svc.predict(
            model_id, image.original_name, image_bytes=image_bytes
        )
        body = CreateDiagnosisRequest(
            disease_name=result.disease_name,
            disease_id=result.disease_id,
            scientific_name=result.scientific_name,
            confidence=result.confidence,
            severity=result.severity,
            description=result.description,
            model_used=result.model_id,
            image_url=None,
            image_name=result.image_name,
            top3=result.top3,
        )
        crop_uuid = (
            inference_svc.disease_catalog[0].crop_id
            if inference_svc.disease_catalog
            else ""
        )
        diagnosis = await diagnosis_svc.create(user_id, body, crop_id=crop_uuid)

        payload = {
            "diagnosis_id": diagnosis.id,
            "disease_name": result.disease_name,
            "disease_id": result.disease_id,
            "scientific_name": result.scientific_name,
            "confidence": result.confidence,
            "severity": str(result.severity),
            "top3": [
                {
                    "rank": p.rank,
                    "disease_name": p.disease_name,
                    "disease_id": p.disease_id,
                    "confidence": p.confidence,
                }
                for p in result.top3
            ],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
                "diagnoses_in_turn": [diagnosis.id],
            }
        )

    return analyze_image
