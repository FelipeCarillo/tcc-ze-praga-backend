"""Tool ``analyze_image`` (state-aware, TCC-079).

Substitui a ``analyze_image`` legada (que recebia ``image_filename`` como arg do
LLM) por uma versão que resolve a imagem do ``state`` (InjectedState), roda a
inferência ONNX real e **persiste o Diagnosis** via ``DiagnosisService``,
registrando o id em ``diagnoses_in_turn`` pra o ``ChatService`` montar o
``ChatResponse.diagnosis``.

O modelo efetivo respeita ``plan_features.diagnosis_models`` — o usuário escolhe
na UI, mas o plano manda. O diagnóstico criado é indexado no Store (pgvector)
pra virar memória semântica entre sessões, espelhando o ``persist_node`` do
``diagnosis_graph``. E a foto vai pro Storage, com a storage key gravada em
``image_url`` — sem isso o histórico ficava sem miniatura, porque a imagem do
chat é efêmera (só existe em base64 no estado do turno).

Use SOMENTE depois que ``inspect_image`` confirmar que a imagem é uma planta.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.domains.chat.agent_state import ChatState, resolve_image
from app.domains.chat.memory import index_diagnosis_in_store
from app.domains.diagnoses.schemas import CreateDiagnosisRequest
from app.domains.inference.service import resolve_allowed_model

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService
    from app.domains.uploads.service import UploadService

logger = logging.getLogger(__name__)


def build_analyze_image_tool(
    inference_svc: InferenceService,
    diagnosis_svc: DiagnosisService,
    store_factory: Callable[[], Awaitable[BaseStore]] | None = None,
    upload_svc: UploadService | None = None,
) -> BaseTool:
    """Factory pra ``analyze_image`` — fecha sobre inference + diagnosis services.

    Args:
        inference_svc: serviço de inferência (ONNX real com fallback mock).
        diagnosis_svc: persistência do diagnóstico.
        store_factory: callable async que devolve o ``BaseStore``. Quando
            passado, cada diagnóstico é indexado em ``("user", uid, "diagnoses")``
            pra busca semântica futura via ``search_my_diagnoses``. ``None``
            mantém o comportamento antigo (só persiste no DB).
        upload_svc: quando passado, a foto do turno vai pro Storage (com dedup
            por sha256) e a storage key é gravada em ``image_url``. ``None``
            mantém o diagnóstico sem imagem.
    """

    async def _store_image(
        user_id: str, image: Any, data: bytes | None
    ) -> str | None:
        """Sobe a foto e devolve a storage key. Best-effort.

        Falha de Storage não pode impedir o diagnóstico: o resultado da
        inferência é o que o usuário veio buscar; a miniatura é acessório.
        """
        if upload_svc is None or not data or not user_id:
            return None
        try:
            row, _dedup = await upload_svc.upload(
                user_id=user_id,
                original_name=image.original_name,
                mime=image.mime or "image/jpeg",
                data=data,
                session_id=None,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Falha ao subir imagem do turno pro Storage")
            return None
        return str(row.storage_key)

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

        user_id = state.get("current_user_id") or ""

        # O plano manda no modelo: se o usuário pediu algo fora do tier, cai no
        # melhor permitido em vez de negar o turno (TCC-051).
        plan_features = state.get("plan_features")
        model_id, downgraded = resolve_allowed_model(
            state.get("selected_model"),
            plan_features.diagnosis_models if plan_features is not None else None,
        )

        image_bytes = base64.b64decode(image.b64) if image.b64 else None
        result = inference_svc.predict(
            model_id, image.original_name, image_bytes=image_bytes
        )
        storage_key = await _store_image(user_id, image, image_bytes)

        body = CreateDiagnosisRequest(
            disease_name=result.disease_name,
            disease_id=result.disease_id,
            scientific_name=result.scientific_name,
            confidence=result.confidence,
            severity=result.severity,
            description=result.description,
            model_used=result.model_id,
            image_url=storage_key,
            image_name=result.image_name,
            top3=result.top3,
        )
        crop_uuid = (
            inference_svc.disease_catalog[0].crop_id
            if inference_svc.disease_catalog
            else ""
        )
        diagnosis = await diagnosis_svc.create(user_id, body, crop_id=crop_uuid)

        # Memória semântica (pgvector) — best-effort: Store fora do ar não pode
        # derrubar um diagnóstico que já foi persistido no DB.
        if store_factory is not None and user_id:
            try:
                store = await store_factory()
                await index_diagnosis_in_store(store, user_id, diagnosis)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Falha ao indexar diagnosis %s no Store", diagnosis.id
                )

        payload = {
            "diagnosis_id": diagnosis.id,
            "model_used": model_id,
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
        if downgraded:
            payload["model_downgraded_to"] = model_id
            payload["note"] = (
                "O modelo pedido não está disponível no plano do usuário; "
                f"a análise usou '{model_id}'. Mencione isso na resposta."
            )
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
