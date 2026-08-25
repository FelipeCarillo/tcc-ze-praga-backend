"""Tool ``deep_diagnose`` — invoca o sub-grafo de diagnostico (TCC-041).

Substitui a antiga ``analyze_image`` (1 imagem only) por uma versao batch
que processa N imagens do turno atual. Quando ``image_ids`` eh None, processa
**todas** as imagens em ``state.uploaded_files``. Quando passado, filtra so
as referenciadas.

InjectedState: ``user_id``, ``session_id``, ``selected_model`` vem do state
e nao sao expostos ao LLM.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_deep_diagnose_tool(
    diagnosis_graph_factory: Callable[[str], CompiledStateGraph[Any]],
) -> BaseTool:
    """Factory pra ``deep_diagnose`` — recebe closure que cria o sub-grafo por crop.

    Args:
        diagnosis_graph_factory: callable ``(crop_id) -> CompiledStateGraph``
            cacheado por crop_id. Em testes pode receber lambda que retorna
            um grafo mockado.

    Returns:
        Tool decorada pronta pra ser anexada ao ``llm.bind_tools()``.
    """

    @tool
    async def deep_diagnose(
        image_ids: list[str] | None = None,
        crop_id: str | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Diagnostica uma ou mais imagens.

        Se ``image_ids`` for omitido, processa TODAS as imagens do turno.

        Exemplos:
        - User mandou 1 foto: chame sem args
        - User mandou 3 e disse "analisa as duas primeiras":
          image_ids=["img-1","img-2"]
        """
        targets = list(state.get("uploaded_files", []) or [])
        if image_ids is not None:
            requested = set(image_ids)
            targets = [f for f in targets if f.id in requested]
        if not targets:
            return json.dumps(
                {"error": "Nenhuma imagem disponivel"}, ensure_ascii=False
            )

        effective_crop = (
            crop_id or state.get("detected_crop_id") or "soja"
        )
        graph = diagnosis_graph_factory(effective_crop)

        user_id = state.get("current_user_id") or ""
        model_id = state.get("selected_model") or "ensemble"

        result = await graph.ainvoke(
            {
                "user_id": user_id,
                "crop_id": effective_crop,
                # Bytes reais (base64) com index alinhado a image_ids — sem
                # isto o run_inference_node cai no mock (TCC-020).
                "image_batch": [f.b64 or "" for f in targets],
                "image_ids": [f.id for f in targets],
                "model_id": model_id,
            }
        )

        persisted = result.get("persisted_ids", [])
        predictions = result.get("predictions", [])

        return json.dumps(
            {
                "count": len(targets),
                "results": [
                    {
                        "image_id": targets[i].id,
                        "image_name": targets[i].original_name,
                        "diagnosis_id": (
                            persisted[i] if i < len(persisted) else None
                        ),
                        "disease": predictions[i]["disease_name"]
                        if i < len(predictions)
                        else None,
                        "disease_id": predictions[i]["disease_id"]
                        if i < len(predictions)
                        else None,
                        "confidence": predictions[i]["confidence"]
                        if i < len(predictions)
                        else None,
                        "severity": predictions[i]["severity"]
                        if i < len(predictions)
                        else None,
                    }
                    for i in range(len(targets))
                ],
            },
            ensure_ascii=False,
        )

    return deep_diagnose
