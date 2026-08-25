"""Tool ``compare_diagnoses`` — roda inferencia em paralelo com N modelos (TCC-050).

Tier Enterprise: o usuario pode pedir "rode resnet50 e vit nessa imagem e
me mostre a comparacao". A tool invoca o sub-grafo de diagnostico uma vez
por modelo na MESMA imagem e retorna JSON tabular com confidence, severity
e disease detectada por modelo.

Diferente de ``deep_diagnose`` (que processa N imagens com 1 modelo), esta
tool processa 1 imagem com N modelos — comparacao lado-a-lado pra apoiar
decisao do agronomo. Nao persiste diagnoses (so' compara — persistencia
fica a cargo do ``deep_diagnose`` quando o usuario escolhe o modelo final).

InjectedState: ``current_user_id``, ``uploaded_files`` vem do state.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState, resolve_image

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_compare_diagnoses_tool(
    diagnosis_graph_factory: Callable[[str], CompiledStateGraph[Any]],
) -> BaseTool:
    """Factory pra ``compare_diagnoses`` — recebe closure cacheada por crop.

    Args:
        diagnosis_graph_factory: callable ``(crop_id) -> CompiledStateGraph``.
            Mesma factory usada por ``deep_diagnose`` — reuso de cache.

    Returns:
        Tool decorada pronta pra bind no LLM.
    """

    @tool
    async def compare_diagnoses(
        image_id: str,
        models: list[str],
        crop_id: str | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Compara multiplos modelos de diagnostico na MESMA imagem (tier Enterprise).

        Roda inferencia em paralelo com cada modelo da lista ``models`` na imagem
        identificada por ``image_id`` e retorna um JSON com comparacao tabular.

        Args:
            image_id: id da imagem em ``state.uploaded_files``.
            models: lista de model_ids (ex: ["resnet50", "vit"]). Tipicamente 2-4.
            crop_id: cultivo opcional (default: ``state.detected_crop_id`` ou "soja").

        Returns:
            JSON-string com chaves ``image_id``, ``image_name``, ``models``,
            ``comparison`` (list de dicts: model_id, disease, disease_id,
            confidence, severity, agreement) e ``consensus`` (disease mais
            comum entre os modelos).
        """
        target = resolve_image(state, image_id)
        if target is None:
            return json.dumps(
                {"error": f"Imagem '{image_id}' nao encontrada no turno atual."},
                ensure_ascii=False,
            )

        if not models:
            return json.dumps(
                {"error": "Lista de models vazia — passe ao menos 1 model_id."},
                ensure_ascii=False,
            )

        effective_crop = (
            crop_id or state.get("detected_crop_id") or "soja"
        )
        graph = diagnosis_graph_factory(effective_crop)
        user_id = state.get("current_user_id") or ""

        # Invoca o sub-grafo uma vez por modelo, em paralelo via asyncio.gather.
        async def _run_with_model(model_id: str) -> dict[str, Any]:
            try:
                result = await graph.ainvoke(
                    {
                        "user_id": user_id,
                        "crop_id": effective_crop,
                        # Mesma imagem pra todos os modelos — bytes reais.
                        "image_batch": [target.b64 or ""],
                        "image_ids": [target.id],
                        "model_id": model_id,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — tool nunca propaga
                return {
                    "model_id": model_id,
                    "error": str(exc),
                }
            preds = result.get("predictions", [])
            if not preds:
                return {"model_id": model_id, "error": "Sem predicao retornada."}
            pred = preds[0]
            return {
                "model_id": model_id,
                "disease": pred.get("disease_name"),
                "disease_id": pred.get("disease_id"),
                "confidence": pred.get("confidence"),
                "severity": pred.get("severity"),
            }

        comparison = await asyncio.gather(*(_run_with_model(m) for m in models))

        # Computa consenso: doenca mais comum (sem contar errors).
        disease_counts: dict[str, int] = {}
        for row in comparison:
            disease_id = row.get("disease_id")
            if disease_id:
                disease_counts[disease_id] = disease_counts.get(disease_id, 0) + 1
        consensus_disease = (
            max(disease_counts.items(), key=lambda kv: kv[1])[0]
            if disease_counts
            else None
        )

        # Anota concordancia com o consenso pra cada linha.
        for row in comparison:
            row["agrees_with_consensus"] = (
                row.get("disease_id") == consensus_disease
                if consensus_disease
                else False
            )

        return json.dumps(
            {
                "image_id": target.id,
                "image_name": target.original_name,
                "models": models,
                "consensus_disease_id": consensus_disease,
                "comparison": comparison,
            },
            ensure_ascii=False,
        )

    return compare_diagnoses
