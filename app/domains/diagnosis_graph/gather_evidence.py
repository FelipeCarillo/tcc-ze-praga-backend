"""Node gather_evidence — coleta evidencia externa em paralelo (TCC-055).

Roda apos ``run_inference`` em paralelo com ``compose_action_plan``: pra cada
predicao, dispara queries em ``search_web`` (Pro+) e/ou ``search_scientific``
(Enterprise) de acordo com ``state.plan_features``.

Design:
- Tier gating: Free skip total; Pro = web only; Enterprise = web + scientific.
- Os callables ``tavily_search`` / ``scielo_search`` sao injetados pra que o
  node nao saiba detalhes da camada externa (testavel + permite cache do
  agent).
- Evidencia retornada eh ``list[list[dict]]`` — index alinhado com
  ``state.predictions``, cada item eh uma lista de resultados (web/scientific
  mesclados na ordem).
- Falhas individuais (Tavily 503, parse de SciELO etc.) sao filtradas via
  ``return_exceptions=True``: o node nunca propaga erros e o grafo segue
  pra persist sem evidencia daquela imagem.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.domains.diagnosis_graph.state import DiagnosisState

logger = logging.getLogger(__name__)

SearchCallable = Callable[[str], Awaitable[str]]


async def gather_evidence_node(
    state: DiagnosisState,
    *,
    tavily_search: SearchCallable | None = None,
    scielo_search: SearchCallable | None = None,
) -> dict[str, Any]:
    """Coleta evidencia externa pra cada predicao do batch.

    Args:
        state: DiagnosisState com ``predictions`` ja preenchidas pelo node
            ``run_inference``. Le ``plan_features`` pra decidir o que rodar.
        tavily_search: callable async ``(query) -> JSON-str`` pra busca web.
            Quando ``None``, busca web eh skipada.
        scielo_search: callable async ``(query) -> JSON-str`` pra busca
            cientifica. Quando ``None``, busca cientifica eh skipada.

    Returns:
        Dict ``{evidence_per_image: list[list[dict]]}`` — index alinhado com
        ``predictions``. Cada elemento eh lista de resultados (web+scientific
        mesclados). Tier que nao habilita nenhuma busca retorna ``[]``.
    """
    plan_features: dict[str, Any] = state.get("plan_features") or {}
    want_web = bool(plan_features.get("search_web")) and tavily_search is not None
    want_scientific = (
        bool(plan_features.get("search_scientific")) and scielo_search is not None
    )

    if not want_web and not want_scientific:
        return {"evidence_per_image": []}

    crop_id = state.get("crop_id", "")
    evidence_per_image: list[list[dict[str, Any]]] = []
    for pred in state.get("predictions", []):
        disease_name = pred.get("disease_name") or pred.get("disease_id") or ""
        query = f"{disease_name} {crop_id} manejo".strip()
        tasks: list[Awaitable[str]] = []
        if want_web:
            tasks.append(tavily_search(query))  # type: ignore[misc]
        if want_scientific:
            tasks.append(scielo_search(query))  # type: ignore[misc]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[dict[str, Any]] = []
        for r in raw_results:
            if isinstance(r, BaseException):
                logger.warning("gather_evidence sub-call failed: %s", r)
                continue
            parsed = _parse_json_payload(r)
            if isinstance(parsed, list):
                merged.extend(parsed)
        evidence_per_image.append(merged)

    return {"evidence_per_image": evidence_per_image}


def _parse_json_payload(raw: str) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Decodifica payload JSON da tool — tolera string nao-JSON."""
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(decoded, list):
        return [d for d in decoded if isinstance(d, dict)]
    return decoded if isinstance(decoded, dict) else None
