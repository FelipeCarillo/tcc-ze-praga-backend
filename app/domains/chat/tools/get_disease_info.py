"""Tool ``get_disease_info`` — lookup do catalogo de doencas (TCC-041).

Refatorada pra usar ``DiseaseRepository`` direto (em vez de ``InferenceService.disease_catalog``
que era cache-limited por crop). Recebe ``db_session_factory`` (async context
manager) pra abrir sessao quando precisar.
"""

from __future__ import annotations

import json
from typing import Annotated, Callable

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState
from app.domains.inference.repository import DiseaseRepository


def build_get_disease_info_tool(
    db_session_factory: Callable[[], object],
) -> BaseTool:
    """Factory pra ``get_disease_info``.

    Args:
        db_session_factory: callable que retorna um async context manager
            cedendo um ``AsyncSession`` (ex: ``AsyncSessionLocal``).
    """

    @tool
    async def get_disease_info(
        disease_id: str,
        crop_id: str | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Retorna informacoes sobre uma doenca via slug."""
        effective_crop = (
            crop_id or state.get("detected_crop_id") or "soja"
        )

        async with db_session_factory() as session:  # type: ignore[union-attr]
            disease_repo = DiseaseRepository(session)
            # Resolve crop_id real se passado como slug ('soja' -> uuid)
            crop_uuid = await _resolve_crop_id(session, effective_crop)
            disease = await disease_repo.get_by_slug(crop_uuid, disease_id)
            if not disease:
                return json.dumps(
                    {"error": f"Doenca {disease_id} nao encontrada"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "id": disease.slug,
                    "slug": disease.slug,
                    "name_pt": disease.name_pt,
                    "scientific_name": disease.scientific_name,
                    "severity_default": disease.severity_default,
                    "description": disease.description_md,
                },
                ensure_ascii=False,
            )

    return get_disease_info


async def _resolve_crop_id(session, crop_ref: str) -> str:
    """Aceita slug ou id. Cache via CropRepository.

    Tenta slug primeiro (caso comum: usuario/state passou "soja", "milho"...).
    So' bate no DB por id quando o slug nao for cache-hit e nao bater no DB.
    """
    from app.domains.inference.repository import CropRepository

    repo = CropRepository(session)
    by_slug = await repo.get_by_slug(crop_ref)
    if by_slug:
        return by_slug.id
    by_id = await repo.get_by_id(crop_ref)
    if by_id:
        return by_id.id
    return crop_ref  # fallback — disease_repo retorna None se nao bater
