"""Tool ``get_action_plan`` — busca plano de acao por disease (TCC-041 / TCC-051).

Recebe ``ActionPlanService`` via closure. Os niveis retornados sao gated pelo
``plan_features.action_plan_levels`` (Sprint A3):
  - Free: ["essencial"]
  - Pro: ["essencial", "campo"]
  - Enterprise: ["essencial", "campo", "especialista"]

O LLM pode pedir um ``level`` especifico — se for permitido pelo plano,
retorna apenas ele. Se for None, retorna TODOS os niveis permitidos.
Se ``preferred_action_level`` esta no state e e permitido, prevalece sobre
``level=None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState

if TYPE_CHECKING:
    from app.domains.action_plans.service import ActionPlanService


def build_get_action_plan_tool(
    action_plan_svc: ActionPlanService,
) -> BaseTool:
    """Factory pra ``get_action_plan``."""

    @tool
    async def get_action_plan(
        disease_id: str,
        level: str | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Plano de acao por doenca, filtrado por nivel permitido pelo plano.

        Args:
            disease_id: slug da doenca (ex: "ferrugem-asiatica").
            level: nivel opcional ("essencial" | "campo" | "especialista").
                Se None, retorna TODOS os niveis permitidos pelo plano. Se
                especificado mas nao permitido pelo plano, retorna apenas os
                permitidos com nota explicativa.

        Returns:
            String com plano formatado + fontes (ou erro se disease nao encontrado).
        """
        try:
            plan = await action_plan_svc.get_by_disease(disease_id)
        except Exception as exc:  # noqa: BLE001 — tool sempre retorna string
            return f"Plano de acao indisponivel para {disease_id}: {exc}"

        # Determina niveis permitidos pelo plano. Sem plan_features (back-compat
        # com testes legacy), aceita todos.
        plan_features = state.get("plan_features")
        if plan_features is not None:
            allowed_levels = set(plan_features.action_plan_levels)
        else:
            allowed_levels = None  # sem gate

        # Determina niveis a retornar:
        # 1. ``level`` arg do LLM (filtra por single level)
        # 2. ``preferred_action_level`` no state (filtra por single level)
        # 3. None = todos permitidos
        preferred = state.get("preferred_action_level")
        target_level = level or preferred

        lines: list[str] = [f"Plano de acao para {plan.disease_id}:"]
        rendered_any = False
        for plan_level in plan.levels:
            level_str = str(plan_level.level)
            # Filtra por plano (se gate ativo).
            if allowed_levels is not None and level_str not in allowed_levels:
                continue
            # Filtra por nivel pedido.
            if target_level and level_str != target_level:
                continue
            lines.append(f"\n[{level_str.upper()}]")
            for action in plan_level.actions:
                lines.append(f"- {action}")
            rendered_any = True

        # Se o usuario pediu um nivel especifico mas o plano nao permite,
        # avisa explicitamente (UX > silencioso).
        if (
            target_level
            and allowed_levels is not None
            and target_level not in allowed_levels
            and not rendered_any
        ):
            permitted = ", ".join(sorted(allowed_levels))
            lines.append(
                f"\n[Nivel '{target_level}' nao disponivel no seu plano. "
                f"Permitidos: {permitted}.]"
            )

        if plan.sources:
            lines.append("\nFontes:")
            for src in plan.sources:
                suffix = f" ({src.url})" if src.url else ""
                lines.append(f"- {src.name}: {src.detail}{suffix}")
        return "\n".join(lines)

    return get_action_plan
