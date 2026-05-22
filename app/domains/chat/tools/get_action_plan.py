"""Tool ``get_action_plan`` — busca plano de acao por disease (TCC-041).

Recebe ``ActionPlanService`` via closure; o filtro por
``preferred_action_level`` (vindo do ChatState) entra em Sprint A3 — por ora
retorna todos os niveis.
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
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Plano de acao por doenca (niveis essencial/campo/especialista + fontes)."""
        try:
            plan = await action_plan_svc.get_by_disease(disease_id)
        except Exception as exc:  # noqa: BLE001 — tool sempre retorna string
            return f"Plano de acao indisponivel para {disease_id}: {exc}"

        preferred = state.get("preferred_action_level")

        lines: list[str] = [f"Plano de acao para {plan.disease_id}:"]
        for level in plan.levels:
            if preferred and str(level.level) != preferred:
                continue
            lines.append(f"\n[{str(level.level).upper()}]")
            for action in level.actions:
                lines.append(f"- {action}")
        if plan.sources:
            lines.append("\nFontes:")
            for src in plan.sources:
                suffix = f" ({src.url})" if src.url else ""
                lines.append(f"- {src.name}: {src.detail}{suffix}")
        return "\n".join(lines)

    return get_action_plan
