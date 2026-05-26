"""Tool Registry — flags globais + features do plano + tier (TCC-039 / Sprint A2).

O registry eh declarativo: cada tool tem um ``ToolConfig`` com flags pra
filtragem (``enabled_globally``, ``required_feature``, ``min_tier``). A
funcao ``get_active_tool_names`` resolve qual subset de tools deve estar
ativo pra um dado plano/feature flags.

Em runtime, o ``ChatService`` consome este modulo pra montar a lista de
tools que sera bindada no LLM. O esquema fica desacoplado de DI: as
factories sao passadas como ``dict[name -> Callable[[], BaseTool]]``
em ``build_tools``.

Plan features (Sprint A3 trara ``PlanFeatures`` tipado):
    - ``tier_name``: "free" | "pro" | "enterprise"
    - flags booleanas opcionais (ex: ``has_search``, ``has_deep_diagnose``)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from app.config import settings


@dataclass(frozen=True)
class ToolConfig:
    """Configuracao declarativa de uma tool — input do registry.

    Args:
        name: identificador unico (matches ``factory_key``).
        version: 1 = lançado em prod; 2 = dev (rollout gradual).
        factory_key: chave usada pra lookup em ``factories`` dict.
        enabled_globally: kill-switch global (deploy-time).
        required_feature: nome de uma flag em ``plan_features`` que deve
            ser ``True`` pra ativar. ``None`` = sem requisito.
        min_tier: tier minimo do plano ("free" | "pro" | "enterprise").
            ``None`` = disponivel pra todos.
        description: doc humana exibida em listagens admin.
    """

    name: str
    version: int
    factory_key: str
    enabled_globally: bool
    required_feature: str | None
    min_tier: str | None
    description: str


# Tier ordering pra comparacao numerica (free < pro < enterprise).
_TIER_ORDER: dict[str, int] = {"free": 0, "pro": 1, "enterprise": 2}


def get_registry() -> list[ToolConfig]:
    """Retorna o registry estatico em Sprint A2.

    Quando essa lista crescer (Sprint A3+), considere mover pra DB/seed pra
    permitir feature flags dinamicas sem deploy.
    """
    return [
        ToolConfig(
            name="deep_diagnose",
            version=1,
            factory_key="deep_diagnose",
            enabled_globally=True,
            required_feature=None,
            min_tier=None,
            description="Diagnostica imagens via sub-grafo ML + plano de acao.",
        ),
        ToolConfig(
            name="get_disease_info",
            version=1,
            factory_key="get_disease_info",
            enabled_globally=True,
            required_feature=None,
            min_tier=None,
            description="Lookup de informacoes sobre uma doenca.",
        ),
        ToolConfig(
            name="get_action_plan",
            version=1,
            factory_key="get_action_plan",
            enabled_globally=True,
            required_feature=None,
            min_tier=None,
            description="Plano de acao por doenca.",
        ),
        ToolConfig(
            name="search_my_diagnoses",
            version=1,
            factory_key="search_my_diagnoses",
            enabled_globally=True,
            required_feature=None,
            min_tier=None,
            description=(
                "Busca diagnosticos passados (sera semantico em A2.5)."
            ),
        ),
        ToolConfig(
            name="ask_user",
            version=1,
            factory_key="ask_user",
            enabled_globally=settings.agent_enable_ask_user,
            required_feature=None,
            min_tier=None,
            description=(
                "Pergunta direta ao usuario via interrupt (human-in-the-loop)."
            ),
        ),
        ToolConfig(
            name="compare_diagnoses",
            version=1,
            factory_key="compare_diagnoses",
            enabled_globally=True,
            required_feature=None,
            min_tier="enterprise",
            description=(
                "Compara multiplos modelos na mesma imagem (Enterprise)."
            ),
        ),
        ToolConfig(
            name="search_web",
            version=1,
            factory_key="search_web",
            enabled_globally=settings.agent_enable_search_web,
            required_feature="search_web",
            min_tier="pro",
            description="Pesquisa web via Tavily (tier Pro+).",
        ),
        ToolConfig(
            name="search_scientific",
            version=1,
            factory_key="search_scientific",
            enabled_globally=settings.agent_enable_search_scientific,
            required_feature="search_scientific",
            min_tier="enterprise",
            description=(
                "Pesquisa literatura cientifica via SciELO (tier Enterprise)."
            ),
        ),
        ToolConfig(
            name="identify_crop",
            version=2,
            factory_key="identify_crop",
            enabled_globally=settings.agent_enable_identify_crop,
            required_feature="identify_crop_auto",
            min_tier="pro",
            description=(
                "Identifica cultivo via gpt-4o vision (V2 — multi-cultivo)."
            ),
        ),
    ]


def get_active_tool_names(
    plan_features: dict[str, Any] | None = None,
) -> list[str]:
    """Filtra tools por flag global + features + tier.

    Args:
        plan_features: dict opcional com flags do plano:
            - ``tier_name``: tier do usuario (default "free")
            - flags booleanas opcionais (matches em ``required_feature``)

    Returns:
        Lista de nomes de tools ativas, na ordem definida em ``get_registry``.
    """
    plan_features = plan_features or {}
    tier_name = plan_features.get("tier_name", "free")
    user_tier_level = _TIER_ORDER.get(tier_name, 0)

    active: list[str] = []
    for cfg in get_registry():
        if not cfg.enabled_globally:
            continue
        if cfg.required_feature and not plan_features.get(
            cfg.required_feature, False
        ):
            continue
        if cfg.min_tier:
            required_level = _TIER_ORDER.get(cfg.min_tier, 0)
            if user_tier_level < required_level:
                continue
        active.append(cfg.name)
    return active


def build_tools(
    factories: dict[str, Callable[[], BaseTool]],
    plan_features: dict[str, Any] | None = None,
) -> list[BaseTool]:
    """Constroi as tools ativas usando factories injetadas.

    Args:
        factories: mapping ``name -> callable que retorna BaseTool``.
            Os callables sao invocados aqui (zero args — capturem services
            via closure quando criarem o dict).
        plan_features: flags do plano (mesma forma de ``get_active_tool_names``).

    Returns:
        Lista de ``BaseTool`` na ordem do registry — tools sem factory
        correspondente sao silenciosamente puladas (defensive).
    """
    active_names = get_active_tool_names(plan_features)
    return [factories[name]() for name in active_names if name in factories]
