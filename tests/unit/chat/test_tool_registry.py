"""Testes do tool_registry (TCC-039)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool, tool

from app.domains.chat import tool_registry
from app.domains.chat.tool_registry import (
    ToolConfig,
    build_tools,
    get_active_tool_names,
    get_registry,
)


@tool
def _dummy_tool() -> str:
    """Dummy tool pra testes."""
    return "ok"


def _factories() -> dict[str, callable]:
    return {
        "deep_diagnose": lambda: _dummy_tool,
        "get_disease_info": lambda: _dummy_tool,
        "get_action_plan": lambda: _dummy_tool,
        "search_my_diagnoses": lambda: _dummy_tool,
    }


# ── get_active_tool_names ─────────────────────────────────────────────────────


def test_default_tier_sees_all_basic_tools() -> None:
    """Free tier sem features ativa todas as 4 tools base (compare_diagnoses fica fora)."""
    names = get_active_tool_names({})
    assert names == [
        "deep_diagnose",
        "get_disease_info",
        "get_action_plan",
        "search_my_diagnoses",
    ]


def test_none_plan_features_treated_as_free() -> None:
    """``None`` deve cair no mesmo caminho que ``{}``."""
    assert get_active_tool_names(None) == get_active_tool_names({})


def test_free_tier_explicit_same_as_default() -> None:
    names = get_active_tool_names({"tier_name": "free"})
    assert "deep_diagnose" in names


def test_pro_tier_sees_base_tools() -> None:
    """Pro nao tem compare_diagnoses (Enterprise-only) — 4 tools."""
    names = get_active_tool_names({"tier_name": "pro"})
    # base 4 tools (sem compare_diagnoses, sem search_web por feature flag)
    assert len(names) == 4
    assert "compare_diagnoses" not in names
    assert "search_web" not in names  # requires plan_features["search_web"]=True


def test_pro_tier_with_search_web_feature_unlocks_tool() -> None:
    """Pro com search_web=True ativa a tool."""
    names = get_active_tool_names(
        {"tier_name": "pro", "search_web": True}
    )
    assert "search_web" in names


def test_enterprise_tier_unlocks_compare_diagnoses() -> None:
    """Enterprise vê todas as 4 base + compare_diagnoses (sem features)."""
    names = get_active_tool_names({"tier_name": "enterprise"})
    assert len(names) == 5
    assert "compare_diagnoses" in names
    # search_web ainda requer feature flag
    assert "search_web" not in names


def test_enterprise_tier_with_search_web_feature_unlocks() -> None:
    """Enterprise + flag ativa search_web."""
    names = get_active_tool_names(
        {"tier_name": "enterprise", "search_web": True}
    )
    assert "search_web" in names


def test_global_flag_off_skips_tool(monkeypatch) -> None:
    """Tool com ``enabled_globally=False`` deve sumir."""
    custom = [
        ToolConfig(
            name="deep_diagnose",
            version=1,
            factory_key="deep_diagnose",
            enabled_globally=False,  # kill-switch on
            required_feature=None,
            min_tier=None,
            description="...",
        ),
        ToolConfig(
            name="get_disease_info",
            version=1,
            factory_key="get_disease_info",
            enabled_globally=True,
            required_feature=None,
            min_tier=None,
            description="...",
        ),
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)
    names = get_active_tool_names({})
    assert names == ["get_disease_info"]


def test_required_feature_blocks_when_missing(monkeypatch) -> None:
    """Tool com ``required_feature=X`` so' ativa se ``plan_features[X] == True``."""
    custom = [
        ToolConfig(
            name="premium_tool",
            version=1,
            factory_key="premium_tool",
            enabled_globally=True,
            required_feature="has_premium",
            min_tier=None,
            description="...",
        )
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    assert get_active_tool_names({}) == []
    assert get_active_tool_names({"has_premium": False}) == []
    assert get_active_tool_names({"has_premium": True}) == ["premium_tool"]


def test_min_tier_gates_by_user_tier(monkeypatch) -> None:
    """Tool com ``min_tier=pro`` nao deve ativar pra free."""
    custom = [
        ToolConfig(
            name="pro_tool",
            version=1,
            factory_key="pro_tool",
            enabled_globally=True,
            required_feature=None,
            min_tier="pro",
            description="...",
        )
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    assert get_active_tool_names({"tier_name": "free"}) == []
    assert get_active_tool_names({"tier_name": "pro"}) == ["pro_tool"]
    assert get_active_tool_names({"tier_name": "enterprise"}) == ["pro_tool"]


def test_min_tier_enterprise_only(monkeypatch) -> None:
    custom = [
        ToolConfig(
            name="ent_tool",
            version=1,
            factory_key="ent_tool",
            enabled_globally=True,
            required_feature=None,
            min_tier="enterprise",
            description="...",
        )
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    assert get_active_tool_names({"tier_name": "free"}) == []
    assert get_active_tool_names({"tier_name": "pro"}) == []
    assert get_active_tool_names({"tier_name": "enterprise"}) == ["ent_tool"]


def test_combined_gates_all_must_pass(monkeypatch) -> None:
    """Quando ha global + feature + tier, todas as 3 precisam passar."""
    custom = [
        ToolConfig(
            name="t1",
            version=1,
            factory_key="t1",
            enabled_globally=True,
            required_feature="has_x",
            min_tier="pro",
            description="...",
        )
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    assert get_active_tool_names({"tier_name": "pro"}) == []
    assert (
        get_active_tool_names({"tier_name": "free", "has_x": True}) == []
    )
    assert (
        get_active_tool_names({"tier_name": "pro", "has_x": True}) == ["t1"]
    )


def test_unknown_tier_treated_as_free() -> None:
    """Tier name desconhecido cai em level 0 (free)."""
    names = get_active_tool_names({"tier_name": "wakanda"})
    # 4 tools base sao todas min_tier=None entao saem ok
    assert len(names) == 4


# ── build_tools ───────────────────────────────────────────────────────────────


def test_build_tools_invokes_factories_in_order() -> None:
    calls: list[str] = []

    def _track(name):
        def _factory():
            calls.append(name)
            return _dummy_tool

        return _factory

    factories = {
        n: _track(n)
        for n in (
            "deep_diagnose",
            "get_disease_info",
            "get_action_plan",
            "search_my_diagnoses",
        )
    }
    tools = build_tools(factories)
    assert len(tools) == 4
    assert calls == [
        "deep_diagnose",
        "get_disease_info",
        "get_action_plan",
        "search_my_diagnoses",
    ]


def test_build_tools_skips_missing_factory() -> None:
    """Se uma tool ativa nao tiver factory, ela e silenciosamente pulada."""
    factories = {"deep_diagnose": lambda: _dummy_tool}
    tools = build_tools(factories)
    assert len(tools) == 1


def test_build_tools_respects_plan_features(monkeypatch) -> None:
    custom = [
        ToolConfig(
            name="t_pro",
            version=1,
            factory_key="t_pro",
            enabled_globally=True,
            required_feature=None,
            min_tier="pro",
            description="...",
        ),
        ToolConfig(
            name="t_free",
            version=1,
            factory_key="t_free",
            enabled_globally=True,
            required_feature=None,
            min_tier=None,
            description="...",
        ),
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    factories = {"t_pro": lambda: _dummy_tool, "t_free": lambda: _dummy_tool}

    free = build_tools(factories, {"tier_name": "free"})
    pro = build_tools(factories, {"tier_name": "pro"})

    assert len(free) == 1
    assert len(pro) == 2


# ── ToolConfig dataclass ──────────────────────────────────────────────────────


def test_tool_config_is_frozen() -> None:
    import dataclasses

    cfg = ToolConfig(
        name="x",
        version=1,
        factory_key="x",
        enabled_globally=True,
        required_feature=None,
        min_tier=None,
        description="",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.name = "y"  # type: ignore[misc]


def test_default_registry_has_seven_tools_post_a4() -> None:
    """Smoke-test do registry default: 6 v1 (4 base + compare_diagnoses + search_web) + 1 v2 (identify_crop)."""
    cfgs = get_registry()
    assert {c.name for c in cfgs} == {
        "deep_diagnose",
        "get_disease_info",
        "get_action_plan",
        "search_my_diagnoses",
        "compare_diagnoses",
        "search_web",
        "identify_crop",
    }
    # 6 tools v1 + 1 tool v2 (identify_crop dormente)
    v1_cfgs = [c for c in cfgs if c.version == 1]
    v2_cfgs = [c for c in cfgs if c.version == 2]
    assert len(v1_cfgs) == 6
    assert len(v2_cfgs) == 1
    # Base 4 sem required_feature/min_tier; compare/search_web têm gating
    by_name = {c.name: c for c in cfgs}
    base_tools = {"deep_diagnose", "get_disease_info", "get_action_plan", "search_my_diagnoses"}
    for n in base_tools:
        assert by_name[n].required_feature is None
        assert by_name[n].min_tier is None
    assert by_name["search_web"].required_feature == "search_web"
    assert by_name["search_web"].min_tier == "pro"
    assert by_name["compare_diagnoses"].min_tier == "enterprise"


def test_compare_diagnoses_gated_for_enterprise_only() -> None:
    """compare_diagnoses deve aparecer SO' pra tier enterprise."""
    free = set(get_active_tool_names({"tier_name": "free"}))
    pro = set(get_active_tool_names({"tier_name": "pro"}))
    enterprise = set(get_active_tool_names({"tier_name": "enterprise"}))

    assert "compare_diagnoses" not in free
    assert "compare_diagnoses" not in pro
    assert "compare_diagnoses" in enterprise


# ── identify_crop V2 — flag + tier + feature gating ──────────────────────────


def test_identify_crop_in_registry_with_v2_gating() -> None:
    """identify_crop deve estar registrado como V2 com gating Pro+ + feature."""
    cfgs = get_registry()
    icrop = next(c for c in cfgs if c.name == "identify_crop")

    assert icrop.version == 2
    assert icrop.factory_key == "identify_crop"
    assert icrop.required_feature == "identify_crop_auto"
    assert icrop.min_tier == "pro"


def test_identify_crop_dormant_by_default_flag_off() -> None:
    """Default (AGENT_ENABLE_IDENTIFY_CROP=false): tool nao deve aparecer
    em get_active_tool_names mesmo com tier Pro + feature ligada.
    """
    from app.config import settings

    # Sanity: o default da settings deve ser False (V2 dormente).
    assert settings.agent_enable_identify_crop is False

    pro_features = {
        "tier_name": "pro",
        "identify_crop_auto": True,
    }
    assert "identify_crop" not in get_active_tool_names(pro_features)


def test_identify_crop_enabled_globally_and_pro_with_feature_activates(
    monkeypatch,
) -> None:
    """Com flag on + tier Pro + feature, identify_crop deve ficar ativo."""
    # Substitui o registry pra simular AGENT_ENABLE_IDENTIFY_CROP=true
    custom = [
        ToolConfig(
            name="identify_crop",
            version=2,
            factory_key="identify_crop",
            enabled_globally=True,
            required_feature="identify_crop_auto",
            min_tier="pro",
            description="...",
        ),
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    pro_features = {"tier_name": "pro", "identify_crop_auto": True}
    assert get_active_tool_names(pro_features) == ["identify_crop"]


def test_identify_crop_blocked_for_free_tier_even_with_flag_and_feature(
    monkeypatch,
) -> None:
    """Free tier nao acessa V2 (min_tier=pro) mesmo com flag + feature on."""
    custom = [
        ToolConfig(
            name="identify_crop",
            version=2,
            factory_key="identify_crop",
            enabled_globally=True,
            required_feature="identify_crop_auto",
            min_tier="pro",
            description="...",
        ),
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    free_features = {"tier_name": "free", "identify_crop_auto": True}
    assert get_active_tool_names(free_features) == []


def test_identify_crop_blocked_without_feature_flag(monkeypatch) -> None:
    """Tier Pro sem ``identify_crop_auto=True`` no plano → bloqueada."""
    custom = [
        ToolConfig(
            name="identify_crop",
            version=2,
            factory_key="identify_crop",
            enabled_globally=True,
            required_feature="identify_crop_auto",
            min_tier="pro",
            description="...",
        ),
    ]
    monkeypatch.setattr(tool_registry, "get_registry", lambda: custom)

    # Pro mas sem feature do plano
    assert get_active_tool_names({"tier_name": "pro"}) == []
    assert (
        get_active_tool_names(
            {"tier_name": "pro", "identify_crop_auto": False}
        )
        == []
    )


def test_identify_crop_uses_settings_flag_in_registry() -> None:
    """A entry de identify_crop usa settings.agent_enable_identify_crop pra
    enabled_globally. O default (False) deixa a tool dormente — verificacao
    indireta via get_active_tool_names.
    """
    from app.config import settings

    cfgs = get_registry()
    icrop = next(c for c in cfgs if c.name == "identify_crop")
    # O kill-switch deve refletir o valor atual da settings.
    assert icrop.enabled_globally == settings.agent_enable_identify_crop
