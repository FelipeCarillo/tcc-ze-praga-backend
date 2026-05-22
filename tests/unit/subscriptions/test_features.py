"""Tests for app/domains/subscriptions/features.py — PlanFeatures (TCC-049)."""

from __future__ import annotations

import pytest

from app.domains.subscriptions.features import (
    ENTERPRISE_FEATURES,
    FEATURES_BY_PLAN_NAME,
    FREE_FEATURES,
    PRO_FEATURES,
    PlanFeatures,
)


# ── defaults / parsing ───────────────────────────────────────────────────────


def test_free_features_defaults() -> None:
    """Free tier deve usar gpt-4o-mini + apenas resnet50 + nivel essencial."""
    assert FREE_FEATURES.tier_name == "free"
    assert FREE_FEATURES.llm_model == "gpt-4o-mini"
    assert FREE_FEATURES.diagnosis_models == ["resnet50"]
    assert FREE_FEATURES.action_plan_levels == ["essencial"]
    assert FREE_FEATURES.allowed_crops is None
    assert FREE_FEATURES.search_web is False
    assert FREE_FEATURES.search_scientific is False
    assert FREE_FEATURES.api_access is False
    assert FREE_FEATURES.export_diagnoses is False
    assert FREE_FEATURES.multi_account is False


def test_pro_features_defaults() -> None:
    """Pro tier: gpt-4o + multi-modelo + essencial+campo + soja only."""
    assert PRO_FEATURES.tier_name == "pro"
    assert PRO_FEATURES.llm_model == "gpt-4o"
    assert "resnet50" in PRO_FEATURES.diagnosis_models
    assert "vit" in PRO_FEATURES.diagnosis_models
    assert "ensemble" not in PRO_FEATURES.diagnosis_models  # ensemble = enterprise only
    assert "essencial" in PRO_FEATURES.action_plan_levels
    assert "campo" in PRO_FEATURES.action_plan_levels
    assert "especialista" not in PRO_FEATURES.action_plan_levels
    assert PRO_FEATURES.allowed_crops == ["soja"]
    assert PRO_FEATURES.search_web is True
    assert PRO_FEATURES.export_diagnoses is True
    assert PRO_FEATURES.api_access is False  # api access = enterprise only


def test_enterprise_features_defaults() -> None:
    """Enterprise: tudo destravado, todos os crops."""
    assert ENTERPRISE_FEATURES.tier_name == "enterprise"
    assert ENTERPRISE_FEATURES.llm_model == "gpt-4o"
    assert "ensemble" in ENTERPRISE_FEATURES.diagnosis_models
    assert "especialista" in ENTERPRISE_FEATURES.action_plan_levels
    assert ENTERPRISE_FEATURES.allowed_crops is None  # todos os cultivos
    assert ENTERPRISE_FEATURES.search_web is True
    assert ENTERPRISE_FEATURES.search_scientific is True
    assert ENTERPRISE_FEATURES.identify_crop_auto is True
    assert ENTERPRISE_FEATURES.api_access is True
    assert ENTERPRISE_FEATURES.export_diagnoses is True
    assert ENTERPRISE_FEATURES.multi_account is True


def test_lookup_by_plan_name() -> None:
    assert FEATURES_BY_PLAN_NAME["free"] is FREE_FEATURES
    assert FEATURES_BY_PLAN_NAME["pro"] is PRO_FEATURES
    assert FEATURES_BY_PLAN_NAME["enterprise"] is ENTERPRISE_FEATURES


def test_parse_from_dict_roundtrip() -> None:
    """PlanFeatures consegue ler/serializar do dict do banco."""
    payload = FREE_FEATURES.model_dump()
    rebuilt = PlanFeatures(**payload)
    assert rebuilt == FREE_FEATURES


def test_custom_features_construction() -> None:
    """Permite criar features customizadas sem usar os defaults."""
    custom = PlanFeatures(
        tier_name="custom",
        llm_model="claude-3-opus",
        diagnosis_models=["my-model"],
        action_plan_levels=["essencial", "especialista"],
        allowed_crops=["soja", "milho"],
        api_access=True,
    )
    assert custom.tier_name == "custom"
    assert custom.llm_model == "claude-3-opus"
    assert custom.api_access is True
    assert custom.search_web is False  # default not overridden


def test_required_field_validation() -> None:
    """tier_name nao tem default e deve ser obrigatorio."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PlanFeatures()  # type: ignore[call-arg]


# ── signature ────────────────────────────────────────────────────────────────


def test_signature_deterministic() -> None:
    """Mesma config -> mesma signature."""
    sig1 = FREE_FEATURES.signature()
    sig2 = FREE_FEATURES.signature()
    assert sig1 == sig2


def test_signature_different_for_different_features() -> None:
    """Free/Pro/Enterprise devem gerar signatures distintas."""
    sigs = {
        FREE_FEATURES.signature(),
        PRO_FEATURES.signature(),
        ENTERPRISE_FEATURES.signature(),
    }
    assert len(sigs) == 3


def test_signature_length() -> None:
    sig = PRO_FEATURES.signature()
    assert len(sig) == 16
    assert all(c in "0123456789abcdef" for c in sig)


def test_signature_changes_when_field_changes() -> None:
    """Mudar um campo deve mudar a signature."""
    a = PlanFeatures(tier_name="free", llm_model="gpt-4o-mini")
    b = PlanFeatures(tier_name="free", llm_model="gpt-4o")
    assert a.signature() != b.signature()


def test_signature_field_order_invariant() -> None:
    """A ordem em que os kwargs sao passados nao deve afetar a signature."""
    a = PlanFeatures(tier_name="free", llm_model="x")
    b = PlanFeatures(llm_model="x", tier_name="free")
    assert a.signature() == b.signature()
