"""Tests for app/shared/enums.py — enum values and StrEnum behavior."""

from app.shared.enums import (
    ActionPlanLevelEnum,
    FeatureTypeEnum,
    ModelEnum,
    SeverityEnum,
    SubscriptionPlanNameEnum,
)


def test_severity_values():
    assert SeverityEnum.ALTA == "alta"
    assert SeverityEnum.MEDIA == "media"
    assert SeverityEnum.BAIXA == "baixa"
    assert SeverityEnum.NENHUMA == "nenhuma"


def test_action_plan_level_values():
    assert ActionPlanLevelEnum.ESSENCIAL == "essencial"
    assert ActionPlanLevelEnum.CAMPO == "campo"
    assert ActionPlanLevelEnum.ESPECIALISTA == "especialista"


def test_model_values():
    assert ModelEnum.RESNET50 == "resnet50"
    assert ModelEnum.EFFICIENTNET == "efficientnet"
    assert ModelEnum.VIT == "vit"
    assert ModelEnum.ENSEMBLE == "ensemble"


def test_feature_type_values():
    assert FeatureTypeEnum.CHAT == "chat"
    assert FeatureTypeEnum.INFERENCE == "inference"
    assert FeatureTypeEnum.API == "api"


def test_subscription_plan_name_values():
    assert SubscriptionPlanNameEnum.FREE == "free"
    assert SubscriptionPlanNameEnum.PRO == "pro"
    assert SubscriptionPlanNameEnum.ENTERPRISE == "enterprise"


def test_severity_is_str():
    assert isinstance(SeverityEnum.ALTA, str)


def test_feature_type_from_string():
    assert FeatureTypeEnum("inference") is FeatureTypeEnum.INFERENCE
