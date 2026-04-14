from enum import StrEnum


class SeverityEnum(StrEnum):
    ALTA = "alta"
    MEDIA = "media"
    BAIXA = "baixa"
    NENHUMA = "nenhuma"


class ActionPlanLevelEnum(StrEnum):
    ESSENCIAL = "essencial"
    CAMPO = "campo"
    ESPECIALISTA = "especialista"


class ModelEnum(StrEnum):
    RESNET50 = "resnet50"
    EFFICIENTNET = "efficientnet"
    VIT = "vit"
    ENSEMBLE = "ensemble"


class FeatureTypeEnum(StrEnum):
    CHAT = "chat"
    INFERENCE = "inference"
    API = "api"


class SubscriptionPlanNameEnum(StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
