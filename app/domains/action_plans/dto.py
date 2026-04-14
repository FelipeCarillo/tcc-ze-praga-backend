from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionPlanSourceDTO:
    id: str
    name: str
    detail: str
    url: str | None
    display_order: int


@dataclass(frozen=True)
class ActionPlanLevelDTO:
    disease_id: str
    level: str
    actions: list[str]


@dataclass(frozen=True)
class ActionPlanDTO:
    disease_id: str
    levels: list[ActionPlanLevelDTO] = field(default_factory=list)
    sources: list[ActionPlanSourceDTO] = field(default_factory=list)
