from pydantic import BaseModel

from app.shared.enums import ActionPlanLevelEnum


class SourceResponse(BaseModel):
    id: str
    name: str
    detail: str
    url: str | None
    display_order: int


class ActionPlanLevelResponse(BaseModel):
    level: ActionPlanLevelEnum
    actions: list[str]


class ActionPlanResponse(BaseModel):
    disease_id: str
    levels: list[ActionPlanLevelResponse]
    sources: list[SourceResponse]
