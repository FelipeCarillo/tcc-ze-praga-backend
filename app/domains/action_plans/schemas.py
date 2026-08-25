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
    """Plano de acao ja filtrado pelos niveis que o plano do usuario libera.

    ``allowed_levels`` acompanha a resposta pra UI conseguir mostrar os niveis
    bloqueados como upsell — sem ele o cliente nao teria como distinguir
    "nivel nao existe pra essa doenca" de "nivel existe mas seu plano nao da".
    """

    disease_id: str
    levels: list[ActionPlanLevelResponse]
    sources: list[SourceResponse]
    allowed_levels: list[ActionPlanLevelEnum] = []
