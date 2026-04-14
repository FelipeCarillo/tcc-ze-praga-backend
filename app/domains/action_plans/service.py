from app.core.exceptions import NotFoundError
from app.domains.action_plans.dto import ActionPlanDTO
from app.domains.action_plans.repository import ActionPlanRepository
from app.domains.action_plans.schemas import (
    ActionPlanLevelResponse,
    ActionPlanResponse,
    SourceResponse,
)
from app.shared.enums import ActionPlanLevelEnum


class ActionPlanService:
    def __init__(self, repo: ActionPlanRepository) -> None:
        self._repo = repo

    async def get_by_disease(self, disease_id: str) -> ActionPlanResponse:
        plan = await self._repo.find_by_disease(disease_id)
        if not plan:
            raise NotFoundError("ActionPlan", disease_id)
        return self._to_response(plan)

    async def get_level(
        self, disease_id: str, level: ActionPlanLevelEnum
    ) -> ActionPlanLevelResponse:
        level_dto = await self._repo.find_level(disease_id, level)
        if not level_dto:
            raise NotFoundError("ActionPlan", f"{disease_id}/{level}")
        return ActionPlanLevelResponse(
            level=ActionPlanLevelEnum(level_dto.level), actions=level_dto.actions
        )

    @staticmethod
    def _to_response(plan: ActionPlanDTO) -> ActionPlanResponse:
        return ActionPlanResponse(
            disease_id=plan.disease_id,
            levels=[
                ActionPlanLevelResponse(
                    level=ActionPlanLevelEnum(lvl.level),
                    actions=lvl.actions,
                )
                for lvl in plan.levels
            ],
            sources=[
                SourceResponse(
                    id=s.id,
                    name=s.name,
                    detail=s.detail,
                    url=s.url,
                    display_order=s.display_order,
                )
                for s in plan.sources
            ],
        )
