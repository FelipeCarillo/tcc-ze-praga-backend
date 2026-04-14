from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.action_plans.dto import ActionPlanDTO, ActionPlanLevelDTO, ActionPlanSourceDTO
from app.models.action_plan import ActionPlan
from app.models.action_plan_source import ActionPlanSource


class ActionPlanRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def find_by_disease(self, disease_id: str) -> ActionPlanDTO | None:
        levels_result = await self._db.execute(
            select(ActionPlan).where(ActionPlan.disease_id == disease_id)
        )
        levels = levels_result.scalars().all()
        if not levels:
            return None

        sources_result = await self._db.execute(
            select(ActionPlanSource)
            .where(ActionPlanSource.disease_id == disease_id)
            .order_by(ActionPlanSource.display_order)
        )
        sources = sources_result.scalars().all()

        return ActionPlanDTO(
            disease_id=disease_id,
            levels=[
                ActionPlanLevelDTO(
                    disease_id=ap.disease_id,
                    level=ap.level,
                    actions=ap.actions,
                )
                for ap in levels
            ],
            sources=[
                ActionPlanSourceDTO(
                    id=s.id,
                    name=s.name,
                    detail=s.detail,
                    url=s.url,
                    display_order=s.display_order,
                )
                for s in sources
            ],
        )

    async def find_level(self, disease_id: str, level: str) -> ActionPlanLevelDTO | None:
        result = await self._db.execute(
            select(ActionPlan).where(
                ActionPlan.disease_id == disease_id,
                ActionPlan.level == level,
            )
        )
        ap = result.scalar_one_or_none()
        if not ap:
            return None
        return ActionPlanLevelDTO(
            disease_id=ap.disease_id,
            level=ap.level,
            actions=ap.actions,
        )
