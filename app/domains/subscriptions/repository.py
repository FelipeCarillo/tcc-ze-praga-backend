from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.domains.subscriptions.dto import PlanDTO, SubscriptionDTO
from app.models.subscription_plan import SubscriptionPlan
from app.models.user_subscription import UserSubscription


class SubscriptionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def find_all_active_plans(self) -> list[PlanDTO]:
        result = await self._db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.is_active == True)  # noqa: E712
        )
        return [self._plan_to_dto(p) for p in result.scalars().all()]

    async def find_plan_by_name(self, name: str) -> PlanDTO | None:
        result = await self._db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.name == name)
        )
        plan = result.scalar_one_or_none()
        return self._plan_to_dto(plan) if plan else None

    async def find_plan_by_id(self, plan_id: str) -> PlanDTO | None:
        result = await self._db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.id == plan_id)
        )
        plan = result.scalar_one_or_none()
        return self._plan_to_dto(plan) if plan else None

    async def find_user_subscription(self, user_id: str) -> SubscriptionDTO | None:
        result = await self._db.execute(
            select(UserSubscription)
            .options(joinedload(UserSubscription.plan))
            .where(UserSubscription.user_id == user_id, UserSubscription.is_active == True)  # noqa: E712
        )
        sub = result.scalar_one_or_none()
        return self._sub_to_dto(sub) if sub else None

    async def upsert_user_subscription(self, user_id: str, plan_id: str) -> SubscriptionDTO:
        result = await self._db.execute(
            select(UserSubscription).where(UserSubscription.user_id == user_id)
        )
        sub = result.scalar_one_or_none()

        if sub:
            sub.plan_id = plan_id
            sub.is_active = True
        else:
            sub = UserSubscription(user_id=user_id, plan_id=plan_id)
            self._db.add(sub)

        await self._db.commit()
        await self._db.refresh(sub)

        # Re-fetch with joined plan
        result = await self._db.execute(
            select(UserSubscription)
            .options(joinedload(UserSubscription.plan))
            .where(UserSubscription.id == sub.id)
        )
        sub = result.scalar_one()
        return self._sub_to_dto(sub)

    @staticmethod
    def _plan_to_dto(plan: SubscriptionPlan) -> PlanDTO:
        return PlanDTO(
            id=plan.id,
            name=plan.name,
            display_name=plan.display_name,
            chat_daily_limit=plan.chat_daily_limit,
            inference_daily_limit=plan.inference_daily_limit,
            api_monthly_limit=plan.api_monthly_limit,
            is_active=plan.is_active,
            features=plan.features,
        )

    @staticmethod
    def _sub_to_dto(sub: UserSubscription) -> SubscriptionDTO:
        return SubscriptionDTO(
            id=sub.id,
            user_id=sub.user_id,
            plan=SubscriptionRepository._plan_to_dto(sub.plan),
            started_at=sub.started_at,
            expires_at=sub.expires_at,
            is_active=sub.is_active,
        )
