from app.core.exceptions import NotFoundError
from app.domains.subscriptions.dto import PlanDTO, SubscriptionDTO
from app.domains.subscriptions.repository import SubscriptionRepository
from app.domains.subscriptions.schemas import (
    PlanResponse,
    SubscribeRequest,
    SubscriptionResponse,
)


class SubscriptionService:
    def __init__(self, repo: SubscriptionRepository) -> None:
        self._repo = repo

    async def list_plans(self) -> list[PlanResponse]:
        plans = await self._repo.find_all_active_plans()
        return [self._plan_to_response(p) for p in plans]

    async def get_user_subscription(self, user_id: str) -> SubscriptionResponse | None:
        sub = await self._repo.find_user_subscription(user_id)
        return self._sub_to_response(sub) if sub else None

    async def subscribe(self, user_id: str, request: SubscribeRequest) -> SubscriptionResponse:
        plan = await self._repo.find_plan_by_name(request.plan_name)
        if not plan:
            raise NotFoundError("Plan", request.plan_name)

        sub = await self._repo.upsert_user_subscription(user_id, plan.id)
        return self._sub_to_response(sub)

    @staticmethod
    def _plan_to_response(plan: PlanDTO) -> PlanResponse:
        return PlanResponse(
            id=plan.id,
            name=plan.name,
            display_name=plan.display_name,
            chat_daily_limit=plan.chat_daily_limit,
            inference_daily_limit=plan.inference_daily_limit,
            api_monthly_limit=plan.api_monthly_limit,
            features=plan.features,
        )

    @staticmethod
    def _sub_to_response(sub: SubscriptionDTO) -> SubscriptionResponse:
        return SubscriptionResponse(
            id=sub.id,
            plan=SubscriptionService._plan_to_response(sub.plan),
            started_at=sub.started_at,
            expires_at=sub.expires_at,
            is_active=sub.is_active,
        )
