from typing import TYPE_CHECKING

from app.core.exceptions import ConflictError, NotFoundError
from app.domains.auth.dto import UserDTO
from app.domains.auth.repository import UserRepository
from app.domains.subscriptions.schemas import PlanResponse
from app.domains.users.schemas import UpdateUserRequest, UserProfileResponse

if TYPE_CHECKING:
    from app.domains.subscriptions.repository import SubscriptionRepository


class UserService:
    def __init__(
        self,
        user_repo: UserRepository,
        sub_repo: "SubscriptionRepository | None" = None,
    ) -> None:
        self._user_repo = user_repo
        self._sub_repo = sub_repo

    async def get_profile(self, user_id: str) -> UserProfileResponse:
        user = await self._user_repo.find_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        plan_response = await self._resolve_plan(user_id)
        return self._to_profile(user, plan_response)

    async def update_profile(self, user_id: str, request: UpdateUserRequest) -> UserProfileResponse:
        if request.email:
            existing = await self._user_repo.find_by_email(request.email)
            if existing and existing.id != user_id:
                raise ConflictError("Email already in use")

        updates = request.model_dump(exclude_none=True)
        user = await self._user_repo.update(user_id, **updates)
        if not user:
            raise NotFoundError("User", user_id)
        plan_response = await self._resolve_plan(user_id)
        return self._to_profile(user, plan_response)

    async def delete_account(self, user_id: str) -> None:
        await self._user_repo.soft_delete(user_id)

    async def _resolve_plan(self, user_id: str) -> PlanResponse | None:
        if self._sub_repo is None:
            return None
        sub = await self._sub_repo.find_user_subscription(user_id)
        if sub is None:
            return None
        plan = sub.plan
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
    def _to_profile(
        user: UserDTO,
        plan: PlanResponse | None = None,
    ) -> UserProfileResponse:
        return UserProfileResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
            plan=plan,
        )
