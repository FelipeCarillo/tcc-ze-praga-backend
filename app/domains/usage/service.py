from app.core.exceptions import QuotaExceededError
from app.domains.subscriptions.repository import SubscriptionRepository
from app.domains.usage.repository import UsageRepository
from app.domains.usage.schemas import (
    FeatureUsageResponse,
    UsageHistoryItemResponse,
    UsageSummaryResponse,
)
from app.shared.enums import FeatureTypeEnum

# Default limits for users without a subscription (treated as free plan)
_FREE_LIMITS: dict[FeatureTypeEnum, int | None] = {
    FeatureTypeEnum.CHAT: 10,
    FeatureTypeEnum.INFERENCE: 5,
    FeatureTypeEnum.API: 0,
}


class UsageService:
    def __init__(
        self,
        usage_repo: UsageRepository,
        sub_repo: SubscriptionRepository,
    ) -> None:
        self._usage_repo = usage_repo
        self._sub_repo = sub_repo

    async def check_quota(self, user_id: str, feature: FeatureTypeEnum) -> None:
        limit = await self._get_limit(user_id, feature)
        if limit is None:
            return  # unlimited

        used = await self._count_usage(user_id, feature)
        if used >= limit:
            raise QuotaExceededError(feature, limit, used)

    async def record_usage(
        self,
        user_id: str,
        feature: FeatureTypeEnum,
        metadata: dict | None = None,
    ) -> None:
        await self._usage_repo.record(user_id, feature, metadata)

    async def get_summary(self, user_id: str) -> UsageSummaryResponse:
        chat_used = await self._count_usage(user_id, FeatureTypeEnum.CHAT)
        inference_used = await self._count_usage(user_id, FeatureTypeEnum.INFERENCE)
        api_used = await self._count_usage(user_id, FeatureTypeEnum.API)

        chat_limit = await self._get_limit(user_id, FeatureTypeEnum.CHAT)
        inference_limit = await self._get_limit(user_id, FeatureTypeEnum.INFERENCE)
        api_limit = await self._get_limit(user_id, FeatureTypeEnum.API)

        return UsageSummaryResponse(
            chat=FeatureUsageResponse.from_counts(chat_used, chat_limit),
            inference=FeatureUsageResponse.from_counts(inference_used, inference_limit),
            api=FeatureUsageResponse.from_counts(api_used, api_limit),
        )

    async def get_history(self, user_id: str, limit: int = 50) -> list[UsageHistoryItemResponse]:
        logs = await self._usage_repo.find_recent(user_id, limit)
        return [
            UsageHistoryItemResponse(id=log.id, feature=log.feature, used_at=log.used_at)
            for log in logs
        ]

    async def _get_limit(self, user_id: str, feature: FeatureTypeEnum) -> int | None:
        sub = await self._sub_repo.find_user_subscription(user_id)
        if not sub:
            return _FREE_LIMITS.get(feature, 0)

        if feature == FeatureTypeEnum.CHAT:
            return sub.plan.chat_daily_limit
        if feature == FeatureTypeEnum.INFERENCE:
            return sub.plan.inference_daily_limit
        if feature == FeatureTypeEnum.API:
            return sub.plan.api_monthly_limit
        return 0

    async def _count_usage(self, user_id: str, feature: FeatureTypeEnum) -> int:
        # API uses monthly count; chat and inference use daily count
        if feature == FeatureTypeEnum.API:
            return await self._usage_repo.count_this_month(user_id, feature)
        return await self._usage_repo.count_today(user_id, feature)
