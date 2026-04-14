"""Tests for app/domains/usage/service.py — UsageService (most complex module)."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import QuotaExceededError
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_plan_dto, make_subscription_dto, make_usage_log_dto


@pytest.fixture
def usage_repo():
    repo = AsyncMock()
    repo.count_today = AsyncMock(return_value=0)
    repo.count_this_month = AsyncMock(return_value=0)
    repo.record = AsyncMock(return_value=make_usage_log_dto())
    repo.find_recent = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def sub_repo():
    repo = AsyncMock()
    repo.find_user_subscription = AsyncMock(return_value=None)
    return repo


# ── check_quota — unlimited plan (None limit) ─────────────────────────────────

async def test_check_quota_unlimited_plan(usage_repo, sub_repo):
    sub = make_subscription_dto(
        plan=make_plan_dto(
            chat_daily_limit=None,
            inference_daily_limit=None,
            api_monthly_limit=None,
        )
    )
    sub_repo.find_user_subscription.return_value = sub
    svc = UsageService(usage_repo, sub_repo)
    # Should not raise
    await svc.check_quota("user-1", FeatureTypeEnum.INFERENCE)
    usage_repo.count_today.assert_not_called()


# ── check_quota — under limit ─────────────────────────────────────────────────

async def test_check_quota_under_limit(usage_repo, sub_repo):
    usage_repo.count_today.return_value = 3  # free plan: limit=5
    svc = UsageService(usage_repo, sub_repo)
    await svc.check_quota("user-1", FeatureTypeEnum.INFERENCE)  # no raise


# ── check_quota — exactly at limit ───────────────────────────────────────────

async def test_check_quota_at_limit_raises(usage_repo, sub_repo):
    usage_repo.count_today.return_value = 5  # used == limit (5)
    svc = UsageService(usage_repo, sub_repo)
    with pytest.raises(QuotaExceededError) as exc_info:
        await svc.check_quota("user-1", FeatureTypeEnum.INFERENCE)
    assert exc_info.value.limit == 5
    assert exc_info.value.used == 5


# ── check_quota — over limit ──────────────────────────────────────────────────

async def test_check_quota_over_limit_raises(usage_repo, sub_repo):
    usage_repo.count_today.return_value = 12  # free chat limit = 10
    svc = UsageService(usage_repo, sub_repo)
    with pytest.raises(QuotaExceededError) as exc_info:
        await svc.check_quota("user-1", FeatureTypeEnum.CHAT)
    assert exc_info.value.feature == FeatureTypeEnum.CHAT


# ── check_quota — API (monthly) ───────────────────────────────────────────────

async def test_check_quota_api_zero_limit(usage_repo, sub_repo):
    """Free plan has api_monthly_limit=0, any usage should fail."""
    usage_repo.count_this_month.return_value = 0
    svc = UsageService(usage_repo, sub_repo)
    with pytest.raises(QuotaExceededError):
        await svc.check_quota("user-1", FeatureTypeEnum.API)


# ── record_usage ──────────────────────────────────────────────────────────────

async def test_record_usage_delegates(usage_repo, sub_repo):
    svc = UsageService(usage_repo, sub_repo)
    await svc.record_usage("user-1", FeatureTypeEnum.INFERENCE, {"k": "v"})
    usage_repo.record.assert_awaited_once_with("user-1", FeatureTypeEnum.INFERENCE, {"k": "v"})


async def test_record_usage_no_metadata(usage_repo, sub_repo):
    svc = UsageService(usage_repo, sub_repo)
    await svc.record_usage("user-1", FeatureTypeEnum.CHAT)
    usage_repo.record.assert_awaited_once_with("user-1", FeatureTypeEnum.CHAT, None)


# ── get_summary ───────────────────────────────────────────────────────────────

async def test_get_summary_no_subscription(usage_repo, sub_repo):
    usage_repo.count_today.return_value = 2
    usage_repo.count_this_month.return_value = 0
    svc = UsageService(usage_repo, sub_repo)
    summary = await svc.get_summary("user-1")
    assert summary.chat.used == 2
    assert summary.chat.limit == 10  # free plan
    assert summary.chat.remaining == 8
    assert summary.inference.limit == 5
    assert summary.api.limit == 0


async def test_get_summary_unlimited_plan(usage_repo, sub_repo):
    sub = make_subscription_dto(
        plan=make_plan_dto(
            chat_daily_limit=None,
            inference_daily_limit=None,
            api_monthly_limit=None,
        )
    )
    sub_repo.find_user_subscription.return_value = sub
    svc = UsageService(usage_repo, sub_repo)
    summary = await svc.get_summary("user-1")
    assert summary.chat.limit is None
    assert summary.chat.remaining is None


# ── get_history ───────────────────────────────────────────────────────────────

async def test_get_history_empty(usage_repo, sub_repo):
    usage_repo.find_recent.return_value = []
    svc = UsageService(usage_repo, sub_repo)
    result = await svc.get_history("user-1")
    assert result == []


async def test_get_history_with_logs(usage_repo, sub_repo):
    logs = [make_usage_log_dto() for _ in range(3)]
    usage_repo.find_recent.return_value = logs
    svc = UsageService(usage_repo, sub_repo)
    result = await svc.get_history("user-1", limit=10)
    assert len(result) == 3
    usage_repo.find_recent.assert_awaited_with("user-1", 10)


# ── _get_limit — all branches ─────────────────────────────────────────────────

async def test_get_limit_no_subscription_chat(usage_repo, sub_repo):
    svc = UsageService(usage_repo, sub_repo)
    limit = await svc._get_limit("user-1", FeatureTypeEnum.CHAT)
    assert limit == 10


async def test_get_limit_no_subscription_inference(usage_repo, sub_repo):
    svc = UsageService(usage_repo, sub_repo)
    limit = await svc._get_limit("user-1", FeatureTypeEnum.INFERENCE)
    assert limit == 5


async def test_get_limit_no_subscription_api(usage_repo, sub_repo):
    svc = UsageService(usage_repo, sub_repo)
    limit = await svc._get_limit("user-1", FeatureTypeEnum.API)
    assert limit == 0


async def test_get_limit_pro_plan_chat(usage_repo, sub_repo):
    sub = make_subscription_dto(plan=make_plan_dto(chat_daily_limit=None))
    sub_repo.find_user_subscription.return_value = sub
    svc = UsageService(usage_repo, sub_repo)
    limit = await svc._get_limit("user-1", FeatureTypeEnum.CHAT)
    assert limit is None


async def test_get_limit_pro_plan_inference(usage_repo, sub_repo):
    sub = make_subscription_dto(plan=make_plan_dto(inference_daily_limit=None))
    sub_repo.find_user_subscription.return_value = sub
    svc = UsageService(usage_repo, sub_repo)
    limit = await svc._get_limit("user-1", FeatureTypeEnum.INFERENCE)
    assert limit is None


async def test_get_limit_pro_plan_api(usage_repo, sub_repo):
    sub = make_subscription_dto(plan=make_plan_dto(api_monthly_limit=500))
    sub_repo.find_user_subscription.return_value = sub
    svc = UsageService(usage_repo, sub_repo)
    limit = await svc._get_limit("user-1", FeatureTypeEnum.API)
    assert limit == 500


async def test_get_limit_unknown_feature_returns_zero(usage_repo, sub_repo):
    """The fallback return 0 when subscription exists but feature is unknown."""
    sub = make_subscription_dto(plan=make_plan_dto())
    sub_repo.find_user_subscription.return_value = sub
    svc = UsageService(usage_repo, sub_repo)
    # Pass a raw string value that doesn't match any FeatureTypeEnum branch
    limit = await svc._get_limit("user-1", "unknown_feature")  # type: ignore[arg-type]
    assert limit == 0


# ── _count_usage — API uses monthly, others use daily ─────────────────────────

async def test_count_usage_chat_uses_daily(usage_repo, sub_repo):
    usage_repo.count_today.return_value = 7
    svc = UsageService(usage_repo, sub_repo)
    count = await svc._count_usage("user-1", FeatureTypeEnum.CHAT)
    assert count == 7
    usage_repo.count_today.assert_awaited()
    usage_repo.count_this_month.assert_not_called()


async def test_count_usage_api_uses_monthly(usage_repo, sub_repo):
    usage_repo.count_this_month.return_value = 99
    svc = UsageService(usage_repo, sub_repo)
    count = await svc._count_usage("user-1", FeatureTypeEnum.API)
    assert count == 99
    usage_repo.count_this_month.assert_awaited()
    usage_repo.count_today.assert_not_called()
