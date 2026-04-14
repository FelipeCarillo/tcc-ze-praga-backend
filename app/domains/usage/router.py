from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_current_user, get_usage_service
from app.domains.auth.dto import UserDTO
from app.domains.usage.schemas import UsageHistoryItemResponse, UsageSummaryResponse
from app.domains.usage.service import UsageService

router = APIRouter(prefix="/usage", tags=["Usage"])


@router.get("/me", response_model=UsageSummaryResponse)
async def get_my_usage(
    current_user: UserDTO = Depends(get_current_user),
    service: UsageService = Depends(get_usage_service),
) -> UsageSummaryResponse:
    return await service.get_summary(current_user.id)


@router.get("/me/history", response_model=list[UsageHistoryItemResponse])
async def get_my_usage_history(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: UserDTO = Depends(get_current_user),
    service: UsageService = Depends(get_usage_service),
) -> list[UsageHistoryItemResponse]:
    return await service.get_history(current_user.id, limit)
