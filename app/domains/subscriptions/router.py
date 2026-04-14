from fastapi import APIRouter, Depends

from app.core.dependencies import get_current_user, get_subscription_service
from app.domains.auth.dto import UserDTO
from app.domains.subscriptions.schemas import PlanResponse, SubscribeRequest, SubscriptionResponse
from app.domains.subscriptions.service import SubscriptionService

router = APIRouter(prefix="/subscriptions", tags=["Subscriptions"])


@router.get("/plans", response_model=list[PlanResponse])
async def list_plans(
    service: SubscriptionService = Depends(get_subscription_service),
) -> list[PlanResponse]:
    return await service.list_plans()


@router.get("/me", response_model=SubscriptionResponse | None)
async def get_my_subscription(
    current_user: UserDTO = Depends(get_current_user),
    service: SubscriptionService = Depends(get_subscription_service),
) -> SubscriptionResponse | None:
    return await service.get_user_subscription(current_user.id)


@router.post("/me", response_model=SubscriptionResponse, status_code=201)
async def subscribe(
    body: SubscribeRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: SubscriptionService = Depends(get_subscription_service),
) -> SubscriptionResponse:
    return await service.subscribe(current_user.id, body)
