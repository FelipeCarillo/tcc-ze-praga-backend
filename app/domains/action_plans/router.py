from fastapi import APIRouter, Depends

from app.core.dependencies import get_action_plan_service, get_current_user
from app.domains.action_plans.schemas import ActionPlanLevelResponse, ActionPlanResponse
from app.domains.action_plans.service import ActionPlanService
from app.shared.enums import ActionPlanLevelEnum

router = APIRouter(prefix="/action-plans", tags=["Action Plans"])


@router.get("/{disease_id}", response_model=ActionPlanResponse)
async def get_action_plan(
    disease_id: str,
    _: object = Depends(get_current_user),
    service: ActionPlanService = Depends(get_action_plan_service),
) -> ActionPlanResponse:
    return await service.get_by_disease(disease_id)


@router.get("/{disease_id}/{level}", response_model=ActionPlanLevelResponse)
async def get_action_plan_level(
    disease_id: str,
    level: ActionPlanLevelEnum,
    _: object = Depends(get_current_user),
    service: ActionPlanService = Depends(get_action_plan_service),
) -> ActionPlanLevelResponse:
    return await service.get_level(disease_id, level)
