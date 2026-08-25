"""Rotas de plano de acao — leitura gated por ``plan_features``.

O gate por nivel (Free=essencial, Pro=+campo, Enterprise=+especialista) existia
so' na tool ``get_action_plan`` do agente; este endpoint REST devolvia todos os
niveis pra qualquer plano. Como o frontend passou a consumi-lo direto (o card de
diagnostico e a pagina de detalhe), o gate precisa valer aqui tambem — senao a
mesma feature fica gated no chat e aberta na tela.
"""

from fastapi import APIRouter, Depends

from app.core.dependencies import (
    get_action_plan_service,
    get_current_user,
    get_plan_features,
)
from app.core.exceptions import ForbiddenError
from app.domains.action_plans.schemas import ActionPlanLevelResponse, ActionPlanResponse
from app.domains.action_plans.service import ActionPlanService
from app.domains.subscriptions.features import PlanFeatures
from app.shared.enums import ActionPlanLevelEnum

router = APIRouter(prefix="/action-plans", tags=["Action Plans"])


@router.get("/{disease_id}", response_model=ActionPlanResponse)
async def get_action_plan(
    disease_id: str,
    _: object = Depends(get_current_user),
    service: ActionPlanService = Depends(get_action_plan_service),
    plan_features: PlanFeatures = Depends(get_plan_features),
) -> ActionPlanResponse:
    """Plano de acao completo, com os niveis filtrados pelo plano do usuario."""
    plan = await service.get_by_disease(disease_id)
    allowed = set(plan_features.action_plan_levels)
    plan.levels = [lvl for lvl in plan.levels if lvl.level in allowed]
    plan.allowed_levels = [
        ActionPlanLevelEnum(lvl) for lvl in plan_features.action_plan_levels
    ]
    return plan


@router.get("/{disease_id}/{level}", response_model=ActionPlanLevelResponse)
async def get_action_plan_level(
    disease_id: str,
    level: ActionPlanLevelEnum,
    _: object = Depends(get_current_user),
    service: ActionPlanService = Depends(get_action_plan_service),
    plan_features: PlanFeatures = Depends(get_plan_features),
) -> ActionPlanLevelResponse:
    """Um nivel especifico. 403 quando o plano nao cobre o nivel pedido.

    Diferente do chat — onde rebaixar silenciosamente e' melhor que quebrar o
    turno do agente —, aqui o cliente pediu um nivel nominal: negar e' a resposta
    honesta.
    """
    if level not in set(plan_features.action_plan_levels):
        permitidos = ", ".join(plan_features.action_plan_levels)
        raise ForbiddenError(
            f"Nivel '{level}' nao disponivel no seu plano. Permitidos: {permitidos}."
        )
    return await service.get_level(disease_id, level)
