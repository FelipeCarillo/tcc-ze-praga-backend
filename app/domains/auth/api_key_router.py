"""Router pra ``/api/v1/auth/api-keys`` — CRUD restrito a tier Enterprise.

Endpoints:
- ``POST   /api/v1/auth/api-keys`` -> ``ApiKeyCreatedResponse`` (plain key 1x)
- ``GET    /api/v1/auth/api-keys`` -> ``list[ApiKeyResponse]``
- ``DELETE /api/v1/auth/api-keys/{id}`` -> 204
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_api_key_service, require_tier_enterprise
from app.domains.auth.api_key_dto import ApiKeyDTO
from app.domains.auth.api_key_schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyResponse,
)
from app.domains.auth.api_key_service import ApiKeyService
from app.domains.auth.dto import UserDTO

router = APIRouter(prefix="/auth/api-keys", tags=["API Keys"])


@router.post(
    "",
    response_model=ApiKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    body: ApiKeyCreateRequest,
    current_user: UserDTO = Depends(require_tier_enterprise),
    service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyCreatedResponse:
    """Gera uma nova API key. O plain text **so'** vem nesse retorno."""
    dto, plain_key = await service.create(current_user.id, body.name)
    return ApiKeyCreatedResponse(
        id=dto.id,
        name=dto.name,
        key=plain_key,
        key_prefix=dto.key_prefix,
        scopes=list(dto.scopes),
        created_at=dto.created_at,
    )


@router.get("", response_model=list[ApiKeyResponse])
async def list_api_keys(
    current_user: UserDTO = Depends(require_tier_enterprise),
    service: ApiKeyService = Depends(get_api_key_service),
) -> list[ApiKeyResponse]:
    keys = await service.list_for_user(current_user.id)
    return [_to_response(k) for k in keys]


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_api_key(
    key_id: str,
    current_user: UserDTO = Depends(require_tier_enterprise),
    service: ApiKeyService = Depends(get_api_key_service),
) -> None:
    ok = await service.revoke(current_user.id, key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="API key not found")


def _to_response(dto: ApiKeyDTO) -> ApiKeyResponse:
    return ApiKeyResponse(
        id=dto.id,
        name=dto.name,
        key_prefix=dto.key_prefix,
        scopes=list(dto.scopes),
        is_active=dto.is_active,
        created_at=dto.created_at,
        last_used_at=dto.last_used_at,
        revoked_at=dto.revoked_at,
    )
