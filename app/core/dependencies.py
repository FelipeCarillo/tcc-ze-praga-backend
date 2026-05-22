from collections.abc import AsyncGenerator

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.db.database import AsyncSessionLocal
from app.domains.auth.dto import UserDTO
from app.domains.auth.repository import UserRepository
from app.domains.auth.service import AuthService
from app.shared.enums import FeatureTypeEnum

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Database ──────────────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# ── Repositories ──────────────────────────────────────────────────────────────

def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_subscription_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.subscriptions.repository import SubscriptionRepository
    return SubscriptionRepository(db)


def get_usage_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.usage.repository import UsageRepository
    return UsageRepository(db)


def get_diagnosis_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.diagnoses.repository import DiagnosisRepository
    return DiagnosisRepository(db)


def get_action_plan_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.action_plans.repository import ActionPlanRepository
    return ActionPlanRepository(db)


# ── Services ──────────────────────────────────────────────────────────────────

def get_auth_service(
    repo: UserRepository = Depends(get_user_repository),
) -> AuthService:
    return AuthService(repo)


def get_user_service(repo: UserRepository = Depends(get_user_repository)):  # type: ignore[no-untyped-def]
    from app.domains.users.service import UserService
    return UserService(repo)


def get_subscription_service(repo=Depends(get_subscription_repository)):  # type: ignore[no-untyped-def]
    from app.domains.subscriptions.service import SubscriptionService
    return SubscriptionService(repo)


def get_usage_service(  # type: ignore[no-untyped-def]
    usage_repo=Depends(get_usage_repository),
    sub_repo=Depends(get_subscription_repository),
):
    from app.domains.usage.service import UsageService
    return UsageService(usage_repo, sub_repo)


def get_diagnosis_service(  # type: ignore[no-untyped-def]
    repo=Depends(get_diagnosis_repository),
):
    from app.domains.diagnoses.service import DiagnosisService
    return DiagnosisService(repo)


def get_action_plan_service(repo=Depends(get_action_plan_repository)):  # type: ignore[no-untyped-def]
    from app.domains.action_plans.service import ActionPlanService
    return ActionPlanService(repo)


def get_inference_service():  # type: ignore[no-untyped-def]
    from app.domains.inference.service import InferenceService
    return InferenceService()


# ── Auth ──────────────────────────────────────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    repo: UserRepository = Depends(get_user_repository),
) -> UserDTO:
    payload = decode_access_token(token)
    user = await repo.find_by_id(payload["sub"])
    if not user or not user.is_active:
        raise UnauthorizedError("User not found or inactive")
    return user


# ── Quota ─────────────────────────────────────────────────────────────────────

def require_quota(feature: FeatureTypeEnum):  # type: ignore[no-untyped-def]
    async def _dependency(
        current_user: UserDTO = Depends(get_current_user),
        usage_svc=Depends(get_usage_service),
    ) -> UserDTO:
        await usage_svc.check_quota(current_user.id, feature)
        return current_user

    return _dependency
