from datetime import datetime

from pydantic import BaseModel, EmailStr

from app.domains.subscriptions.schemas import PlanResponse


class UpdateUserRequest(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None


class UserProfileResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    # Plano + features do usuario (TCC-049) — None quando sem subscription ativa.
    plan: PlanResponse | None = None

    model_config = {"from_attributes": True}
