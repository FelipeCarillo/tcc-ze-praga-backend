from datetime import datetime

from pydantic import BaseModel, EmailStr


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

    model_config = {"from_attributes": True}
