from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class RegistrationPendingResponse(BaseModel):
    """Resposta do cadastro quando a verificação de e-mail está exigida.

    Devolvida com HTTP 202 — a conta existe mas ainda não tem token de acesso,
    porque o usuário só é ativado ao clicar no link enviado.
    """

    verification_required: bool = True
    email: str
    message: str = "Enviamos um link de confirmação para o seu e-mail."
