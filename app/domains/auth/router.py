from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import settings
from app.core.dependencies import get_auth_service, get_current_user
from app.core.exceptions import UnauthorizedError
from app.core.rate_limit import RateLimit
from app.domains.auth.dto import UserDTO
from app.domains.auth.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
)
from app.domains.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])

# Limites por IP (TCC-091). Rotas que disparam e-mail são as mais apertadas —
# além do custo, servem de amplificador: sem freio, um script transforma o
# endpoint numa máquina de mandar mensagem para caixa de entrada alheia.
_LIMITE_LOGIN = RateLimit(10, 300, nome="auth:login")
_LIMITE_CADASTRO = RateLimit(5, 3600, nome="auth:register")
_LIMITE_EMAIL = RateLimit(3, 3600, nome="auth:email")
_LIMITE_RESET = RateLimit(10, 3600, nome="auth:reset")


@router.post(
    "/register",
    status_code=201,
    response_model=None,
    dependencies=[Depends(_LIMITE_CADASTRO)],
)
async def register(
    body: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse | JSONResponse:
    """Cria a conta.

    Devolve **201 + token** quando o cadastro é direto e **202** quando a
    verificação de e-mail está exigida (TCC-090) — nesse caso não há token,
    porque a conta ainda não está ativa.
    """
    result = await service.register(body)
    if isinstance(result, TokenResponse):
        return result
    return JSONResponse(status_code=202, content=result.model_dump())


@router.get("/verify", include_in_schema=False)
async def verify_email(
    token: str = Query(..., min_length=16),
    service: AuthService = Depends(get_auth_service),
) -> RedirectResponse:
    """Destino do link do e-mail.

    Sempre redireciona pro frontend em vez de renderizar JSON — quem abre isso é
    um navegador vindo do cliente de e-mail, não a SPA.
    """
    base = settings.frontend_url.rstrip("/")
    try:
        await service.verify_email(token)
    except UnauthorizedError:
        return RedirectResponse(url=f"{base}/login?verificado=erro", status_code=303)
    return RedirectResponse(url=f"{base}/login?verificado=1", status_code=303)


@router.post(
    "/resend-verification",
    status_code=202,
    dependencies=[Depends(_LIMITE_EMAIL)],
)
async def resend_verification(
    body: ResendVerificationRequest,
    service: AuthService = Depends(get_auth_service),
) -> dict[str, str]:
    """Reenvia o link de confirmação.

    Responde 202 sempre, mesmo pra e-mail inexistente — não confirmar se a
    conta existe evita transformar o endpoint em enumerador de usuários.
    """
    await service.resend_verification(body.email)
    return {"message": "Se houver uma conta pendente com esse e-mail, o link foi reenviado."}


@router.post(
    "/forgot-password",
    status_code=202,
    dependencies=[Depends(_LIMITE_EMAIL)],
)
async def forgot_password(
    body: ForgotPasswordRequest,
    service: AuthService = Depends(get_auth_service),
) -> dict[str, str]:
    """Pede o link de redefinição de senha.

    Responde 202 mesmo para e-mail inexistente, pelo mesmo motivo do reenvio:
    uma resposta diferente aqui entregaria quem tem conta.
    """
    await service.request_password_reset(body.email)
    return {"message": "Se houver uma conta com esse e-mail, enviamos o link de redefinição."}


@router.post("/reset-password", status_code=204, dependencies=[Depends(_LIMITE_RESET)])
async def reset_password(
    body: ResetPasswordRequest,
    service: AuthService = Depends(get_auth_service),
) -> None:
    """Troca a senha usando o token do e-mail. Uso único."""
    await service.reset_password(body)


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(_LIMITE_LOGIN)])
async def login(
    body: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await service.login(body)


@router.get("/me", response_model=UserResponse)
async def me(current_user: UserDTO = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
    )
