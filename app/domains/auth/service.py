import hashlib
import html
import logging
import secrets
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.core.email import EmailSender, get_email_sender
from app.core.exceptions import ConflictError, UnauthorizedError
from app.core.security import create_access_token, hash_password, verify_password
from app.domains.auth.dto import UserCreateDTO, UserDTO
from app.domains.auth.repository import (
    EmailVerificationRepository,
    PasswordResetRepository,
    UserRepository,
)
from app.domains.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    RegistrationPendingResponse,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)

_TOKEN_BYTES = 32


def _hash_token(token: str) -> str:
    """SHA-256 do token cru.

    Não é bcrypt de propósito: o token tem 256 bits de entropia vindos do
    ``secrets``, então não há o que um hash lento proteja — não existe ataque de
    dicionário contra um valor aleatório desse tamanho.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def _as_utc(moment: datetime) -> datetime:
    """Normaliza pra aware-UTC — o driver pode devolver naive dependendo do banco."""
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


class AuthService:
    def __init__(
        self,
        user_repo: UserRepository,
        verification_repo: EmailVerificationRepository | None = None,
        email_sender: EmailSender | None = None,
        reset_repo: PasswordResetRepository | None = None,
    ) -> None:
        self._user_repo = user_repo
        self._verification_repo = verification_repo
        self._email_sender = email_sender
        self._reset_repo = reset_repo

    # ── Cadastro ──────────────────────────────────────────────────────────────

    async def register(
        self, request: RegisterRequest
    ) -> TokenResponse | RegistrationPendingResponse:
        """Cria a conta.

        Com ``require_email_verification`` ligado o usuário nasce **inativo** e
        recebe um link por e-mail; sem verificação o comportamento é o antigo
        (conta ativa + token na hora), preservando dev e a suíte existente.
        """
        if await self._user_repo.find_by_email(request.email):
            raise ConflictError("Email already in use")

        hashed = hash_password(request.password)
        gated = self._verification_enabled()
        user = await self._user_repo.create(
            UserCreateDTO(
                email=request.email,
                password_hash=hashed,
                full_name=request.full_name,
            ),
            is_active=not gated,
        )

        if not gated:
            return self._build_token_response(user)

        await self._issue_verification(user)
        return RegistrationPendingResponse(email=user.email)

    async def verify_email(self, token: str) -> UserDTO:
        """Ativa a conta a partir do token do e-mail. Uso único."""
        if self._verification_repo is None:
            raise UnauthorizedError("Verificação de e-mail indisponível")

        record = await self._verification_repo.find_by_hash(_hash_token(token))
        if record is None:
            raise UnauthorizedError("Link de verificação inválido")
        if record.used_at is not None:
            raise UnauthorizedError("Este link já foi utilizado")
        if _as_utc(record.expires_at) < datetime.now(UTC):
            raise UnauthorizedError("Link de verificação expirado")

        user = await self._user_repo.update(record.user_id, is_active=True)
        if user is None:
            raise UnauthorizedError("Usuário não encontrado")

        await self._verification_repo.mark_used(record.id)
        return user

    async def resend_verification(self, email: str) -> None:
        """Reemite o link.

        Silencioso de propósito: nunca revela se o e-mail existe ou se já está
        verificado — senão o endpoint vira um oráculo de enumeração de contas.
        """
        if not self._verification_enabled():
            return
        user = await self._user_repo.find_by_email(email)
        if user is None or user.is_active:
            return
        await self._issue_verification(user)

    # ── Senha esquecida ───────────────────────────────────────────────────────

    async def request_password_reset(self, email: str) -> None:
        """Emite o link de redefinição.

        Silencioso como o reenvio de verificação: nunca revela se o e-mail
        existe. Um "não encontrei essa conta" aqui entrega ao atacante a lista
        de quem tem cadastro.
        """
        if self._reset_repo is None:
            return
        user = await self._user_repo.find_by_email(email)
        if user is None:
            return

        await self._reset_repo.invalidate_pending(user.id)
        raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
        expires_at = datetime.now(UTC) + timedelta(hours=settings.password_reset_ttl_hours)
        await self._reset_repo.create(user.id, _hash_token(raw_token), expires_at)

        # Este link vai pro **frontend**, não pra API: o usuário precisa de uma
        # tela pra digitar a senha nova. É o oposto da verificação de e-mail,
        # que só precisa de um clique e por isso aponta direto pro backend.
        base = settings.frontend_url.rstrip("/")
        link = f"{base}/redefinir-senha?token={raw_token}"

        sender = self._email_sender or get_email_sender()
        await sender.send(
            to=user.email,
            subject="Redefinir sua senha — Zé Praga",
            html=_password_reset_html(user.full_name, link),
        )
        logger.info("Link de redefinição emitido para user_id=%s", user.id)

    async def reset_password(self, request: ResetPasswordRequest) -> None:
        """Troca a senha a partir do token do e-mail. Uso único."""
        if self._reset_repo is None:
            raise UnauthorizedError("Redefinição de senha indisponível")

        record = await self._reset_repo.find_by_hash(_hash_token(request.token))
        if record is None:
            raise UnauthorizedError("Link de redefinição inválido")
        if record.used_at is not None:
            raise UnauthorizedError("Este link já foi utilizado")
        if _as_utc(record.expires_at) < datetime.now(UTC):
            raise UnauthorizedError("Link de redefinição expirado")

        user = await self._user_repo.update(
            record.user_id, password_hash=hash_password(request.password)
        )
        if user is None:
            raise UnauthorizedError("Usuário não encontrado")

        await self._reset_repo.mark_used(record.id)
        # Queima os outros links pendentes: quem pediu reset duas vezes não deve
        # deixar um segundo link vivo depois de já ter trocado a senha.
        await self._reset_repo.invalidate_pending(user.id)
        logger.info("Senha redefinida para user_id=%s", user.id)

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(self, request: LoginRequest) -> TokenResponse:
        user = await self._user_repo.find_by_email(request.email)
        if not user or not verify_password(request.password, user.password_hash):
            raise UnauthorizedError("Invalid email or password")
        if not user.is_active:
            # Com o gate ligado, "inativo" quase sempre significa "não verificou
            # o e-mail" — a mensagem diz o que fazer em vez de só negar.
            if self._verification_enabled():
                raise UnauthorizedError("Confirme seu e-mail antes de entrar")
            raise UnauthorizedError("Account is inactive")
        return self._build_token_response(user)

    # ── Internos ──────────────────────────────────────────────────────────────

    def _verification_enabled(self) -> bool:
        return settings.require_email_verification and self._verification_repo is not None

    async def _issue_verification(self, user: UserDTO) -> None:
        """Queima os links pendentes, emite um novo e dispara o e-mail."""
        if self._verification_repo is None:  # pragma: no cover — guardado por _verification_enabled
            return

        await self._verification_repo.invalidate_pending(user.id)
        raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
        expires_at = datetime.now(UTC) + timedelta(hours=settings.email_verification_ttl_hours)
        await self._verification_repo.create(user.id, _hash_token(raw_token), expires_at)

        sender = self._email_sender or get_email_sender()
        base = settings.public_api_url.rstrip("/")
        link = f"{base}/api/v1/auth/verify?token={raw_token}"
        await sender.send(
            to=user.email,
            subject="Confirme seu e-mail — Zé Praga",
            html=_verification_html(user.full_name, link),
        )
        logger.info("Link de verificação emitido para user_id=%s", user.id)

    @staticmethod
    def _build_token_response(user: UserDTO) -> TokenResponse:
        token = create_access_token(user.id)
        return TokenResponse(
            access_token=token,
            user=UserResponse(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                created_at=user.created_at,
            ),
        )


def _verification_html(full_name: str | None, link: str) -> str:
    """Corpo do e-mail. ``full_name`` vem do usuário — escapado antes de virar HTML."""
    greeting = f"Olá, {html.escape(full_name)}!" if full_name else "Olá!"
    horas = settings.email_verification_ttl_hours
    return f"""<!doctype html>
<html lang="pt-BR">
  <body style="margin:0;padding:24px;background:#f6f7f5;
               font-family:Arial,Helvetica,sans-serif;color:#1a2e1a;">
    <div style="max-width:520px;margin:0 auto;background:#ffffff;border-radius:12px;padding:32px;">
      <h1 style="margin:0 0 16px;font-size:20px;">Zé Praga</h1>
      <p style="margin:0 0 12px;font-size:15px;line-height:1.6;">{greeting}</p>
      <p style="margin:0 0 24px;font-size:15px;line-height:1.6;">
        Falta um passo para ativar sua conta no Zé Praga. Clique no botão abaixo
        para confirmar seu e-mail.
      </p>
      <p style="margin:0 0 24px;">
        <a href="{link}"
           style="display:inline-block;background:#2e7d32;color:#ffffff;text-decoration:none;
                  padding:12px 24px;border-radius:8px;font-size:15px;font-weight:bold;">
          Confirmar e-mail
        </a>
      </p>
      <p style="margin:0 0 8px;font-size:13px;color:#5a6b5a;line-height:1.6;">
        O link vale por {horas} horas e só pode ser usado uma vez.
        Se o botão não funcionar, copie e cole este endereço no navegador:
      </p>
      <p style="margin:0 0 24px;font-size:12px;color:#5a6b5a;word-break:break-all;">{link}</p>
      <p style="margin:0;font-size:12px;color:#8a978a;line-height:1.6;">
        Se você não criou essa conta, ignore este e-mail — nada será ativado.
      </p>
    </div>
  </body>
</html>
"""


def _password_reset_html(full_name: str | None, link: str) -> str:
    """Corpo do e-mail de redefinição. ``full_name`` é escapado antes de virar HTML."""
    greeting = f"Olá, {html.escape(full_name)}!" if full_name else "Olá!"
    horas = settings.password_reset_ttl_hours
    return f"""<!doctype html>
<html lang="pt-BR">
  <body style="margin:0;padding:24px;background:#f6f7f5;
               font-family:Arial,Helvetica,sans-serif;color:#1a2e1a;">
    <div style="max-width:520px;margin:0 auto;background:#ffffff;border-radius:12px;padding:32px;">
      <h1 style="margin:0 0 16px;font-size:20px;">Zé Praga</h1>
      <p style="margin:0 0 12px;font-size:15px;line-height:1.6;">{greeting}</p>
      <p style="margin:0 0 24px;font-size:15px;line-height:1.6;">
        Recebemos um pedido para redefinir a senha da sua conta. Clique no botão
        abaixo para escolher uma nova.
      </p>
      <p style="margin:0 0 24px;">
        <a href="{link}"
           style="display:inline-block;background:#2e7d32;color:#ffffff;text-decoration:none;
                  padding:12px 24px;border-radius:8px;font-size:15px;font-weight:bold;">
          Redefinir senha
        </a>
      </p>
      <p style="margin:0 0 8px;font-size:13px;color:#5a6b5a;line-height:1.6;">
        O link vale por {horas} horas e só pode ser usado uma vez.
        Se o botão não funcionar, copie e cole este endereço no navegador:
      </p>
      <p style="margin:0 0 24px;font-size:12px;color:#5a6b5a;word-break:break-all;">{link}</p>
      <p style="margin:0;font-size:12px;color:#8a978a;line-height:1.6;">
        Se você não pediu isso, ignore este e-mail — sua senha continua a mesma.
      </p>
    </div>
  </body>
</html>
"""
