"""Testes do fluxo de redefinição de senha (TCC-092)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.core.exceptions import UnauthorizedError
from app.domains.auth.dto import EmailVerificationTokenDTO
from app.domains.auth.schemas import ResetPasswordRequest
from app.domains.auth.service import AuthService, _hash_token
from tests.conftest import make_user_dto

NOW = datetime.now(UTC)


def make_token_dto(**kwargs) -> EmailVerificationTokenDTO:
    defaults = dict(
        id="reset-token-1",
        user_id="user-uuid-1",
        token_hash=_hash_token("raw-token"),
        expires_at=NOW + timedelta(hours=2),
        used_at=None,
        created_at=NOW,
    )
    return EmailVerificationTokenDTO(**{**defaults, **kwargs})


@pytest.fixture
def user_repo():
    repo = AsyncMock()
    repo.find_by_email = AsyncMock(return_value=make_user_dto())
    repo.update = AsyncMock(return_value=make_user_dto())
    return repo


@pytest.fixture
def reset_repo():
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=make_token_dto())
    repo.find_by_hash = AsyncMock(return_value=make_token_dto())
    repo.mark_used = AsyncMock()
    repo.invalidate_pending = AsyncMock()
    return repo


@pytest.fixture
def sender():
    fake = AsyncMock()
    fake.send = AsyncMock()
    return fake


@pytest.fixture
def svc(user_repo, reset_repo, sender):
    return AuthService(user_repo, None, sender, reset_repo=reset_repo)


@pytest.fixture
def settings_reset():
    with patch("app.domains.auth.service.settings") as mock:
        mock.password_reset_ttl_hours = 2
        mock.frontend_url = "https://ze-praga.vercel.app"
        yield mock


# ── request_password_reset ────────────────────────────────────────────────────


async def test_envia_link_apontando_pro_frontend(settings_reset, svc, sender):
    """O link vai pro frontend, não pra API — o usuário precisa digitar a senha."""
    await svc.request_password_reset("test@example.com")

    sender.send.assert_awaited_once()
    corpo = sender.send.await_args.kwargs["html"]
    assert "https://ze-praga.vercel.app/redefinir-senha?token=" in corpo
    assert "/api/v1/" not in corpo.split("?token=")[0].split('href="')[-1]


async def test_persiste_apenas_o_hash(settings_reset, svc, reset_repo, sender):
    await svc.request_password_reset("test@example.com")

    guardado = reset_repo.create.await_args.args[1]
    corpo = sender.send.await_args.kwargs["html"]
    cru = corpo.split("?token=")[1].split('"')[0]
    assert guardado == _hash_token(cru)
    assert cru not in guardado


async def test_queima_pendentes_antes_de_emitir(settings_reset, svc, reset_repo):
    """Só o último link recebido deve funcionar."""
    await svc.request_password_reset("test@example.com")
    reset_repo.invalidate_pending.assert_awaited_once_with("user-uuid-1")


async def test_silencioso_para_email_inexistente(settings_reset, svc, user_repo, sender):
    """Não pode virar oráculo de quem tem conta."""
    user_repo.find_by_email.return_value = None
    await svc.request_password_reset("ghost@test.com")
    sender.send.assert_not_awaited()


async def test_noop_sem_repo_configurado(user_repo, sender):
    svc = AuthService(user_repo, None, sender)
    await svc.request_password_reset("test@example.com")
    sender.send.assert_not_awaited()


# ── reset_password ────────────────────────────────────────────────────────────


async def test_troca_a_senha(svc, user_repo, reset_repo):
    with patch("app.domains.auth.service.hash_password", return_value="novo-hash"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))

    user_repo.update.assert_awaited_once_with("user-uuid-1", password_hash="novo-hash")
    reset_repo.mark_used.assert_awaited_once_with("reset-token-1")


async def test_queima_os_outros_links_apos_trocar(svc, reset_repo):
    """Quem pediu reset duas vezes não pode ficar com um segundo link vivo."""
    with patch("app.domains.auth.service.hash_password", return_value="novo-hash"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))
    reset_repo.invalidate_pending.assert_awaited_once_with("user-uuid-1")


async def test_token_inexistente(svc, reset_repo, user_repo):
    reset_repo.find_by_hash.return_value = None
    with pytest.raises(UnauthorizedError, match="inválido"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))
    user_repo.update.assert_not_awaited()


async def test_token_ja_usado(svc, reset_repo, user_repo):
    reset_repo.find_by_hash.return_value = make_token_dto(used_at=NOW)
    with pytest.raises(UnauthorizedError, match="já foi utilizado"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))
    user_repo.update.assert_not_awaited()


async def test_token_expirado(svc, reset_repo, user_repo):
    reset_repo.find_by_hash.return_value = make_token_dto(expires_at=NOW - timedelta(minutes=1))
    with pytest.raises(UnauthorizedError, match="expirado"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))
    user_repo.update.assert_not_awaited()


async def test_aceita_expires_at_naive(svc, user_repo, reset_repo):
    reset_repo.find_by_hash.return_value = make_token_dto(
        expires_at=(NOW + timedelta(hours=1)).replace(tzinfo=None)
    )
    with patch("app.domains.auth.service.hash_password", return_value="novo-hash"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))
    user_repo.update.assert_awaited_once()


async def test_sem_repo_configurado(user_repo, sender):
    svc = AuthService(user_repo, None, sender)
    with pytest.raises(UnauthorizedError, match="indisponível"):
        await svc.reset_password(ResetPasswordRequest(token="x" * 32, password="novasenha1"))


async def test_senha_curta_e_rejeitada_no_schema():
    """O reset não pode aceitar senha mais fraca do que o cadastro exige."""
    with pytest.raises(ValueError):
        ResetPasswordRequest(token="x" * 32, password="123")


async def test_token_curto_e_rejeitado_no_schema():
    with pytest.raises(ValueError):
        ResetPasswordRequest(token="curto", password="novasenha1")
