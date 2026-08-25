"""Testes do gate de verificacao de e-mail (TCC-090).

Cobre o AuthService com ``require_email_verification`` ligado, os estados de
token (valido, usado, expirado, inexistente), o reenvio silencioso e o cliente
Resend em ``app/core/email.py``.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.core.email import NullEmailSender, ResendEmailSender, get_email_sender
from app.core.exceptions import UnauthorizedError
from app.domains.auth.dto import EmailVerificationTokenDTO
from app.domains.auth.schemas import LoginRequest, RegisterRequest
from app.domains.auth.service import AuthService, _hash_token
from tests.conftest import make_user_dto

NOW = datetime.now(UTC)


def make_token_dto(**kwargs) -> EmailVerificationTokenDTO:
    defaults = dict(
        id="token-uuid-1",
        user_id="user-uuid-1",
        token_hash=_hash_token("raw-token"),
        expires_at=NOW + timedelta(hours=24),
        used_at=None,
        created_at=NOW,
    )
    return EmailVerificationTokenDTO(**{**defaults, **kwargs})


@pytest.fixture
def user_repo():
    repo = AsyncMock()
    repo.find_by_email = AsyncMock(return_value=None)
    repo.create = AsyncMock(return_value=make_user_dto(is_active=False))
    repo.update = AsyncMock(return_value=make_user_dto(is_active=True))
    return repo


@pytest.fixture
def verification_repo():
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
def gate_on():
    """Liga ``require_email_verification`` so no escopo do teste."""
    with patch("app.domains.auth.service.settings") as mock_settings:
        mock_settings.require_email_verification = True
        mock_settings.email_verification_ttl_hours = 24
        mock_settings.public_api_url = "https://felipe-ze-praga.hf.space"
        yield mock_settings


# ── register com o gate ligado ────────────────────────────────────────────────


async def test_register_gated_cria_usuario_inativo(gate_on, user_repo, verification_repo, sender):
    svc = AuthService(user_repo, verification_repo, sender)
    with patch("app.domains.auth.service.hash_password", return_value="hashed"):
        resp = await svc.register(RegisterRequest(email="novo@test.com", password="secret1"))

    assert resp.verification_required is True
    assert resp.email == "test@example.com"
    # o ponto central do gate: a conta nasce desligada
    assert user_repo.create.await_args.kwargs["is_active"] is False


async def test_register_gated_envia_email_com_link(gate_on, user_repo, verification_repo, sender):
    svc = AuthService(user_repo, verification_repo, sender)
    with patch("app.domains.auth.service.hash_password", return_value="hashed"):
        await svc.register(RegisterRequest(email="novo@test.com", password="secret1"))

    sender.send.assert_awaited_once()
    html_body = sender.send.await_args.kwargs["html"]
    assert "https://felipe-ze-praga.hf.space/api/v1/auth/verify?token=" in html_body


async def test_register_gated_persiste_apenas_o_hash(gate_on, user_repo, verification_repo, sender):
    """O token cru vai pro e-mail; o banco so ve o SHA-256."""
    svc = AuthService(user_repo, verification_repo, sender)
    with patch("app.domains.auth.service.hash_password", return_value="hashed"):
        await svc.register(RegisterRequest(email="novo@test.com", password="secret1"))

    stored_hash = verification_repo.create.await_args.args[1]
    html_body = sender.send.await_args.kwargs["html"]
    raw_token = html_body.split("?token=")[1].split('"')[0]

    assert stored_hash == _hash_token(raw_token)
    assert raw_token not in stored_hash


async def test_register_sem_gate_mantem_comportamento_antigo(user_repo, verification_repo, sender):
    """Com a flag desligada o cadastro segue devolvendo token na hora."""
    user_repo.create.return_value = make_user_dto(is_active=True)
    svc = AuthService(user_repo, verification_repo, sender)
    with (
        patch("app.domains.auth.service.hash_password", return_value="hashed"),
        patch("app.domains.auth.service.create_access_token", return_value="tok"),
    ):
        resp = await svc.register(RegisterRequest(email="novo@test.com", password="secret1"))

    assert resp.access_token == "tok"
    assert user_repo.create.await_args.kwargs["is_active"] is True
    sender.send.assert_not_awaited()


# ── verify_email ──────────────────────────────────────────────────────────────


async def test_verify_email_ativa_a_conta(user_repo, verification_repo, sender):
    svc = AuthService(user_repo, verification_repo, sender)
    user = await svc.verify_email("raw-token")

    assert user.is_active is True
    user_repo.update.assert_awaited_once_with("user-uuid-1", is_active=True)
    verification_repo.mark_used.assert_awaited_once_with("token-uuid-1")


async def test_verify_email_token_inexistente(user_repo, verification_repo, sender):
    verification_repo.find_by_hash.return_value = None
    svc = AuthService(user_repo, verification_repo, sender)
    with pytest.raises(UnauthorizedError, match="inválido"):
        await svc.verify_email("nao-existe")


async def test_verify_email_token_ja_usado(user_repo, verification_repo, sender):
    verification_repo.find_by_hash.return_value = make_token_dto(used_at=NOW)
    svc = AuthService(user_repo, verification_repo, sender)
    with pytest.raises(UnauthorizedError, match="já foi utilizado"):
        await svc.verify_email("raw-token")
    user_repo.update.assert_not_awaited()


async def test_verify_email_token_expirado(user_repo, verification_repo, sender):
    verification_repo.find_by_hash.return_value = make_token_dto(
        expires_at=NOW - timedelta(hours=1)
    )
    svc = AuthService(user_repo, verification_repo, sender)
    with pytest.raises(UnauthorizedError, match="expirado"):
        await svc.verify_email("raw-token")
    user_repo.update.assert_not_awaited()


async def test_verify_email_aceita_expires_at_naive(user_repo, verification_repo, sender):
    """Driver pode devolver datetime sem tzinfo — nao pode estourar comparacao."""
    verification_repo.find_by_hash.return_value = make_token_dto(
        expires_at=(NOW + timedelta(hours=5)).replace(tzinfo=None)
    )
    svc = AuthService(user_repo, verification_repo, sender)
    user = await svc.verify_email("raw-token")
    assert user.is_active is True


async def test_verify_email_sem_repo_configurado(user_repo):
    svc = AuthService(user_repo)
    with pytest.raises(UnauthorizedError, match="indisponível"):
        await svc.verify_email("raw-token")


# ── resend_verification ───────────────────────────────────────────────────────


async def test_resend_reemite_para_conta_pendente(gate_on, user_repo, verification_repo, sender):
    user_repo.find_by_email.return_value = make_user_dto(is_active=False)
    svc = AuthService(user_repo, verification_repo, sender)
    await svc.resend_verification("test@example.com")

    verification_repo.invalidate_pending.assert_awaited_once_with("user-uuid-1")
    sender.send.assert_awaited_once()


async def test_resend_silencioso_para_email_inexistente(
    gate_on, user_repo, verification_repo, sender
):
    user_repo.find_by_email.return_value = None
    svc = AuthService(user_repo, verification_repo, sender)
    await svc.resend_verification("ghost@test.com")
    sender.send.assert_not_awaited()


async def test_resend_silencioso_para_conta_ja_ativa(gate_on, user_repo, verification_repo, sender):
    user_repo.find_by_email.return_value = make_user_dto(is_active=True)
    svc = AuthService(user_repo, verification_repo, sender)
    await svc.resend_verification("test@example.com")
    sender.send.assert_not_awaited()


async def test_resend_noop_com_gate_desligado(user_repo, verification_repo, sender):
    user_repo.find_by_email.return_value = make_user_dto(is_active=False)
    svc = AuthService(user_repo, verification_repo, sender)
    await svc.resend_verification("test@example.com")
    sender.send.assert_not_awaited()


# ── login com o gate ligado ───────────────────────────────────────────────────


async def test_login_inativo_com_gate_pede_confirmacao(
    gate_on, user_repo, verification_repo, sender
):
    user_repo.find_by_email.return_value = make_user_dto(is_active=False)
    svc = AuthService(user_repo, verification_repo, sender)
    with patch("app.domains.auth.service.verify_password", return_value=True):
        with pytest.raises(UnauthorizedError, match="Confirme seu e-mail"):
            await svc.login(LoginRequest(email="test@example.com", password="pass"))


async def test_login_inativo_sem_gate_mantem_mensagem_antiga(user_repo, verification_repo, sender):
    user_repo.find_by_email.return_value = make_user_dto(is_active=False)
    svc = AuthService(user_repo, verification_repo, sender)
    with patch("app.domains.auth.service.verify_password", return_value=True):
        with pytest.raises(UnauthorizedError, match="Account is inactive"):
            await svc.login(LoginRequest(email="test@example.com", password="pass"))


# ── cliente de e-mail ─────────────────────────────────────────────────────────


async def test_null_sender_nao_chama_rede():
    """Nao deve levantar nem tocar httpx — so registra no log."""
    await NullEmailSender().send(to="a@b.com", subject="x", html="<p>y</p>")


def test_factory_cai_no_null_sem_api_key():
    with patch("app.core.email.settings") as mock_settings:
        mock_settings.resend_api_key = None
        assert isinstance(get_email_sender(), NullEmailSender)


def test_factory_usa_resend_com_api_key():
    with patch("app.core.email.settings") as mock_settings:
        mock_settings.resend_api_key = "re_fake"
        mock_settings.email_from = "Ze Praga <no-reply@test.com>"
        assert isinstance(get_email_sender(), ResendEmailSender)


async def test_resend_sender_monta_payload_da_api():
    response = type("R", (), {"status_code": 200, "text": "ok"})()
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.core.email.httpx.AsyncClient", return_value=client):
        await ResendEmailSender("re_fake", "Ze <no-reply@test.com>").send(
            to="destino@test.com", subject="Assunto", html="<p>oi</p>"
        )

    payload = client.post.await_args.kwargs["json"]
    headers = client.post.await_args.kwargs["headers"]
    assert payload["to"] == ["destino@test.com"]
    assert payload["from"] == "Ze <no-reply@test.com>"
    assert headers["Authorization"] == "Bearer re_fake"


async def test_resend_sender_engole_erro_da_api():
    """Falha de e-mail nao pode derrubar o cadastro que ja foi gravado."""
    response = type("R", (), {"status_code": 422, "text": "domain not verified"})()
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.core.email.httpx.AsyncClient", return_value=client):
        await ResendEmailSender("re_fake", "Ze <no-reply@test.com>").send(
            to="destino@test.com", subject="Assunto", html="<p>oi</p>"
        )
