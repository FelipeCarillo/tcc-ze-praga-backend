"""Integration tests for /api/v1/auth router."""

from unittest.mock import AsyncMock

import pytest
from app.core.dependencies import get_auth_service, get_current_user
from app.core.exceptions import ConflictError, UnauthorizedError
from app.main import app
from tests.conftest import make_user_dto
from tests.integration.conftest import make_token_response, make_user_response


@pytest.fixture
def mock_auth_svc():
    svc = AsyncMock()
    svc.register = AsyncMock(return_value=make_token_response())
    svc.login = AsyncMock(return_value=make_token_response())
    return svc


@pytest.fixture
async def client_auth(mock_auth_svc):
    from httpx import ASGITransport, AsyncClient

    app.dependency_overrides[get_auth_service] = lambda: mock_auth_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── POST /auth/register ───────────────────────────────────────────────────────

async def test_register_201(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/register",
        json={"email": "new@test.com", "password": "secret123"},
    )
    assert r.status_code == 201
    assert r.json()["access_token"] == "fake-token"


async def test_register_conflict_409(client_auth, mock_auth_svc):
    mock_auth_svc.register.side_effect = ConflictError("Email already in use")
    r = await client_auth.post(
        "/api/v1/auth/register",
        json={"email": "used@test.com", "password": "secret123"},
    )
    assert r.status_code == 409


async def test_register_invalid_body(client_auth):
    r = await client_auth.post("/api/v1/auth/register", json={"email": "bad"})
    assert r.status_code == 422


# ── POST /auth/login ──────────────────────────────────────────────────────────

async def test_login_200(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/login",
        json={"email": "test@test.com", "password": "pass"},
    )
    assert r.status_code == 200
    assert "access_token" in r.json()


async def test_login_unauthorized_401(client_auth, mock_auth_svc):
    mock_auth_svc.login.side_effect = UnauthorizedError("Invalid email or password")
    r = await client_auth.post(
        "/api/v1/auth/login",
        json={"email": "x@x.com", "password": "wrong"},
    )
    assert r.status_code == 401


# ── GET /auth/me ──────────────────────────────────────────────────────────────

async def test_me_200(client_auth):
    r = await client_auth.get("/api/v1/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == "test@example.com"


# ── Verificacao de e-mail (TCC-090) ───────────────────────────────────────────


async def test_register_202_quando_verificacao_exigida(client_auth, mock_auth_svc):
    """Com o gate ligado o cadastro responde 202 e nao entrega token."""
    from app.domains.auth.schemas import RegistrationPendingResponse

    mock_auth_svc.register.return_value = RegistrationPendingResponse(email="novo@test.com")
    r = await client_auth.post(
        "/api/v1/auth/register",
        json={"email": "novo@test.com", "password": "secret123"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["verification_required"] is True
    assert body["email"] == "novo@test.com"
    assert "access_token" not in body


async def test_verify_redireciona_com_sucesso(client_auth, mock_auth_svc):
    mock_auth_svc.verify_email = AsyncMock(return_value=make_user_dto(is_active=True))
    r = await client_auth.get(
        "/api/v1/auth/verify", params={"token": "x" * 32}, follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"].endswith("/login?verificado=1")


async def test_verify_redireciona_com_erro_em_token_invalido(client_auth, mock_auth_svc):
    """Token ruim nao vira 401 cru — quem abre isso e um navegador."""
    mock_auth_svc.verify_email = AsyncMock(side_effect=UnauthorizedError("Link invalido"))
    r = await client_auth.get(
        "/api/v1/auth/verify", params={"token": "x" * 32}, follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"].endswith("/login?verificado=erro")


async def test_verify_rejeita_token_curto(client_auth):
    r = await client_auth.get("/api/v1/auth/verify", params={"token": "curto"})
    assert r.status_code == 422


async def test_resend_verification_202(client_auth, mock_auth_svc):
    mock_auth_svc.resend_verification = AsyncMock()
    r = await client_auth.post(
        "/api/v1/auth/resend-verification", json={"email": "novo@test.com"}
    )
    assert r.status_code == 202
    mock_auth_svc.resend_verification.assert_awaited_once_with("novo@test.com")


async def test_resend_verification_nao_revela_conta_inexistente(client_auth, mock_auth_svc):
    """Mesma resposta pra e-mail que existe e pra que nao existe."""
    mock_auth_svc.resend_verification = AsyncMock()
    r = await client_auth.post(
        "/api/v1/auth/resend-verification", json={"email": "ghost@test.com"}
    )
    assert r.status_code == 202
    assert "reenviado" in r.json()["message"]


# ── Redefinição de senha (TCC-092) ────────────────────────────────────────────


async def test_forgot_password_202(client_auth, mock_auth_svc):
    mock_auth_svc.request_password_reset = AsyncMock()
    r = await client_auth.post("/api/v1/auth/forgot-password", json={"email": "a@test.com"})
    assert r.status_code == 202
    mock_auth_svc.request_password_reset.assert_awaited_once_with("a@test.com")


async def test_forgot_password_nao_revela_conta_inexistente(client_auth, mock_auth_svc):
    """Mesma resposta para e-mail que existe e para o que não existe."""
    mock_auth_svc.request_password_reset = AsyncMock()
    r = await client_auth.post("/api/v1/auth/forgot-password", json={"email": "ghost@test.com"})
    assert r.status_code == 202
    assert "Se houver uma conta" in r.json()["message"]


async def test_reset_password_204(client_auth, mock_auth_svc):
    mock_auth_svc.reset_password = AsyncMock()
    r = await client_auth.post(
        "/api/v1/auth/reset-password", json={"token": "x" * 32, "password": "novasenha1"}
    )
    assert r.status_code == 204


async def test_reset_password_token_invalido_401(client_auth, mock_auth_svc):
    mock_auth_svc.reset_password = AsyncMock(
        side_effect=UnauthorizedError("Link de redefinição inválido")
    )
    r = await client_auth.post(
        "/api/v1/auth/reset-password", json={"token": "x" * 32, "password": "novasenha1"}
    )
    assert r.status_code == 401


async def test_reset_password_rejeita_senha_curta_422(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/reset-password", json={"token": "x" * 32, "password": "123"}
    )
    assert r.status_code == 422


# ── Rate limit (TCC-091) ──────────────────────────────────────────────────────


async def test_login_devolve_429_depois_do_limite(client_auth):
    """10 tentativas em 5 min; a 11ª é barrada com Retry-After."""
    corpo = {"email": "a@test.com", "password": "seja-o-que-for"}
    for _ in range(10):
        await client_auth.post("/api/v1/auth/login", json=corpo)

    r = await client_auth.post("/api/v1/auth/login", json=corpo)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) > 0
    assert r.json()["retry_after"] > 0


async def test_register_tem_limite_proprio(client_auth):
    """Gastar o limite de login não pode fechar o cadastro."""
    for _ in range(10):
        await client_auth.post(
            "/api/v1/auth/login", json={"email": "a@test.com", "password": "x"}
        )
    r = await client_auth.post(
        "/api/v1/auth/register", json={"email": "novo@test.com", "password": "secret123"}
    )
    assert r.status_code == 201


async def test_forgot_password_limite_mais_apertado(client_auth, mock_auth_svc):
    """Rota que dispara e-mail: 3 por hora, senão vira máquina de spam."""
    mock_auth_svc.request_password_reset = AsyncMock()
    for _ in range(3):
        await client_auth.post("/api/v1/auth/forgot-password", json={"email": "a@test.com"})

    r = await client_auth.post("/api/v1/auth/forgot-password", json={"email": "a@test.com"})
    assert r.status_code == 429
