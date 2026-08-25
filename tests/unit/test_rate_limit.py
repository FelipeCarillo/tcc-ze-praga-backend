"""Testes do rate limit por IP (TCC-091)."""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import RateLimitedError
from app.core.rate_limit import RateLimit, client_ip, limpar_rate_limit


def make_request(ip: str = "203.0.113.7", *, forwarded: str | None = None, path: str = "/x"):
    req = MagicMock()
    req.headers = {"x-forwarded-for": forwarded} if forwarded else {}
    req.client = MagicMock(host=ip)
    req.url = MagicMock(path=path)
    req.scope = {}
    return req


@pytest.fixture(autouse=True)
def _limpa():
    limpar_rate_limit()
    yield
    limpar_rate_limit()


# ── client_ip ─────────────────────────────────────────────────────────────────


def test_usa_o_primeiro_ip_do_x_forwarded_for():
    """Atrás do proxy do Cloud Run, o cliente é o primeiro da cadeia."""
    req = make_request(forwarded="198.51.100.5, 10.0.0.1, 10.0.0.2")
    assert client_ip(req) == "198.51.100.5"


def test_cai_no_client_host_sem_header():
    assert client_ip(make_request(ip="192.0.2.10")) == "192.0.2.10"


def test_ignora_header_vazio():
    assert client_ip(make_request(ip="192.0.2.10", forwarded="   ")) == "192.0.2.10"


def test_sem_client_nem_header():
    req = make_request()
    req.client = None
    req.headers = {}
    assert client_ip(req) == "desconhecido"


# ── RateLimit ─────────────────────────────────────────────────────────────────


async def test_permite_ate_o_limite():
    limite = RateLimit(3, 60, nome="t1")
    req = make_request()
    for _ in range(3):
        await limite(req)  # não levanta


async def test_bloqueia_a_partir_do_limite():
    limite = RateLimit(3, 60, nome="t2")
    req = make_request()
    for _ in range(3):
        await limite(req)
    with pytest.raises(RateLimitedError) as exc:
        await limite(req)
    assert exc.value.retry_after > 0


async def test_ips_diferentes_tem_baldes_separados():
    """Um IP abusando não pode derrubar o resto do mundo."""
    limite = RateLimit(2, 60, nome="t3")
    a, b = make_request(ip="198.51.100.1"), make_request(ip="198.51.100.2")
    for _ in range(2):
        await limite(a)
    with pytest.raises(RateLimitedError):
        await limite(a)
    await limite(b)  # o outro IP segue livre


async def test_rotas_diferentes_tem_baldes_separados():
    """Gastar as tentativas de login não pode consumir as de cadastro."""
    login = RateLimit(1, 60, nome="auth:login")
    cadastro = RateLimit(1, 60, nome="auth:register")
    req = make_request()
    await login(req)
    with pytest.raises(RateLimitedError):
        await login(req)
    await cadastro(req)  # balde independente


async def test_janela_desliza(monkeypatch):
    """Passada a janela, as batidas antigas saem e liberam vaga."""
    relogio = {"t": 1000.0}
    monkeypatch.setattr("app.core.rate_limit.time.monotonic", lambda: relogio["t"])

    limite = RateLimit(2, 60, nome="t4")
    req = make_request()
    await limite(req)
    await limite(req)
    with pytest.raises(RateLimitedError):
        await limite(req)

    relogio["t"] += 61  # janela passou
    await limite(req)  # liberado de novo


async def test_retry_after_reflete_o_tempo_restante(monkeypatch):
    relogio = {"t": 500.0}
    monkeypatch.setattr("app.core.rate_limit.time.monotonic", lambda: relogio["t"])

    limite = RateLimit(1, 60, nome="t5")
    req = make_request()
    await limite(req)

    relogio["t"] += 20  # faltam 40s para a primeira batida sair da janela
    with pytest.raises(RateLimitedError) as exc:
        await limite(req)
    assert 39 <= exc.value.retry_after <= 42


async def test_usa_o_path_quando_nao_ha_nome():
    limite = RateLimit(1, 60)
    req = make_request(path="/api/v1/qualquer")
    await limite(req)
    with pytest.raises(RateLimitedError):
        await limite(req)


async def test_limpar_zera_a_contagem():
    limite = RateLimit(1, 60, nome="t6")
    req = make_request()
    await limite(req)
    limpar_rate_limit()
    await limite(req)  # contagem zerada
