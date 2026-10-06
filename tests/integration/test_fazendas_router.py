"""TCC-096 — rotas /api/v1/fazendas e o PATCH de talhão."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_fazenda_service, get_talhao_service
from app.core.exceptions import NotFoundError
from app.domains.fazendas.schemas import FazendaResponse
from app.domains.talhoes.schemas import TalhaoResponse
from app.main import app
from tests.conftest import NOW, make_user_dto

API = "/api/v1"


def _talhao(**kw) -> TalhaoResponse:
    base = dict(id="t-1", fazenda_id="f-1", nome="Sede", apelido=None, hectares=60.0,
                cultura="soja", data_semeadura=None, created_at=NOW)
    return TalhaoResponse(**{**base, **kw})


def _fazenda(**kw) -> FazendaResponse:
    base = dict(id="f-1", nome="Boa Vista", municipio="Rio Verde", uf="GO", hectares=182.0,
                agronomo_nome=None, agronomo_crea=None, created_at=NOW, talhoes=[_talhao()])
    return FazendaResponse(**{**base, **kw})


@pytest.fixture
def svc():
    s = AsyncMock()
    s.list_for_user = AsyncMock(return_value=[_fazenda()])
    s.create = AsyncMock(return_value=_fazenda(id="f-2", talhoes=[]))
    s.update = AsyncMock(return_value=_fazenda(nome="Boa Vista II"))
    s.delete = AsyncMock(return_value=None)
    return s


@pytest.fixture
def talhao_svc():
    s = AsyncMock()
    s.update = AsyncMock(return_value=_talhao(fazenda_id="f-2"))
    s.list_for_user = AsyncMock(return_value=[_talhao()])
    return s


@pytest.fixture
async def client(svc, talhao_svc):
    app.dependency_overrides[get_fazenda_service] = lambda: svc
    app.dependency_overrides[get_talhao_service] = lambda: talhao_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_lista_fazendas_com_talhoes(client, svc):
    r = await client.get(f"{API}/fazendas")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["nome"] == "Boa Vista"
    assert body[0]["talhoes"][0]["fazenda_id"] == "f-1"
    svc.list_for_user.assert_awaited_once_with("user-uuid-1")


async def test_cria_fazenda(client, svc):
    r = await client.post(f"{API}/fazendas", json={"nome": "Sítio", "uf": "go"})
    assert r.status_code == 201
    assert svc.create.await_args.args[1].uf == "GO"


async def test_cria_fazenda_exige_nome(client):
    r = await client.post(f"{API}/fazendas", json={"nome": ""})
    assert r.status_code == 422


async def test_uf_com_tamanho_errado(client):
    r = await client.post(f"{API}/fazendas", json={"nome": "X", "uf": "GOI"})
    assert r.status_code == 422


async def test_atualiza_fazenda(client, svc):
    r = await client.patch(f"{API}/fazendas/f-1", json={"nome": "Boa Vista II"})
    assert r.status_code == 200 and r.json()["nome"] == "Boa Vista II"


async def test_atualiza_fazenda_inexistente(client, svc):
    svc.update.side_effect = NotFoundError("Fazenda", "x")
    r = await client.patch(f"{API}/fazendas/x", json={"nome": "A"})
    assert r.status_code == 404


async def test_apaga_fazenda(client, svc):
    r = await client.delete(f"{API}/fazendas/f-1")
    assert r.status_code == 204
    svc.delete.assert_awaited_once_with("f-1", "user-uuid-1")


async def test_move_talhao_de_fazenda(client, talhao_svc):
    r = await client.patch(f"{API}/talhoes/t-1", json={"fazenda_id": "f-2"})
    assert r.status_code == 200 and r.json()["fazenda_id"] == "f-2"


async def test_lista_talhoes_filtrando_por_fazenda(client, talhao_svc):
    r = await client.get(f"{API}/talhoes?fazenda_id=f-1")
    assert r.status_code == 200
    talhao_svc.list_for_user.assert_awaited_once_with("user-uuid-1", "f-1")


async def test_fazendas_exige_login():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get(f"{API}/fazendas")
    assert r.status_code == 401
