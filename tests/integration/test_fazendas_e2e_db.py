"""TCC-096 — fazendas, talhões e histórico por talhão pela API, num Postgres real.

Sobe o app de verdade (rotas + services + repositories + SQL) e só troca o
usuário autenticado e a sessão do banco. Pulado sem ``ZP_TEST_PG_URL``.
"""

import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.dependencies import get_current_user, get_db
from app.main import app
from app.models.user import User
from tests.conftest import make_user_dto

PG_URL = os.environ.get("ZP_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="sem ZP_TEST_PG_URL (Postgres real)")
API = "/api/v1"


@pytest.fixture
async def client():
    engine = create_async_engine(PG_URL)  # type: ignore[arg-type]
    maker = async_sessionmaker(engine, expire_on_commit=False)
    tag = uuid.uuid4().hex[:8]
    async with maker() as s:
        user = User(email=f"e2e-{tag}@t.test", password_hash="x")
        s.add(user)
        await s.commit()
        uid = user.id

    async def _db():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = lambda: make_user_dto(id=uid)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    await engine.dispose()


async def test_fluxo_fazenda_talhao(client):
    # Talhão sem fazenda cria a "Minha fazenda".
    r = await client.post(f"{API}/talhoes", json={"nome": "Talhão 1"})
    assert r.status_code == 201
    padrao = r.json()["fazenda_id"]

    r = await client.post(f"{API}/fazendas", json={"nome": "Fazenda Boa Vista", "uf": "go", "municipio": "Rio Verde"})
    assert r.status_code == 201 and r.json()["uf"] == "GO"
    boa = r.json()["id"]

    r = await client.post(f"{API}/talhoes", json={"nome": "Talhão 3", "apelido": "Sede", "hectares": 60, "fazenda_id": boa})
    sede = r.json()
    assert sede["fazenda_id"] == boa

    # Hierarquia: cada fazenda com os seus talhões.
    fazendas = (await client.get(f"{API}/fazendas")).json()
    by_nome = {f["nome"]: f for f in fazendas}
    assert [t["nome"] for t in by_nome["Fazenda Boa Vista"]["talhoes"]] == ["Talhão 3"]
    assert [t["nome"] for t in by_nome["Minha fazenda"]["talhoes"]] == ["Talhão 1"]

    # Filtro por fazenda e mover talhão.
    so_boa = (await client.get(f"{API}/talhoes", params={"fazenda_id": boa})).json()
    assert [t["id"] for t in so_boa] == [sede["id"]]
    r = await client.patch(f"{API}/talhoes/{sede['id']}", json={"fazenda_id": padrao})
    assert r.status_code == 200 and r.json()["fazenda_id"] == padrao

    # Fazenda de outro usuário é recusada.
    r = await client.post(f"{API}/talhoes", json={"nome": "X", "fazenda_id": str(uuid.uuid4())})
    assert r.status_code == 404

    # Histórico por talhão traz a fazenda de cada grupo.
    grupos = (await client.get(f"{API}/diagnoses/por-talhao")).json()
    assert {g["talhao_nome"]: g["fazenda_nome"] for g in grupos} == {
        "Talhão 1": "Minha fazenda",
        "Talhão 3": "Minha fazenda",
    }

    # Editar e apagar fazenda (apaga os talhões dela).
    r = await client.patch(f"{API}/fazendas/{boa}", json={"agronomo_nome": "Ana Lima", "agronomo_crea": "GO-1"})
    assert r.json()["agronomo_nome"] == "Ana Lima"
    assert (await client.delete(f"{API}/fazendas/{padrao}")).status_code == 204
    restantes = (await client.get(f"{API}/talhoes")).json()
    assert restantes == []
