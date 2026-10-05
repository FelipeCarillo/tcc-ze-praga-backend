"""TCC-093 — rotas de laudos por talhao em /api/v1/diagnoses."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_diagnosis_service
from app.core.exceptions import NotFoundError
from app.domains.diagnoses.schemas import TalhaoGroupResponse
from app.main import app
from app.shared.pagination import PaginatedResponse
from tests.conftest import NOW, make_user_dto
from tests.integration.conftest import make_diagnosis_response


@pytest.fixture
def svc():
    s = AsyncMock()
    s.list_for_user = AsyncMock(
        return_value=PaginatedResponse(items=[], total=0, page=1, limit=20)
    )
    s.group_by_talhao = AsyncMock(
        return_value=[
            TalhaoGroupResponse(
                talhao_id="t-1",
                talhao_nome="Sede",
                total=2,
                last_at=NOW,
                severity_trend=["baixa", "alta"],
                recent=[make_diagnosis_response()],
            ),
            TalhaoGroupResponse(
                talhao_id=None,
                talhao_nome=None,
                total=1,
                last_at=NOW,
                severity_trend=[],
                recent=[],
            ),
        ]
    )
    s.set_talhao = AsyncMock(return_value=make_diagnosis_response())
    return s


@pytest.fixture
async def client(svc):
    app.dependency_overrides[get_diagnosis_service] = lambda: svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_por_talhao_devolve_grupos(client, svc):
    r = await client.get("/api/v1/diagnoses/por-talhao")
    assert r.status_code == 200
    body = r.json()
    assert [g["talhao_nome"] for g in body] == ["Sede", None]
    assert body[0]["severity_trend"] == ["baixa", "alta"]
    svc.group_by_talhao.assert_awaited_once_with("user-uuid-1", 3)


async def test_por_talhao_nao_e_capturada_como_id_de_laudo(client, svc):
    """A rota fixa precisa vir antes de ``/{diagnosis_id}``."""
    await client.get("/api/v1/diagnoses/por-talhao?per_group=5")
    svc.get_by_id.assert_not_called()
    svc.group_by_talhao.assert_awaited_once_with("user-uuid-1", 5)


async def test_por_talhao_limita_per_group(client):
    r = await client.get("/api/v1/diagnoses/por-talhao?per_group=50")
    assert r.status_code == 422


async def test_lista_filtra_por_talhao(client, svc):
    r = await client.get("/api/v1/diagnoses?talhao_id=sem-talhao")
    assert r.status_code == 200
    filters = svc.list_for_user.await_args.args[1]
    assert filters.talhao_id == "sem-talhao"


async def test_patch_talhao_move_laudo(client, svc):
    r = await client.patch("/api/v1/diagnoses/diag-uuid-1/talhao", json={"talhao_id": "t-1"})
    assert r.status_code == 200
    svc.set_talhao.assert_awaited_once_with("diag-uuid-1", "user-uuid-1", "t-1")


async def test_patch_talhao_null_desfaz_vinculo(client, svc):
    r = await client.patch("/api/v1/diagnoses/diag-uuid-1/talhao", json={"talhao_id": None})
    assert r.status_code == 200
    svc.set_talhao.assert_awaited_once_with("diag-uuid-1", "user-uuid-1", None)


async def test_patch_talhao_alheio_404(client, svc):
    svc.set_talhao.side_effect = NotFoundError("Talhao", "t-x")
    r = await client.patch("/api/v1/diagnoses/diag-uuid-1/talhao", json={"talhao_id": "t-x"})
    assert r.status_code == 404
