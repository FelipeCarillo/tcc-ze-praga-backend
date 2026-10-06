"""TCC-096 — TalhaoRepository: criação na fazenda, filtro e PATCH."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.talhoes.repository import TalhaoRepository
from app.domains.talhoes.schemas import CreateTalhaoRequest, UpdateTalhaoRequest
from app.models.talhao import Talhao
from tests.conftest import NOW


def _row(fazenda_id: str = "f-1") -> Talhao:
    t = Talhao(id="t-1", user_id="u-1", fazenda_id=fazenda_id, nome="Sede", cultura="soja")
    t.created_at = NOW
    return t


def _result(rows: list[Talhao]) -> MagicMock:
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    res.scalar_one_or_none.return_value = rows[0] if rows else None
    return res


@pytest.fixture
def db():
    s = MagicMock()
    s.execute = AsyncMock()
    s.commit = AsyncMock()
    s.refresh = AsyncMock()
    s.delete = AsyncMock()
    return s


async def test_create_grava_a_fazenda(db):
    async def _refresh(obj):
        obj.id, obj.created_at = "t-novo", NOW

    db.refresh.side_effect = _refresh
    out = await TalhaoRepository(db).create("u-1", "f-9", CreateTalhaoRequest(nome="Rio"))
    assert out.fazenda_id == "f-9" and out.id == "t-novo"


async def test_find_all_by_fazenda(db):
    db.execute.return_value = _result([_row()])
    out = await TalhaoRepository(db).find_all_by_fazenda("f-1", "u-1")
    assert [t.fazenda_id for t in out] == ["f-1"]


async def test_find_by_id(db):
    db.execute.return_value = _result([_row()])
    assert (await TalhaoRepository(db).find_by_id("t-1", "u-1")).nome == "Sede"


async def test_update_move_de_fazenda(db):
    db.execute.return_value = _result([_row()])
    out = await TalhaoRepository(db).update("t-1", "u-1", UpdateTalhaoRequest(fazenda_id="f-2"))
    assert out is not None and out.fazenda_id == "f-2" and out.nome == "Sede"


async def test_update_ausente(db):
    db.execute.return_value = _result([])
    assert await TalhaoRepository(db).update("x", "u-1", UpdateTalhaoRequest()) is None


async def test_delete_ausente(db):
    db.execute.return_value = _result([])
    assert await TalhaoRepository(db).delete("x", "u-1") is False
