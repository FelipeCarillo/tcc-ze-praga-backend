"""TCC-096 — FazendaRepository com a sessão mockada."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.fazendas.repository import DEFAULT_NOME, FazendaRepository
from app.domains.fazendas.schemas import CreateFazendaRequest, UpdateFazendaRequest
from app.models.fazenda import Fazenda
from tests.conftest import NOW


def _row(id_: str = "f-1", nome: str = "Boa Vista") -> Fazenda:
    f = Fazenda(id=id_, user_id="u-1", nome=nome, municipio=None, uf="GO", hectares=10)
    f.created_at = NOW
    return f


def _result(rows: list[Fazenda]) -> MagicMock:
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


async def test_create(db):
    async def _refresh(obj):
        obj.id, obj.created_at = "f-novo", NOW

    db.refresh.side_effect = _refresh
    out = await FazendaRepository(db).create("u-1", CreateFazendaRequest(nome="Nova", uf="mt"))
    assert out.id == "f-novo" and out.uf == "MT" and out.user_id == "u-1"
    db.add.assert_called_once()


async def test_find_all_converte_hectares(db):
    db.execute.return_value = _result([_row()])
    out = await FazendaRepository(db).find_all_by_user("u-1")
    assert out[0].hectares == 10.0 and isinstance(out[0].hectares, float)


async def test_default_reaproveita_a_mais_antiga(db):
    db.execute.return_value = _result([_row("f-velha"), _row("f-nova")])
    out = await FazendaRepository(db).get_or_create_default("u-1")
    assert out.id == "f-velha"
    db.add.assert_not_called()


async def test_default_cria_minha_fazenda(db):
    db.execute.return_value = _result([])

    async def _refresh(obj):
        obj.id, obj.created_at = "f-padrao", NOW

    db.refresh.side_effect = _refresh
    out = await FazendaRepository(db).get_or_create_default("u-1")
    assert out.nome == DEFAULT_NOME and out.id == "f-padrao"


async def test_find_by_id_ausente(db):
    db.execute.return_value = _result([])
    assert await FazendaRepository(db).find_by_id("x", "u-1") is None


async def test_update_parcial(db):
    row = _row()
    db.execute.return_value = _result([row])
    out = await FazendaRepository(db).update("f-1", "u-1", UpdateFazendaRequest(municipio="Jataí"))
    assert out is not None and out.municipio == "Jataí" and out.nome == "Boa Vista"


async def test_update_ausente(db):
    db.execute.return_value = _result([])
    assert await FazendaRepository(db).update("x", "u-1", UpdateFazendaRequest()) is None


async def test_delete(db):
    db.execute.return_value = _result([_row()])
    assert await FazendaRepository(db).delete("f-1", "u-1") is True
    db.delete.assert_awaited_once()


async def test_delete_ausente(db):
    db.execute.return_value = _result([])
    assert await FazendaRepository(db).delete("x", "u-1") is False
