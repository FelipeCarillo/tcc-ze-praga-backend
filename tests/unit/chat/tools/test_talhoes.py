"""TCC-098 — tools de talhão do agente (listar, escolher, cadastrar)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Command

from app.core.exceptions import NotFoundError
from app.domains.chat.tools.talhoes import build_talhao_tools
from app.domains.fazendas.dto import FazendaDTO
from app.domains.talhoes.dto import TalhaoDTO
from app.domains.talhoes.schemas import TalhaoResponse
from tests.conftest import NOW

MOD = "app.domains.chat.tools.talhoes"


@asynccontextmanager
async def _session():
    yield MagicMock()


def _fazenda(id_="f-1", nome="Boa Vista") -> FazendaDTO:
    return FazendaDTO(id_, "u-1", nome, None, None, None, None, None, NOW)


def _talhao(id_="t-1", fazenda_id="f-1", nome="Talhão 3", apelido="Sede") -> TalhaoDTO:
    return TalhaoDTO(id_, "u-1", fazenda_id, nome, apelido, 60.0, "soja", None, NOW)


def _call(tool, args, state):
    return tool.ainvoke(
        {"name": tool.name, "args": {**args, "state": state}, "id": "call-1", "type": "tool_call"}
    )


@pytest.fixture
def tools():
    return {name: factory() for name, factory in build_talhao_tools(_session).items()}


async def test_lista_fazendas_com_talhoes_e_escolhido(tools):
    faz = MagicMock(find_all_by_user=AsyncMock(return_value=[_fazenda(), _fazenda("f-2", "São José")]))
    tal = MagicMock(find_all_by_user=AsyncMock(return_value=[_talhao(), _talhao("t-2", "f-2", "T1", None)]))
    with patch(f"{MOD}.FazendaRepository", return_value=faz), patch(f"{MOD}.TalhaoRepository", return_value=tal):
        out = await tools["list_my_talhoes"].ainvoke(
            {"state": {"current_user_id": "u-1", "selected_talhao_id": "t-1"}}
        )
    body = json.loads(out)
    assert body["selected_talhao_id"] == "t-1"
    assert [f["nome"] for f in body["fazendas"]] == ["Boa Vista", "São José"]
    assert body["fazendas"][0]["talhoes"] == [{"id": "t-1", "nome": "Talhão 3", "apelido": "Sede"}]
    faz.find_all_by_user.assert_awaited_once_with("u-1")


async def test_use_talhao_escolhe_e_avisa_a_ui(tools):
    tal = MagicMock(find_by_id=AsyncMock(return_value=_talhao()))
    faz = MagicMock(find_by_id=AsyncMock(return_value=_fazenda()))
    with patch(f"{MOD}.FazendaRepository", return_value=faz), patch(f"{MOD}.TalhaoRepository", return_value=tal):
        result = await _call(tools["use_talhao"], {"talhao_id": "t-1"}, {"current_user_id": "u-1"})
    assert isinstance(result, Command)
    assert result.update["selected_talhao_id"] == "t-1"
    sel = result.update["talhao_selected"]
    assert sel["nome"] == "Talhão 3" and sel["fazenda_nome"] == "Boa Vista" and sel["created"] is False
    tal.find_by_id.assert_awaited_once_with("t-1", "u-1")


async def test_use_talhao_alheio_nao_escolhe(tools):
    tal = MagicMock(find_by_id=AsyncMock(return_value=None))
    with patch(f"{MOD}.FazendaRepository", return_value=MagicMock()), patch(f"{MOD}.TalhaoRepository", return_value=tal):
        result = await _call(tools["use_talhao"], {"talhao_id": "t-x"}, {"current_user_id": "u-1"})
    assert "selected_talhao_id" not in result.update
    assert "error" in json.loads(result.update["messages"][0].content)


def _created(fazenda_id="f-1", **kw) -> TalhaoResponse:
    base = dict(id="t-9", fazenda_id=fazenda_id, nome="Talhão 9", apelido="Rio", hectares=40.0,
                cultura="soja", data_semeadura=date(2026, 9, 12), created_at=NOW)
    return TalhaoResponse(**{**base, **kw})


async def test_register_talhao_cria_e_escolhe(tools):
    svc = MagicMock(create=AsyncMock(return_value=_created()))
    faz = MagicMock(find_by_id=AsyncMock(return_value=_fazenda()))
    with patch(f"{MOD}.FazendaRepository", return_value=faz), patch(f"{MOD}.TalhaoRepository"), patch(
        f"{MOD}.TalhaoService", return_value=svc
    ):
        result = await _call(
            tools["register_talhao"],
            {"nome": " Talhão 9 ", "apelido": "Rio", "hectares": 40, "data_semeadura": "2026-09-12"},
            {"current_user_id": "u-1"},
        )
    req = svc.create.await_args.args[1]
    assert req.nome == "Talhão 9" and req.hectares == 40 and req.data_semeadura == date(2026, 9, 12)
    assert req.fazenda_id is None
    assert result.update["selected_talhao_id"] == "t-9"
    sel = result.update["talhao_selected"]
    assert sel["created"] is True and sel["data_semeadura"] == "2026-09-12" and sel["fazenda_nome"] == "Boa Vista"


async def test_register_talhao_data_invalida_e_area_negativa(tools):
    svc = MagicMock(create=AsyncMock(return_value=_created(data_semeadura=None)))
    with patch(f"{MOD}.FazendaRepository", return_value=MagicMock(find_by_id=AsyncMock(return_value=None))), patch(
        f"{MOD}.TalhaoRepository"
    ), patch(f"{MOD}.TalhaoService", return_value=svc):
        result = await _call(
            tools["register_talhao"],
            {"nome": "T", "hectares": -3, "data_semeadura": "12 de setembro"},
            {"current_user_id": "u-1"},
        )
    req = svc.create.await_args.args[1]
    assert req.hectares is None and req.data_semeadura is None
    assert result.update["talhao_selected"]["fazenda_nome"] is None


async def test_register_talhao_fazenda_inventada_cai_na_padrao(tools):
    svc = MagicMock(create=AsyncMock(side_effect=[NotFoundError("Fazenda", "x"), _created()]))
    with patch(f"{MOD}.FazendaRepository", return_value=MagicMock(find_by_id=AsyncMock(return_value=_fazenda()))), patch(
        f"{MOD}.TalhaoRepository"
    ), patch(f"{MOD}.TalhaoService", return_value=svc):
        result = await _call(
            tools["register_talhao"], {"nome": "T9", "fazenda_id": "inventada"}, {"current_user_id": "u-1"}
        )
    assert svc.create.await_count == 2
    assert svc.create.await_args_list[1].args[1].fazenda_id is None
    assert result.update["selected_talhao_id"] == "t-9"


async def test_use_talhao_depois_de_cadastrar_mantem_criado(tools):
    """Visto com o GPT real: register_talhao seguido de use_talhao no mesmo talhão."""
    tal = MagicMock(find_by_id=AsyncMock(return_value=_talhao("t-9")))
    faz = MagicMock(find_by_id=AsyncMock(return_value=_fazenda()))
    state = {"current_user_id": "u-1", "talhao_selected": {"id": "t-9", "created": True}}
    with patch(f"{MOD}.FazendaRepository", return_value=faz), patch(f"{MOD}.TalhaoRepository", return_value=tal):
        result = await _call(tools["use_talhao"], {"talhao_id": "t-9"}, state)
    assert result.update["talhao_selected"]["created"] is True
