"""TCC-096 — fazendas com os talhões aninhados e talhões que pertencem a uma fazenda."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import NotFoundError
from app.domains.fazendas.dto import FazendaDTO
from app.domains.fazendas.schemas import CreateFazendaRequest, UpdateFazendaRequest
from app.domains.fazendas.service import FazendaService
from app.domains.talhoes.dto import TalhaoDTO
from app.domains.talhoes.schemas import CreateTalhaoRequest, UpdateTalhaoRequest
from app.domains.talhoes.service import TalhaoService
from tests.conftest import NOW


def _fazenda(id_: str = "f-1", nome: str = "Boa Vista") -> FazendaDTO:
    return FazendaDTO(
        id=id_,
        user_id="u-1",
        nome=nome,
        municipio="Rio Verde",
        uf="GO",
        hectares=182.0,
        agronomo_nome=None,
        agronomo_crea=None,
        created_at=NOW,
    )


def _talhao(id_: str, fazenda_id: str, nome: str = "Sede") -> TalhaoDTO:
    return TalhaoDTO(
        id=id_,
        user_id="u-1",
        fazenda_id=fazenda_id,
        nome=nome,
        apelido=None,
        hectares=60.0,
        cultura="soja",
        data_semeadura=None,
        created_at=NOW,
    )


@pytest.fixture
def fazendas():
    r = AsyncMock()
    r.find_all_by_user = AsyncMock(return_value=[_fazenda("f-1"), _fazenda("f-2", "São José")])
    r.find_by_id = AsyncMock(return_value=_fazenda())
    r.get_or_create_default = AsyncMock(return_value=_fazenda("f-padrao", "Minha fazenda"))
    r.create = AsyncMock(return_value=_fazenda("f-nova", "Nova"))
    r.update = AsyncMock(return_value=_fazenda(nome="Boa Vista II"))
    r.delete = AsyncMock(return_value=True)
    return r


@pytest.fixture
def talhoes():
    r = AsyncMock()
    r.find_all_by_user = AsyncMock(
        return_value=[_talhao("t-1", "f-1"), _talhao("t-2", "f-1", "Baixada"), _talhao("t-3", "f-2")]
    )
    r.find_all_by_fazenda = AsyncMock(return_value=[_talhao("t-1", "f-1")])
    r.create = AsyncMock(side_effect=lambda user, faz, data: _talhao("t-novo", faz, data.nome))
    r.update = AsyncMock(return_value=_talhao("t-1", "f-2"))
    r.delete = AsyncMock(return_value=True)
    return r


# ── FazendaService ────────────────────────────────────────────────────────────


async def test_lista_fazendas_com_talhoes_aninhados(fazendas, talhoes):
    out = await FazendaService(fazendas, talhoes).list_for_user("u-1")

    assert [f.id for f in out] == ["f-1", "f-2"]
    assert [t.id for t in out[0].talhoes] == ["t-1", "t-2"]
    assert [t.id for t in out[1].talhoes] == ["t-3"]
    assert all(t.fazenda_id == "f-1" for t in out[0].talhoes)


async def test_fazenda_sem_talhao_vem_com_lista_vazia(fazendas, talhoes):
    talhoes.find_all_by_user.return_value = []
    out = await FazendaService(fazendas, talhoes).list_for_user("u-1")
    assert all(f.talhoes == [] for f in out)


async def test_cria_fazenda(fazendas, talhoes):
    out = await FazendaService(fazendas, talhoes).create("u-1", CreateFazendaRequest(nome="Nova"))
    assert out.id == "f-nova" and out.talhoes == []


async def test_atualiza_fazenda_devolve_os_talhoes(fazendas, talhoes):
    out = await FazendaService(fazendas, talhoes).update(
        "f-1", "u-1", UpdateFazendaRequest(nome="Boa Vista II")
    )
    assert out.nome == "Boa Vista II"
    assert [t.id for t in out.talhoes] == ["t-1"]
    talhoes.find_all_by_fazenda.assert_awaited_once_with("f-1", "u-1")


async def test_atualiza_fazenda_inexistente(fazendas, talhoes):
    fazendas.update.return_value = None
    with pytest.raises(NotFoundError):
        await FazendaService(fazendas, talhoes).update("x", "u-1", UpdateFazendaRequest())


async def test_apaga_fazenda_inexistente(fazendas, talhoes):
    fazendas.delete.return_value = False
    with pytest.raises(NotFoundError):
        await FazendaService(fazendas, talhoes).delete("x", "u-1")


async def test_apaga_fazenda(fazendas, talhoes):
    await FazendaService(fazendas, talhoes).delete("f-1", "u-1")
    fazendas.delete.assert_awaited_once_with("f-1", "u-1")


def test_uf_vira_maiuscula():
    assert CreateFazendaRequest(nome="X", uf="go").uf == "GO"


# ── TalhaoService ─────────────────────────────────────────────────────────────


async def test_talhao_sem_fazenda_vai_para_a_padrao(fazendas, talhoes):
    out = await TalhaoService(talhoes, fazendas).create("u-1", CreateTalhaoRequest(nome="Rio"))

    assert out.fazenda_id == "f-padrao"
    fazendas.get_or_create_default.assert_awaited_once_with("u-1")


async def test_talhao_na_fazenda_escolhida(fazendas, talhoes):
    out = await TalhaoService(talhoes, fazendas).create(
        "u-1", CreateTalhaoRequest(nome="Rio", fazenda_id="f-1")
    )
    assert out.fazenda_id == "f-1"
    fazendas.find_by_id.assert_awaited_once_with("f-1", "u-1")
    fazendas.get_or_create_default.assert_not_called()


async def test_talhao_em_fazenda_alheia_e_recusado(fazendas, talhoes):
    fazendas.find_by_id.return_value = None
    with pytest.raises(NotFoundError):
        await TalhaoService(talhoes, fazendas).create(
            "u-1", CreateTalhaoRequest(nome="Rio", fazenda_id="f-de-outro")
        )
    talhoes.create.assert_not_called()


async def test_lista_talhoes_de_uma_fazenda(fazendas, talhoes):
    out = await TalhaoService(talhoes, fazendas).list_for_user("u-1", "f-1")
    assert [t.id for t in out] == ["t-1"]
    talhoes.find_all_by_user.assert_not_called()


async def test_lista_todos_os_talhoes(fazendas, talhoes):
    out = await TalhaoService(talhoes, fazendas).list_for_user("u-1")
    assert len(out) == 3


async def test_move_talhao_de_fazenda(fazendas, talhoes):
    out = await TalhaoService(talhoes, fazendas).update(
        "t-1", "u-1", UpdateTalhaoRequest(fazenda_id="f-2")
    )
    assert out.fazenda_id == "f-2"
    fazendas.find_by_id.assert_awaited_once_with("f-2", "u-1")


async def test_mover_para_fazenda_alheia_e_recusado(fazendas, talhoes):
    fazendas.find_by_id.return_value = None
    with pytest.raises(NotFoundError):
        await TalhaoService(talhoes, fazendas).update(
            "t-1", "u-1", UpdateTalhaoRequest(fazenda_id="f-de-outro")
        )
    talhoes.update.assert_not_called()


async def test_editar_sem_mover_nao_consulta_fazenda(fazendas, talhoes):
    await TalhaoService(talhoes, fazendas).update("t-1", "u-1", UpdateTalhaoRequest(nome="Sede 2"))
    fazendas.find_by_id.assert_not_called()


async def test_editar_talhao_inexistente(fazendas, talhoes):
    talhoes.update.return_value = None
    with pytest.raises(NotFoundError):
        await TalhaoService(talhoes, fazendas).update("x", "u-1", UpdateTalhaoRequest(nome="A"))


async def test_apagar_talhao_inexistente(fazendas, talhoes):
    talhoes.delete.return_value = False
    with pytest.raises(NotFoundError):
        await TalhaoService(talhoes, fazendas).delete("x", "u-1")
