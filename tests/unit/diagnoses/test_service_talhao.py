"""TCC-093 — laudos ligados a talhao no DiagnosisService."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import NotFoundError
from app.domains.diagnoses.dto import TalhaoGroupDTO
from app.domains.diagnoses.schemas import CreateDiagnosisRequest
from app.domains.diagnoses.service import RECENT_PER_TALHAO, DiagnosisService
from app.shared.enums import SeverityEnum
from tests.conftest import NOW, make_diagnosis_dto


def _request(talhao_id: str | None) -> CreateDiagnosisRequest:
    return CreateDiagnosisRequest(
        disease_name="Ferrugem",
        disease_id="ferrugem-asiatica",
        confidence=0.9,
        severity=SeverityEnum.ALTA,
        model_used="ensemble",
        talhao_id=talhao_id,
    )


@pytest.fixture
def repo():
    r = AsyncMock()
    r.create = AsyncMock(return_value=make_diagnosis_dto())
    r.talhao_belongs_to_user = AsyncMock(return_value=True)
    r.set_talhao = AsyncMock(
        return_value=make_diagnosis_dto(talhao_id="t-1", talhao_nome="Sede")
    )
    r.group_by_talhao = AsyncMock(return_value=[])
    return r


# ── create ────────────────────────────────────────────────────────────────────


async def test_create_mantem_talhao_do_proprio_usuario(repo):
    await DiagnosisService(repo).create("u-1", _request("t-1"), crop_id="c")
    repo.talhao_belongs_to_user.assert_awaited_once_with("t-1", "u-1")
    assert repo.create.await_args.args[1].talhao_id == "t-1"


async def test_create_descarta_talhao_de_outro_usuario_sem_falhar(repo):
    """Talhao alheio vira "Sem talhao": o laudo nao pode ser perdido."""
    repo.talhao_belongs_to_user.return_value = False
    result = await DiagnosisService(repo).create("u-1", _request("t-alheio"), crop_id="c")
    assert repo.create.await_args.args[1].talhao_id is None
    assert result.disease_id == "ferrugem-asiatica"


async def test_create_sem_talhao_nao_consulta_posse(repo):
    await DiagnosisService(repo).create("u-1", _request(None), crop_id="c")
    repo.talhao_belongs_to_user.assert_not_awaited()


async def test_resposta_expoe_talhao(repo):
    repo.create.return_value = make_diagnosis_dto(talhao_id="t-1", talhao_nome="Sede")
    result = await DiagnosisService(repo).create("u-1", _request("t-1"), crop_id="c")
    assert (result.talhao_id, result.talhao_nome) == ("t-1", "Sede")


# ── set_talhao ────────────────────────────────────────────────────────────────


async def test_set_talhao_move_o_laudo(repo):
    result = await DiagnosisService(repo).set_talhao("d-1", "u-1", "t-1")
    repo.set_talhao.assert_awaited_once_with("d-1", "u-1", "t-1")
    assert result.talhao_nome == "Sede"


async def test_set_talhao_recusa_talhao_de_outro_usuario(repo):
    repo.talhao_belongs_to_user.return_value = False
    with pytest.raises(NotFoundError, match="Talhao"):
        await DiagnosisService(repo).set_talhao("d-1", "u-1", "t-alheio")
    repo.set_talhao.assert_not_awaited()


async def test_set_talhao_none_desfaz_vinculo_sem_checar_posse(repo):
    repo.set_talhao.return_value = make_diagnosis_dto()
    result = await DiagnosisService(repo).set_talhao("d-1", "u-1", None)
    repo.talhao_belongs_to_user.assert_not_awaited()
    assert result.talhao_id is None


async def test_set_talhao_laudo_inexistente(repo):
    repo.set_talhao.return_value = None
    with pytest.raises(NotFoundError, match="Diagnosis"):
        await DiagnosisService(repo).set_talhao("sumiu", "u-1", "t-1")


# ── group_by_talhao ───────────────────────────────────────────────────────────


async def test_group_by_talhao_tendencia_do_mais_antigo_ao_mais_recente(repo):
    recentes = [  # o repo devolve do mais recente pro mais antigo
        make_diagnosis_dto(id="d3", severity="alta", created_at=NOW),
        make_diagnosis_dto(id="d2", severity="media", created_at=NOW - timedelta(days=7)),
        make_diagnosis_dto(id="d1", severity="nenhuma", created_at=NOW - timedelta(days=30)),
    ]
    repo.group_by_talhao.return_value = [
        TalhaoGroupDTO(talhao_id="t-1", talhao_nome="Sede", total=5, last_at=NOW, recent=recentes),
        TalhaoGroupDTO(talhao_id=None, talhao_nome=None, total=1, last_at=NOW, recent=[]),
    ]
    groups = await DiagnosisService(repo).group_by_talhao("u-1")

    repo.group_by_talhao.assert_awaited_once_with("u-1", RECENT_PER_TALHAO)
    assert groups[0].severity_trend == ["nenhuma", "media", "alta"]
    assert [d.id for d in groups[0].recent] == ["d3", "d2", "d1"]
    assert groups[0].total == 5
    assert groups[1].talhao_id is None and groups[1].recent == []


async def test_group_by_talhao_assina_todas_as_miniaturas_de_uma_vez(repo):
    repo.group_by_talhao.return_value = [
        TalhaoGroupDTO(
            talhao_id="t-1", talhao_nome="Sede", total=1, last_at=NOW,
            recent=[make_diagnosis_dto(id="a", image_url="users/u/a.jpg")],
        ),
        TalhaoGroupDTO(
            talhao_id="t-2", talhao_nome="Baixada", total=1, last_at=NOW,
            recent=[make_diagnosis_dto(id="b", image_url="users/u/b.jpg")],
        ),
    ]
    resolver = MagicMock(
        return_value={"users/u/a.jpg": "https://s/a", "users/u/b.jpg": "https://s/b"}
    )
    groups = await DiagnosisService(repo, image_url_resolver=resolver).group_by_talhao("u-1")

    resolver.assert_called_once()
    assert sorted(resolver.call_args.args[0]) == ["users/u/a.jpg", "users/u/b.jpg"]
    assert groups[0].recent[0].image_url == "https://s/a"
    assert groups[1].recent[0].image_url == "https://s/b"
