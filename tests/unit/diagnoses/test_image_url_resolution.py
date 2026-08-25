"""``image_url`` sai como URL utilizável, não como storage key crua.

Os três caminhos de persistência gravavam ``image_url=None``: a foto do chat é
efêmera (base64 no estado do turno) e nunca ia pro Storage. O `HistoryItem` do
frontend renderiza ``diagnosis.imageUrl`` e nunca recebia nada — a miniatura era
sempre o gradiente verde.

Agora o banco guarda a **storage key** (URL assinada expira; guardar uma URL
morta seria pior) e o `DiagnosisService` a converte na leitura, em lote.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.domains.diagnoses.dto import DiagnosisDTO
from app.domains.diagnoses.schemas import DiagnosisFilters
from app.domains.diagnoses.service import DiagnosisService

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _dto(diag_id: str = "d1", image_url: str | None = None) -> DiagnosisDTO:
    return DiagnosisDTO(
        id=diag_id,
        user_id="user-1",
        disease_name="Ferrugem Asiática",
        disease_id="ferrugem-asiatica",
        scientific_name="Phakopsora pachyrhizi",
        confidence=0.92,
        severity="alta",
        description="...",
        model_used="ensemble",
        image_url=image_url,
        image_name="folha.jpg",
        created_at=NOW,
        top3=[],
        sources=[],
    )


def _svc(resolver=None, dtos=None) -> DiagnosisService:
    repo = AsyncMock()
    repo.find_by_id = AsyncMock(return_value=(dtos or [_dto()])[0])
    repo.find_all_by_user = AsyncMock(return_value=(dtos or [_dto()], len(dtos or [1])))
    repo.create = AsyncMock(return_value=(dtos or [_dto()])[0])
    return DiagnosisService(repo, image_url_resolver=resolver)


async def test_storage_key_vira_url_assinada_na_leitura() -> None:
    resolver = MagicMock(return_value={"users/u/abc-folha.jpg": "https://signed/abc"})
    svc = _svc(resolver, [_dto(image_url="users/u/abc-folha.jpg")])

    resp = await svc.get_by_id("d1", "user-1")

    assert resp.image_url == "https://signed/abc"
    resolver.assert_called_once_with(["users/u/abc-folha.jpg"])


async def test_listagem_assina_tudo_em_uma_chamada_so() -> None:
    """20 diagnósticos por página não podem virar 20 round-trips ao storage."""
    dtos = [_dto(f"d{i}", f"users/u/key-{i}.jpg") for i in range(20)]
    resolver = MagicMock(
        return_value={f"users/u/key-{i}.jpg": f"https://signed/{i}" for i in range(20)}
    )
    svc = _svc(resolver, dtos)

    page = await svc.list_for_user("user-1", DiagnosisFilters())

    assert resolver.call_count == 1
    assert len(resolver.call_args.args[0]) == 20
    assert page.items[3].image_url == "https://signed/3"


async def test_diagnostico_sem_imagem_nao_chama_o_storage() -> None:
    """Diagnósticos antigos (pré-upload) têm image_url None."""
    resolver = MagicMock(return_value={})
    svc = _svc(resolver, [_dto(image_url=None)])

    resp = await svc.get_by_id("d1", "user-1")

    assert resp.image_url is None
    resolver.assert_not_called()


async def test_url_completa_passa_direto_sem_assinar() -> None:
    svc = _svc(MagicMock(return_value={}), [_dto(image_url="https://cdn/x.jpg")])

    resp = await svc.get_by_id("d1", "user-1")

    assert resp.image_url == "https://cdn/x.jpg"


async def test_chave_que_falhou_ao_assinar_nao_quebra_a_resposta() -> None:
    """Storage fora do ar devolve {} — a tela abre sem miniatura, não com erro."""
    svc = _svc(MagicMock(return_value={}), [_dto(image_url="users/u/abc.jpg")])

    resp = await svc.get_by_id("d1", "user-1")

    # Sem URL assinada sobra a key crua; o <img> falha silenciosamente e o
    # placeholder aparece — melhor que derrubar o histórico inteiro.
    assert resp.image_url == "users/u/abc.jpg"


async def test_sem_resolvedor_mantem_comportamento_antigo() -> None:
    svc = _svc(None, [_dto(image_url="users/u/abc.jpg")])

    resp = await svc.get_by_id("d1", "user-1")

    assert resp.image_url == "users/u/abc.jpg"
