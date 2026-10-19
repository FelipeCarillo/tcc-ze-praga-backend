"""Integration tests for POST /api/v1/diagnoses/analyze (TCC-042).

Endpoint REST direto pra diagnostico — invoca o sub-grafo diagnosis_graph
e retorna a lista de Diagnosis persistidos. Mockamos o factory + o repo
de diagnoses, deixando o roteamento real.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_current_user,
    get_current_user_or_api_key,
    get_diagnosis_graph_factory,
    get_diagnosis_repository,
    get_diagnosis_service,
    get_plan_features_dual,
    get_subscription_repository,
    get_usage_repository,
    get_usage_service,
    require_quota,
    require_quota_dual,
)
from app.domains.diagnoses.service import DiagnosisService
from app.domains.subscriptions.features import ENTERPRISE_FEATURES, FREE_FEATURES
from app.main import app
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_diagnosis_dto, make_user_dto


def _fake_signer(keys: list[str]) -> dict[str, str]:
    return {k: f"https://storage.test/signed/{k}?token=abc" for k in keys}


@pytest.fixture
def mock_graph_factory():
    """Factory mock que retorna um grafo simulado por crop_id."""
    factory = MagicMock()

    async def _ainvoke(state):
        n = len(state.get("image_ids", []))
        return {
            "persisted_ids": [f"diag-{i}" for i in range(n)],
            "predictions": [{} for _ in range(n)],
        }

    graph = MagicMock()
    graph.ainvoke = _ainvoke
    factory.return_value = graph

    def _factory_fn(crop_id):
        factory(crop_id)
        return graph

    _factory_fn.calls = factory  # type: ignore[attr-defined]
    return _factory_fn


@pytest.fixture
def mock_diag_repo():
    repo = AsyncMock()
    repo.find_by_id = AsyncMock(side_effect=lambda diag_id, _user_id: make_diagnosis_dto(id=diag_id))
    return repo


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    return svc


@pytest.fixture
async def client_analyze(mock_graph_factory, mock_diag_repo, mock_usage_svc):
    app.dependency_overrides[require_quota(FeatureTypeEnum.INFERENCE)] = (
        lambda: make_user_dto()
    )
    app.dependency_overrides[require_quota_dual] = lambda: make_user_dto()
    app.dependency_overrides[get_diagnosis_graph_factory] = (
        lambda: mock_graph_factory
    )
    app.dependency_overrides[get_diagnosis_repository] = lambda: mock_diag_repo
    # TCC-105: o router le pelo DiagnosisService (que assina as imagens).
    app.dependency_overrides[get_diagnosis_service] = lambda: DiagnosisService(
        mock_diag_repo, image_url_resolver=_fake_signer
    )
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_usage_repository] = lambda: AsyncMock()
    app.dependency_overrides[get_subscription_repository] = lambda: AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_current_user_or_api_key] = lambda: make_user_dto()
    app.dependency_overrides[get_plan_features_dual] = lambda: ENTERPRISE_FEATURES
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Single image ──────────────────────────────────────────────────────────────


async def test_analyze_single_image_returns_one_diagnosis(
    client_analyze, mock_graph_factory, mock_diag_repo, mock_usage_svc
):
    files = {"images": ("leaf.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")}
    data = {"crop_id": "soja", "model": "ensemble"}

    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", data=data, files=files
    )

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["id"] == "diag-0"
    assert mock_graph_factory.calls.call_args.args[0] == "soja"  # type: ignore[attr-defined]
    mock_diag_repo.find_by_id.assert_awaited_once_with(
        "diag-0", make_user_dto().id
    )
    mock_usage_svc.record_usage.assert_awaited_once()


async def test_analyze_default_crop_and_model(client_analyze):
    """Quando crop_id/model nao sao passados, defaults sao usados."""
    files = {"images": ("leaf.jpg", io.BytesIO(b"X"), "image/jpeg")}
    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", files=files
    )
    assert r.status_code == 200


# ── Batch images ──────────────────────────────────────────────────────────────


async def test_analyze_batch_returns_one_per_image(
    client_analyze, mock_graph_factory, mock_diag_repo
):
    files = [
        ("images", ("a.jpg", io.BytesIO(b"a"), "image/jpeg")),
        ("images", ("b.jpg", io.BytesIO(b"b"), "image/jpeg")),
        ("images", ("c.jpg", io.BytesIO(b"c"), "image/jpeg")),
    ]
    data = {"crop_id": "soja", "model": "vit"}

    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", data=data, files=files
    )

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert [d["id"] for d in body] == ["diag-0", "diag-1", "diag-2"]
    assert mock_diag_repo.find_by_id.await_count == 3


async def test_analyze_records_usage_with_batch_metadata(
    client_analyze, mock_usage_svc
):
    files = [
        ("images", ("a.jpg", io.BytesIO(b"a"), "image/jpeg")),
        ("images", ("b.jpg", io.BytesIO(b"b"), "image/jpeg")),
    ]

    await client_analyze.post(
        "/api/v1/diagnoses/analyze",
        data={"crop_id": "milho", "model": "vit"},
        files=files,
    )

    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert metadata == {
        "crop_id": "milho",
        # O router normaliza o id pro canonico do registro antes de rodar o
        # sub-grafo, e guarda o pedido original em ``model_requested``.
        "model": "vit_b16",
        "model_requested": "vit",
        "batch_size": 2,
        "images_failed": 0,
        "auth_method": "jwt",
    }


async def test_analyze_missing_images_returns_422(client_analyze):
    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", data={"crop_id": "soja"}
    )
    assert r.status_code == 422


async def test_analyze_returns_404_when_persisted_diag_missing(
    client_analyze, mock_diag_repo
):
    """Se o repo nao encontrar um id persistido, levanta NotFound."""
    mock_diag_repo.find_by_id.side_effect = None
    mock_diag_repo.find_by_id.return_value = None

    files = {"images": ("a.jpg", io.BytesIO(b"a"), "image/jpeg")}
    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", data={"crop_id": "soja"}, files=files
    )
    assert r.status_code == 404


# ── TCC-105: resposta completa do /analyze ───────────────────────────────────


def _graph_returning(result):
    """Factory de grafo cujo ``ainvoke`` devolve ``result`` fixo."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=result)
    return lambda _crop_id: graph


_ACTION_PLAN_RAW = {
    "disease_id": "ferrugem-asiatica",
    "levels": [{"level": "essencial", "actions": ["Aplicar fungicida"]}],
    "sources": [{"name": "EMBRAPA", "url": "https://embrapa.br"}],
}


async def test_analyze_signs_image_url_and_returns_sources(
    client_analyze, mock_diag_repo
):
    """A storage key vira URL assinada e as fontes persistidas voltam."""
    source = {
        "type": "scientific",
        "url": "https://scielo.br/x",
        "title": "Ferrugem na soja",
        "snippet": None,
        "doi": "10.1590/x",
    }
    mock_diag_repo.find_by_id.side_effect = lambda diag_id, _uid: make_diagnosis_dto(
        id=diag_id, image_url="users/u1/abc-leaf.jpg", sources=[source]
    )
    files = {"images": ("leaf.jpg", io.BytesIO(b"a"), "image/jpeg")}

    r = await client_analyze.post("/api/v1/diagnoses/analyze", files=files)

    assert r.status_code == 200
    body = r.json()[0]
    assert body["image_url"] == (
        "https://storage.test/signed/users/u1/abc-leaf.jpg?token=abc"
    )
    assert body["sources"] == [source]


async def test_analyze_returns_action_plan_with_allowed_levels(
    client_analyze,
):
    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: _graph_returning(
        {"persisted_ids": ["diag-0"], "action_plans": [_ACTION_PLAN_RAW]}
    )
    files = {"images": ("leaf.jpg", io.BytesIO(b"a"), "image/jpeg")}

    r = await client_analyze.post("/api/v1/diagnoses/analyze", files=files)

    plan = r.json()[0]["action_plan"]
    assert plan["levels"] == [
        {"level": "essencial", "actions": ["Aplicar fungicida"]}
    ]
    assert plan["allowed_levels"] == ["essencial", "campo", "especialista"]
    assert plan["sources"] == [{"name": "EMBRAPA", "url": "https://embrapa.br"}]


async def test_analyze_action_plan_is_null_without_plan(client_analyze):
    """Folha saudavel: o node devolve plano vazio e a resposta traz ``null``."""
    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: _graph_returning(
        {
            "persisted_ids": ["diag-0"],
            "action_plans": [{"disease_id": "saudavel", "levels": [], "sources": []}],
        }
    )
    files = {"images": ("leaf.jpg", io.BytesIO(b"a"), "image/jpeg")}

    r = await client_analyze.post("/api/v1/diagnoses/analyze", files=files)

    assert r.json()[0]["action_plan"] is None


async def test_analyze_reports_partial_failures_in_header(
    client_analyze, mock_usage_svc
):
    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: _graph_returning(
        {
            "persisted_ids": ["diag-0"],
            "errors": [{"image_name": "borrada.jpg", "error": "Imagem invalida."}],
        }
    )
    files = [
        ("images", ("ok.jpg", io.BytesIO(b"a"), "image/jpeg")),
        ("images", ("borrada.jpg", io.BytesIO(b"b"), "image/jpeg")),
    ]

    r = await client_analyze.post("/api/v1/diagnoses/analyze", files=files)

    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.headers["X-Images-Failed"] == "1"
    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert metadata["images_failed"] == 1


async def test_analyze_all_images_failed_returns_503_without_usage(
    client_analyze, mock_usage_svc
):
    """Lote sem nenhuma classificacao nao e' ``200 []`` nem consome cota."""
    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: _graph_returning(
        {
            "persisted_ids": [],
            "errors": [{"image_name": "borrada.jpg", "error": "Imagem invalida."}],
        }
    )
    files = {"images": ("borrada.jpg", io.BytesIO(b"b"), "image/jpeg")}

    r = await client_analyze.post("/api/v1/diagnoses/analyze", files=files)

    assert r.status_code == 503
    assert "borrada.jpg" in r.json()["detail"]
    mock_usage_svc.record_usage.assert_not_awaited()


async def test_analyze_reports_model_downgrade_in_headers(client_analyze):
    """Free pedindo ensemble roda ResNet-50 e avisa nos headers."""
    app.dependency_overrides[get_plan_features_dual] = lambda: FREE_FEATURES
    files = {"images": ("leaf.jpg", io.BytesIO(b"a"), "image/jpeg")}

    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", data={"model": "ensemble"}, files=files
    )

    assert r.status_code == 200
    assert r.headers["X-Model-Used"] == "resnet50"
    assert r.headers["X-Model-Downgraded"] == "true"
    assert r.headers["X-Images-Failed"] == "0"


async def test_analyze_no_downgrade_when_model_allowed(client_analyze):
    files = {"images": ("leaf.jpg", io.BytesIO(b"a"), "image/jpeg")}

    r = await client_analyze.post(
        "/api/v1/diagnoses/analyze", data={"model": "ensemble"}, files=files
    )

    assert r.headers["X-Model-Used"] == "ensemble"
    assert r.headers["X-Model-Downgraded"] == "false"
