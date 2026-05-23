"""Testes do gather_evidence_node + paralelismo no diagnosis_graph (TCC-055).

Cobertura:
- Tier gating: Free, Pro, Enterprise (e dicts vazios)
- Sem callables injetadas: skipa mesmo com plan_features ativos
- Sucesso: web + scientific mergeados em ordem
- Falha em uma sub-call: filtra exception, mantem outras
- JSON invalido retornado pela tool: ignora
- Empty predictions: retorna []
- Full graph: gather_evidence + compose_action_plan rodam em paralelo (smoke test)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.diagnoses.schemas import DiagnosisResponse
from app.domains.diagnosis_graph.gather_evidence import (
    _parse_json_payload,
    gather_evidence_node,
)
from app.domains.diagnosis_graph.graph import build_diagnosis_graph
from tests.conftest import NOW


def _pred(disease_name: str = "Ferrugem Asiática") -> dict:
    return {
        "disease_id": "ferrugem-asiatica",
        "disease_name": disease_name,
        "confidence": 0.91,
        "severity": "alta",
    }


# ── tier gating ────────────────────────────────────────────────────────────


async def test_free_tier_skips_evidence() -> None:
    """plan_features sem search_web/scientific -> retorna []."""
    tavily = AsyncMock(return_value=json.dumps([{"title": "T"}]))
    scielo = AsyncMock(return_value=json.dumps([{"title": "S"}]))
    state = {
        "crop_id": "soja",
        "plan_features": {"tier_name": "free"},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(
        state, tavily_search=tavily, scielo_search=scielo
    )

    assert update == {"evidence_per_image": []}
    tavily.assert_not_called()
    scielo.assert_not_called()


async def test_pro_tier_calls_web_only() -> None:
    """Pro tier ativa search_web; nao scientific."""
    tavily = AsyncMock(return_value=json.dumps([{"title": "Web"}]))
    scielo = AsyncMock(return_value=json.dumps([{"title": "Sci"}]))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True, "search_scientific": False},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(
        state, tavily_search=tavily, scielo_search=scielo
    )

    assert tavily.await_count == 1
    scielo.assert_not_called()
    assert update["evidence_per_image"][0] == [{"title": "Web"}]


async def test_enterprise_tier_calls_both() -> None:
    """Enterprise ativa web + scientific."""
    tavily = AsyncMock(return_value=json.dumps([{"title": "Web"}]))
    scielo = AsyncMock(return_value=json.dumps([{"title": "Sci"}]))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True, "search_scientific": True},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(
        state, tavily_search=tavily, scielo_search=scielo
    )

    assert tavily.await_count == 1
    assert scielo.await_count == 1
    # Merge mantem ordem (web vem antes de scientific)
    merged = update["evidence_per_image"][0]
    assert {"title": "Web"} in merged
    assert {"title": "Sci"} in merged
    assert len(merged) == 2


async def test_missing_plan_features_treated_as_free() -> None:
    """Sem plan_features no state, comporta como free."""
    tavily = AsyncMock(return_value="[]")
    state = {"crop_id": "soja", "predictions": [_pred()]}

    update = await gather_evidence_node(state, tavily_search=tavily)

    assert update == {"evidence_per_image": []}
    tavily.assert_not_called()


async def test_no_callables_injected_skipa() -> None:
    """plan_features quer mas callables sao None -> skipa."""
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True, "search_scientific": True},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(state)

    assert update == {"evidence_per_image": []}


# ── multiple predictions + errors ───────────────────────────────────────────


async def test_multiple_predictions_each_get_evidence() -> None:
    """Batch de N imagens -> evidence_per_image tem N entries."""
    tavily = AsyncMock(return_value=json.dumps([{"title": "Web"}]))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True},
        "predictions": [_pred("D1"), _pred("D2"), _pred("D3")],
    }

    update = await gather_evidence_node(state, tavily_search=tavily)

    assert len(update["evidence_per_image"]) == 3
    assert tavily.await_count == 3


async def test_sub_call_failure_filtered() -> None:
    """Quando tavily levanta, scielo result eh mantido."""
    tavily = AsyncMock(side_effect=RuntimeError("tavily down"))
    scielo = AsyncMock(return_value=json.dumps([{"title": "Sci"}]))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True, "search_scientific": True},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(
        state, tavily_search=tavily, scielo_search=scielo
    )

    # Apenas scientific result presente
    assert update["evidence_per_image"][0] == [{"title": "Sci"}]


async def test_both_sub_calls_failing_returns_empty_for_image() -> None:
    tavily = AsyncMock(side_effect=RuntimeError("net"))
    scielo = AsyncMock(side_effect=RuntimeError("net"))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True, "search_scientific": True},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(
        state, tavily_search=tavily, scielo_search=scielo
    )

    assert update["evidence_per_image"] == [[]]


async def test_invalid_json_payload_ignored() -> None:
    """Tool retornando string nao-JSON nao quebra o node."""
    tavily = AsyncMock(return_value="not a json")
    scielo = AsyncMock(return_value=json.dumps([{"title": "Sci"}]))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True, "search_scientific": True},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(
        state, tavily_search=tavily, scielo_search=scielo
    )

    assert update["evidence_per_image"][0] == [{"title": "Sci"}]


async def test_tool_returns_error_dict_treated_as_empty() -> None:
    """Tool retorna {"error": "..."} (dict, nao lista) -> ignorado."""
    tavily = AsyncMock(return_value=json.dumps({"error": "boom"}))
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True},
        "predictions": [_pred()],
    }

    update = await gather_evidence_node(state, tavily_search=tavily)

    assert update["evidence_per_image"] == [[]]


async def test_empty_predictions_returns_empty_list() -> None:
    """Predictions vazias -> evidence_per_image=[]."""
    tavily = AsyncMock(return_value="[]")
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True},
        "predictions": [],
    }

    update = await gather_evidence_node(state, tavily_search=tavily)

    assert update["evidence_per_image"] == []
    tavily.assert_not_called()


async def test_query_includes_disease_and_crop() -> None:
    """Query enviada inclui disease_name + crop_id."""
    tavily = AsyncMock(return_value="[]")
    state = {
        "crop_id": "soja",
        "plan_features": {"search_web": True},
        "predictions": [_pred("Ferrugem Asiática")],
    }

    await gather_evidence_node(state, tavily_search=tavily)

    call_arg = tavily.await_args.args[0]
    assert "Ferrugem Asiática" in call_arg
    assert "soja" in call_arg
    assert "manejo" in call_arg


# ── _parse_json_payload helper ──────────────────────────────────────────────


def test_parse_json_payload_list_of_dicts() -> None:
    assert _parse_json_payload('[{"a":1}]') == [{"a": 1}]


def test_parse_json_payload_dict_returns_dict() -> None:
    out = _parse_json_payload('{"error": "x"}')
    assert out == {"error": "x"}


def test_parse_json_payload_invalid_returns_none() -> None:
    assert _parse_json_payload("not json") is None


def test_parse_json_payload_filters_non_dicts_in_list() -> None:
    out = _parse_json_payload('[{"a":1}, "string", 42, {"b":2}]')
    assert out == [{"a": 1}, {"b": 2}]


def test_parse_json_payload_non_string_returns_none() -> None:
    assert _parse_json_payload(123) is None  # type: ignore[arg-type]


# ── Full graph integration: paralelismo ────────────────────────────────────


def _diagnosis_response(diagnosis_id: str = "diag-1") -> DiagnosisResponse:
    return DiagnosisResponse(
        id=diagnosis_id,
        disease_name="Ferrugem Asiática",
        disease_id="ferrugem-asiatica",
        scientific_name="Phakopsora pachyrhizi",
        confidence=0.91,
        severity="alta",
        description="Doença severa.",
        model_used="ensemble",
        image_url=None,
        image_name="leaf.jpg",
        created_at=NOW,
        top3=[],
    )


@pytest.fixture
def mock_inference_svc():
    from app.domains.inference.schemas import InferenceResult
    from app.shared.enums import SeverityEnum

    svc = MagicMock()
    svc.predict.return_value = InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença severa.",
        confidence=0.91,
        model_id="ensemble",
        image_name="leaf.jpg",
        top3=[],
    )
    return svc


@pytest.fixture
def mock_action_plan_svc():
    from app.domains.action_plans.schemas import (
        ActionPlanLevelResponse,
        ActionPlanResponse,
    )
    from app.shared.enums import ActionPlanLevelEnum

    svc = AsyncMock()
    svc.get_by_disease.return_value = ActionPlanResponse(
        disease_id="ferrugem-asiatica",
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL,
                actions=["Aplicar fungicida"],
            )
        ],
        sources=[],
    )
    return svc


@pytest.fixture
def mock_diagnosis_svc():
    svc = AsyncMock()
    svc.create.return_value = _diagnosis_response()
    return svc


async def test_full_graph_runs_evidence_in_parallel(
    mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
):
    """Smoke test do grafo completo com gather_evidence ativo (Enterprise)."""
    tavily = AsyncMock(return_value=json.dumps([{"title": "Web result"}]))
    scielo = AsyncMock(
        return_value=json.dumps([{"title": "Scientific paper"}])
    )

    graph = build_diagnosis_graph(
        mock_inference_svc,
        mock_action_plan_svc,
        mock_diagnosis_svc,
        tavily_search=tavily,
        scielo_search=scielo,
    )

    final = await graph.ainvoke(
        {
            "user_id": "user-uuid-1",
            "crop_id": "soja-id",
            "image_batch": [],
            "image_ids": ["leaf.jpg"],
            "model_id": "ensemble",
            "plan_features": {
                "search_web": True,
                "search_scientific": True,
            },
        }
    )

    assert final["persisted_ids"] == ["diag-1"]
    assert "action_plans" in final
    assert "evidence_per_image" in final
    assert len(final["evidence_per_image"]) == 1
    assert len(final["evidence_per_image"][0]) == 2
    # Both searches were invoked.
    assert tavily.await_count == 1
    assert scielo.await_count == 1
    # action_plan e persist tambem rodaram.
    mock_action_plan_svc.get_by_disease.assert_awaited_once()
    mock_diagnosis_svc.create.assert_awaited_once()


async def test_full_graph_free_tier_skips_evidence(
    mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
):
    """Free tier: evidence_per_image=[]; tools nao chamadas."""
    tavily = AsyncMock(return_value="[]")
    scielo = AsyncMock(return_value="[]")

    graph = build_diagnosis_graph(
        mock_inference_svc,
        mock_action_plan_svc,
        mock_diagnosis_svc,
        tavily_search=tavily,
        scielo_search=scielo,
    )

    final = await graph.ainvoke(
        {
            "user_id": "user-uuid-1",
            "crop_id": "soja-id",
            "image_batch": [],
            "image_ids": ["leaf.jpg"],
            "model_id": "resnet50",
            "plan_features": {"tier_name": "free"},
        }
    )

    assert final["persisted_ids"] == ["diag-1"]
    assert final["evidence_per_image"] == []
    tavily.assert_not_called()
    scielo.assert_not_called()


async def test_full_graph_without_search_callables(
    mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
):
    """Sem search callables (modo back-compat), grafo ainda roda."""
    graph = build_diagnosis_graph(
        mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
    )

    final = await graph.ainvoke(
        {
            "user_id": "user-uuid-1",
            "crop_id": "soja-id",
            "image_batch": [],
            "image_ids": ["leaf.jpg"],
            "model_id": "resnet50",
            "plan_features": {
                "search_web": True,
                "search_scientific": True,
            },
        }
    )

    assert final["persisted_ids"] == ["diag-1"]
    assert final["evidence_per_image"] == []
