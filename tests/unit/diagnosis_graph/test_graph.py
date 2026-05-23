"""Testes do sub-grafo de diagnostico (TCC-040).

Estrategia:
    - Mockamos os 3 services (InferenceService, ActionPlanService, DiagnosisService).
    - Verificamos que o grafo passa pelos 4 nodes e produz o estado final esperado.
    - Cobrimos edge cases: crop_id ausente, action_plan svc levanta excecao, batch
      com varias imagens.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.action_plans.schemas import (
    ActionPlanLevelResponse,
    ActionPlanResponse,
    SourceResponse,
)
from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema
from app.domains.diagnosis_graph.graph import build_diagnosis_graph
from app.domains.diagnosis_graph.nodes import (
    compose_action_plan_node,
    load_model_node,
    persist_node,
    run_inference_node,
)
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import ActionPlanLevelEnum, SeverityEnum
from tests.conftest import NOW


# ── Helpers ──────────────────────────────────────────────────────────────────


def _inference_result(
    disease_id: str = "ferrugem-asiatica",
    disease_name: str = "Ferrugem Asiática",
) -> InferenceResult:
    return InferenceResult(
        disease_id=disease_id,
        disease_name=disease_name,
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença severa.",
        confidence=0.91,
        model_id="ensemble",
        image_name="leaf.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name=disease_name,
                disease_id=disease_id,
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.91,
                severity="alta",
            ),
            Top3PredictionSchema(
                rank=2,
                disease_name="Mancha-Alvo",
                disease_id="mancha-alvo",
                scientific_name="Corynespora cassiicola",
                confidence=0.05,
                severity="media",
            ),
            Top3PredictionSchema(
                rank=3,
                disease_name="Antracnose",
                disease_id="antracnose",
                scientific_name="Colletotrichum truncatum",
                confidence=0.04,
                severity="media",
            ),
        ],
    )


def _action_plan_response(
    disease_id: str = "ferrugem-asiatica",
) -> ActionPlanResponse:
    return ActionPlanResponse(
        disease_id=disease_id,
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL,
                actions=["Aplicar fungicida"],
            ),
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.CAMPO,
                actions=["Rotacionar culturas"],
            ),
        ],
        sources=[
            SourceResponse(
                id="src-1",
                name="EMBRAPA",
                detail="Fonte técnica",
                url="https://embrapa.br",
                display_order=0,
            )
        ],
    )


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
    svc = MagicMock()
    svc.predict.return_value = _inference_result()
    return svc


@pytest.fixture
def mock_action_plan_svc():
    svc = AsyncMock()
    svc.get_by_disease.return_value = _action_plan_response()
    return svc


@pytest.fixture
def mock_diagnosis_svc():
    svc = AsyncMock()
    svc.create.return_value = _diagnosis_response()
    return svc


# ── Node-level tests ─────────────────────────────────────────────────────────


async def test_load_model_node_accepts_valid_state(mock_inference_svc):
    update = await load_model_node(
        {"crop_id": "soja-id"}, inference_svc=mock_inference_svc
    )
    assert update == {}


async def test_load_model_node_flags_missing_crop(mock_inference_svc):
    update = await load_model_node({}, inference_svc=mock_inference_svc)
    assert "errors" in update
    assert update["errors"][0]["node"] == "load_model"


async def test_run_inference_node_populates_predictions(mock_inference_svc):
    state = {
        "image_ids": ["img-1", "img-2"],
        "model_id": "ensemble",
        "crop_id": "soja-id",
    }
    update = await run_inference_node(state, inference_svc=mock_inference_svc)

    assert len(update["predictions"]) == 2
    assert mock_inference_svc.predict.call_count == 2
    first = update["predictions"][0]
    assert first["disease_id"] == "ferrugem-asiatica"
    assert first["severity"] == "alta"
    assert len(first["top3"]) == 3


async def test_run_inference_handles_empty_batch(mock_inference_svc):
    update = await run_inference_node(
        {"image_ids": [], "model_id": "ensemble"},
        inference_svc=mock_inference_svc,
    )
    assert update["predictions"] == []
    mock_inference_svc.predict.assert_not_called()


async def test_compose_action_plan_node_picks_up_plans(mock_action_plan_svc):
    state = {
        "predictions": [
            {"disease_id": "ferrugem-asiatica"},
            {"disease_id": "mancha-alvo"},
        ]
    }
    update = await compose_action_plan_node(
        state, action_plan_svc=mock_action_plan_svc
    )
    assert len(update["action_plans"]) == 2
    assert update["action_plans"][0]["disease_id"] == "ferrugem-asiatica"
    assert len(update["action_plans"][0]["levels"]) == 2


async def test_compose_action_plan_handles_lookup_failure(mock_action_plan_svc):
    mock_action_plan_svc.get_by_disease.side_effect = ValueError("nope")
    state = {"predictions": [{"disease_id": "ghost-disease"}]}

    update = await compose_action_plan_node(
        state, action_plan_svc=mock_action_plan_svc
    )

    assert update["action_plans"] == [
        {"disease_id": "ghost-disease", "levels": [], "sources": []}
    ]


async def test_persist_node_creates_one_diagnosis_per_image(mock_diagnosis_svc):
    state = {
        "user_id": "user-uuid-1",
        "model_id": "ensemble",
        "image_ids": ["leaf-1.jpg", "leaf-2.jpg"],
        "predictions": [
            {
                "disease_id": "ferrugem-asiatica",
                "disease_name": "Ferrugem Asiática",
                "scientific_name": "Phakopsora pachyrhizi",
                "severity": "alta",
                "confidence": 0.91,
                "description": None,
                "top3": [],
            },
            {
                "disease_id": "mancha-alvo",
                "disease_name": "Mancha-Alvo",
                "scientific_name": "Corynespora cassiicola",
                "severity": "media",
                "confidence": 0.82,
                "description": None,
                "top3": [],
            },
        ],
    }
    mock_diagnosis_svc.create.side_effect = [
        _diagnosis_response("diag-1"),
        _diagnosis_response("diag-2"),
    ]

    update = await persist_node(state, diagnosis_svc=mock_diagnosis_svc)

    assert update["persisted_ids"] == ["diag-1", "diag-2"]
    assert mock_diagnosis_svc.create.await_count == 2
    args_first = mock_diagnosis_svc.create.await_args_list[0]
    assert args_first.args[0] == "user-uuid-1"
    assert args_first.args[1].image_name == "leaf-1.jpg"


async def test_persist_node_indexes_in_store_when_provided(mock_diagnosis_svc):
    """TCC-045: quando store eh passado, cada diagnosis vira aput no namespace correto."""
    state = {
        "user_id": "user-uuid-1",
        "model_id": "ensemble",
        "image_ids": ["leaf-1.jpg"],
        "predictions": [
            {
                "disease_id": "ferrugem-asiatica",
                "disease_name": "Ferrugem Asiática",
                "scientific_name": "Phakopsora pachyrhizi",
                "severity": "alta",
                "confidence": 0.91,
                "description": None,
                "top3": [],
            }
        ],
    }
    mock_diagnosis_svc.create.side_effect = [_diagnosis_response("diag-1")]
    store = AsyncMock()

    update = await persist_node(
        state, diagnosis_svc=mock_diagnosis_svc, store=store
    )

    assert update["persisted_ids"] == ["diag-1"]
    store.aput.assert_awaited_once()
    aput_kwargs = store.aput.call_args.kwargs
    assert aput_kwargs["namespace"] == ("user", "user-uuid-1", "diagnoses")
    assert aput_kwargs["key"] == "diag-1"


async def test_persist_node_swallows_store_errors(mock_diagnosis_svc, caplog):
    """Erros do Store nao quebram o persist — diagnosis fica no DB mesmo assim."""
    state = {
        "user_id": "user-uuid-1",
        "model_id": "ensemble",
        "image_ids": ["leaf-1.jpg"],
        "predictions": [
            {
                "disease_id": "ferrugem-asiatica",
                "disease_name": "Ferrugem Asiática",
                "scientific_name": "Phakopsora pachyrhizi",
                "severity": "alta",
                "confidence": 0.91,
                "description": None,
                "top3": [],
            }
        ],
    }
    mock_diagnosis_svc.create.side_effect = [_diagnosis_response("diag-1")]
    store = AsyncMock()
    store.aput.side_effect = RuntimeError("Store offline")

    update = await persist_node(
        state, diagnosis_svc=mock_diagnosis_svc, store=store
    )

    assert update["persisted_ids"] == ["diag-1"]
    assert "Failed to index diagnosis diag-1 in Store" in caplog.text


async def test_persist_node_writes_sources_from_evidence(mock_diagnosis_svc):
    """TCC-056: persist_node grava state.evidence_per_image em diagnoses.sources."""
    from app.domains.diagnosis_graph.nodes import _build_diagnosis_sources

    state = {
        "user_id": "user-uuid-1",
        "model_id": "ensemble",
        "image_ids": ["leaf-1.jpg"],
        "predictions": [
            {
                "disease_id": "ferrugem-asiatica",
                "disease_name": "Ferrugem Asiática",
                "scientific_name": "Phakopsora pachyrhizi",
                "severity": "alta",
                "confidence": 0.91,
                "description": None,
                "top3": [],
            }
        ],
        "evidence_per_image": [
            [
                {
                    "title": "Manejo de ferrugem",
                    "url": "https://embrapa.br/x",
                    "snippet": "Aplicar triazol em V4",
                },
                {
                    "title": "Phakopsora resistance study",
                    "url": "https://scielo.br/y",
                    "abstract": "Estudo de campo com isolados resistentes.",
                    "doi": "10.1590/x",
                },
            ]
        ],
    }
    mock_diagnosis_svc.create.side_effect = [_diagnosis_response("diag-1")]

    update = await persist_node(state, diagnosis_svc=mock_diagnosis_svc)

    assert update["persisted_ids"] == ["diag-1"]
    body = mock_diagnosis_svc.create.await_args_list[0].args[1]
    # Verifica que sources foi populado com tipos corretos.
    assert len(body.sources) == 2
    web_src = next(s for s in body.sources if s.type == "web")
    sci_src = next(s for s in body.sources if s.type == "scientific")
    assert web_src.url == "https://embrapa.br/x"
    assert sci_src.doi == "10.1590/x"
    assert sci_src.snippet == "Estudo de campo com isolados resistentes."


async def test_persist_node_handles_empty_evidence(mock_diagnosis_svc):
    """Sem evidence_per_image, sources fica []."""
    state = {
        "user_id": "user-uuid-1",
        "model_id": "ensemble",
        "image_ids": ["leaf-1.jpg"],
        "predictions": [
            {
                "disease_id": "ferrugem-asiatica",
                "disease_name": "Ferrugem Asiática",
                "scientific_name": "Phakopsora pachyrhizi",
                "severity": "alta",
                "confidence": 0.91,
                "description": None,
                "top3": [],
            }
        ],
    }
    mock_diagnosis_svc.create.side_effect = [_diagnosis_response("diag-1")]

    await persist_node(state, diagnosis_svc=mock_diagnosis_svc)
    body = mock_diagnosis_svc.create.await_args_list[0].args[1]
    assert body.sources == []


def test_build_diagnosis_sources_classifies_web_vs_scientific() -> None:
    """Helper classifica via presenca de doi/abstract."""
    from app.domains.diagnosis_graph.nodes import _build_diagnosis_sources

    raw = [
        {"title": "Web", "url": "https://x", "snippet": "y"},
        {"title": "Paper", "url": "https://p", "doi": "10.1/x"},
        {"title": "Paper2", "url": "https://p2", "abstract": "z"},
        "not-a-dict",  # filtrado
        {},  # vazio mas valido (vira web)
    ]
    out = _build_diagnosis_sources(raw)
    assert len(out) == 4
    assert out[0].type == "web"
    assert out[1].type == "scientific"  # doi presente
    assert out[2].type == "scientific"  # abstract presente
    assert out[3].type == "web"  # dict vazio default


def test_build_diagnosis_sources_handles_none_input() -> None:
    from app.domains.diagnosis_graph.nodes import _build_diagnosis_sources

    assert _build_diagnosis_sources([]) == []
    assert _build_diagnosis_sources(None) == []  # type: ignore[arg-type]


async def test_persist_node_no_store_skips_indexing(mock_diagnosis_svc):
    """Sem store, nada acontece com indexing — back-compat."""
    state = {
        "user_id": "user-uuid-1",
        "model_id": "ensemble",
        "image_ids": ["leaf-1.jpg"],
        "predictions": [
            {
                "disease_id": "ferrugem-asiatica",
                "disease_name": "Ferrugem Asiática",
                "scientific_name": "Phakopsora pachyrhizi",
                "severity": "alta",
                "confidence": 0.91,
                "description": None,
                "top3": [],
            }
        ],
    }
    mock_diagnosis_svc.create.side_effect = [_diagnosis_response("diag-1")]

    update = await persist_node(state, diagnosis_svc=mock_diagnosis_svc)

    assert update["persisted_ids"] == ["diag-1"]


# ── End-to-end graph ─────────────────────────────────────────────────────────


async def test_build_diagnosis_graph_runs_full_pipeline_single_image(
    mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
):
    graph = build_diagnosis_graph(
        mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
    )

    final = await graph.ainvoke(
        {
            "user_id": "user-uuid-1",
            "crop_id": "soja-id",
            "image_batch": [],
            "image_ids": ["leaf.jpg"],
            "model_id": "ensemble",
        }
    )

    assert final["persisted_ids"] == ["diag-1"]
    assert len(final["predictions"]) == 1
    assert len(final["action_plans"]) == 1
    mock_inference_svc.predict.assert_called_once()
    mock_action_plan_svc.get_by_disease.assert_awaited_once()
    mock_diagnosis_svc.create.assert_awaited_once()


async def test_build_diagnosis_graph_runs_batch(
    mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
):
    mock_diagnosis_svc.create.side_effect = [
        _diagnosis_response(f"diag-{i}") for i in range(3)
    ]
    graph = build_diagnosis_graph(
        mock_inference_svc, mock_action_plan_svc, mock_diagnosis_svc
    )

    final = await graph.ainvoke(
        {
            "user_id": "user-uuid-1",
            "crop_id": "soja-id",
            "image_batch": [],
            "image_ids": ["a.jpg", "b.jpg", "c.jpg"],
            "model_id": "vit",
        }
    )

    assert final["persisted_ids"] == ["diag-0", "diag-1", "diag-2"]
    assert mock_inference_svc.predict.call_count == 3
    assert mock_diagnosis_svc.create.await_count == 3
