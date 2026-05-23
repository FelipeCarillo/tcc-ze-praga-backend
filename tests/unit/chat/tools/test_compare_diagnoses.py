"""Testes da tool compare_diagnoses (TCC-050).

Cobertura:
- imagem nao encontrada -> error
- lista de models vazia -> error
- 2 modelos com mesma doenca -> consenso + agrees_with_consensus=True em ambos
- 2 modelos com doencas diferentes -> consenso pega o primeiro, agreement
  refletindo isso
- erro em 1 modelo (gather propaga exception caught dentro do worker)
  -> entry com "error" + outros funcionam
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.tools.compare_diagnoses import build_compare_diagnoses_tool


def _file(file_id: str = "img-1") -> UploadedFileDTO:
    return UploadedFileDTO(
        id=file_id,
        original_name=f"{file_id}.jpg",
        mime="image/jpeg",
        storage_key=f"uploads/u1/{file_id}.jpg",
        size_bytes=1024,
    )


def _graph_with_responses(responses_by_model: dict[str, dict]):
    """Cria mock graph cujo ainvoke retorna predicao baseada em model_id."""
    graph = MagicMock()

    async def _ainvoke(state):
        model_id = state.get("model_id")
        if model_id in responses_by_model:
            return responses_by_model[model_id]
        return {"predictions": []}

    graph.ainvoke = _ainvoke
    return graph


def _failing_graph_for(failing_models: set[str], responses: dict[str, dict]):
    """Graph que levanta excecao pra modelos em ``failing_models``."""
    graph = MagicMock()

    async def _ainvoke(state):
        model_id = state.get("model_id")
        if model_id in failing_models:
            raise RuntimeError(f"boom-{model_id}")
        return responses.get(model_id, {"predictions": []})

    graph.ainvoke = _ainvoke
    return graph


def _factory(graph):
    def _factory_fn(_crop_id: str):
        return graph

    return _factory_fn


# ── basic errors ─────────────────────────────────────────────────────────────


async def test_image_not_found_returns_error() -> None:
    graph = _graph_with_responses({})
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {
            "image_id": "ghost",
            "models": ["resnet50"],
            "crop_id": None,
            "state": state,
        }
    )
    parsed = json.loads(raw)
    assert "error" in parsed
    assert "ghost" in parsed["error"]


async def test_empty_models_returns_error() -> None:
    graph = _graph_with_responses({})
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {"image_id": "a", "models": [], "crop_id": None, "state": state}
    )
    parsed = json.loads(raw)
    assert "error" in parsed


# ── happy path ───────────────────────────────────────────────────────────────


async def test_two_models_same_disease_yields_consensus() -> None:
    graph = _graph_with_responses(
        {
            "resnet50": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem Asiática",
                        "confidence": 0.91,
                        "severity": "alta",
                    }
                ]
            },
            "vit": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem Asiática",
                        "confidence": 0.88,
                        "severity": "alta",
                    }
                ]
            },
        }
    )
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50", "vit"],
            "crop_id": None,
            "state": state,
        }
    )
    parsed = json.loads(raw)
    assert parsed["image_id"] == "a"
    assert parsed["models"] == ["resnet50", "vit"]
    assert parsed["consensus_disease_id"] == "ferrugem-asiatica"
    assert len(parsed["comparison"]) == 2
    assert all(row["agrees_with_consensus"] for row in parsed["comparison"])
    assert parsed["comparison"][0]["confidence"] == 0.91
    assert parsed["comparison"][1]["confidence"] == 0.88


async def test_two_models_different_diseases_picks_one_as_consensus() -> None:
    graph = _graph_with_responses(
        {
            "resnet50": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem",
                        "confidence": 0.85,
                        "severity": "alta",
                    }
                ]
            },
            "vit": {
                "predictions": [
                    {
                        "disease_id": "mancha-alvo",
                        "disease_name": "Mancha-Alvo",
                        "confidence": 0.62,
                        "severity": "media",
                    }
                ]
            },
        }
    )
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50", "vit"],
            "crop_id": None,
            "state": state,
        }
    )
    parsed = json.loads(raw)
    # Quando ha empate (1-1), max() retorna a primeira chave inserida.
    assert parsed["consensus_disease_id"] in {"ferrugem-asiatica", "mancha-alvo"}
    # Exatamente 1 modelo concorda com o consenso.
    agreements = [r["agrees_with_consensus"] for r in parsed["comparison"]]
    assert sum(agreements) == 1


async def test_three_models_majority_disease_wins_consensus() -> None:
    graph = _graph_with_responses(
        {
            "resnet50": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem",
                        "confidence": 0.85,
                        "severity": "alta",
                    }
                ]
            },
            "vit": {
                "predictions": [
                    {
                        "disease_id": "mancha-alvo",
                        "disease_name": "Mancha-Alvo",
                        "confidence": 0.62,
                        "severity": "media",
                    }
                ]
            },
            "efficientnet": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem",
                        "confidence": 0.78,
                        "severity": "alta",
                    }
                ]
            },
        }
    )
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50", "vit", "efficientnet"],
            "crop_id": None,
            "state": state,
        }
    )
    parsed = json.loads(raw)
    assert parsed["consensus_disease_id"] == "ferrugem-asiatica"
    agreements = [r["agrees_with_consensus"] for r in parsed["comparison"]]
    assert sum(agreements) == 2


# ── error handling ──────────────────────────────────────────────────────────


async def test_one_model_failing_others_succeed() -> None:
    graph = _failing_graph_for(
        {"resnet50"},
        {
            "vit": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem",
                        "confidence": 0.91,
                        "severity": "alta",
                    }
                ]
            }
        },
    )
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50", "vit"],
            "crop_id": None,
            "state": state,
        }
    )
    parsed = json.loads(raw)
    # Verifica que tool nao explodiu — retornou comparison com 2 rows.
    assert len(parsed["comparison"]) == 2
    # resnet50 deve ter "error", vit deve ter a doenca.
    by_model = {row["model_id"]: row for row in parsed["comparison"]}
    assert "error" in by_model["resnet50"]
    assert "boom-resnet50" in by_model["resnet50"]["error"]
    assert by_model["vit"]["disease_id"] == "ferrugem-asiatica"
    assert parsed["consensus_disease_id"] == "ferrugem-asiatica"


async def test_model_returns_empty_predictions() -> None:
    graph = _graph_with_responses(
        {
            "resnet50": {"predictions": []},  # sem predicao
            "vit": {
                "predictions": [
                    {
                        "disease_id": "ferrugem-asiatica",
                        "disease_name": "Ferrugem",
                        "confidence": 0.91,
                        "severity": "alta",
                    }
                ]
            },
        }
    )
    tool = build_compare_diagnoses_tool(_factory(graph))

    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    raw = await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50", "vit"],
            "crop_id": None,
            "state": state,
        }
    )
    parsed = json.loads(raw)
    by_model = {row["model_id"]: row for row in parsed["comparison"]}
    assert "error" in by_model["resnet50"]
    assert by_model["vit"]["disease_id"] == "ferrugem-asiatica"
    assert parsed["consensus_disease_id"] == "ferrugem-asiatica"


# ── crop resolution ──────────────────────────────────────────────────────────


async def test_explicit_crop_id_used() -> None:
    calls: list[str] = []

    def _factory_recording(crop_id: str):
        calls.append(crop_id)
        return _graph_with_responses(
            {
                "resnet50": {
                    "predictions": [
                        {
                            "disease_id": "x",
                            "disease_name": "X",
                            "confidence": 0.5,
                            "severity": "media",
                        }
                    ]
                }
            }
        )

    tool = build_compare_diagnoses_tool(_factory_recording)
    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50"],
            "crop_id": "milho-id",
            "state": state,
        }
    )
    assert "milho-id" in calls


async def test_falls_back_to_state_detected_crop() -> None:
    calls: list[str] = []

    def _factory_recording(crop_id: str):
        calls.append(crop_id)
        return _graph_with_responses(
            {
                "resnet50": {
                    "predictions": [
                        {
                            "disease_id": "x",
                            "disease_name": "X",
                            "confidence": 0.5,
                            "severity": "media",
                        }
                    ]
                }
            }
        )

    tool = build_compare_diagnoses_tool(_factory_recording)
    state = {
        "current_user_id": "u-1",
        "detected_crop_id": "feijao-id",
        "uploaded_files": [_file("a")],
    }
    await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50"],
            "crop_id": None,
            "state": state,
        }
    )
    assert "feijao-id" in calls


async def test_falls_back_to_soja_when_no_crop() -> None:
    calls: list[str] = []

    def _factory_recording(crop_id: str):
        calls.append(crop_id)
        return _graph_with_responses(
            {
                "resnet50": {
                    "predictions": [
                        {
                            "disease_id": "x",
                            "disease_name": "X",
                            "confidence": 0.5,
                            "severity": "media",
                        }
                    ]
                }
            }
        )

    tool = build_compare_diagnoses_tool(_factory_recording)
    state = {"current_user_id": "u-1", "uploaded_files": [_file("a")]}
    await tool.ainvoke(
        {
            "image_id": "a",
            "models": ["resnet50"],
            "crop_id": None,
            "state": state,
        }
    )
    assert "soja" in calls
