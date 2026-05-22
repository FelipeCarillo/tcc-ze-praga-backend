"""Testes da tool deep_diagnose (TCC-041)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.tools.deep_diagnose import build_deep_diagnose_tool


def _file(file_id: str = "img-1") -> UploadedFileDTO:
    return UploadedFileDTO(
        id=file_id,
        original_name=f"{file_id}.jpg",
        mime="image/jpeg",
        storage_key=f"uploads/u1/{file_id}.jpg",
        size_bytes=1024,
    )


def _graph_invoker(persisted_count: int = 1):
    """Cria mock de grafo compilado que retorna persisted_ids + predictions."""
    graph = MagicMock()

    async def _ainvoke(state):
        n = len(state.get("image_ids", []))
        return {
            "persisted_ids": [f"diag-{i}" for i in range(n)],
            "predictions": [
                {
                    "disease_id": "ferrugem-asiatica",
                    "disease_name": "Ferrugem Asiática",
                    "confidence": 0.91,
                    "severity": "alta",
                }
                for _ in range(n)
            ],
        }

    graph.ainvoke = _ainvoke
    return graph


def _factory(graph):
    factories: dict = {}

    def _factory_fn(crop_id: str):
        factories["last_crop"] = crop_id
        return graph

    _factory_fn.factories = factories  # type: ignore[attr-defined]
    return _factory_fn


async def test_deep_diagnose_returns_error_when_no_files() -> None:
    graph = _graph_invoker()
    tool = build_deep_diagnose_tool(_factory(graph))

    raw = await tool.ainvoke(
        {"image_ids": None, "crop_id": None, "state": {"uploaded_files": []}}
    )
    parsed = json.loads(raw)
    assert parsed == {"error": "Nenhuma imagem disponivel"}


async def test_deep_diagnose_filters_by_image_ids() -> None:
    graph = _graph_invoker()
    tool = build_deep_diagnose_tool(_factory(graph))

    state = {
        "current_user_id": "u-1",
        "selected_model": "ensemble",
        "uploaded_files": [_file("a"), _file("b"), _file("c")],
    }
    raw = await tool.ainvoke(
        {"image_ids": ["a", "c"], "crop_id": None, "state": state}
    )
    parsed = json.loads(raw)
    assert parsed["count"] == 2
    assert {r["image_id"] for r in parsed["results"]} == {"a", "c"}


async def test_deep_diagnose_processes_all_when_image_ids_none() -> None:
    graph = _graph_invoker()
    tool = build_deep_diagnose_tool(_factory(graph))

    state = {
        "current_user_id": "u-1",
        "selected_model": "ensemble",
        "uploaded_files": [_file("a"), _file("b")],
    }
    raw = await tool.ainvoke(
        {"image_ids": None, "crop_id": None, "state": state}
    )
    parsed = json.loads(raw)
    assert parsed["count"] == 2
    assert parsed["results"][0]["disease"] == "Ferrugem Asiática"
    assert parsed["results"][0]["diagnosis_id"] == "diag-0"


async def test_deep_diagnose_uses_detected_crop_when_arg_missing() -> None:
    graph = _graph_invoker()
    factory = _factory(graph)
    tool = build_deep_diagnose_tool(factory)

    state = {
        "current_user_id": "u-1",
        "selected_model": "vit",
        "detected_crop_id": "milho-id",
        "uploaded_files": [_file("a")],
    }
    await tool.ainvoke({"image_ids": None, "crop_id": None, "state": state})

    assert factory.factories["last_crop"] == "milho-id"  # type: ignore[attr-defined]


async def test_deep_diagnose_falls_back_to_soja() -> None:
    graph = _graph_invoker()
    factory = _factory(graph)
    tool = build_deep_diagnose_tool(factory)

    state = {
        "current_user_id": "u-1",
        "selected_model": "ensemble",
        "uploaded_files": [_file("a")],
    }
    await tool.ainvoke({"image_ids": None, "crop_id": None, "state": state})
    assert factory.factories["last_crop"] == "soja"  # type: ignore[attr-defined]


async def test_deep_diagnose_explicit_crop_id_overrides_state() -> None:
    graph = _graph_invoker()
    factory = _factory(graph)
    tool = build_deep_diagnose_tool(factory)

    state = {
        "current_user_id": "u-1",
        "selected_model": "ensemble",
        "detected_crop_id": "milho-id",
        "uploaded_files": [_file("a")],
    }
    await tool.ainvoke(
        {"image_ids": None, "crop_id": "soja-id", "state": state}
    )
    assert factory.factories["last_crop"] == "soja-id"  # type: ignore[attr-defined]


async def test_deep_diagnose_skips_when_filter_excludes_all() -> None:
    graph = _graph_invoker()
    tool = build_deep_diagnose_tool(_factory(graph))

    state = {
        "current_user_id": "u-1",
        "selected_model": "ensemble",
        "uploaded_files": [_file("a")],
    }
    raw = await tool.ainvoke(
        {"image_ids": ["nonexistent"], "crop_id": None, "state": state}
    )
    parsed = json.loads(raw)
    assert parsed == {"error": "Nenhuma imagem disponivel"}
