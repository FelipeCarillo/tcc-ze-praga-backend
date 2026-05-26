"""Testes da tool inspect_image (TCC-079) — gate de visão."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.tools.inspect_image import build_inspect_image_tool


def _file(file_id: str = "img-1", b64: str | None = "ZmFrZQ==") -> UploadedFileDTO:
    return UploadedFileDTO(
        id=file_id,
        original_name=f"{file_id}.jpg",
        mime="image/jpeg",
        storage_key="",
        size_bytes=1024,
        b64=b64,
    )


def _patch_llm(content: str):
    fake = AsyncMock()
    fake.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return patch(
        "app.domains.chat.tools.inspect_image.get_chat_model", return_value=fake
    ), fake


async def test_inspect_image_classifies_plant_true() -> None:
    patcher, _ = _patch_llm(
        '{"is_analyzable_plant": true, "subject": "folha", "reason": "folha de soja"}'
    )
    with patcher:
        tool = build_inspect_image_tool()
        result = await tool.ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    data = json.loads(result)
    assert data["is_analyzable_plant"] is True
    assert data["subject"] == "folha"


async def test_inspect_image_classifies_non_plant_false() -> None:
    patcher, _ = _patch_llm(
        '{"is_analyzable_plant": false, "subject": "produto", "reason": "embalagem"}'
    )
    with patcher:
        tool = build_inspect_image_tool()
        result = await tool.ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    data = json.loads(result)
    assert data["is_analyzable_plant"] is False
    assert data["subject"] == "produto"


async def test_inspect_image_strips_json_fence() -> None:
    patcher, _ = _patch_llm(
        '```json\n{"is_analyzable_plant": true, "subject": "planta", "reason": "ok"}\n```'
    )
    with patcher:
        tool = build_inspect_image_tool()
        result = await tool.ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    assert json.loads(result)["is_analyzable_plant"] is True


async def test_inspect_image_no_image_returns_false_without_llm() -> None:
    patcher, fake = _patch_llm('{"is_analyzable_plant": true}')
    with patcher as mock_get:
        tool = build_inspect_image_tool()
        result = await tool.ainvoke({"image_id": None, "state": {"uploaded_files": []}})
    data = json.loads(result)
    assert data["is_analyzable_plant"] is False
    mock_get.assert_not_called()


async def test_inspect_image_without_b64_returns_false() -> None:
    patcher, _ = _patch_llm('{"is_analyzable_plant": true}')
    with patcher as mock_get:
        tool = build_inspect_image_tool()
        result = await tool.ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file(b64=None)]}}
        )
    assert json.loads(result)["is_analyzable_plant"] is False
    mock_get.assert_not_called()


async def test_inspect_image_invalid_json_fallbacks_to_plant() -> None:
    """JSON malformado da LLM → fallback resiliente (na dúvida, planta)."""
    patcher, _ = _patch_llm("isso nao e json")
    with patcher:
        tool = build_inspect_image_tool()
        result = await tool.ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    assert json.loads(result)["is_analyzable_plant"] is True
