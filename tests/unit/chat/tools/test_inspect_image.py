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


# ── TCC-103: gate pelo Decisions API, com o LLM de visão como reserva ───────


def _decisions(prob: float, subject: str):
    return {
        "is_analyzable_plant": {"type": "predicate", "name": "is_analyzable_plant", "probability": prob},
        "subject": {"type": "choice", "name": "subject", "choice": subject, "confidence": 0.9},
    }


async def test_inspect_image_usa_decisions_sem_chamar_llm(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "vision_gate_decisions", True)
    decide = AsyncMock(return_value=_decisions(0.97, "folha"))
    patcher, fake_llm = _patch_llm("{}")
    with patcher, patch("app.domains.chat.tools.inspect_image.decide", decide):
        result = await build_inspect_image_tool().ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    data = json.loads(result)
    assert data == {
        "is_analyzable_plant": True,
        "subject": "folha",
        "reason": "Classificada como folha ou folíolo em close.",
    }
    fake_llm.ainvoke.assert_not_called()
    sent_input = decide.await_args.args[0]
    assert sent_input[0]["content"][1]["image_url"] == "data:image/jpeg;base64,ZmFrZQ=="


async def test_inspect_image_decisions_baixa_probabilidade_e_false(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "vision_gate_decisions", True)
    with patch(
        "app.domains.chat.tools.inspect_image.decide",
        AsyncMock(return_value=_decisions(0.02, "pessoa")),
    ):
        result = await build_inspect_image_tool().ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    data = json.loads(result)
    assert data["is_analyzable_plant"] is False and data["subject"] == "pessoa"


async def test_inspect_image_decisions_falha_cai_no_llm(monkeypatch) -> None:
    from app.config import settings
    from app.core.decisions import DecisionsError

    monkeypatch.setattr(settings, "vision_gate_decisions", True)
    patcher, fake_llm = _patch_llm(
        '{"is_analyzable_plant": true, "subject": "folha", "reason": "folha"}'
    )
    with patcher, patch(
        "app.domains.chat.tools.inspect_image.decide",
        AsyncMock(side_effect=DecisionsError("timeout")),
    ):
        result = await build_inspect_image_tool().ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    assert json.loads(result)["is_analyzable_plant"] is True
    fake_llm.ainvoke.assert_awaited_once()


async def test_inspect_image_decisions_resposta_estranha_cai_no_llm(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "vision_gate_decisions", True)
    patcher, fake_llm = _patch_llm(
        '{"is_analyzable_plant": false, "subject": "produto", "reason": "x"}'
    )
    with patcher, patch(
        "app.domains.chat.tools.inspect_image.decide", AsyncMock(return_value={})
    ):
        result = await build_inspect_image_tool().ainvoke(
            {"image_id": None, "state": {"uploaded_files": [_file()]}}
        )
    assert json.loads(result)["subject"] == "produto"
    fake_llm.ainvoke.assert_awaited_once()
