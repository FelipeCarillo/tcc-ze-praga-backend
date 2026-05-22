"""Testes da tool identify_crop (TCC-065, V2 dormente)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.tools.identify_crop import build_identify_crop_tool


def _file(file_id: str = "img-1", b64: str | None = "ZmFrZQ==") -> UploadedFileDTO:
    return UploadedFileDTO(
        id=file_id,
        original_name=f"{file_id}.jpg",
        mime="image/jpeg",
        storage_key=f"uploads/u1/{file_id}.jpg",
        size_bytes=1024,
        b64=b64,
    )


class _FakeVisionLLM:
    """Stand-in pra ``ChatOpenAI`` que devolve ``AIMessage`` configurado."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.ainvoke = AsyncMock(return_value=AIMessage(content=content))


def _patch_llm(fake_content: str):
    """Factory de patch — substitui ``ChatOpenAI`` no modulo da tool."""
    fake = _FakeVisionLLM(fake_content)
    return (
        patch(
            "app.domains.chat.tools.identify_crop.ChatOpenAI",
            return_value=fake,
        ),
        fake,
    )


# ── happy path ────────────────────────────────────────────────────────────────


async def test_identify_crop_updates_state_when_confidence_high() -> None:
    """confidence >= 0.7 + crop_id valido → State recebe detected_crop_id."""
    patcher, _ = _patch_llm(
        '{"crop_id":"soja","confidence":0.95,"reason":"folhas trifolioladas"}'
    )
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert isinstance(result, Command)
    assert result.update == {"detected_crop_id": "soja"}


async def test_identify_crop_skips_state_update_when_low_confidence() -> None:
    """confidence < 0.7 → fica desconhecido, state nao e' atualizado."""
    patcher, _ = _patch_llm(
        '{"crop_id":"milho","confidence":0.4,"reason":"meio borrado"}'
    )
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert isinstance(result, Command)
    assert result.update == {}


async def test_identify_crop_skips_when_llm_returns_desconhecido() -> None:
    """LLM retorna explicitly 'desconhecido' → sem update mesmo com alta confianca."""
    patcher, _ = _patch_llm(
        '{"crop_id":"desconhecido","confidence":0.99,"reason":"sem cultivo"}'
    )
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert isinstance(result, Command)
    assert result.update == {}


# ── error / graceful fallbacks ────────────────────────────────────────────────


async def test_identify_crop_returns_empty_when_llm_returns_invalid_json() -> None:
    """JSON malformado da LLM → graceful, sem update no state."""
    patcher, _ = _patch_llm("isso nao e' json")
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert isinstance(result, Command)
    assert result.update == {"messages": []}


async def test_identify_crop_returns_empty_when_confidence_not_numeric() -> None:
    """confidence nao-coercivel a float → graceful."""
    patcher, _ = _patch_llm(
        '{"crop_id":"soja","confidence":"alta","reason":"x"}'
    )
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert isinstance(result, Command)
    assert result.update == {"messages": []}


async def test_identify_crop_handles_missing_image() -> None:
    """image_id inexistente → graceful, LLM nem e' chamado."""
    patcher, _ = _patch_llm(
        '{"crop_id":"soja","confidence":0.9,"reason":"x"}'
    )
    with patcher as mock_chat:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "nonexistent", "state": state})

    assert isinstance(result, Command)
    assert result.update == {"messages": []}
    # Garante que nao gastamos chamada de LLM
    mock_chat.assert_not_called()


async def test_identify_crop_handles_image_without_b64() -> None:
    """Imagem existe mas b64 ainda nao foi carregado → graceful."""
    patcher, _ = _patch_llm(
        '{"crop_id":"soja","confidence":0.9,"reason":"x"}'
    )
    with patcher as mock_chat:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1", b64=None)]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert isinstance(result, Command)
    assert result.update == {"messages": []}
    mock_chat.assert_not_called()


# ── plano / allowed_crops ─────────────────────────────────────────────────────


async def test_identify_crop_uses_default_crops_when_plan_unspecified() -> None:
    """Sem plan_features.allowed_crops, usa o conjunto default."""
    patcher, fake = _patch_llm(
        '{"crop_id":"soja","confidence":0.9,"reason":"x"}'
    )
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        await tool.ainvoke({"image_id": "img-1", "state": state})

    # Garante que o prompt mencionou os cultivos default
    call_args = fake.ainvoke.call_args
    messages = call_args.args[0]
    prompt_text = messages[0].content[0]["text"]
    assert "soja" in prompt_text
    assert "milho" in prompt_text
    assert "trigo" in prompt_text


async def test_identify_crop_respects_plan_allowed_crops() -> None:
    """Quando plan_features traz allowed_crops, usa eles no prompt."""
    patcher, fake = _patch_llm(
        '{"crop_id":"cafe","confidence":0.92,"reason":"x"}'
    )
    with patcher:
        from app.domains.subscriptions.features import PlanFeatures

        tool = build_identify_crop_tool()
        state = {
            "uploaded_files": [_file("img-1")],
            "plan_features": PlanFeatures(
                tier_name="pro", allowed_crops=["cafe", "cacau"]
            ),
        }

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert result.update == {"detected_crop_id": "cafe"}
    call_args = fake.ainvoke.call_args
    prompt_text = call_args.args[0][0].content[0]["text"]
    assert "cafe" in prompt_text
    assert "cacau" in prompt_text
    # Cultivos default que NAO estao no plan devem ficar fora
    assert "milho" not in prompt_text


# ── confidence borderline ────────────────────────────────────────────────────


async def test_identify_crop_threshold_at_exact_0_7() -> None:
    """confidence == 0.7 e' o limite inferior (>=) → atualiza."""
    patcher, _ = _patch_llm(
        '{"crop_id":"trigo","confidence":0.7,"reason":"x"}'
    )
    with patcher:
        tool = build_identify_crop_tool()
        state = {"uploaded_files": [_file("img-1")]}

        result = await tool.ainvoke({"image_id": "img-1", "state": state})

    assert result.update == {"detected_crop_id": "trigo"}
