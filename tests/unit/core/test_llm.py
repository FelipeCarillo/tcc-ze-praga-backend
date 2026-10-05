"""Factory agnóstica de LLM: temperatura opcional, embeddings e texto da resposta."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from app.core import llm as llm_module


def test_get_chat_model_omits_temperature_by_default(monkeypatch):
    """Modelos de raciocínio (gpt-5, o-series) recusam ``temperature``."""
    monkeypatch.setattr(llm_module.settings, "llm_temperature", None)
    with patch.object(llm_module, "init_chat_model", return_value=MagicMock()) as init:
        llm_module.get_chat_model("openai:gpt-5-mini")
    assert "temperature" not in init.call_args.kwargs
    assert init.call_args.args[0] == "openai:gpt-5-mini"


def test_get_chat_model_sends_configured_temperature(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "llm_temperature", 0.0)
    with patch.object(llm_module, "init_chat_model", return_value=MagicMock()) as init:
        llm_module.get_chat_model("anthropic:claude-x")
    assert init.call_args.kwargs["temperature"] == 0.0


def test_get_chat_model_forwards_provider_kwargs_from_settings(monkeypatch):
    """LLM_MODEL_KWARGS chega ao construtor (ex.: Responses API do gpt-6)."""
    monkeypatch.setattr(llm_module.settings, "llm_temperature", None)
    monkeypatch.setattr(
        llm_module.settings, "llm_model_kwargs", {"use_responses_api": True}
    )
    with patch.object(llm_module, "init_chat_model", return_value=MagicMock()) as init:
        llm_module.get_chat_model("openai:gpt-6-luna", timeout=5)
    assert init.call_args.kwargs["use_responses_api"] is True
    assert init.call_args.kwargs["timeout"] == 5


def test_get_embeddings_normalizes_legacy_id(monkeypatch):
    with patch("langchain.embeddings.init_embeddings", return_value="emb") as init:
        assert llm_module.get_embeddings("text-embedding-3-small") == "emb"
    init.assert_called_once_with("openai:text-embedding-3-small")


def test_get_embeddings_defaults_to_settings(monkeypatch):
    monkeypatch.setattr(
        llm_module.settings, "embeddings_model", "google_genai:gemini-embedding-001"
    )
    with patch("langchain.embeddings.init_embeddings", return_value="emb") as init:
        llm_module.get_embeddings()
    init.assert_called_once_with("google_genai:gemini-embedding-001")


def test_message_text_plain_string():
    assert llm_module.message_text(AIMessage(content="olá")) == "olá"


def test_message_text_keeps_only_text_blocks():
    """Anthropic/Gemini/Responses API: blocos de raciocínio e tool call ficam fora."""
    msg = AIMessage(
        content=[
            {"type": "reasoning", "summary": [{"text": "pensando"}]},
            {"type": "text", "text": "Ferrugem "},
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            {"type": "text", "text": "asiática."},
        ]
    )
    assert llm_module.message_text(msg) == "Ferrugem asiática."


def test_message_text_none_and_raw_values():
    assert llm_module.message_text(MagicMock(content=None)) == ""
    assert llm_module.message_text("cru") == "cru"
