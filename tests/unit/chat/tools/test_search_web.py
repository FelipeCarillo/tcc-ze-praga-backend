"""Testes da tool search_web (TCC-053).

Cobertura:
- Sem API key configurada -> retorna JSON com error
- Tavily AsyncTavilyClient.search levanta excecao -> retorna JSON com error
- Sucesso com resultados -> JSON com [{title, url, snippet}]
- Snippet eh truncado a 300 chars
- crop_context injetado quando detected_crop_id presente no state
- Cache hit pula chamada a Tavily
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domains.chat.tools._cache import clear_cache
from app.domains.chat.tools.search_web import build_search_web_tool


@pytest.fixture(autouse=True)
def _reset_cache():
    """Garante cache limpo entre testes."""
    clear_cache()
    yield
    clear_cache()


def _state(detected_crop_id: str | None = None) -> dict:
    state: dict = {"current_user_id": "u-1"}
    if detected_crop_id:
        state["detected_crop_id"] = detected_crop_id
    return state


async def test_no_api_key_returns_error(monkeypatch) -> None:
    """Sem tavily_api_key configurada, tool retorna error JSON."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", None)

    tool = build_search_web_tool()
    raw = await tool.ainvoke(
        {"query": "ferrugem asiatica", "max_results": 3, "state": _state()}
    )
    parsed = json.loads(raw)
    assert "error" in parsed
    assert "tavily_api_key" in parsed["error"]


async def test_tavily_failure_returns_error(monkeypatch) -> None:
    """Quando AsyncTavilyClient.search levanta, tool retorna error JSON."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    fake_client = MagicMock()
    fake_client.search = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "tavily.AsyncTavilyClient",
        return_value=fake_client,
    ):
        tool = build_search_web_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
    parsed = json.loads(raw)
    assert "error" in parsed
    assert "boom" in parsed["error"]


async def test_success_returns_results(monkeypatch) -> None:
    """Sucesso retorna JSON com lista de {title, url, snippet}."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    fake_client = MagicMock()
    fake_client.search = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Manejo Ferrugem Asiatica",
                    "url": "https://embrapa.br/x",
                    "content": "Aplicar fungicida triazol em V4-V6.",
                },
                {
                    "title": "Boletim Soja 2025",
                    "url": "https://exemplo.com/y",
                    "content": "Ocorrencia regional em MT.",
                },
            ]
        }
    )
    with patch("tavily.AsyncTavilyClient", return_value=fake_client):
        tool = build_search_web_tool()
        raw = await tool.ainvoke(
            {
                "query": "ferrugem asiatica manejo",
                "max_results": 2,
                "state": _state(),
            }
        )
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Manejo Ferrugem Asiatica"
    assert parsed[0]["url"] == "https://embrapa.br/x"
    assert "Aplicar fungicida" in parsed[0]["snippet"]


async def test_snippet_truncated_to_300_chars(monkeypatch) -> None:
    """Conteudo longo eh cortado a 300 caracteres."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    long_content = "x" * 500
    fake_client = MagicMock()
    fake_client.search = AsyncMock(
        return_value={
            "results": [
                {"title": "T", "url": "https://x", "content": long_content}
            ]
        }
    )
    with patch("tavily.AsyncTavilyClient", return_value=fake_client):
        tool = build_search_web_tool()
        raw = await tool.ainvoke(
            {"query": "q", "max_results": 1, "state": _state()}
        )
    parsed = json.loads(raw)
    assert len(parsed[0]["snippet"]) == 300


async def test_crop_context_injected_in_query(monkeypatch) -> None:
    """detected_crop_id no state vira sufixo da query."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    fake_client = MagicMock()
    fake_client.search = AsyncMock(return_value={"results": []})
    with patch("tavily.AsyncTavilyClient", return_value=fake_client):
        tool = build_search_web_tool()
        await tool.ainvoke(
            {
                "query": "ferrugem",
                "max_results": 1,
                "state": _state(detected_crop_id="soja"),
            }
        )
    call_kwargs = fake_client.search.await_args.kwargs
    assert "soja" in call_kwargs["query"]
    assert "ferrugem" in call_kwargs["query"]


async def test_no_crop_context_uses_raw_query(monkeypatch) -> None:
    """Sem detected_crop_id, query nao recebe sufixo."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    fake_client = MagicMock()
    fake_client.search = AsyncMock(return_value={"results": []})
    with patch("tavily.AsyncTavilyClient", return_value=fake_client):
        tool = build_search_web_tool()
        await tool.ainvoke(
            {"query": "ferrugem", "max_results": 1, "state": _state()}
        )
    call_kwargs = fake_client.search.await_args.kwargs
    assert call_kwargs["query"] == "ferrugem"


async def test_cache_hit_skips_tavily(monkeypatch) -> None:
    """Segunda chamada com mesma query bate cache, nao chama Tavily de novo."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    fake_client = MagicMock()
    fake_client.search = AsyncMock(
        return_value={
            "results": [
                {"title": "Cached", "url": "https://c", "content": "ok"}
            ]
        }
    )
    with patch("tavily.AsyncTavilyClient", return_value=fake_client):
        tool = build_search_web_tool()
        first = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
        second = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
    assert first == second
    assert fake_client.search.await_count == 1


async def test_empty_results_returns_empty_list(monkeypatch) -> None:
    """Resposta sem results retorna [] (JSON)."""
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    fake_client = MagicMock()
    fake_client.search = AsyncMock(return_value={"results": []})
    with patch("tavily.AsyncTavilyClient", return_value=fake_client):
        tool = build_search_web_tool()
        raw = await tool.ainvoke(
            {"query": "obscure", "max_results": 5, "state": _state()}
        )
    parsed = json.loads(raw)
    assert parsed == []
