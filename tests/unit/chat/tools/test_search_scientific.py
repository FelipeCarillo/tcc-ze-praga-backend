"""Testes da tool search_scientific (TCC-054).

Cobertura:
- httpx falha -> error JSON
- SciELO retorna 500 -> error JSON
- Schema Solr-like (response.docs) -> articles extraidos e normalizados
- Schema top-level docs -> extraido
- Schema top-level results -> extraido
- Schema ES-like (hits.hits._source) -> extraido
- Sem articles -> [] (lista vazia)
- abstract truncado a 500 chars
- crop_context injetado quando detected_crop_id presente
- Cache hit pula chamada HTTP
- Helpers internos (_extract_articles, _format_article, _first)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domains.chat.tools._cache import clear_cache
from app.domains.chat.tools.search_scientific import (
    _extract_articles,
    _first,
    _format_article,
    build_search_scientific_tool,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


def _state(detected_crop_id: str | None = None) -> dict:
    state: dict = {"current_user_id": "u-1"}
    if detected_crop_id:
        state["detected_crop_id"] = detected_crop_id
    return state


def _mock_async_client(response_data: dict | None = None, raise_exc: Exception | None = None):
    """Cria um mock pra httpx.AsyncClient suportando async ctx manager."""
    fake_response = MagicMock()
    if raise_exc is not None:
        fake_response.raise_for_status = MagicMock(side_effect=raise_exc)
        fake_response.json = MagicMock(return_value={})
    else:
        fake_response.raise_for_status = MagicMock(return_value=None)
        fake_response.json = MagicMock(return_value=response_data or {"response": {"docs": []}})

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    return fake_client


# ── basic error paths ───────────────────────────────────────────────────────


async def test_httpx_failure_returns_error_json() -> None:
    """Quando httpx levanta, tool retorna error JSON sem propagar."""
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.get = AsyncMock(side_effect=httpx.ConnectError("net down"))

    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "ferrugem", "max_results": 3, "state": _state()}
        )
    parsed = json.loads(raw)
    assert "error" in parsed
    assert "SciELO" in parsed["error"]


async def test_http_error_returns_error_json() -> None:
    """500 response gera raise_for_status -> error JSON."""
    fake_client = _mock_async_client(
        raise_exc=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
    )
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
    parsed = json.loads(raw)
    assert "error" in parsed


# ── schema variations ──────────────────────────────────────────────────────


async def test_solr_like_schema_extracts_articles() -> None:
    """Schema Solr ``{"response": {"docs": [...]}}`` eh extraido corretamente."""
    payload = {
        "response": {
            "docs": [
                {
                    "ti_pt": "Manejo de Phakopsora pachyrhizi",
                    "doi": "10.1590/x",
                    "id": "https://scielo.br/x",
                    "ab_pt": "Estudo sobre fungicidas e resistencia.",
                    "year": "2024",
                },
                {
                    "ti_pt": "Outro estudo",
                    "doi": "10.1590/y",
                    "id": "https://scielo.br/y",
                    "ab_pt": "abstract 2",
                    "year": "2023",
                },
            ]
        }
    }
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "ferrugem", "max_results": 5, "state": _state()}
        )
    parsed = json.loads(raw)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Manejo de Phakopsora pachyrhizi"
    assert parsed[0]["doi"] == "10.1590/x"
    assert parsed[0]["url"] == "https://scielo.br/x"
    assert "fungicidas" in parsed[0]["abstract"]
    assert parsed[0]["year"] == "2024"


async def test_top_level_docs_schema() -> None:
    """Schema ``{"docs": [...]}`` (sem response wrapper) tambem funciona."""
    payload = {
        "docs": [
            {"title": "T1", "doi": "10.x/1", "url": "https://a"},
        ]
    }
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 1, "state": _state()}
        )
    parsed = json.loads(raw)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "T1"


async def test_top_level_results_schema() -> None:
    """Schema ``{"results": [...]}`` tambem funciona."""
    payload = {"results": [{"title": "T", "doi": "", "url": "https://r"}]}
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 1, "state": _state()}
        )
    parsed = json.loads(raw)
    assert len(parsed) == 1


async def test_es_like_hits_schema() -> None:
    """Schema ES-like ``{"hits": {"hits": [{"_source": {...}}]}}`` funciona."""
    payload = {
        "hits": {
            "hits": [
                {"_source": {"title": "ES Title", "doi": "10.es/1"}},
            ]
        }
    }
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 1, "state": _state()}
        )
    parsed = json.loads(raw)
    assert parsed[0]["title"] == "ES Title"


async def test_unknown_schema_returns_empty() -> None:
    """Schema desconhecido (sem articles) retorna []."""
    payload = {"unexpected": "thing"}
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 5, "state": _state()}
        )
    parsed = json.loads(raw)
    assert parsed == []


# ── article formatting ─────────────────────────────────────────────────────


async def test_abstract_truncated_to_500_chars() -> None:
    """Abstracts longos sao truncados a 500 caracteres."""
    long_abstract = "y" * 800
    payload = {
        "response": {
            "docs": [
                {
                    "ti_pt": "T",
                    "ab_pt": long_abstract,
                    "doi": "10.x/1",
                    "id": "https://a",
                }
            ]
        }
    }
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 1, "state": _state()}
        )
    parsed = json.loads(raw)
    assert len(parsed[0]["abstract"]) == 500


async def test_max_results_truncates_payload() -> None:
    """Mais articles que max_results sao cortados."""
    payload = {
        "response": {
            "docs": [
                {"ti_pt": f"T{i}", "doi": f"10.x/{i}", "id": f"https://a/{i}"}
                for i in range(10)
            ]
        }
    }
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        raw = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
    parsed = json.loads(raw)
    assert len(parsed) == 3


# ── context + cache ───────────────────────────────────────────────────────


async def test_crop_context_injected_in_query() -> None:
    """detected_crop_id no state vira sufixo da query enviada ao SciELO."""
    fake_client = _mock_async_client(response_data={"response": {"docs": []}})
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        await tool.ainvoke(
            {
                "query": "ferrugem",
                "max_results": 1,
                "state": _state(detected_crop_id="soja"),
            }
        )
    # check that get was called with q containing both
    call_kwargs = fake_client.get.await_args.kwargs
    assert "soja" in call_kwargs["params"]["q"]
    assert "ferrugem" in call_kwargs["params"]["q"]


async def test_no_crop_context_uses_raw_query() -> None:
    """Sem detected_crop_id, query nao recebe sufixo."""
    fake_client = _mock_async_client(response_data={"response": {"docs": []}})
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        await tool.ainvoke(
            {"query": "ferrugem", "max_results": 1, "state": _state()}
        )
    call_kwargs = fake_client.get.await_args.kwargs
    assert call_kwargs["params"]["q"] == "ferrugem"


async def test_cache_hit_skips_http() -> None:
    """Segunda chamada com mesma query bate cache, nao chama httpx de novo."""
    payload = {
        "response": {
            "docs": [
                {
                    "ti_pt": "Cached",
                    "doi": "10.x/c",
                    "id": "https://c",
                    "ab_pt": "ok",
                }
            ]
        }
    }
    fake_client = _mock_async_client(response_data=payload)
    with patch("httpx.AsyncClient", return_value=fake_client):
        tool = build_search_scientific_tool()
        first = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
        second = await tool.ainvoke(
            {"query": "x", "max_results": 3, "state": _state()}
        )
    assert first == second
    assert fake_client.get.await_count == 1


# ── helper unit tests ─────────────────────────────────────────────────────


def test_extract_articles_handles_list_input() -> None:
    data = [{"title": "a"}, {"title": "b"}, "not-a-dict"]
    out = _extract_articles(data)
    assert len(out) == 2


def test_extract_articles_handles_none_input() -> None:
    assert _extract_articles(None) == []
    assert _extract_articles(123) == []


def test_format_article_fills_defaults_for_missing_fields() -> None:
    out = _format_article({})
    assert out["title"] == ""
    assert out["doi"] == ""
    assert out["url"] == ""
    assert out["abstract"] == ""
    assert out["year"] == ""


def test_format_article_prefers_pt_over_en() -> None:
    out = _format_article(
        {"ti_pt": "Titulo PT", "ti_en": "EN Title", "ab_pt": "Resumo PT"}
    )
    assert out["title"] == "Titulo PT"
    assert "Resumo PT" in out["abstract"]


def test_first_returns_first_str_of_list() -> None:
    assert _first(["a", "b"]) == "a"
    assert _first([]) == ""
    assert _first("x") == "x"
    assert _first(None) == ""
    assert _first([123]) == ""  # nao-string vira ""
