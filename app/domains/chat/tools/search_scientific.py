"""Tool ``search_scientific`` — pesquisa literatura cientifica via SciELO (TCC-054).

Tier Enterprise: o agente pode buscar evidencia citavel em artigos cientificos
pra apoiar diagnosticos com referencias. Contextualiza pelo cultivo detectado
em ``state.detected_crop_id`` quando disponivel.

A API publica do SciELO retorna JSON via parametro ``output=site``; o schema
pode variar — fazemos parse defensivo pra extrair os campos comuns. Schema
tipico (Solr-like): ``{"response": {"docs": [{...}]}}``.

Cache 24h aplicado via ``_cache`` modulo (TCC-056) — queries identicas pulam
o GET HTTP.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import httpx
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState
from app.domains.chat.tools._cache import get_cached, make_cache_key, set_cached

SCIELO_URL = "https://search.scielo.org/"


def build_search_scientific_tool() -> BaseTool:
    """Factory pra ``search_scientific`` — sem dependencias injetadas.

    Returns:
        Tool decorada pronta pra bind no LLM.
    """

    @tool
    async def search_scientific(
        query: str,
        max_results: int = 5,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Pesquisa literatura cientifica via SciELO (tier Enterprise).

        Use pra evidencia citavel em diagnosticos (DOI, titulo, abstract).
        A query eh contextualizada pelo cultivo detectado em
        ``state.detected_crop_id`` quando presente.

        Args:
            query: termo de busca (ex: "Phakopsora pachyrhizi resistance").
            max_results: numero maximo de resultados (default 5).

        Returns:
            JSON-string com lista de ``{title, doi, url, abstract, year}`` ou
            ``{"error": "..."}`` em caso de falha.
        """
        crop_context = state.get("detected_crop_id")
        full_query = f"{query} {crop_context}" if crop_context else query

        cache_key = make_cache_key("search_scientific", full_query, max_results)
        cached: str | None = get_cached(cache_key)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(
                    SCIELO_URL,
                    params={
                        "q": full_query,
                        "output": "site",
                        "lang": "pt-br",
                        "count": max_results,
                        "format": "json",
                    },
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001 — tool nunca propaga
                return json.dumps(
                    {"error": f"SciELO falhou: {exc}"},
                    ensure_ascii=False,
                )

        articles = _extract_articles(data)
        payload = [_format_article(a) for a in articles[:max_results]]
        encoded = json.dumps(payload, ensure_ascii=False)
        set_cached(cache_key, encoded)
        return encoded

    return search_scientific


def _extract_articles(data: Any) -> list[dict[str, Any]]:
    """Extrai lista de articles de payload SciELO com varios formatos possiveis.

    SciELO pode retornar Solr ({"response": {"docs": [...]}}), top-level docs,
    results, ou hits.hits (ES-like). Cobre todos esses caminhos defensivamente.
    """
    if isinstance(data, list):
        return [a for a in data if isinstance(a, dict)]
    if not isinstance(data, dict):
        return []
    response = data.get("response")
    if isinstance(response, dict):
        docs = response.get("docs")
        if isinstance(docs, list):
            return [d for d in docs if isinstance(d, dict)]
    docs = data.get("docs")
    if isinstance(docs, list):
        return [d for d in docs if isinstance(d, dict)]
    results = data.get("results")
    if isinstance(results, list):
        return [d for d in results if isinstance(d, dict)]
    hits = data.get("hits")
    if isinstance(hits, dict):
        inner = hits.get("hits")
        if isinstance(inner, list):
            return [
                h.get("_source", h) if isinstance(h, dict) else {}
                for h in inner
            ]
    return []


def _format_article(a: dict[str, Any]) -> dict[str, Any]:
    """Normaliza dict de artigo pra schema final.

    Defensive: campos podem nao existir ou vir como listas; retorna strings
    vazias quando ausentes.
    """
    title = (
        a.get("ti_pt")
        or a.get("title")
        or a.get("ti_en")
        or _first(a.get("ti"))
        or ""
    )
    abstract = (
        a.get("ab_pt")
        or a.get("abstract")
        or a.get("ab_en")
        or _first(a.get("ab"))
        or ""
    )
    abstract_str = abstract if isinstance(abstract, str) else ""
    return {
        "title": title if isinstance(title, str) else "",
        "doi": a.get("doi", "") or "",
        "url": a.get("id") or a.get("url") or a.get("link") or "",
        "abstract": abstract_str[:500],
        "year": a.get("year") or a.get("publication_year") or "",
    }


def _first(val: Any) -> str:
    """Retorna primeiro elemento de lista, ou val (se str), ou ''."""
    if isinstance(val, list):
        return val[0] if val and isinstance(val[0], str) else ""
    return val if isinstance(val, str) else ""
