"""Tool ``search_web`` — pesquisa web via Tavily (TCC-053).

Tier Pro+: o agente pode buscar informacoes atualizadas na web pra trazer
tratamentos novos, fungicidas atualizados, ocorrencias regionais e material de
extensao recente. Contextualiza pelo cultivo detectado em ``state.detected_crop_id``
quando disponivel.

Cache 24h aplicado via ``app.domains.chat.tools._cache`` (TCC-056) — queries
identicas pulam a chamada a API.

Notas de design:
- Factory pattern segue o resto do registry; nao ha services injetados.
- ``AsyncTavilyClient`` eh instanciado a cada chamada — barato, e evita prender
  conexao quando settings.tavily_api_key muda em runtime (testes).
- Falhas (sem API key, network, schema) sempre viram JSON com chave ``error``
  pra que o LLM possa reagir em vez de quebrar a cadeia de tools.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.config import settings
from app.domains.chat.agent_state import ChatState
from app.domains.chat.tools._cache import get_cached, make_cache_key, set_cached


def build_search_web_tool() -> BaseTool:
    """Factory pra ``search_web`` — sem dependencias injetadas.

    Returns:
        Tool decorada pronta pra bind no LLM.
    """

    @tool
    async def search_web(
        query: str,
        max_results: int = 5,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Pesquisa na web por informacoes atualizadas (tier Pro+).

        Use pra: tratamentos novos, fungicidas atualizados, ocorrencias regionais,
        material de extensao recente da Embrapa. A query eh contextualizada pelo
        cultivo detectado em ``state.detected_crop_id`` quando presente.

        Args:
            query: termo de busca (ex: "fungicida ferrugem asiatica 2025").
            max_results: numero maximo de resultados (default 5, max 10).

        Returns:
            JSON-string com lista de ``{title, url, snippet}`` ou
            ``{"error": "..."}`` em caso de falha.
        """
        if not settings.tavily_api_key:
            return json.dumps(
                {"error": "tavily_api_key nao configurada"},
                ensure_ascii=False,
            )

        crop_context = state.get("detected_crop_id")
        full_query = f"{query} {crop_context}" if crop_context else query

        # Cache 24h — chave baseada em (query, max_results) ja contextualizada.
        cache_key = make_cache_key("search_web", full_query, max_results)
        cached: str | None = get_cached(cache_key)
        if cached is not None:
            return cached

        # Import local: evita import-time error se tavily nao estiver instalado
        # em ambientes de teste / dev minimo.
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        try:
            results = await client.search(
                query=full_query,
                max_results=max_results,
                search_depth="basic",
            )
        except Exception as exc:  # noqa: BLE001 — tool nunca propaga
            return json.dumps(
                {"error": f"Tavily search falhou: {exc}"},
                ensure_ascii=False,
            )

        payload = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content", "") or "")[:300],
            }
            for r in results.get("results", [])
        ]
        encoded = json.dumps(payload, ensure_ascii=False)
        set_cached(cache_key, encoded)
        return encoded

    return search_web
