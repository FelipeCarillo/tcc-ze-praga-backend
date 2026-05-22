"""Tool ``search_my_diagnoses`` — busca semantica de diagnoses (TCC-046).

A tool consulta o Store em ``("user", uid, "diagnoses")`` por similaridade
de embedding com a ``query`` do usuario e retorna a lista de summaries +
metadados em JSON-string pro LLM consumir.

Em Sprint A2.5 (TCC-044) o Store passa a indexar diagnoses automaticamente
no ``persist_node``; esta tool consome esse indice.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


def build_search_my_diagnoses_tool(
    store_factory: Callable[[], Awaitable[BaseStore]] | None = None,
) -> BaseTool:
    """Factory pra ``search_my_diagnoses`` semantica via Store.

    Args:
        store_factory: callable async que retorna o ``BaseStore`` quando
            invocado. Em runtime, usa-se ``app.db.store.get_store``. Em
            testes, monkeypatch passando um AsyncMock. Quando ``None``,
            a tool ainda eh registrada mas retorna lista vazia (back-compat
            com testes que nao injetam store).
    """

    @tool
    async def search_my_diagnoses(
        query: str,
        limit: int = 5,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Busca diagnosticos passados do usuario por similaridade semantica.

        Args:
            query: descricao em linguagem natural do que voce procura
                (ex: "ferrugem ano passado", "doenca em soja").
            limit: numero maximo de resultados a retornar.

        Returns:
            JSON-string com a lista dos diagnoses mais relevantes — cada
            item contem ``summary_text``, ``diagnosis_id``, ``disease_id``,
            ``disease_name``, ``confidence``, ``severity``, ``created_at``.
        """
        user_id = state.get("current_user_id") if state else None
        if not user_id:
            logger.warning(
                "search_my_diagnoses chamado sem current_user_id no state"
            )
            return json.dumps([], ensure_ascii=False)

        if store_factory is None:
            return json.dumps([], ensure_ascii=False)

        try:
            store = await store_factory()
            results = await store.asearch(
                ("user", user_id, "diagnoses"),
                query=query,
                limit=limit,
            )
        except Exception:  # noqa: BLE001 — tool sempre retorna string
            logger.exception("search_my_diagnoses Store query failed")
            return json.dumps([], ensure_ascii=False)

        payload = [
            r.value if isinstance(r.value, dict) else dict(r.value)
            for r in results
        ]
        return json.dumps(payload, ensure_ascii=False, default=str)

    return search_my_diagnoses
