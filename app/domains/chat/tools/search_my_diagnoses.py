"""Tool ``search_my_diagnoses`` — placeholder (TCC-041).

Sera semantico em Sprint A2.5 (embeddings + pgvector). Por ora retorna lista
vazia em formato JSON-string pra que o LLM ainda possa ser exposto a uma
ferramenta com mesma assinatura (compatibilidade futura).
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.domains.chat.agent_state import ChatState


def build_search_my_diagnoses_tool() -> BaseTool:
    """Factory pra ``search_my_diagnoses`` (placeholder)."""

    @tool
    async def search_my_diagnoses(
        query: str,  # noqa: ARG001 — placeholder
        limit: int = 5,  # noqa: ARG001 — placeholder
        *,
        state: Annotated[ChatState, InjectedState],  # noqa: ARG001
    ) -> str:
        """Busca diagnosticos passados do usuario (placeholder — sera semantico em A2.5)."""
        return json.dumps([], ensure_ascii=False)

    return search_my_diagnoses
