"""Tool ``ask_user`` — pergunta direta ao usuario via langgraph interrupt (TCC-057).

Implementa o padrao human-in-the-loop do LangGraph: quando o agente precisa
de mais informacao do usuario antes de continuar (cultivo ambiguo, top-2
doencas com confianca proxima, confirmacao antes de um plano de acao caro),
ele invoca ``ask_user`` que dispara ``langgraph.types.interrupt()``. O grafo
pausa, o checkpointer persiste o snapshot, e o turno e retomado via
``Command(resume=<resposta>)`` quando o usuario responder.

InjectedState eh aceito (mesmo nao usado pela tool) pra manter o mesmo
contrato das demais tools e abrir espaco pra futuras heuristicas baseadas
em contexto (ex: nao perguntar se ``preferred_action_level`` ja foi
escolhido em turno anterior).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from app.domains.chat.agent_state import ChatState


def build_ask_user_tool() -> BaseTool:
    """Factory pra ``ask_user`` — sem dependencias externas (tool pura).

    Returns:
        Tool decorada pronta pra ser bindada no LLM.
    """

    @tool
    def ask_user(
        question: str,
        response_kind: Literal["text", "choice", "boolean", "confirm"] = "text",
        options: list[str] | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Pergunta diretamente ao usuario antes de continuar.

        Use quando:
        - Cultivo eh ambiguo (varias possibilidades visuais).
        - Top-2 doencas com confianca proxima — pedir confirmacao do usuario.
        - Confirmar substituicao de imagem ambigua.
        - Validar contexto antes de gerar plano de acao caro.

        Args:
            question: pergunta clara em pt-br pro usuario.
            response_kind: tipo de resposta esperada — ``"text"`` (livre),
                ``"choice"`` (escolha 1 de N), ``"boolean"`` (sim/nao),
                ``"confirm"`` (apenas OK pra continuar).
            options: lista de opcoes — obrigatorio quando
                ``response_kind="choice"``, ignorado nos demais.

        Returns:
            String com a resposta do usuario apos resume. O LLM continua
            o raciocinio com essa resposta no contexto.
        """
        # state aceito por contrato (futuras heuristicas).
        _ = state

        user_response = interrupt(
            {
                "kind": "ask_user",
                "question": question,
                "response_kind": response_kind,
                "options": options,
                "asked_at": datetime.now(UTC).isoformat(),
            }
        )
        return f"Usuario respondeu: {user_response}"

    return ask_user
