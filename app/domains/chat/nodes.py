"""Nodes auxiliares do chatbot_graph (Sprint A2.5 / TCC-047).

Centraliza os nodes que nao sao o llm/tools default — por ora apenas o
``maybe_summarize_node`` que faz compressao de historico (rolling summary)
quando o contexto cresce acima do limite.

Por convencao do langgraph, todo node retorna um dict de updates parciais
do state; ``messages`` segue o reducer ``add_messages``, entao um update
``{"messages": [...]}`` faz append (nao replace). Pra fazer **replace** do
historico (como precisamos no rolling summary), usamos o pattern de
``RemoveMessage`` ou substituicao do estado inteiro via ``messages``
contendo a lista nova completa apos um ``REPLACE_ALL`` token.

Implementacao Sprint A2.5: substituimos as primeiras N-10 mensagens por
1 ``SystemMessage`` contendo o resumo. Como o reducer eh ``add_messages``,
fazemos isso retornando ``messages`` com ``RemoveMessage`` para cada msg
antiga + a ``SystemMessage`` nova + as 10 ultimas reapendadas.

Simpler approach: o node retorna o slice completo das mensagens via
``REMOVE_ALL_MESSAGES`` + reconstrucao — mas o add_messages reducer ja
suporta `RemoveMessage` por id pra remover seletivamente.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import RemoveMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from app.domains.chat.agent_state import ChatState

# Numero de mensagens recentes que ficam intactas (nao sao compressas).
_KEEP_RECENT = 10
# Threshold absoluto pra disparar compressao.
_COMPRESS_THRESHOLD = 20

_SUMMARY_PROMPT = (
    "Resuma em ate 150 palavras o conteudo destas trocas, preservando: "
    "1) cultivos mencionados, 2) doencas identificadas, 3) decisoes de "
    "manejo, 4) duvidas pendentes. Use linguagem natural em portugues."
)


async def maybe_summarize_node(
    state: ChatState,
    *,
    llm: BaseChatModel,
) -> dict:
    """Compress historico longo via LLM, mantendo as ultimas 10 mensagens.

    Quando ``len(messages) > 20``, gera um resumo das mensagens iniciais e
    substitui o historico antigo por um ``SystemMessage`` com esse resumo
    + as 10 mensagens mais recentes. Quando ``<= 20``, retorna noop.

    Args:
        state: estado do grafo (deve conter ``messages``).
        llm: chat model usado pra gerar o resumo.

    Returns:
        Update dict — vazio quando nao ha compressao; senao, contem o
        ``messages`` com ``RemoveMessage`` IDs antigos + summary + keep.
    """
    messages = state.get("messages", [])
    if len(messages) <= _COMPRESS_THRESHOLD:
        return {}

    # Tudo que NAO esta nas 10 ultimas vai pro resumo.
    to_compress = messages[:-_KEEP_RECENT]
    keep = messages[-_KEEP_RECENT:]

    summary_resp = await llm.ainvoke(
        [SystemMessage(content=_SUMMARY_PROMPT), *to_compress]
    )
    summary_content = getattr(summary_resp, "content", "")
    if isinstance(summary_content, list):
        # Provedor pode mandar content como parts.
        summary_content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in summary_content
        )

    summary_msg = SystemMessage(
        content=f"[Conversa anterior resumida]: {summary_content}"
    )

    # Pattern do add_messages reducer:
    # - RemoveMessage(id=x) remove a msg com aquele id
    # - Mensagens com id ja existente sobrescrevem (idempotent)
    # Removemos TODAS as msgs antigas + recentes (com id), e adicionamos
    # a ordem desejada [summary, ...keep] de uma vez. As mensagens em
    # ``keep`` reaparecem com mesmos ids — add_messages dedup-ara por id
    # mas o RemoveMessage primeiro garante a limpeza.
    remove_updates = [
        RemoveMessage(id=msg.id)
        for msg in messages
        if getattr(msg, "id", None) is not None
    ]

    return {"messages": [*remove_updates, summary_msg, *keep]}
