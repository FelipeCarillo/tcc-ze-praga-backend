"""Tools de talhão do agente (TCC-098).

O Zé pergunta em qual talhão a foto foi tirada (via ``ask_user``) e registra a
escolha — ou cadastra um talhão novo descrito na conversa ("cria o talhão 9
do Rio, 40 hectares"). As três tools mexem em ``selected_talhao_id``, que é o
que ``analyze_image``/``deep_diagnose`` usam para ligar o laudo ao talhão.

- ``list_my_talhoes``: fazendas do usuário com os talhões e o talhão já
  escolhido no turno (para o agente não perguntar à toa).
- ``use_talhao``: escolhe um talhão existente (dono validado).
- ``register_talhao``: cria o talhão (na fazenda pedida ou na padrão) e já o
  escolhe.

``talhao_selected`` no estado leva a escolha até a resposta: a UI torna o
talhão ativo e, quando ``created`` é verdadeiro, mostra o cartão "Talhão
criado".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.core.exceptions import NotFoundError
from app.domains.chat.agent_state import ChatState
from app.domains.fazendas.repository import FazendaRepository
from app.domains.talhoes.repository import TalhaoRepository
from app.domains.talhoes.schemas import CreateTalhaoRequest
from app.domains.talhoes.service import TalhaoService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _reply(tool_call_id: str, payload: dict[str, Any], **update: Any) -> Command[Any]:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False, default=str),
                    tool_call_id=tool_call_id,
                )
            ],
            **update,
        }
    )


def _selected(talhao: Any, fazenda_nome: str | None, created: bool) -> dict[str, Any]:
    return {
        "id": talhao.id,
        "nome": talhao.nome,
        "apelido": talhao.apelido,
        "hectares": talhao.hectares,
        "data_semeadura": talhao.data_semeadura.isoformat() if talhao.data_semeadura else None,
        "fazenda_id": talhao.fazenda_id,
        "fazenda_nome": fazenda_nome,
        "created": created,
    }


def build_talhao_tools(
    db_session_factory: Callable[[], AsyncSession],
) -> dict[str, Callable[[], BaseTool]]:
    """Factories zero-arg das três tools, para o ``tool_registry``."""

    def build_list() -> BaseTool:
        @tool
        async def list_my_talhoes(
            *, state: Annotated[ChatState, InjectedState]
        ) -> str:
            """Lista as fazendas do usuário com os talhões de cada uma.

            Chame antes de analisar uma foto: se ``selected_talhao_id`` vier
            preenchido, o usuário já escolheu o talhão — não pergunte.
            """
            user_id = state.get("current_user_id") or ""
            async with db_session_factory() as session:
                fazendas = await FazendaRepository(session).find_all_by_user(user_id)
                talhoes = await TalhaoRepository(session).find_all_by_user(user_id)
            return json.dumps(
                {
                    "selected_talhao_id": state.get("selected_talhao_id"),
                    "fazendas": [
                        {
                            "id": f.id,
                            "nome": f.nome,
                            "talhoes": [
                                {"id": t.id, "nome": t.nome, "apelido": t.apelido}
                                for t in talhoes
                                if t.fazenda_id == f.id
                            ],
                        }
                        for f in fazendas
                    ],
                },
                ensure_ascii=False,
            )

        return list_my_talhoes

    def build_use() -> BaseTool:
        @tool
        async def use_talhao(
            talhao_id: str,
            *,
            state: Annotated[ChatState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command[Any]:
            """Escolhe um talhão existente para os laudos deste turno."""
            user_id = state.get("current_user_id") or ""
            async with db_session_factory() as session:
                talhao = await TalhaoRepository(session).find_by_id(talhao_id, user_id)
                fazenda = (
                    await FazendaRepository(session).find_by_id(talhao.fazenda_id, user_id)
                    if talhao
                    else None
                )
            if talhao is None:
                return _reply(tool_call_id, {"error": "Talhão não encontrado."})
            selected = _selected(talhao, fazenda.nome if fazenda else None, created=False)
            return _reply(
                tool_call_id,
                {"ok": True, "talhao": selected},
                selected_talhao_id=talhao.id,
                talhao_selected=selected,
            )

        return use_talhao

    def build_register() -> BaseTool:
        @tool
        async def register_talhao(
            nome: str,
            apelido: str | None = None,
            hectares: float | None = None,
            data_semeadura: str | None = None,
            fazenda_id: str | None = None,
            *,
            state: Annotated[ChatState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command[Any]:
            """Cadastra um talhão novo e já o escolhe para os laudos do turno.

            Use quando o usuário descrever um talhão que ainda não existe
            ("talhão 9 do Rio, 40 hectares, plantei dia 12 de setembro").

            Args:
                nome: nome curto, ex. "Talhão 9".
                apelido: apelido opcional, ex. "Rio".
                hectares: área em hectares, se o usuário disser.
                data_semeadura: data de semeadura em AAAA-MM-DD, se disser.
                fazenda_id: fazenda de ``list_my_talhoes``; sem ela, a padrão.
            """
            user_id = state.get("current_user_id") or ""
            try:
                semeadura = date.fromisoformat(data_semeadura) if data_semeadura else None
            except ValueError:
                semeadura = None
            request = CreateTalhaoRequest(
                nome=nome.strip()[:120] or "Novo talhão",
                apelido=(apelido or None) and apelido.strip()[:120],
                hectares=hectares if hectares is None or hectares >= 0 else None,
                data_semeadura=semeadura,
                fazenda_id=fazenda_id,
            )
            async with db_session_factory() as session:
                fazendas = FazendaRepository(session)
                svc = TalhaoService(TalhaoRepository(session), fazendas)
                try:
                    created = await svc.create(user_id, request)
                except NotFoundError:
                    # Fazenda inventada pelo modelo: cai na padrão do usuário.
                    created = await svc.create(
                        user_id, request.model_copy(update={"fazenda_id": None})
                    )
                fazenda = await fazendas.find_by_id(created.fazenda_id, user_id)
            selected = _selected(created, fazenda.nome if fazenda else None, created=True)
            return _reply(
                tool_call_id,
                {"ok": True, "talhao": selected},
                selected_talhao_id=created.id,
                talhao_selected=selected,
            )

        return register_talhao

    return {
        "list_my_talhoes": build_list,
        "use_talhao": build_use,
        "register_talhao": build_register,
    }
