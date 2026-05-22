"""ChatService — orquestra grafo LangGraph + persistência de mensagens.

Substitui o keyword-matching antigo do chat/router.py por chamada real ao agente.
Mantém a responsabilidade de persistir histórico (ChatSession + ChatMessage) e,
quando o turno produz um diagnóstico, persistir o Diagnosis também.

A camada de transporte (router) continua simples: parseia messages/image/session,
delega ao service, retorna ChatResponse.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.domains.chat.agent import build_graph
from app.domains.chat.repository import ChatMessageRepository, ChatSessionRepository
from app.domains.chat.schemas import ChatResponse, CloseSessionResponse
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisResponse

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from app.domains.action_plans.service import ActionPlanService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        session_repo: ChatSessionRepository,
        message_repo: ChatMessageRepository,
        inference_svc: "InferenceService",
        action_plan_svc: "ActionPlanService",
        diagnosis_svc: "DiagnosisService",
        store_factory: Callable[[], Awaitable["BaseStore"]] | None = None,
    ) -> None:
        self._session_repo = session_repo
        self._message_repo = message_repo
        self._inference_svc = inference_svc
        self._action_plan_svc = action_plan_svc
        self._diagnosis_svc = diagnosis_svc
        self._store_factory = store_factory

    async def chat(
        self,
        user_id: str,
        session_id: str | None,
        message_text: str,
        image_filename: str | None,
        model_id: str,
    ) -> ChatResponse:
        """Roda o agente, persiste o turno e devolve a resposta consolidada.

        Pipeline:
          1) get_or_create session
          2) persist user message
          3) (opt) se há imagem, roda inferência e persiste Diagnosis ANTES
             do agente — assim o LLM recebe o resultado real no histórico
          4) invoke graph com mensagens + estado
          5) persist assistant message (com diagnosis_id se houver)
        """
        session = await self._session_repo.get_or_create_for_user(user_id, session_id)

        await self._message_repo.create(
            session_id=session.id,
            role="user",
            content=message_text,
            metadata={"image_filename": image_filename} if image_filename else None,
        )

        diagnosis: DiagnosisResponse | None = None
        seed_messages = [HumanMessage(content=message_text)]

        # Quando há imagem: roda inferência direto, persiste Diagnosis e injeta
        # o resultado no histórico via HumanMessage sintética. Isso garante que
        # o LLM (mesmo sem visão) "veja" o diagnóstico real, e que o Diagnosis
        # seja persistido independente das decisões do agente.
        if image_filename:
            diagnosis = await self._run_inference_and_persist(user_id, image_filename, model_id)
            seed_messages.append(
                HumanMessage(
                    content=(
                        "[Resultado da análise da imagem]\n"
                        f"Doença detectada: {diagnosis.disease_name}\n"
                        f"ID: {diagnosis.disease_id}\n"
                        f"Confiança: {diagnosis.confidence:.2%}\n"
                        f"Severidade: {diagnosis.severity}\n"
                        "Explique de forma amigável ao usuário e, se relevante, "
                        "use get_action_plan pra trazer recomendações."
                    )
                )
            )

        # Sprint A2.5 pre-fetch: busca diagnoses passados relevantes ao
        # conteudo da mensagem atual, alimenta o state pro LLM (e tools).
        recent_relevant = await self._prefetch_relevant_diagnoses(
            user_id, message_text
        )

        graph = build_graph(self._inference_svc, self._action_plan_svc)
        result = await graph.ainvoke(
            {
                "messages": seed_messages,
                "current_user_id": user_id,
                "image_filename": image_filename,
                "model_id": model_id,
                "last_diagnosis_id": diagnosis.id if diagnosis else None,
                "recent_relevant_diagnoses": recent_relevant,
            }
        )

        assistant_text = self._extract_final_text(result["messages"])

        await self._message_repo.create(
            session_id=session.id,
            role="assistant",
            content=assistant_text,
            diagnosis_id=diagnosis.id if diagnosis else None,
        )

        return ChatResponse(
            role="assistant",
            content=assistant_text,
            diagnosis=diagnosis,
            session_id=session.id,
        )

    async def chat_stream(
        self,
        user_id: str,
        session_id: str | None,
        message_text: str,
        image_filename: str | None,
        model_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Variante streaming — yields dicts {event, data} pro SSE endpoint.

        Eventos:
          - token:        chunk de texto do LLM
          - tool_call:    nome da tool sendo invocada
          - tool_result:  output da tool (string)
          - diagnosis:    DiagnosisResponse serializado (JSON-string)
          - done:         marcador final
        """
        session = await self._session_repo.get_or_create_for_user(user_id, session_id)
        await self._message_repo.create(
            session_id=session.id,
            role="user",
            content=message_text,
            metadata={"image_filename": image_filename} if image_filename else None,
        )

        diagnosis: DiagnosisResponse | None = None
        seed_messages = [HumanMessage(content=message_text)]

        if image_filename:
            diagnosis = await self._run_inference_and_persist(user_id, image_filename, model_id)
            seed_messages.append(
                HumanMessage(
                    content=(
                        "[Resultado da análise da imagem]\n"
                        f"Doença detectada: {diagnosis.disease_name}\n"
                        f"ID: {diagnosis.disease_id}\n"
                        f"Confiança: {diagnosis.confidence:.2%}\n"
                        f"Severidade: {diagnosis.severity}\n"
                        "Explique de forma amigável ao usuário e, se relevante, "
                        "use get_action_plan pra trazer recomendações."
                    )
                )
            )
            yield {
                "event": "diagnosis",
                "data": diagnosis.model_dump_json(),
            }

        graph = build_graph(self._inference_svc, self._action_plan_svc)
        initial_state = {
            "messages": seed_messages,
            "current_user_id": user_id,
            "image_filename": image_filename,
            "model_id": model_id,
            "last_diagnosis_id": diagnosis.id if diagnosis else None,
        }

        collected_chunks: list[str] = []
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                token = getattr(chunk, "content", None) or ""
                if token:
                    collected_chunks.append(token)
                    yield {"event": "token", "data": token}
            elif kind == "on_tool_start":
                yield {"event": "tool_call", "data": event.get("name", "")}
            elif kind == "on_tool_end":
                output = event["data"].get("output")
                tool_text = self._tool_output_to_text(output)
                yield {"event": "tool_result", "data": tool_text}

        assistant_text = "".join(collected_chunks).strip()
        if not assistant_text:
            # Fallback: ChatModel pode não ter emitido tokens streamados (FakeLLM,
            # erro de streaming, etc). Recupera o final via invoke não-stream
            # como safety net pra persistir algo coerente.
            result = await graph.ainvoke(initial_state)
            assistant_text = self._extract_final_text(result["messages"])

        await self._message_repo.create(
            session_id=session.id,
            role="assistant",
            content=assistant_text,
            diagnosis_id=diagnosis.id if diagnosis else None,
        )

        yield {"event": "done", "data": session.id}

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run_inference_and_persist(
        self, user_id: str, image_filename: str, model_id: str
    ) -> DiagnosisResponse:
        result = self._inference_svc.predict(model_id, image_filename)
        body = CreateDiagnosisRequest(
            disease_name=result.disease_name,
            disease_id=result.disease_id,
            scientific_name=result.scientific_name,
            confidence=result.confidence,
            severity=result.severity,
            description=result.description,
            model_used=result.model_id,
            image_url=None,
            image_name=result.image_name,
            top3=result.top3,
        )
        return await self._diagnosis_svc.create(user_id, body)

    @staticmethod
    def _extract_final_text(messages: list) -> str:
        """Pega o conteúdo do último AIMessage no histórico."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                # OpenAI as vezes manda content como list de parts
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in msg.content
                )
        return ""

    @staticmethod
    def _tool_output_to_text(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if hasattr(output, "content"):
            content = output.content
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False, default=str)
        return str(output)

    # ── Sprint A2.5: Store-backed memory ──────────────────────────────────────

    async def _prefetch_relevant_diagnoses(
        self, user_id: str, query: str, limit: int = 3
    ) -> list[dict]:
        """Pre-busca diagnoses passados relevantes pra o turno atual.

        Best-effort: erros do Store ou ausencia de factory retornam [].
        """
        if not self._store_factory or not query:
            return []
        try:
            store = await self._store_factory()
            results = await store.asearch(
                ("user", user_id, "diagnoses"),
                query=query,
                limit=limit,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Pre-fetch relevant diagnoses failed")
            return []
        return [
            r.value if isinstance(r.value, dict) else dict(r.value)
            for r in results
        ]

    async def close_session(
        self,
        user_id: str,
        session_id: str,
        llm: Any = None,
    ) -> CloseSessionResponse:
        """Fecha a sessao gerando + persistindo um resumo da conversa.

        Pipeline:
            1) busca sessao + mensagens
            2) gera resumo via LLM (mesmo prompt do rolling summary)
            3) persiste em ``chat_sessions.summary_text``
            4) indexa no Store em ``("user", uid, "session_summaries")``

        Args:
            user_id: dono da sessao.
            session_id: id da sessao a fechar.
            llm: LLM pra gerar o resumo. Quando ``None``, usa ChatOpenAI default.
        """
        session = await self._session_repo.get_by_id(session_id, user_id)
        if session is None:
            return CloseSessionResponse(session_id=session_id, summary_text=None)

        messages = await self._message_repo.list_by_session(session_id)
        if not messages:
            return CloseSessionResponse(
                session_id=session_id, summary_text=None
            )

        summary_text = await self._generate_session_summary(messages, llm)

        await self._session_repo.update_summary(
            session_id, user_id, summary_text
        )

        # Indexa no Store em ('user', uid, 'session_summaries').
        if self._store_factory and summary_text:
            try:
                from app.domains.chat.memory import (
                    index_session_summary_in_store,
                )

                store = await self._store_factory()
                await index_session_summary_in_store(
                    store, user_id, session_id, summary_text
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to index session summary in Store"
                )

        return CloseSessionResponse(
            session_id=session_id, summary_text=summary_text
        )

    async def _generate_session_summary(
        self, messages: list, llm: Any = None
    ) -> str:
        """Roda LLM com o prompt do rolling summary para gerar o resumo final."""
        if llm is None:
            from langchain_openai import ChatOpenAI

            from app.config import settings

            llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
            )

        history = [
            HumanMessage(content=m.content)
            if m.role == "user"
            else AIMessage(content=m.content)
            for m in messages
        ]
        prompt = (
            "Resuma em ate 200 palavras o conteudo desta conversa, "
            "preservando: 1) cultivos mencionados, 2) doencas identificadas, "
            "3) decisoes de manejo, 4) duvidas pendentes do usuario. "
            "Use linguagem natural em portugues."
        )
        response = await llm.ainvoke(
            [SystemMessage(content=prompt), *history]
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return content.strip() if isinstance(content, str) else str(content)
