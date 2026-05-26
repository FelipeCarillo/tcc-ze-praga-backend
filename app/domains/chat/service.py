"""ChatService — orquestra grafo LangGraph + persistência de mensagens.

Substitui o keyword-matching antigo do chat/router.py por chamada real ao agente.
Mantém a responsabilidade de persistir histórico (ChatSession + ChatMessage) e,
quando o turno produz um diagnóstico, persistir o Diagnosis também.

A camada de transporte (router) continua simples: parseia messages/image/session,
delega ao service, retorna ChatResponse.

Sprint A3 (TCC-051): aceita ``SubscriptionRepository`` opcional pra carregar
``PlanFeatures`` do usuario e passar pro ``build_graph`` (escolhe LLM model
dinamico por tier). Quando sub_repo for None, usa default settings.

Sprint A4.5 (TCC-058/059): introduz HITL via ``langgraph.interrupt()`` —
quando uma tool dispara interrupt, o grafo pausa, o checkpointer persiste o
snapshot, e o turno e retomado via ``Command(resume=<resposta>)`` no
endpoint ``POST /chat/resume``. O endpoint ``GET /chat/interrupts`` lista
threads do usuario com interrupts pendentes.
"""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.domains.chat.agent import build_graph
from app.domains.chat.repository import ChatMessageRepository, ChatSessionRepository
from app.domains.chat.schemas import (
    ChatResponse,
    CloseSessionResponse,
    InterruptInfo,
    PendingInterrupt,
)
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisResponse
from app.domains.subscriptions.features import FREE_FEATURES, PlanFeatures

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

    from app.domains.action_plans.service import ActionPlanService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService
    from app.domains.subscriptions.repository import SubscriptionRepository

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
        checkpointer_factory: (
            Callable[[], Awaitable["BaseCheckpointSaver"]] | None
        ) = None,
        sub_repo: "SubscriptionRepository | None" = None,
    ) -> None:
        self._session_repo = session_repo
        self._message_repo = message_repo
        self._inference_svc = inference_svc
        self._action_plan_svc = action_plan_svc
        self._diagnosis_svc = diagnosis_svc
        self._store_factory = store_factory
        self._checkpointer_factory = checkpointer_factory
        self._sub_repo = sub_repo
        # Cached compiled graph por (plan_features_key) — built lazily.
        # Chaveado por hash de plan_features pra permitir LLM-switching por tier.
        self._graph_cache: dict[str, Any] = {}

    async def _resolve_plan_features(self, user_id: str) -> PlanFeatures:
        """Carrega PlanFeatures do plano ativo. Fallback: FREE_FEATURES."""
        if self._sub_repo is None:
            return FREE_FEATURES
        sub = await self._sub_repo.find_user_subscription(user_id)
        if sub is None or sub.plan.features is None:
            return FREE_FEATURES
        try:
            return PlanFeatures(**sub.plan.features)
        except Exception:  # noqa: BLE001 — fallback resilient
            return FREE_FEATURES

    async def _get_graph(self, plan_features: PlanFeatures | None = None) -> Any:
        """Resolve o grafo compilado lazy-init — com checkpointer + plan_features.

        O checkpointer eh compartilhado entre chamadas (chat / resume /
        interrupts list) pra garantir que snapshots persistidos sobrevivem
        a chamadas distintas — requisito do ciclo HITL.

        Cacheia por plan_features (hash da config) — assim tiers diferentes
        usam grafos com LLM models diferentes sem rebuild a cada turno.
        """
        # Cache key: plan_features tier_name (default 'free').
        tier_key = "free" if plan_features is None else plan_features.tier_name
        if tier_key in self._graph_cache:
            return self._graph_cache[tier_key]

        checkpointer = None
        if self._checkpointer_factory is not None:
            try:
                checkpointer = await self._checkpointer_factory()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Checkpointer factory failed — graph runs sem persistencia"
                )

        graph = build_graph(
            self._inference_svc,
            self._action_plan_svc,
            plan_features=plan_features,
            checkpointer=checkpointer,
        )
        self._graph_cache[tier_key] = graph
        return graph

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

        # Sprint A3: carrega features do plano ativo pra escolher LLM model.
        plan_features = await self._resolve_plan_features(user_id)
        graph = await self._get_graph(plan_features=plan_features)
        config = {"configurable": {"thread_id": session.id}}
        result = await graph.ainvoke(
            {
                "messages": seed_messages,
                "current_user_id": user_id,
                "image_filename": image_filename,
                "model_id": model_id,
                "last_diagnosis_id": diagnosis.id if diagnosis else None,
                "recent_relevant_diagnoses": recent_relevant,
            },
            config=config,
        )

        # Detecta interrupt — quando o grafo pausa via ask_user, o "result"
        # contem o payload do interrupt em vez do estado final. Ainda assim
        # devolvemos a sessao com diagnosis (se houve) e content vazio +
        # info do interrupt, deixando o cliente disparar /chat/resume.
        interrupt_info = self._extract_interrupt_from_result(result)
        if interrupt_info is not None:
            return ChatResponse(
                role="assistant",
                content="",
                diagnosis=diagnosis,
                session_id=session.id,
                interrupt=interrupt_info,
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

        plan_features = await self._resolve_plan_features(user_id)
        graph = await self._get_graph(plan_features=plan_features)
        config = {"configurable": {"thread_id": session.id}}
        initial_state = {
            "messages": seed_messages,
            "current_user_id": user_id,
            "image_filename": image_filename,
            "model_id": model_id,
            "last_diagnosis_id": diagnosis.id if diagnosis else None,
        }

        collected_chunks: list[str] = []
        async for event in graph.astream_events(
            initial_state, version="v2", config=config
        ):
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

        # Apos o streaming completar, verifica se o grafo pausou em interrupt.
        # Quando pausou, emite evento `interrupt` em vez de persistir resposta
        # vazia — o cliente deve renderizar dialog + chamar /chat/resume.
        interrupt_info = await self._extract_pending_interrupt(graph, config)
        if interrupt_info is not None:
            yield {
                "event": "interrupt",
                "data": interrupt_info.model_dump(),
            }
            yield {"event": "done", "data": session.id}
            return

        assistant_text = "".join(collected_chunks).strip()
        if not assistant_text:
            # Fallback: ChatModel pode não ter emitido tokens streamados (FakeLLM,
            # erro de streaming, etc). Recupera o final via invoke não-stream
            # como safety net pra persistir algo coerente.
            result = await graph.ainvoke(initial_state, config=config)
            assistant_text = self._extract_final_text(result["messages"])

        await self._message_repo.create(
            session_id=session.id,
            role="assistant",
            content=assistant_text,
            diagnosis_id=diagnosis.id if diagnosis else None,
        )

        yield {"event": "done", "data": session.id}

    # ── Sprint A4.5: HITL resume + interrupts listing ─────────────────────────

    async def resume(
        self, user_id: str, thread_id: str, response: str
    ) -> ChatResponse:
        """Retoma uma sessao interrompida via ``Command(resume=response)``.

        Args:
            user_id: dono da sessao — usado pra validar ownership.
            thread_id: id da sessao (= chat_session.id).
            response: texto da resposta do usuario ao interrupt.

        Returns:
            ``ChatResponse`` com o conteudo final apos retomada. Pode
            conter outro ``interrupt`` se o agente disparou nova pergunta.
        """
        session = await self._session_repo.get_by_id(thread_id, user_id=user_id)
        if session is None:
            return ChatResponse(
                role="assistant",
                content="",
                session_id=thread_id,
            )

        plan_features = await self._resolve_plan_features(user_id)
        graph = await self._get_graph(plan_features=plan_features)
        config = {"configurable": {"thread_id": thread_id}}

        # Persiste a resposta do usuario como ChatMessage (role=user) pra
        # historico no DB — diferente da seed message do turno original.
        await self._message_repo.create(
            session_id=thread_id,
            role="user",
            content=response,
            metadata={"resume": True},
        )

        result = await graph.ainvoke(Command(resume=response), config=config)

        interrupt_info = self._extract_interrupt_from_result(result)
        if interrupt_info is not None:
            return ChatResponse(
                role="assistant",
                content="",
                session_id=thread_id,
                interrupt=interrupt_info,
            )

        assistant_text = self._extract_final_text(result["messages"])
        await self._message_repo.create(
            session_id=thread_id,
            role="assistant",
            content=assistant_text,
        )
        return ChatResponse(
            role="assistant",
            content=assistant_text,
            session_id=thread_id,
        )

    async def resume_stream(
        self, user_id: str, thread_id: str, response: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Variante SSE do resume — emite token/tool_call/interrupt/done."""
        session = await self._session_repo.get_by_id(thread_id, user_id=user_id)
        if session is None:
            yield {"event": "done", "data": thread_id}
            return

        plan_features = await self._resolve_plan_features(user_id)
        graph = await self._get_graph(plan_features=plan_features)
        config = {"configurable": {"thread_id": thread_id}}

        await self._message_repo.create(
            session_id=thread_id,
            role="user",
            content=response,
            metadata={"resume": True},
        )

        collected_chunks: list[str] = []
        async for event in graph.astream_events(
            Command(resume=response), version="v2", config=config
        ):
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

        interrupt_info = await self._extract_pending_interrupt(graph, config)
        if interrupt_info is not None:
            yield {"event": "interrupt", "data": interrupt_info.model_dump()}
            yield {"event": "done", "data": thread_id}
            return

        assistant_text = "".join(collected_chunks).strip()
        if not assistant_text:
            result = await graph.ainvoke(None, config=config)
            assistant_text = self._extract_final_text(result["messages"])

        await self._message_repo.create(
            session_id=thread_id,
            role="assistant",
            content=assistant_text,
        )
        yield {"event": "done", "data": thread_id}

    async def list_pending_interrupts(
        self, user_id: str
    ) -> list[PendingInterrupt]:
        """Lista threads do user com interrupt ativo.

        Itera ``chat_sessions`` do usuario; pra cada sessao, consulta o
        snapshot do checkpointer e detecta se ha tasks com interrupts
        pendentes. Sessoes sem snapshot persistido (ou erro de leitura)
        sao silenciosamente puladas.
        """
        if self._checkpointer_factory is None:
            return []

        sessions = await self._session_repo.list_for_user(user_id)
        plan_features = await self._resolve_plan_features(user_id)
        graph = await self._get_graph(plan_features=plan_features)
        pending: list[PendingInterrupt] = []
        for sess in sessions:
            config = {"configurable": {"thread_id": sess.id}}
            try:
                snapshot = await graph.aget_state(config)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to read state for session %s", sess.id
                )
                continue

            interrupts = [
                i for task in snapshot.tasks for i in (task.interrupts or [])
            ]
            if not interrupts:
                continue

            payload = interrupts[0].value
            if not isinstance(payload, dict):
                continue

            try:
                info = InterruptInfo(
                    kind=payload.get("kind", "ask_user"),
                    question=payload.get("question", ""),
                    response_kind=payload.get("response_kind", "text"),
                    options=payload.get("options"),
                    asked_at=payload.get("asked_at"),
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Invalid interrupt payload for session %s", sess.id
                )
                continue

            created_at = None
            try:
                created_at = (
                    snapshot.created_at if snapshot.created_at else None
                )
            except Exception:  # noqa: BLE001
                created_at = None

            pending.append(
                PendingInterrupt(
                    session_id=sess.id,
                    interrupt=info,
                    created_at=created_at,
                )
            )

        return pending

    # ── Interrupt helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_interrupt_from_result(result: Any) -> InterruptInfo | None:
        """Detecta interrupt no retorno de ``graph.ainvoke``.

        Quando o grafo pausa em ``interrupt()``, o LangGraph anexa o payload
        em ``result["__interrupt__"]`` (uma tupla de ``Interrupt``). Quando
        nao ha interrupt, retorna ``None``.
        """
        if not isinstance(result, dict):
            return None
        raw = result.get("__interrupt__")
        if not raw:
            return None
        # raw eh uma tupla de Interrupt(value=...) — pega o primeiro.
        try:
            first = raw[0]
        except (IndexError, TypeError):
            return None
        payload = getattr(first, "value", None)
        if not isinstance(payload, dict):
            return None
        try:
            return InterruptInfo(
                kind=payload.get("kind", "ask_user"),
                question=payload.get("question", ""),
                response_kind=payload.get("response_kind", "text"),
                options=payload.get("options"),
                asked_at=payload.get("asked_at"),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Invalid interrupt payload from ainvoke result")
            return None

    @staticmethod
    async def _extract_pending_interrupt(
        graph: Any, config: dict
    ) -> InterruptInfo | None:
        """Consulta o snapshot pra detectar interrupt pendente apos streaming."""
        try:
            snapshot = await graph.aget_state(config)
        except Exception:  # noqa: BLE001
            return None
        interrupts = [
            i for task in snapshot.tasks for i in (task.interrupts or [])
        ]
        if not interrupts:
            return None
        payload = interrupts[0].value
        if not isinstance(payload, dict):
            return None
        try:
            return InterruptInfo(
                kind=payload.get("kind", "ask_user"),
                question=payload.get("question", ""),
                response_kind=payload.get("response_kind", "text"),
                options=payload.get("options"),
                asked_at=payload.get("asked_at"),
            )
        except Exception:  # noqa: BLE001
            return None

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
        # ``Diagnosis.crop_id`` eh NOT NULL desde 0004 — extrai do catalogo
        # carregado no InferenceService (todas as diseases sao do mesmo crop).
        crop_uuid = (
            self._inference_svc.disease_catalog[0].crop_id
            if self._inference_svc.disease_catalog
            else ""
        )
        return await self._diagnosis_svc.create(user_id, body, crop_id=crop_uuid)

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
            from app.config import settings
            from app.core.llm import get_chat_model

            llm = get_chat_model(settings.chat_model)

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
