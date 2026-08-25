"""ChatService — orquestra grafo LangGraph + persistência de mensagens.

Substitui o keyword-matching antigo do chat/router.py por chamada real ao agente.
Mantém a responsabilidade de persistir histórico (ChatSession + ChatMessage) e,
quando o turno produz um diagnóstico, persistir o Diagnosis também.

A camada de transporte (router) continua simples: parseia messages/image/session,
delega ao service, retorna ChatResponse.

Sprint A3 (TCC-051): aceita ``SubscriptionRepository`` opcional pra carregar
``PlanFeatures`` do usuario e passar pro ``build_graph`` (escolhe LLM model
dinamico por tier). Quando sub_repo for None, usa default settings.

O conjunto de tools vem do ``tool_registry``: ``_build_tool_factories`` monta o
dict ``name -> factory`` e ``build_tools`` filtra por ``enabled_globally`` +
``required_feature`` + ``min_tier`` do plano ativo. Antes disso o service montava
uma lista fixa de 4 tools e o registry (com as 11) só era exercitado em testes.

Sprint A4.5 (TCC-058/059): introduz HITL via ``langgraph.interrupt()`` —
quando uma tool dispara interrupt, o grafo pausa, o checkpointer persiste o
snapshot, e o turno e retomado via ``Command(resume=<resposta>)`` no
endpoint ``POST /chat/resume``. O endpoint ``GET /chat/interrupts`` lista
threads do usuario com interrupts pendentes.
"""

import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from app.domains.chat import agent_state
from app.domains.chat.agent import build_graph
from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.repository import ChatMessageRepository, ChatSessionRepository
from app.domains.chat.schemas import (
    ChatMessageResponse,
    ChatResponse,
    ChatSessionSummary,
    CloseSessionResponse,
    InterruptInfo,
    PendingInterrupt,
)
from app.domains.chat.tool_registry import build_tools
from app.domains.diagnoses.schemas import DiagnosisResponse
from app.domains.subscriptions.features import FREE_FEATURES, PlanFeatures

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore

    from app.domains.action_plans.service import ActionPlanService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService
    from app.domains.subscriptions.repository import SubscriptionRepository
    from app.domains.uploads.service import UploadService

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
            Callable[[], Awaitable["BaseCheckpointSaver[Any]"]] | None
        ) = None,
        sub_repo: "SubscriptionRepository | None" = None,
        diagnosis_graph_factory: (
            Callable[..., "CompiledStateGraph[Any]"] | None
        ) = None,
        db_session_factory: Callable[[], Any] | None = None,
        upload_svc: "UploadService | None" = None,
    ) -> None:
        self._session_repo = session_repo
        self._message_repo = message_repo
        self._inference_svc = inference_svc
        self._action_plan_svc = action_plan_svc
        self._diagnosis_svc = diagnosis_svc
        self._store_factory = store_factory
        self._checkpointer_factory = checkpointer_factory
        self._sub_repo = sub_repo
        self._diagnosis_graph_factory = diagnosis_graph_factory
        if db_session_factory is None:
            from app.db.database import AsyncSessionLocal

            db_session_factory = AsyncSessionLocal
        self._db_session_factory = db_session_factory
        self._upload_svc = upload_svc
        # Cached compiled graph por plan_features.signature() — built lazily.
        # A assinatura cobre TODAS as flags do plano (não só o tier), porque o
        # conjunto de tools ativas varia com elas, não apenas o LLM model.
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

    def _build_tool_factories(
        self, store: "BaseStore | None" = None
    ) -> dict[str, Callable[[], "BaseTool"]]:
        """Mapa ``nome -> factory`` consumido pelo ``tool_registry``.

        As factories sao zero-arg: capturam os services por closure. Tools cuja
        dependencia nao esta disponivel (ex: ``deep_diagnose`` sem
        ``diagnosis_graph_factory``) simplesmente nao entram no dict — o
        ``build_tools`` pula nomes ativos sem factory correspondente.

        Args:
            store: ``BaseStore`` ja resolvido, repassado ao sub-grafo pra que o
                ``persist_node`` indexe os diagnosticos do ``deep_diagnose`` no
                mesmo namespace que o ``analyze_image`` usa.
        """
        from app.domains.chat.tools.analyze_image import build_analyze_image_tool
        from app.domains.chat.tools.ask_user import build_ask_user_tool
        from app.domains.chat.tools.compare_diagnoses import (
            build_compare_diagnoses_tool,
        )
        from app.domains.chat.tools.deep_diagnose import build_deep_diagnose_tool
        from app.domains.chat.tools.get_action_plan import (
            build_get_action_plan_tool,
        )
        from app.domains.chat.tools.get_disease_info import (
            build_get_disease_info_tool,
        )
        from app.domains.chat.tools.identify_crop import build_identify_crop_tool
        from app.domains.chat.tools.inspect_image import build_inspect_image_tool
        from app.domains.chat.tools.search_my_diagnoses import (
            build_search_my_diagnoses_tool,
        )
        from app.domains.chat.tools.search_scientific import (
            build_search_scientific_tool,
        )
        from app.domains.chat.tools.search_web import build_search_web_tool

        factories: dict[str, Callable[[], BaseTool]] = {
            "inspect_image": build_inspect_image_tool,
            "analyze_image": lambda: build_analyze_image_tool(
                self._inference_svc,
                self._diagnosis_svc,
                store_factory=self._store_factory,
                upload_svc=self._upload_svc,
            ),
            "get_disease_info": lambda: build_get_disease_info_tool(
                self._db_session_factory
            ),
            "get_action_plan": lambda: build_get_action_plan_tool(
                self._action_plan_svc
            ),
            "search_my_diagnoses": lambda: build_search_my_diagnoses_tool(
                self._store_factory
            ),
            "ask_user": build_ask_user_tool,
            "search_web": build_search_web_tool,
            "search_scientific": build_search_scientific_tool,
            "identify_crop": build_identify_crop_tool,
        }

        # deep_diagnose e compare_diagnoses dependem do sub-grafo; sem a factory
        # injetada (testes legados, DI parcial) ficam de fora em vez de quebrar.
        if self._diagnosis_graph_factory is not None:
            graph_factory = self._diagnosis_graph_factory

            def _subgraph_for(crop_id: str) -> Any:
                """Fecha o ``store`` no factory — as tools chamam com 1 arg."""
                return graph_factory(crop_id, store)

            factories["deep_diagnose"] = lambda: build_deep_diagnose_tool(
                _subgraph_for
            )
            factories["compare_diagnoses"] = lambda: build_compare_diagnoses_tool(
                _subgraph_for
            )

        return factories

    async def _get_graph(self, plan_features: PlanFeatures | None = None) -> Any:
        """Resolve o grafo compilado lazy-init — com checkpointer + plan_features.

        O checkpointer eh compartilhado entre chamadas (chat / resume /
        interrupts list) pra garantir que snapshots persistidos sobrevivem
        a chamadas distintas — requisito do ciclo HITL.

        As tools vem do ``tool_registry``, filtradas pelas features do plano:
        Free nao ve ``search_web``/``compare_diagnoses``, Enterprise ve tudo.

        Cacheia por ``plan_features.signature()`` — hash de TODAS as flags, nao
        so' do tier, porque o tool set muda com elas.
        """
        features = plan_features or FREE_FEATURES
        cache_key = features.signature()
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        checkpointer = None
        if self._checkpointer_factory is not None:
            try:
                checkpointer = await self._checkpointer_factory()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Checkpointer factory failed — graph runs sem persistencia"
                )

        # Store resolvido aqui (uma vez por feature-set) pra ser fechado tanto
        # no analyze_image quanto no sub-grafo do deep_diagnose.
        store: BaseStore | None = None
        if self._store_factory is not None:
            try:
                store = await self._store_factory()
            except Exception:  # noqa: BLE001 — memoria semantica e' best-effort
                logger.exception("Store factory failed — grafo roda sem memoria")

        tools = build_tools(
            self._build_tool_factories(store), features.model_dump()
        )
        logger.info(
            "Grafo do chat montado pra tier=%s com tools: %s",
            features.tier_name,
            ", ".join(t.name for t in tools),
        )
        graph = build_graph(
            tools=tools,
            state_schema=agent_state.ChatState,
            plan_features=features,
            checkpointer=checkpointer,
        )
        self._graph_cache[cache_key] = graph
        return graph

    async def chat(
        self,
        user_id: str,
        session_id: str | None,
        message_text: str,
        image_bytes: bytes | None,
        image_mime: str | None,
        image_filename: str | None,
        model_id: str,
    ) -> ChatResponse:
        """Roda o agente, persiste o turno e devolve a resposta consolidada.

        Pipeline (TCC-079):
          1) get_or_create session
          2) persist user message
          3) se há imagem, ela é disponibilizada ao agente (base64 em
             ``uploaded_files``) — o agente decide via tools (inspect_image →
             analyze_image) se diagnostica; NÃO forçamos mais inferência.
          4) invoke graph com mensagens + estado
          5) persist assistant message (com diagnosis_id se o agente diagnosticou)
        """
        session = await self._session_repo.get_or_create_for_user(user_id, session_id)

        await self._message_repo.create(
            session_id=session.id,
            role="user",
            content=message_text,
            metadata={"image_filename": image_filename} if image_filename else None,
        )

        uploaded_files = self._build_uploaded_files(image_bytes, image_mime, image_filename)
        seed_messages = [
            HumanMessage(content=self._seed_text(message_text, bool(uploaded_files)))
        ]

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
                "current_session_id": session.id,
                "selected_model": model_id,
                "plan_features": plan_features,
                "uploaded_files": uploaded_files,
                "diagnoses_in_turn": [],
                "recent_relevant_diagnoses": recent_relevant,
            },
            config=config,
        )

        # Detecta interrupt — quando o grafo pausa via ask_user, o "result"
        # contem o payload do interrupt em vez do estado final.
        interrupt_info = self._extract_interrupt_from_result(result)
        if interrupt_info is not None:
            return ChatResponse(
                role="assistant",
                content="",
                session_id=session.id,
                interrupt=interrupt_info,
            )

        diagnosis = await self._diagnosis_from_state(result, user_id)
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
        image_bytes: bytes | None,
        image_mime: str | None,
        image_filename: str | None,
        model_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Variante streaming — yields dicts {event, data} pro SSE endpoint.

        Eventos:
          - token:        chunk de texto do LLM
          - tool_call:    nome da tool sendo invocada
          - tool_result:  output da tool (string)
          - diagnosis:    DiagnosisResponse serializado (JSON-string) — emitido
                          ao final, se o agente diagnosticou via analyze_image
          - done:         marcador final
        """
        session = await self._session_repo.get_or_create_for_user(user_id, session_id)
        await self._message_repo.create(
            session_id=session.id,
            role="user",
            content=message_text,
            metadata={"image_filename": image_filename} if image_filename else None,
        )

        uploaded_files = self._build_uploaded_files(image_bytes, image_mime, image_filename)
        seed_messages = [
            HumanMessage(content=self._seed_text(message_text, bool(uploaded_files)))
        ]

        # O streaming e' o caminho que a UI realmente usa — sem este pre-fetch
        # a memoria semantica so' existia no endpoint sincrono.
        recent_relevant = await self._prefetch_relevant_diagnoses(
            user_id, message_text
        )

        plan_features = await self._resolve_plan_features(user_id)
        graph = await self._get_graph(plan_features=plan_features)
        config = {"configurable": {"thread_id": session.id}}
        initial_state = {
            "messages": seed_messages,
            "current_user_id": user_id,
            "current_session_id": session.id,
            "selected_model": model_id,
            "plan_features": plan_features,
            "uploaded_files": uploaded_files,
            "diagnoses_in_turn": [],
            "recent_relevant_diagnoses": recent_relevant,
        }

        collected_chunks: list[str] = []
        try:
            async for event in graph.astream_events(
                initial_state, version="v2", config=config
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    # Só transmite tokens do nó `llm` (agente principal). LLMs
                    # aninhados — o de visão no inspect_image, o summary no
                    # maybe_summarize — também emitem on_chat_model_stream e
                    # vazariam pro balão (ex.: o JSON do inspect_image).
                    if event.get("metadata", {}).get("langgraph_node") != "llm":
                        continue
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

            # Se o agente diagnosticou via analyze_image, o id ficou em
            # ``diagnoses_in_turn`` no snapshot — carrega e emite o evento.
            diagnosis = await self._diagnosis_from_snapshot(graph, config, user_id)
            if diagnosis is not None:
                yield {"event": "diagnosis", "data": diagnosis.model_dump_json()}

            await self._message_repo.create(
                session_id=session.id,
                role="assistant",
                content=assistant_text,
                diagnosis_id=diagnosis.id if diagnosis else None,
            )
        except Exception:
            # Sem isto, uma exceção no meio do stream mata o gerador SSE sem
            # emitir `done` — o cliente fica com o balão travado e o erro some
            # dos logs. Logamos a causa e emitimos um evento `error` terminal.
            logger.exception("chat_stream falhou (session=%s)", session.id)
            yield {
                "event": "error",
                "data": "Não foi possível gerar a resposta agora. Tente novamente.",
            }

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
        try:
            async for event in graph.astream_events(
                Command(resume=response), version="v2", config=config
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    # Só transmite tokens do nó `llm` (agente principal). LLMs
                    # aninhados — o de visão no inspect_image, o summary no
                    # maybe_summarize — também emitem on_chat_model_stream e
                    # vazariam pro balão (ex.: o JSON do inspect_image).
                    if event.get("metadata", {}).get("langgraph_node") != "llm":
                        continue
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
        except Exception:
            logger.exception("resume_stream falhou (thread=%s)", thread_id)
            yield {
                "event": "error",
                "data": "Não foi possível retomar a conversa agora. Tente novamente.",
            }

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
        graph: Any, config: dict[str, Any]
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

    # ── TCC-079: imagem efêmera (base64) + diagnosis do turno ──────────────────

    @staticmethod
    def _build_uploaded_files(
        image_bytes: bytes | None,
        image_mime: str | None,
        image_filename: str | None,
    ) -> list[UploadedFileDTO]:
        """Monta o ``UploadedFileDTO`` efêmero do turno (base64 in-memory).

        O b64 fica SÓ em ``uploaded_files`` (estado transitório), nunca em
        ``messages`` — assim não vaza pro checkpointer nem pro summarizer.
        """
        if not image_bytes:
            return []
        return [
            UploadedFileDTO(
                id=uuid4().hex,
                original_name=image_filename or "imagem.jpg",
                mime=image_mime or "image/jpeg",
                storage_key="",
                size_bytes=len(image_bytes),
                b64=base64.b64encode(image_bytes).decode("ascii"),
            )
        ]

    @staticmethod
    def _seed_text(message_text: str, has_image: bool) -> str:
        """Texto da seed message; sinaliza a imagem ao LLM sem expor os bytes."""
        if not has_image:
            return message_text
        note = "[O usuário anexou uma imagem para análise.]"
        base = (message_text or "").strip()
        return f"{base}\n\n{note}" if base else note

    async def _diagnosis_from_state(
        self, result: Any, user_id: str
    ) -> DiagnosisResponse | None:
        """Carrega o Diagnosis criado no turno (analyze_image) a partir do result."""
        ids = result.get("diagnoses_in_turn") if isinstance(result, dict) else None
        return await self._fetch_turn_diagnosis(ids, user_id)

    async def _diagnosis_from_snapshot(
        self, graph: Any, config: dict[str, Any], user_id: str
    ) -> DiagnosisResponse | None:
        """Versão pro streaming: lê ``diagnoses_in_turn`` do snapshot do grafo."""
        try:
            snapshot = await graph.aget_state(config)
        except Exception:  # noqa: BLE001
            return None
        ids = snapshot.values.get("diagnoses_in_turn") if snapshot else None
        return await self._fetch_turn_diagnosis(ids, user_id)

    async def _fetch_turn_diagnosis(
        self, ids: list[str] | None, user_id: str
    ) -> DiagnosisResponse | None:
        if not ids:
            return None
        try:
            return await self._diagnosis_svc.get_by_id(ids[-1], user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Falha ao carregar diagnosis %s do turno", ids[-1])
            return None

    @staticmethod
    def _extract_final_text(messages: list[Any]) -> str:
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
    ) -> list[dict[str, Any]]:
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

    # ── Historico de conversas ────────────────────────────────────────────────

    async def list_sessions(
        self, user_id: str, limit: int = 50
    ) -> list[ChatSessionSummary]:
        """Conversas do usuario, mais recentes primeiro.

        Sem isto o chat recomecava do zero a cada reload: o backend persistia
        ``chat_sessions``/``chat_messages`` desde sempre, mas nao havia endpoint
        de leitura e o frontend nao tinha como voltar numa conversa.
        """
        rows = await self._session_repo.list_with_preview(user_id, limit=limit)
        return [
            ChatSessionSummary(
                id=sess.id,
                title=sess.title,
                preview=preview,
                message_count=count,
                summary_text=sess.summary_text,
                created_at=sess.created_at,
                updated_at=sess.updated_at,
            )
            for sess, count, preview in rows
        ]

    async def get_session_messages(
        self, user_id: str, session_id: str
    ) -> list[ChatMessageResponse]:
        """Mensagens de uma conversa, em ordem cronologica.

        Retorna lista vazia quando a sessao nao existe ou nao e' do usuario —
        o router traduz isso em 404 pra nao vazar existencia de sessao alheia.
        """
        session = await self._session_repo.get_by_id(session_id, user_id=user_id)
        if session is None:
            return []
        messages = await self._message_repo.list_by_session(session_id)
        return [
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                diagnosis_id=m.diagnosis_id,
                created_at=m.created_at,
            )
            for m in messages
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
        self, messages: list[Any], llm: Any = None
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
