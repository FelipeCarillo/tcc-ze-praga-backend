"""Roteador do Zé Praga — orquestra agente LangGraph + persistência.

Substituiu (TCC-010) o keyword-matching antigo por chamada real ao agente via
ChatService. TCC-011 adicionou endpoint streaming /chat/stream (SSE) que reusa
ChatService.chat_stream() — mantém /chat síncrono pra back-compat.

Sprint A2.5 (TCC-048) adicionou POST /sessions/{id}/close que gera o resumo
final da sessao e o indexa no Store.

Sprint A4.5 (TCC-058) adicionou:
- POST /chat/resume + /chat/resume/stream pra retomar interrupts via
  ``Command(resume=...)``.
- GET /chat/interrupts pra listar threads pendentes.

Helper _extract_last_message preserva fix do TCC-005 (parsing JSON robusto).
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.core.dependencies import (
    get_chat_service,
    get_current_user,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.chat.schemas import (
    ChatResponse,
    CloseSessionResponse,
    PendingInterrupt,
    ResumeRequest,
)
from app.domains.chat.service import ChatService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, ModelEnum

router = APIRouter(prefix="/chat", tags=["Chat"])
sessions_router = APIRouter(prefix="/sessions", tags=["Chat"])


def _extract_last_message(messages: str) -> str:
    """Parseia messages como JSON array de turnos; fallback retorna o input cru.

    Regressão TCC-005: o try/except aninhado quebrava com TypeError quando
    `messages` não era JSON e parsed era None.
    """
    try:
        parsed = json.loads(messages)
        return parsed[-1].get("content", "") if parsed else ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return messages


@router.post("", response_model=ChatResponse, status_code=200)
async def send_message(
    messages: str = Form(...),
    model: str = Form(default=ModelEnum.ENSEMBLE),
    image: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.CHAT)),
    chat_svc: ChatService = Depends(get_chat_service),
    usage_svc: UsageService = Depends(get_usage_service),
) -> ChatResponse:
    await usage_svc.record_usage(
        current_user.id,
        FeatureTypeEnum.CHAT,
        {"model": model, "has_image": image is not None},
    )

    message_text = _extract_last_message(messages)
    image_filename = image.filename if image else None

    return await chat_svc.chat(
        user_id=current_user.id,
        session_id=session_id,
        message_text=message_text,
        image_filename=image_filename,
        model_id=model,
    )


@router.post("/stream", status_code=200)
async def send_message_stream(
    messages: str = Form(...),
    model: str = Form(default=ModelEnum.ENSEMBLE),
    image: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.CHAT)),
    chat_svc: ChatService = Depends(get_chat_service),
    usage_svc: UsageService = Depends(get_usage_service),
) -> EventSourceResponse:
    """SSE streaming endpoint — yields token/tool_call/tool_result/diagnosis/done."""
    await usage_svc.record_usage(
        current_user.id,
        FeatureTypeEnum.CHAT,
        {"model": model, "has_image": image is not None, "streaming": True},
    )

    message_text = _extract_last_message(messages)
    image_filename = image.filename if image else None

    async def _event_generator() -> AsyncIterator[dict[str, str]]:
        async for event in chat_svc.chat_stream(
            user_id=current_user.id,
            session_id=session_id,
            message_text=message_text,
            image_filename=image_filename,
            model_id=model,
        ):
            # sse-starlette espera dict {event, data}; data é serializado pra string.
            data = event.get("data", "")
            if not isinstance(data, str):
                data = json.dumps(data, ensure_ascii=False, default=str)
            yield {"event": event["event"], "data": data}

    return EventSourceResponse(_event_generator())


@router.post("/resume", response_model=ChatResponse, status_code=200)
async def resume_chat(
    body: ResumeRequest,
    current_user: UserDTO = Depends(get_current_user),
    chat_svc: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Retoma uma sessao interrompida (HITL) via ``Command(resume=...)``.

    Espera ``{thread_id, response}`` — onde ``thread_id`` eh o id da sessao
    (= chat_session.id) e ``response`` eh a resposta do usuario ao
    interrupt previamente disparado pela tool ``ask_user``.
    """
    return await chat_svc.resume(
        user_id=current_user.id,
        thread_id=body.thread_id,
        response=body.response,
    )


@router.post("/resume/stream", status_code=200)
async def resume_chat_stream(
    body: ResumeRequest,
    current_user: UserDTO = Depends(get_current_user),
    chat_svc: ChatService = Depends(get_chat_service),
) -> EventSourceResponse:
    """SSE streaming do resume — yields token/tool_call/tool_result/interrupt/done."""

    async def _event_generator() -> AsyncIterator[dict[str, str]]:
        async for event in chat_svc.resume_stream(
            user_id=current_user.id,
            thread_id=body.thread_id,
            response=body.response,
        ):
            data = event.get("data", "")
            if not isinstance(data, str):
                data = json.dumps(data, ensure_ascii=False, default=str)
            yield {"event": event["event"], "data": data}

    return EventSourceResponse(_event_generator())


@router.get(
    "/interrupts",
    response_model=list[PendingInterrupt],
    status_code=200,
)
async def list_interrupts(
    current_user: UserDTO = Depends(get_current_user),
    chat_svc: ChatService = Depends(get_chat_service),
) -> list[PendingInterrupt]:
    """Lista sessoes do usuario com interrupt pendente aguardando resposta."""
    return await chat_svc.list_pending_interrupts(current_user.id)


@sessions_router.post(
    "/{session_id}/close", response_model=CloseSessionResponse, status_code=200
)
async def close_session(
    session_id: str,
    current_user: UserDTO = Depends(get_current_user),
    chat_svc: ChatService = Depends(get_chat_service),
) -> CloseSessionResponse:
    """Encerra a sessao gerando + persistindo um resumo da conversa.

    O resumo eh salvo em ``chat_sessions.summary_text`` e tambem indexado
    no Store sob ``("user", uid, "session_summaries")`` pra busca futura.
    """
    return await chat_svc.close_session(current_user.id, session_id)
