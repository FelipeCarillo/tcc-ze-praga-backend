"""Roteador do Zé Praga — orquestra agente LangGraph + persistência.

Substituiu (TCC-010) o keyword-matching antigo por chamada real ao agente via
ChatService. Mantém o helper _extract_last_message do TCC-005 (regressão do
parsing JSON robusto).
"""

import json

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependencies import (
    get_chat_service,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.chat.schemas import ChatResponse
from app.domains.chat.service import ChatService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, ModelEnum

router = APIRouter(prefix="/chat", tags=["Chat"])


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
