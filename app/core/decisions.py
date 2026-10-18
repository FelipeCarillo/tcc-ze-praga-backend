"""Cliente fino do Decisions API da OpenAI (TCC-103).

O Decisions (``POST /v1/decisions``, em beta desde 06/10/2026) responde
perguntas fechadas — predicate (probabilidade 0–1), choice (uma opção de uma
lista) ou score — sem gerar texto: não cobra tokens de saída e responde em
fração do tempo de um chat. Só aceita o ``gpt-6-luna``.

No teste de 17/10/2026 (7 imagens × 3 repetições) o gate de imagem pelo
Decisions acertou 100%, como o LLM de visão, com p50 de 0,38 s contra 1,67 s
do ``gpt-6-luna`` por chat. Por ser beta e de modelo único, quem usa este
cliente deve sempre ter um caminho de reserva — ``DecisionsError`` cobre
qualquer falha (rede, timeout, 4xx/5xx, resposta fora do formato).

Formato descoberto na prática (a referência oficial dava 404):
- ``choices``: objetos ``{"value", "description"?}`` — a descrição melhora
  muito o acerto em opções parecidas;
- ``levels`` (score): objetos ``{"label"}``;
- imagem: ``input=[{"role": "user", "content": [{"type": "input_image",
  "image_url": "data:..."}]}]`` (só data URL inline).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.config import settings


class DecisionsError(RuntimeError):
    """Falha ao consultar o Decisions API — o chamador cai no caminho de reserva."""


def _base_url() -> str:
    # Mesmo override do SDK da OpenAI (o local_dev aponta para um fake).
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


async def decide(
    input_: str | list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    timeout: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Faz as perguntas e devolve as respostas indexadas pelo ``name``.

    Raises:
        DecisionsError: sem chave, erro HTTP/rede/timeout ou resposta inválida.
    """
    if not settings.openai_api_key:
        raise DecisionsError("OPENAI_API_KEY ausente")
    body = {"model": settings.decisions_model, "input": input_, "questions": questions}
    try:
        async with httpx.AsyncClient(
            timeout=timeout or settings.decisions_timeout_seconds
        ) as client:
            resp = await client.post(
                f"{_base_url()}/decisions",
                json=body,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
        resp.raise_for_status()
        answers = resp.json()["answers"]
        return {a["name"]: a for a in answers}
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise DecisionsError(f"{type(exc).__name__}: {exc}") from exc


def image_input(text: str, data_url: str) -> list[dict[str, Any]]:
    """Entrada multimodal (texto + uma imagem inline) no formato do Decisions."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": data_url},
            ],
        }
    ]
