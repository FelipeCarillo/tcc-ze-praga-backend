"""Envio de e-mail transacional via Resend (TCC-090).

O backend não depende do SDK do Resend — a API é um único POST JSON, então
usamos o ``httpx`` que já está no projeto. Menos uma dependência pra travar no
build do Space.

Sem ``RESEND_API_KEY`` o factory devolve o ``NullEmailSender``. Ele registra a
causa e levanta uma exceção controlada: seguir como se o e-mail tivesse saído
seria enganoso para a pessoa que está tentando recuperar o acesso.
"""

import logging
from typing import Protocol

import httpx

from app.config import settings
from app.core.exceptions import EmailDeliveryError

logger = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10.0


class EmailSender(Protocol):
    """Contrato de envio — permite injetar um fake nos testes."""

    async def send(self, *, to: str, subject: str, html: str) -> None: ...


class NullEmailSender:
    """Sender inerte: registra a causa e sinaliza indisponibilidade.

    Usado quando ``RESEND_API_KEY`` não está configurada. Não revela o link no
    log: um token de recuperação não deve aparecer em saída operacional.
    """

    async def send(self, *, to: str, subject: str, html: str) -> None:
        logger.warning(
            "[NullEmailSender] e-mail NÃO enviado (sem RESEND_API_KEY) — "
            "to=%s subject=%s",
            to,
            subject,
        )
        raise EmailDeliveryError()


class ResendEmailSender:
    """Envio real via API do Resend."""

    def __init__(self, api_key: str, sender: str) -> None:
        self._api_key = api_key
        self._sender = sender

    async def send(self, *, to: str, subject: str, html: str) -> None:
        payload = {"from": self._sender, "to": [to], "subject": subject, "html": html}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(RESEND_ENDPOINT, json=payload, headers=headers)
        except httpx.HTTPError:
            logger.exception("Não foi possível alcançar o Resend")
            raise EmailDeliveryError() from None
        if response.status_code >= 400:
            logger.error(
                "Resend recusou o envio — status=%s body=%s", response.status_code, response.text
            )
            raise EmailDeliveryError()
        logger.info("E-mail enviado via Resend — to=%s subject=%s", to, subject)


def get_email_sender() -> EmailSender:
    """Factory com fallback: sem chave configurada, devolve o sender inerte."""
    if not settings.resend_api_key:
        return NullEmailSender()
    return ResendEmailSender(settings.resend_api_key, settings.email_from)
