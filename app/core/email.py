"""Envio de e-mail transacional via Resend (TCC-090).

O backend não depende do SDK do Resend — a API é um único POST JSON, então
usamos o ``httpx`` que já está no projeto. Menos uma dependência pra travar no
build do Space.

Degradação graciosa, mesmo padrão do ``InferenceService``: sem
``RESEND_API_KEY`` o factory devolve o ``NullEmailSender``, que loga em vez de
enviar. Dev e a suíte de testes rodam sem credencial nenhuma.
"""

import logging
import re
from typing import Protocol

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10.0


class EmailSender(Protocol):
    """Contrato de envio — permite injetar um fake nos testes."""

    async def send(self, *, to: str, subject: str, html: str) -> None: ...


class NullEmailSender:
    """Sender inerte: registra o envio no log e não chama rede.

    Usado quando ``RESEND_API_KEY`` não está configurada. Imprime no log o
    primeiro link encontrado no corpo — sem isso não haveria como confirmar uma
    conta em dev, já que o token cru só existe dentro do e-mail.
    """

    async def send(self, *, to: str, subject: str, html: str) -> None:
        match = re.search(r'href="(https?://[^"]+)"', html)
        link = match.group(1) if match else "(nenhum link no corpo)"
        logger.warning(
            "[NullEmailSender] e-mail NÃO enviado (sem RESEND_API_KEY) — "
            "to=%s subject=%s\n    link: %s",
            to,
            subject,
            link,
        )


class ResendEmailSender:
    """Envio real via API do Resend."""

    def __init__(self, api_key: str, sender: str) -> None:
        self._api_key = api_key
        self._sender = sender

    async def send(self, *, to: str, subject: str, html: str) -> None:
        payload = {"from": self._sender, "to": [to], "subject": subject, "html": html}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(RESEND_ENDPOINT, json=payload, headers=headers)
        if response.status_code >= 400:
            # Não estoura pro usuário: o cadastro já foi gravado e ele pode
            # pedir reenvio. Falha de e-mail não deve derrubar o registro.
            logger.error(
                "Resend recusou o envio — status=%s body=%s", response.status_code, response.text
            )
            return
        logger.info("E-mail enviado via Resend — to=%s subject=%s", to, subject)


def get_email_sender() -> EmailSender:
    """Factory com fallback: sem chave configurada, devolve o sender inerte."""
    if not settings.resend_api_key:
        return NullEmailSender()
    return ResendEmailSender(settings.resend_api_key, settings.email_from)
