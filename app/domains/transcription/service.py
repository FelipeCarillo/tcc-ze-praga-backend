"""TranscriptionService — STT (voz → texto) via OpenAI (TCC-081).

Wrapper fino sobre ``openai.AsyncOpenAI`` (SDK já no projeto). Recebe os bytes
do áudio gravado no front (webm/opus, m4a, etc.) e devolve o texto transcrito,
que o ``/chat`` usa como ``message_text``. Reusa a ``OPENAI_API_KEY`` do
ambiente (mesma credencial do chat/embeddings).
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings


class TranscriptionService:
    """Transcreve áudio usando o modelo ``settings.transcription_model``."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        # Client lazy: só é construído quando ``transcribe`` é chamado de fato
        # (evita exigir OPENAI_API_KEY na mera resolução da dependency — ex: em
        # testes ou turnos sem áudio). Em testes injeta-se um AsyncMock aqui.
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key or None)
        return self._client

    async def transcribe(self, *, data: bytes, filename: str, mime: str) -> str:
        """Transcreve ``data`` (bytes do áudio) e retorna o texto.

        Args:
            data: bytes do arquivo de áudio.
            filename: nome com extensão coerente (ex: ``voice.webm``) — o SDK usa
                pra inferir o formato.
            mime: content-type do upload (ex: ``audio/webm``).

        Returns:
            Texto transcrito (pode ser string vazia se o áudio for silêncio).
        """
        result = await self._get_client().audio.transcriptions.create(
            model=settings.transcription_model,
            file=(filename, data, mime),
        )
        return (getattr(result, "text", "") or "").strip()
