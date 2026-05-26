"""Testes do TranscriptionService (TCC-081) — STT via OpenAI."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.domains.transcription.service import TranscriptionService


def _fake_client(text: str) -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text=text)
    )
    return client


async def test_transcribe_returns_text() -> None:
    client = _fake_client("  olá zé praga  ")
    svc = TranscriptionService(client=client)

    out = await svc.transcribe(data=b"audio-bytes", filename="voice.webm", mime="audio/webm")

    assert out == "olá zé praga"
    client.audio.transcriptions.create.assert_awaited_once()
    kwargs = client.audio.transcriptions.create.await_args.kwargs
    assert kwargs["file"] == ("voice.webm", b"audio-bytes", "audio/webm")


async def test_transcribe_empty_text() -> None:
    svc = TranscriptionService(client=_fake_client(""))
    assert await svc.transcribe(data=b"x", filename="v.webm", mime="audio/webm") == ""
