"""TCC-103 — cliente do Decisions API (sem rede: transporte HTTP mockado)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import settings
from app.core import decisions
from app.core.decisions import DecisionsError, decide, image_input


@pytest.fixture
def transport(monkeypatch):
    """Troca o AsyncClient por um com MockTransport; guarda os requests."""
    seen: list[httpx.Request] = []
    state: dict = {"handler": None}
    real_client = httpx.AsyncClient

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return state["handler"](request)

    def _client(**kwargs):
        return real_client(transport=httpx.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(decisions.httpx, "AsyncClient", _client)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    return seen, state


async def test_decide_indexa_respostas_por_nome(transport):
    seen, state = transport
    state["handler"] = lambda r: httpx.Response(
        200,
        json={
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": "ok", "probability": 0.9},
                {"type": "choice", "name": "cor", "choice": "verde", "confidence": 0.8},
            ],
        },
    )
    out = await decide("folha verde", [{"type": "predicate", "name": "ok", "instructions": "?"}])
    assert out["ok"]["probability"] == 0.9 and out["cor"]["choice"] == "verde"
    body = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/decisions")
    assert body["model"] == settings.decisions_model
    assert seen[0].headers["Authorization"] == "Bearer sk-test"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": {"message": "bad"}}),
        httpx.Response(200, json={"sem": "answers"}),
        httpx.Response(200, text="não é json"),
    ],
)
async def test_decide_falhas_viram_decisions_error(transport, response):
    _, state = transport
    state["handler"] = lambda r: response
    with pytest.raises(DecisionsError):
        await decide("x", [])


async def test_decide_erro_de_rede_vira_decisions_error(transport):
    _, state = transport

    def _boom(r):
        raise httpx.ConnectTimeout("timeout", request=r)

    state["handler"] = _boom
    with pytest.raises(DecisionsError):
        await decide("x", [])


async def test_decide_sem_chave_nem_tenta(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(DecisionsError):
        await decide("x", [])


def test_image_input_formato_message():
    out = image_input("texto", "data:image/jpeg;base64,AA==")
    assert out[0]["role"] == "user"
    assert out[0]["content"][0] == {"type": "input_text", "text": "texto"}
    assert out[0]["content"][1] == {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="}
