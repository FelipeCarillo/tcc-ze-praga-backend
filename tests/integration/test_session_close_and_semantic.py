"""Integration tests do TCC-048 — close_session + semantic search endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_chat_service,
    get_current_user,
    get_store_dep,
)
from app.domains.chat.schemas import CloseSessionResponse
from app.main import app
from tests.conftest import make_user_dto


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def fake_user():
    return make_user_dto()


@pytest.fixture
def auth_headers(valid_token):
    return {"Authorization": f"Bearer {valid_token}"}


# ── POST /api/v1/sessions/{id}/close ──────────────────────────────────────────


async def test_close_session_returns_summary(client, fake_user, auth_headers):
    chat_svc = AsyncMock()
    chat_svc.close_session.return_value = CloseSessionResponse(
        session_id="sess-1",
        summary_text="Resumo: conversa sobre ferrugem na soja.",
    )
    app.dependency_overrides[get_chat_service] = lambda: chat_svc
    app.dependency_overrides[get_current_user] = lambda: fake_user

    resp = await client.post(
        "/api/v1/sessions/sess-1/close", headers=auth_headers
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess-1"
    assert "ferrugem" in body["summary_text"]
    chat_svc.close_session.assert_awaited_once_with(fake_user.id, "sess-1")


async def test_close_session_unauthenticated_returns_401(client):
    resp = await client.post("/api/v1/sessions/sess-1/close")
    assert resp.status_code == 401


# ── GET /api/v1/diagnoses/semantic ────────────────────────────────────────────


async def test_semantic_search_returns_hits(client, fake_user, auth_headers):
    fake_item = MagicMock()
    fake_item.value = {
        "summary_text": "Diagnostico de Ferrugem em 2026-05",
        "diagnosis_id": "diag-old",
        "disease_id": "ferrugem-asiatica",
        "disease_name": "Ferrugem Asiatica",
        "crop_id": "crop-soja",
        "confidence": 0.91,
        "severity": "alta",
        "created_at": "2026-05-10T12:00:00+00:00",
    }
    store = MagicMock()
    store.asearch = AsyncMock(return_value=[fake_item])

    app.dependency_overrides[get_store_dep] = lambda: store
    app.dependency_overrides[get_current_user] = lambda: fake_user

    resp = await client.get(
        "/api/v1/diagnoses/semantic?q=ferrugem&limit=5",
        headers=auth_headers,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["diagnosis_id"] == "diag-old"
    assert body[0]["disease_id"] == "ferrugem-asiatica"
    assert body[0]["confidence"] == 0.91

    store.asearch.assert_awaited_once()
    call_args = store.asearch.call_args
    assert call_args.args[0] == ("user", fake_user.id, "diagnoses")
    assert call_args.kwargs["query"] == "ferrugem"
    assert call_args.kwargs["limit"] == 5


async def test_semantic_search_requires_query_param(
    client, fake_user, auth_headers
):
    store = MagicMock()
    app.dependency_overrides[get_store_dep] = lambda: store
    app.dependency_overrides[get_current_user] = lambda: fake_user

    resp = await client.get(
        "/api/v1/diagnoses/semantic", headers=auth_headers
    )
    # FastAPI validation — falta de ``q`` retorna 422
    assert resp.status_code == 422


async def test_semantic_search_returns_empty_on_store_error(
    client, fake_user, auth_headers
):
    store = MagicMock()
    store.asearch = AsyncMock(side_effect=RuntimeError("offline"))
    app.dependency_overrides[get_store_dep] = lambda: store
    app.dependency_overrides[get_current_user] = lambda: fake_user

    resp = await client.get(
        "/api/v1/diagnoses/semantic?q=ferrugem", headers=auth_headers
    )
    assert resp.status_code == 200
    assert resp.json() == []


async def test_semantic_search_unauthenticated_returns_401(client):
    resp = await client.get("/api/v1/diagnoses/semantic?q=ferrugem")
    assert resp.status_code == 401
