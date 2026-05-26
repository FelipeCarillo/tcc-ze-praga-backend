"""Smoke tests for Talhoes, Uploads, and Health endpoints (TCC-073).

Pattern: each authenticated domain gets its own fixture-client that applies
dependency overrides before instantiating AsyncClient, then clears them in
teardown. smoke_client (from conftest) is used for unauthenticated (401) tests.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_current_user,
    get_talhao_service,
    get_upload_service,
)
from app.domains.talhoes.schemas import TalhaoResponse
from app.domains.uploads.router import MAX_FILE_SIZE_BYTES, MAX_FILES_PER_REQUEST
from app.domains.uploads.schemas import UploadResponse
from app.main import app
from tests.conftest import NOW, make_user_dto
from tests.smoke.conftest import bypass_auth_overrides

API = "/api/v1"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_talhao_response(**kwargs) -> TalhaoResponse:
    defaults = dict(
        id="talhao-uuid-1",
        nome="Talhao Norte",
        apelido="Norte",
        hectares=42.5,
        cultura="soja",
        data_semeadura=None,
        created_at=NOW,
    )
    return TalhaoResponse(**{**defaults, **kwargs})


def _make_upload_row(
    *,
    id_: str = "upload-uuid-1",
    original_name: str = "folha.jpg",
    storage_key: str = "users/user-uuid-1/abc-folha.jpg",
    mime: str = "image/jpeg",
    size_bytes: int = 1234,
    hash_sha256: str = "0" * 64,
    uploaded_at: datetime | None = None,
):
    """Builds a minimal object that UploadResponse.model_validate can consume."""
    from app.models.uploaded_file import UploadedFile

    row = UploadedFile()
    row.id = id_
    row.user_id = "user-uuid-1"
    row.session_id = None
    row.original_name = original_name
    row.mime = mime
    row.storage_key = storage_key
    row.size_bytes = size_bytes
    row.hash_sha256 = hash_sha256
    row.uploaded_at = uploaded_at or NOW
    return row


# ── Talhoes fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
async def client_talhoes():
    """Authenticated client with TalhaoService mocked."""
    mock_svc = AsyncMock()
    bypass_auth_overrides()
    app.dependency_overrides[get_talhao_service] = lambda: mock_svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        ac._mock_svc = mock_svc  # type: ignore[attr-defined]
        yield ac
    app.dependency_overrides.clear()


# ── Talhoes tests ─────────────────────────────────────────────────────────────


async def test_talhoes_list_returns_200(client_talhoes):
    """GET /talhoes -> 200 with list (may be empty)."""
    client_talhoes._mock_svc.list_for_user = AsyncMock(return_value=[])

    r = await client_talhoes.get(f"{API}/talhoes")

    assert r.status_code == 200
    assert isinstance(r.json(), list)


async def test_talhoes_list_returns_items(client_talhoes):
    """GET /talhoes -> 200 with the mocked talhao in list."""
    talhao = _make_talhao_response()
    client_talhoes._mock_svc.list_for_user = AsyncMock(return_value=[talhao])

    r = await client_talhoes.get(f"{API}/talhoes")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["id"] == "talhao-uuid-1"
    assert body[0]["nome"] == "Talhao Norte"


async def test_talhoes_create_returns_201(client_talhoes):
    """POST /talhoes -> 201 with TalhaoResponse."""
    talhao = _make_talhao_response()
    client_talhoes._mock_svc.create = AsyncMock(return_value=talhao)

    payload = {"nome": "Talhao Norte", "cultura": "soja"}
    r = await client_talhoes.post(f"{API}/talhoes", json=payload)

    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "talhao-uuid-1"
    assert body["nome"] == "Talhao Norte"
    assert body["cultura"] == "soja"


async def test_talhoes_delete_returns_204(client_talhoes):
    """DELETE /talhoes/{id} -> 204 no content."""
    client_talhoes._mock_svc.delete = AsyncMock(return_value=None)

    r = await client_talhoes.delete(f"{API}/talhoes/talhao-uuid-1")

    assert r.status_code == 204


async def test_talhoes_requires_auth(smoke_client):
    """GET /talhoes without auth -> 401."""
    r = await smoke_client.get(f"{API}/talhoes")
    assert r.status_code == 401


# ── Uploads fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
async def client_uploads():
    """Authenticated client with UploadService mocked."""
    mock_svc = AsyncMock()
    bypass_auth_overrides()
    app.dependency_overrides[get_upload_service] = lambda: mock_svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        ac._mock_svc = mock_svc  # type: ignore[attr-defined]
        yield ac
    app.dependency_overrides.clear()


# ── Uploads tests ─────────────────────────────────────────────────────────────


async def test_uploads_single_file_returns_201(client_uploads):
    """POST /uploads with 1 valid file -> 201 with UploadResponse."""
    row = _make_upload_row()
    client_uploads._mock_svc.upload = AsyncMock(return_value=(row, False))

    files = {"files": ("folha.jpg", io.BytesIO(b"fake-image-data"), "image/jpeg")}
    r = await client_uploads.post(f"{API}/uploads", files=files)

    assert r.status_code == 201
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["id"] == "upload-uuid-1"
    assert body[0]["original_name"] == "folha.jpg"


async def test_uploads_exceeds_max_files_returns_422(client_uploads):
    """POST /uploads with >5 files -> 422."""
    files = [
        ("files", (f"img-{i}.jpg", io.BytesIO(b"x"), "image/jpeg"))
        for i in range(MAX_FILES_PER_REQUEST + 1)
    ]
    r = await client_uploads.post(f"{API}/uploads", files=files)

    assert r.status_code == 422


async def test_uploads_file_exceeds_max_size_returns_413(client_uploads):
    """POST /uploads with file >10MB -> 413."""
    big_data = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    files = {"files": ("big.jpg", io.BytesIO(big_data), "image/jpeg")}
    r = await client_uploads.post(f"{API}/uploads", files=files)

    assert r.status_code == 413


# ── Health test ───────────────────────────────────────────────────────────────


async def test_health_returns_200_healthy(smoke_client):
    """GET /health -> 200 with status healthy (no mock needed)."""
    r = await smoke_client.get(f"{API}/health")

    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
