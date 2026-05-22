"""Integration tests for POST /api/v1/uploads (TCC-037).

Mockamos o ``UploadService`` inteiro pra nao depender do Supabase real nem
do banco. Cobertura:
    - upload de 1 arquivo -> 201 com UploadResponse
    - upload em batch (3 arquivos) -> 201 com lista de 3
    - dedup (mesmo hash retorna deduplicated=True)
    - max 5 files -> 422
    - file > 10MB -> 413
    - 0 files -> 422
"""

from __future__ import annotations

import io
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_upload_service
from app.domains.uploads.router import MAX_FILE_SIZE_BYTES, MAX_FILES_PER_REQUEST
from app.main import app
from app.models.uploaded_file import UploadedFile
from tests.conftest import NOW, make_user_dto


def _make_row(
    *,
    id_: str = "upload-uuid-1",
    original_name: str = "folha.jpg",
    storage_key: str = "users/user-uuid-1/abc-folha.jpg",
    mime: str = "image/jpeg",
    size_bytes: int = 1234,
    hash_sha256: str = "0" * 64,
    uploaded_at: datetime | None = None,
) -> UploadedFile:
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


@pytest.fixture
def mock_upload_svc():
    svc = AsyncMock()
    return svc


@pytest.fixture
async def client_uploads(mock_upload_svc):
    app.dependency_overrides[get_upload_service] = lambda: mock_upload_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_upload_single_file_returns_201(client_uploads, mock_upload_svc):
    """POST com 1 arquivo retorna 201 + UploadResponse."""
    row = _make_row()
    mock_upload_svc.upload = AsyncMock(return_value=(row, False))

    files = {"files": ("folha.jpg", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"fake-image"), "image/jpeg")}

    r = await client_uploads.post("/api/v1/uploads", files=files)

    assert r.status_code == 201
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["id"] == "upload-uuid-1"
    assert body[0]["original_name"] == "folha.jpg"
    assert body[0]["mime"] == "image/jpeg"
    assert body[0]["storage_key"] == "users/user-uuid-1/abc-folha.jpg"
    assert body[0]["deduplicated"] is False

    mock_upload_svc.upload.assert_awaited_once()
    call_kwargs = mock_upload_svc.upload.await_args.kwargs
    assert call_kwargs["user_id"] == "user-uuid-1"
    assert call_kwargs["original_name"] == "folha.jpg"
    assert call_kwargs["mime"] == "image/jpeg"


async def test_upload_batch_three_files_returns_list(client_uploads, mock_upload_svc):
    """POST com 3 arquivos retorna lista com 3 UploadResponse."""
    rows = [
        _make_row(id_=f"upload-{i}", original_name=f"img-{i}.jpg", hash_sha256=str(i) * 64)
        for i in range(3)
    ]
    mock_upload_svc.upload = AsyncMock(side_effect=[(rows[0], False), (rows[1], False), (rows[2], False)])

    files = [
        ("files", ("img-0.jpg", io.BytesIO(b"contents-0"), "image/jpeg")),
        ("files", ("img-1.jpg", io.BytesIO(b"contents-1"), "image/jpeg")),
        ("files", ("img-2.jpg", io.BytesIO(b"contents-2"), "image/jpeg")),
    ]

    r = await client_uploads.post("/api/v1/uploads", files=files)

    assert r.status_code == 201
    body = r.json()
    assert len(body) == 3
    assert [b["id"] for b in body] == ["upload-0", "upload-1", "upload-2"]
    assert all(b["deduplicated"] is False for b in body)
    assert mock_upload_svc.upload.await_count == 3


async def test_upload_dedup_returns_deduplicated_true(client_uploads, mock_upload_svc):
    """Upload de arquivo com mesmo hash retorna deduplicated=True."""
    row = _make_row()
    # Service ja indica que era dedup
    mock_upload_svc.upload = AsyncMock(return_value=(row, True))

    files = {"files": ("folha.jpg", io.BytesIO(b"same-bytes"), "image/jpeg")}

    r = await client_uploads.post("/api/v1/uploads", files=files)

    assert r.status_code == 201
    body = r.json()
    assert len(body) == 1
    assert body[0]["deduplicated"] is True
    assert body[0]["id"] == "upload-uuid-1"


async def test_upload_dedup_two_calls_same_file(client_uploads, mock_upload_svc):
    """Dois POSTs do mesmo arquivo: primeiro deduplicated=False, segundo True."""
    row = _make_row()
    mock_upload_svc.upload = AsyncMock(side_effect=[(row, False), (row, True)])

    files = {"files": ("folha.jpg", io.BytesIO(b"identical"), "image/jpeg")}
    r1 = await client_uploads.post("/api/v1/uploads", files=files)
    files = {"files": ("folha.jpg", io.BytesIO(b"identical"), "image/jpeg")}
    r2 = await client_uploads.post("/api/v1/uploads", files=files)

    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()[0]["deduplicated"] is False
    assert r2.json()[0]["deduplicated"] is True
    # Mesmo storage_key (reaproveitado)
    assert r1.json()[0]["storage_key"] == r2.json()[0]["storage_key"]


async def test_upload_exceeds_max_files_returns_422(client_uploads, mock_upload_svc):
    """6 arquivos (limite e' 5) -> 422 e service NAO e' chamado."""
    files = [
        ("files", (f"img-{i}.jpg", io.BytesIO(b"x"), "image/jpeg"))
        for i in range(MAX_FILES_PER_REQUEST + 1)
    ]

    r = await client_uploads.post("/api/v1/uploads", files=files)

    assert r.status_code == 422
    assert "Maximo" in r.json()["detail"] or "5" in r.json()["detail"]
    mock_upload_svc.upload.assert_not_awaited()


async def test_upload_file_exceeds_max_size_returns_413(client_uploads, mock_upload_svc):
    """Arquivo > 10MB -> 413."""
    big_bytes = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    files = {"files": ("big.jpg", io.BytesIO(big_bytes), "image/jpeg")}

    r = await client_uploads.post("/api/v1/uploads", files=files)

    assert r.status_code == 413
    assert "excede" in r.json()["detail"].lower() or "10" in r.json()["detail"]
    mock_upload_svc.upload.assert_not_awaited()


async def test_upload_file_at_exactly_max_size_succeeds(client_uploads, mock_upload_svc):
    """Arquivo de exatamente 10MB e' aceito (< MAX, nao <=)."""
    row = _make_row(size_bytes=MAX_FILE_SIZE_BYTES)
    mock_upload_svc.upload = AsyncMock(return_value=(row, False))

    files = {"files": ("ok.jpg", io.BytesIO(b"x" * MAX_FILE_SIZE_BYTES), "image/jpeg")}

    r = await client_uploads.post("/api/v1/uploads", files=files)

    assert r.status_code == 201
    mock_upload_svc.upload.assert_awaited_once()


async def test_upload_without_files_returns_422(client_uploads, mock_upload_svc):
    """POST sem campo ``files`` -> 422 (FastAPI validation)."""
    r = await client_uploads.post("/api/v1/uploads", files={})

    # FastAPI retorna 422 quando o body obrigatorio nao vem
    assert r.status_code == 422
    mock_upload_svc.upload.assert_not_awaited()


async def test_upload_requires_authentication():
    """Sem token -> 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"files": ("folha.jpg", io.BytesIO(b"data"), "image/jpeg")}
        r = await ac.post("/api/v1/uploads", files=files)

    assert r.status_code == 401


async def test_upload_passes_session_id_to_service(client_uploads, mock_upload_svc):
    """Quando session_id e' fornecido, repassa pro service."""
    row = _make_row()
    mock_upload_svc.upload = AsyncMock(return_value=(row, False))

    files = {"files": ("folha.jpg", io.BytesIO(b"x"), "image/jpeg")}
    data = {"session_id": "chat-session-123"}

    r = await client_uploads.post("/api/v1/uploads", files=files, data=data)

    assert r.status_code == 201
    call_kwargs = mock_upload_svc.upload.await_args.kwargs
    assert call_kwargs["session_id"] == "chat-session-123"
