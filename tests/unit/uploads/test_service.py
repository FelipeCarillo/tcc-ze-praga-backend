"""Unit tests for UploadService (TCC-037)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.uploads.service import (
    SupabaseStorageUploader,
    UploadService,
)
from app.models.uploaded_file import UploadedFile


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.find_by_hash = AsyncMock(return_value=None)
    repo.create = AsyncMock()
    return repo


@pytest.fixture
def mock_uploader():
    """StorageUploader que so registra a chamada e retorna o path."""
    uploader = MagicMock()
    uploader.upload = MagicMock(side_effect=lambda *, bucket, path, data, content_type: path)
    return uploader


@pytest.fixture
def service(mock_repo, mock_uploader):
    return UploadService(mock_repo, mock_uploader)


async def test_upload_new_file_calls_storage_and_persists(service, mock_repo, mock_uploader):
    """Hash inedito -> upload no storage + create no DB, deduplicated=False."""
    data = b"image-bytes"
    expected_hash = hashlib.sha256(data).hexdigest()

    fake_row = UploadedFile()
    fake_row.id = "uuid-1"
    fake_row.hash_sha256 = expected_hash
    mock_repo.create = AsyncMock(return_value=fake_row)

    row, deduplicated = await service.upload(
        user_id="user-1",
        original_name="folha.jpg",
        mime="image/jpeg",
        data=data,
    )

    assert deduplicated is False
    assert row is fake_row

    mock_repo.find_by_hash.assert_awaited_once_with("user-1", expected_hash)
    mock_uploader.upload.assert_called_once()
    upload_kwargs = mock_uploader.upload.call_args.kwargs
    assert upload_kwargs["bucket"] == "uploads"
    assert upload_kwargs["data"] == data
    assert upload_kwargs["content_type"] == "image/jpeg"
    assert upload_kwargs["path"].startswith("users/user-1/")

    mock_repo.create.assert_awaited_once()
    create_kwargs = mock_repo.create.await_args.kwargs
    assert create_kwargs["hash_sha256"] == expected_hash
    assert create_kwargs["size_bytes"] == len(data)
    assert create_kwargs["original_name"] == "folha.jpg"


async def test_upload_dedup_skips_storage_and_create(service, mock_repo, mock_uploader):
    """Hash ja existe -> nao chama storage nem create, deduplicated=True."""
    existing = UploadedFile()
    existing.id = "uuid-existing"
    existing.hash_sha256 = hashlib.sha256(b"img").hexdigest()
    mock_repo.find_by_hash = AsyncMock(return_value=existing)

    row, deduplicated = await service.upload(
        user_id="user-1",
        original_name="folha.jpg",
        mime="image/jpeg",
        data=b"img",
    )

    assert deduplicated is True
    assert row is existing
    mock_uploader.upload.assert_not_called()
    mock_repo.create.assert_not_awaited()


async def test_upload_sanitizes_path_separators_in_storage_key(service, mock_repo, mock_uploader):
    """Nome com / ou \\ nao gera path traversal (separadores viram _)."""
    fake_row = UploadedFile()
    fake_row.id = "uuid-1"
    mock_repo.create = AsyncMock(return_value=fake_row)

    await service.upload(
        user_id="user-1",
        original_name="../../etc/passwd",
        mime="application/octet-stream",
        data=b"x",
    )

    upload_path = mock_uploader.upload.call_args.kwargs["path"]
    # users/user-1/ deve ser o unico prefixo de path antes do leaf
    assert upload_path.startswith("users/user-1/")
    leaf = upload_path[len("users/user-1/") :]
    # leaf nao pode conter / nem \\ (sanitizados)
    assert "/" not in leaf
    assert "\\" not in leaf


async def test_upload_passes_session_id_to_repo(service, mock_repo, mock_uploader):
    fake_row = UploadedFile()
    fake_row.id = "uuid-1"
    mock_repo.create = AsyncMock(return_value=fake_row)

    await service.upload(
        user_id="user-1",
        original_name="folha.jpg",
        mime="image/jpeg",
        data=b"x",
        session_id="sess-42",
    )

    assert mock_repo.create.await_args.kwargs["session_id"] == "sess-42"


async def test_storage_key_uses_hash_prefix(service, mock_repo, mock_uploader):
    """storage_key inclui prefixo do hash sha256 (16 chars)."""
    fake_row = UploadedFile()
    fake_row.id = "uuid-1"
    mock_repo.create = AsyncMock(return_value=fake_row)

    data = b"image-content"
    expected_prefix = hashlib.sha256(data).hexdigest()[:16]

    await service.upload(
        user_id="user-1",
        original_name="folha.jpg",
        mime="image/jpeg",
        data=data,
    )

    path = mock_uploader.upload.call_args.kwargs["path"]
    assert expected_prefix in path
    assert path.startswith("users/user-1/")


def test_supabase_storage_uploader_calls_client():
    """SupabaseStorageUploader passa args corretos pro client."""
    client = MagicMock()
    uploader = SupabaseStorageUploader(client)

    result = uploader.upload(
        bucket="uploads",
        path="users/u1/file.jpg",
        data=b"bytes",
        content_type="image/jpeg",
    )

    assert result == "users/u1/file.jpg"
    client.storage.from_.assert_called_once_with("uploads")
    client.storage.from_().upload.assert_called_once()
    call_kwargs = client.storage.from_().upload.call_args.kwargs
    assert call_kwargs["path"] == "users/u1/file.jpg"
    assert call_kwargs["file"] == b"bytes"
    assert call_kwargs["file_options"]["content-type"] == "image/jpeg"
