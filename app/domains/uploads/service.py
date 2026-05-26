"""UploadService — orquestra upload + dedup + persistencia.

Validacoes nao-trivais (tamanho/quantidade) ficam no router porque dependem
do FastAPI ``UploadFile.size``; aqui o service trata: ler bytes, calcular
sha256, dedup por hash, upload Supabase, persistir row.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Protocol

from app.domains.uploads.repository import UploadedFileRepository
from app.models.uploaded_file import UploadedFile


class StorageUploader(Protocol):
    """Interface minima do client de storage — facilita mock em testes."""

    def upload(self, *, bucket: str, path: str, data: bytes, content_type: str) -> str:
        """Faz upload e retorna a storage_key (path no bucket). Sincrono — Supabase v2."""
        ...


class SupabaseStorageUploader:
    """Implementacao real wrapando o client do Supabase."""

    def __init__(self, client: Any, bucket: str = "uploads") -> None:
        self._client = client
        self._bucket = bucket

    @property
    def bucket(self) -> str:
        return self._bucket

    def upload(self, *, bucket: str, path: str, data: bytes, content_type: str) -> str:
        self._client.storage.from_(bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return path


class UploadService:
    """Salva um arquivo enviado pelo usuario.

    Fluxo:
        1. Calcula sha256 do conteudo
        2. Se ja existe um upload deste user com mesmo hash, retorna o existente
        3. Senao, upload pra Supabase Storage e persistencia da linha
    """

    BUCKET = "uploads"

    def __init__(
        self,
        repo: UploadedFileRepository,
        uploader: StorageUploader,
    ) -> None:
        self._repo = repo
        self._uploader = uploader

    async def upload(
        self,
        *,
        user_id: str,
        original_name: str,
        mime: str,
        data: bytes,
        session_id: str | None = None,
    ) -> tuple[UploadedFile, bool]:
        """Upload (ou dedup) de um arquivo. Retorna ``(row, deduplicated)``."""
        hash_sha256 = hashlib.sha256(data).hexdigest()

        existing = await self._repo.find_by_hash(user_id, hash_sha256)
        if existing is not None:
            return existing, True

        storage_key = self._build_storage_key(user_id, hash_sha256, original_name)
        self._uploader.upload(
            bucket=self.BUCKET,
            path=storage_key,
            data=data,
            content_type=mime,
        )

        row = await self._repo.create(
            user_id=user_id,
            original_name=original_name,
            mime=mime,
            storage_key=storage_key,
            size_bytes=len(data),
            hash_sha256=hash_sha256,
            session_id=session_id,
        )
        return row, False

    @staticmethod
    def _build_storage_key(user_id: str, hash_sha256: str, original_name: str) -> str:
        """Path no bucket: ``users/<user_id>/<hash[:16]>-<uuid>-<name>``.

        Usamos prefixo do hash pra agrupar dedup-friendly e um uuid pra
        evitar colisao se o mesmo hash for re-uploadado por algum motivo
        (race condition no flush).
        """
        suffix = uuid.uuid4().hex[:8]
        # sanitiza nome (sem path traversal)
        safe_name = original_name.replace("/", "_").replace("\\", "_")
        return f"users/{user_id}/{hash_sha256[:16]}-{suffix}-{safe_name}"
