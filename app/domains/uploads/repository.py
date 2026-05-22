from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.uploaded_file import UploadedFile


class UploadedFileRepository:
    """CRUD basico de uploads.

    Nao retornamos DTOs aqui porque o caller imediato (service) e' quem
    decide o shape final (UploadResponse). Trabalhamos com ORM direto.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        user_id: str,
        original_name: str,
        mime: str,
        storage_key: str,
        size_bytes: int,
        hash_sha256: str,
        session_id: str | None = None,
    ) -> UploadedFile:
        row = UploadedFile(
            user_id=user_id,
            session_id=session_id,
            original_name=original_name,
            mime=mime,
            storage_key=storage_key,
            size_bytes=size_bytes,
            hash_sha256=hash_sha256,
        )
        self._db.add(row)
        await self._db.flush()
        await self._db.refresh(row)
        return row

    async def find_by_hash(
        self, user_id: str, hash_sha256: str
    ) -> UploadedFile | None:
        result = await self._db.execute(
            select(UploadedFile).where(
                UploadedFile.user_id == user_id,
                UploadedFile.hash_sha256 == hash_sha256,
            )
        )
        return result.scalar_one_or_none()
