"""POST /api/v1/uploads — multipart, batch ate 5 arquivos.

DoD (TCC-037):
    - aceita ``files[]`` multipart
    - max 5 files / request
    - max 10MB / file
    - dedup por sha256 + user_id
    - upload pra Supabase Storage (bucket ``uploads``)
    - retorna ``list[UploadResponse]`` com ``deduplicated`` flag
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.dependencies import get_current_user, get_upload_service
from app.domains.auth.dto import UserDTO
from app.domains.uploads.schemas import UploadResponse
from app.domains.uploads.service import UploadService

router = APIRouter(prefix="/uploads", tags=["Uploads"])


# Limites da spec (TCC-037 DoD)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILES_PER_REQUEST = 5


@router.post(
    "",
    response_model=list[UploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload de imagens (multipart)",
)
async def create_uploads(
    files: list[UploadFile] = File(..., description="Ate 5 arquivos de no maximo 10MB cada"),
    session_id: str | None = Form(default=None, description="Chat session opcional"),
    current_user: UserDTO = Depends(get_current_user),
    upload_svc: UploadService = Depends(get_upload_service),
) -> list[UploadResponse]:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Pelo menos um arquivo e' necessario.",
        )

    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Maximo {MAX_FILES_PER_REQUEST} arquivos por request "
                f"(recebido: {len(files)})."
            ),
        )

    results: list[UploadResponse] = []
    for file in files:
        data = await file.read()
        if len(data) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Arquivo '{file.filename}' excede o limite de "
                    f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
                ),
            )

        row, deduplicated = await upload_svc.upload(
            user_id=current_user.id,
            original_name=file.filename or "arquivo.bin",
            mime=file.content_type or "application/octet-stream",
            data=data,
            session_id=session_id,
        )
        response = UploadResponse.model_validate(row)
        response.deduplicated = deduplicated
        results.append(response)

    return results
