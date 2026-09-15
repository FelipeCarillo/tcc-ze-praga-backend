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

import mimetypes

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

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


@router.get(
    "/local/{key:path}",
    include_in_schema=False,
    summary="Serve imagem do storage local (somente storage_backend=local)",
)
async def serve_local_upload(key: str, exp: int, sig: str) -> Response:
    """Entrega um arquivo do storage em disco validando a URL assinada.

    Deliberadamente **sem** ``Depends(get_current_user)``: quem chama é a tag
    ``<img>`` do navegador, que não manda ``Authorization``. A credencial aqui é
    o HMAC em ``sig`` — mesma escolha do Supabase, que também autoriza a leitura
    pela query string. Só existe no modo local; com Supabase responde 404.
    """
    from app.config import settings
    from app.core.dependencies import get_local_storage_uploader

    if settings.storage_backend != "local":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    uploader = get_local_storage_uploader()
    if not uploader.verify(key, exp, sig):
        # Mesma resposta para assinatura inválida e link vencido: não vale
        # entregar ao cliente a informação de qual dos dois foi.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Link expirado ou inválido."
        )

    try:
        data = uploader.read(uploader.bucket, key)
    except (OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Arquivo não encontrado."
        ) from None

    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(content=data, media_type=media_type)
