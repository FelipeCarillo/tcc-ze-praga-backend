from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    """Resposta de upload de um unico arquivo.

    ``deduplicated`` indica que o arquivo ja existia (mesmo user + hash) e
    o storage_key/row foi reaproveitado em vez de re-uploadado.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    mime: str
    storage_key: str
    size_bytes: int
    hash_sha256: str
    uploaded_at: datetime
    deduplicated: bool = False
