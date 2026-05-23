from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class ApiKeyDTO:
    id: str
    user_id: str
    name: str
    key_hash: str
    key_prefix: str
    scopes: list[str]
    is_active: bool
    last_used_at: datetime | None
    created_at: datetime
    revoked_at: datetime | None


@dataclass(frozen=True)
class ApiKeyCreateDTO:
    user_id: str
    name: str
    key_hash: str
    key_prefix: str
    scopes: list[str] = field(default_factory=lambda: ["diagnoses:analyze"])
