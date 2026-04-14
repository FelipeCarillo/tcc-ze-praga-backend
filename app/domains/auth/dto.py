from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class UserDTO:
    id: str
    email: str
    password_hash: str
    full_name: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class UserCreateDTO:
    email: str
    password_hash: str
    full_name: str | None = None
