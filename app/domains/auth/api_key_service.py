"""API key lifecycle: generate, list, revoke, verify.

Token format: ``zp_live_<32 bytes urlsafe>`` (~50 chars total).
``key_prefix`` = first 12 chars = ``zp_live_XXXX`` (4 chars de entropia
no prefix bastam pra restringir bcrypt verify a poucos candidatos).
``key_hash`` = bcrypt(plain_key).

Verify flow:
1. Extrai prefix dos 12 primeiros chars do header.
2. ``find_by_prefix_active(prefix)`` retorna 0+ candidatos.
3. Pra cada candidato, ``bcrypt.checkpw`` — primeiro match vence.
4. Atualiza ``last_used_at`` async (fire-and-forget seria ideal, mas
   commit sincrono eh aceitavel pra carga atual).
"""

import secrets

import bcrypt

from app.domains.auth.api_key_dto import ApiKeyCreateDTO, ApiKeyDTO
from app.domains.auth.api_key_repository import ApiKeyRepository

_PREFIX_LEN = 12
_TOKEN_PREFIX = "zp_live_"
_DEFAULT_SCOPES: list[str] = ["diagnoses:analyze"]


class ApiKeyService:
    def __init__(self, repo: ApiKeyRepository) -> None:
        self._repo = repo

    async def create(
        self,
        user_id: str,
        name: str,
        scopes: list[str] | None = None,
    ) -> tuple[ApiKeyDTO, str]:
        """Gera token, persiste hash + prefix, retorna (DTO, plain_key).

        ``plain_key`` so' existe nesse retorno — chamador deve enviar pro
        cliente UMA vez. Nao loggar, nao persistir.
        """
        plain_key = self._generate_token()
        key_hash = self._hash(plain_key)
        prefix = plain_key[:_PREFIX_LEN]

        dto = await self._repo.create(
            ApiKeyCreateDTO(
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                key_prefix=prefix,
                scopes=list(scopes) if scopes else list(_DEFAULT_SCOPES),
            )
        )
        return dto, plain_key

    async def list_for_user(self, user_id: str) -> list[ApiKeyDTO]:
        return await self._repo.find_active_by_user(user_id)

    async def revoke(self, user_id: str, key_id: str) -> bool:
        return await self._repo.revoke(key_id, user_id)

    async def verify(self, plain_key: str) -> ApiKeyDTO | None:
        """Resolve API key plain text -> ApiKeyDTO ativa ou ``None``.

        Rejeita keys com prefix invalido sem bater no DB (early return).
        """
        if not plain_key or len(plain_key) < _PREFIX_LEN:
            return None
        if not plain_key.startswith(_TOKEN_PREFIX):
            return None

        prefix = plain_key[:_PREFIX_LEN]
        candidates = await self._repo.find_by_prefix_active(prefix)
        for candidate in candidates:
            if self._check(plain_key, candidate.key_hash):
                return candidate
        return None

    async def touch_last_used(self, key_id: str) -> None:
        await self._repo.touch_last_used(key_id)

    # ── Helpers (testaveis estaticamente) ─────────────────────────────────────

    @staticmethod
    def _generate_token() -> str:
        """Token URL-safe com prefix amigavel."""
        return f"{_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"

    @staticmethod
    def _hash(plain: str) -> str:
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def _check(plain: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(plain.encode(), hashed.encode())
        except (ValueError, TypeError):
            return False
