"""Cache TTL in-memory pra queries externas (Tavily/SciELO) — TCC-056.

Cache simples de processo, sem persistencia. Adequado pra reduzir custos de API
em queries repetidas dentro do mesmo turno/sessao curta. Quando o backend
escalar pra multiplas replicas, deve ser substituido por Redis ou similar.

API:
    - ``make_cache_key(*parts)``: gera chave determinista a partir de strings
      e numeros (sha256 hex 16 chars).
    - ``get_cached(key)``: retorna valor ou ``None`` se expirado/ausente.
    - ``set_cached(key, value)``: guarda valor com timestamp atual.
    - ``clear_cache()``: limpa tudo (uso em testes).

Notas:
- ``_TTL`` eh 24h por padrao. Trocavel via ``configure_cache(ttl=...)``.
- Sem locking — risco de double-fetch sob alta concorrencia eh aceitavel,
  pois resultado final ainda eh consistente (ultimo write vence).
- Cleanup eh lazy: chaves expiradas so somem quando consultadas.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

_CACHE: dict[str, tuple[datetime, Any]] = {}
_TTL: timedelta = timedelta(hours=24)


def make_cache_key(*parts: Any) -> str:
    """Gera chave determinista a partir de partes arbitrarias.

    Args:
        *parts: strings, ints, ou outros tipos serializaveis via ``str()``.

    Returns:
        Hex string (16 chars) — colisao improvavel pro volume esperado.
    """
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_cached(key: str) -> Any | None:
    """Retorna valor cacheado ou ``None`` se expirado/ausente.

    Cleanup lazy: chaves expiradas sao removidas no momento da consulta.
    """
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, value = entry
    if datetime.now(timezone.utc) - ts >= _TTL:
        del _CACHE[key]
        return None
    return value


def set_cached(key: str, value: Any) -> None:
    """Guarda valor com timestamp atual (UTC)."""
    _CACHE[key] = (datetime.now(timezone.utc), value)


def clear_cache() -> None:
    """Limpa todo o cache (usado em testes)."""
    _CACHE.clear()


def configure_cache(ttl: timedelta | None = None) -> None:
    """Reconfigura TTL global do cache (usado em testes).

    Args:
        ttl: novo TTL. ``None`` reseta pro default (24h).
    """
    global _TTL
    _TTL = ttl if ttl is not None else timedelta(hours=24)
