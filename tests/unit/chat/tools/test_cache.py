"""Testes do cache TTL in-memory pra queries externas (TCC-056).

Cobertura:
- get/set basico (hit + miss)
- TTL expirado -> miss + cleanup
- clear_cache zera tudo
- configure_cache permite TTL custom (testes)
- make_cache_key eh determinista e cobre tipos diversos
- make_cache_key gera chaves distintas pra args distintos
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.domains.chat.tools._cache import (
    clear_cache,
    configure_cache,
    get_cached,
    make_cache_key,
    set_cached,
)


@pytest.fixture(autouse=True)
def _reset():
    clear_cache()
    configure_cache(None)  # reset TTL pra default
    yield
    clear_cache()
    configure_cache(None)


def test_get_miss_returns_none() -> None:
    assert get_cached("missing") is None


def test_set_then_get_hits() -> None:
    set_cached("k", {"value": 1})
    assert get_cached("k") == {"value": 1}


def test_set_overwrites_previous_value() -> None:
    set_cached("k", "v1")
    set_cached("k", "v2")
    assert get_cached("k") == "v2"


def test_clear_cache_removes_all_entries() -> None:
    set_cached("a", 1)
    set_cached("b", 2)
    clear_cache()
    assert get_cached("a") is None
    assert get_cached("b") is None


def test_ttl_expired_returns_none_and_cleans_up() -> None:
    """TTL muito curto -> entrada vira None apos sleep simulado."""
    import time

    configure_cache(timedelta(milliseconds=1))
    set_cached("k", "v")
    time.sleep(0.005)
    assert get_cached("k") is None
    # Cleanup lazy: chave foi removida do dict
    from app.domains.chat.tools import _cache

    assert "k" not in _cache._CACHE


def test_ttl_within_window_returns_value() -> None:
    """Quando TTL eh longo, get retorna mesmo apos varias chamadas."""
    configure_cache(timedelta(hours=1))
    set_cached("k", "v")
    for _ in range(5):
        assert get_cached("k") == "v"


def test_configure_cache_with_none_resets_default() -> None:
    """``configure_cache(None)`` volta pra 24h."""
    from app.domains.chat.tools import _cache

    configure_cache(timedelta(seconds=1))
    assert _cache._TTL == timedelta(seconds=1)
    configure_cache(None)
    assert _cache._TTL == timedelta(hours=24)


def test_make_cache_key_is_deterministic() -> None:
    k1 = make_cache_key("a", 1, "b")
    k2 = make_cache_key("a", 1, "b")
    assert k1 == k2


def test_make_cache_key_returns_16_chars() -> None:
    k = make_cache_key("anything")
    assert len(k) == 16


def test_make_cache_key_different_args_differ() -> None:
    a = make_cache_key("query", 5)
    b = make_cache_key("query", 6)
    assert a != b


def test_make_cache_key_order_matters() -> None:
    a = make_cache_key("x", "y")
    b = make_cache_key("y", "x")
    assert a != b


def test_make_cache_key_handles_mixed_types() -> None:
    """Strings, ints, bools, None — todos convertidos via str()."""
    k = make_cache_key("q", 1, True, None)
    assert len(k) == 16
    # Reproduzivel
    assert make_cache_key("q", 1, True, None) == k


def test_cache_stores_strings_lists_dicts() -> None:
    set_cached("str", "value")
    set_cached("list", [1, 2, 3])
    set_cached("dict", {"a": "b"})
    assert get_cached("str") == "value"
    assert get_cached("list") == [1, 2, 3]
    assert get_cached("dict") == {"a": "b"}
