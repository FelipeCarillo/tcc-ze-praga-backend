"""Testes pro modulo de memoria semantica (TCC-045)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.domains.chat.memory import (
    format_diagnosis_summary,
    index_diagnosis_in_store,
    index_session_summary_in_store,
)


def make_fake_diagnosis(**overrides) -> SimpleNamespace:
    defaults = dict(
        id="diag-uuid-1",
        disease_name="Ferrugem Asiatica",
        disease_id="ferrugem-asiatica",
        crop_id="crop-soja",
        confidence=0.942,
        severity="alta",
        created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── format_diagnosis_summary ──────────────────────────────────────────────────


def test_format_diagnosis_summary_basic():
    diag = make_fake_diagnosis()
    out = format_diagnosis_summary(diag)
    assert "Ferrugem Asiatica" in out
    assert "ferrugem-asiatica" in out
    assert "94.2%" in out
    assert "alta" in out
    assert "2026-05-22" in out


def test_format_diagnosis_summary_handles_missing_attrs():
    diag = SimpleNamespace()  # tudo ausente
    out = format_diagnosis_summary(diag)
    # Nao explode, retorna string mesmo com placeholders
    assert isinstance(out, str)
    assert "doenca desconhecida" in out


def test_format_diagnosis_summary_with_none_confidence():
    diag = make_fake_diagnosis(confidence=None)
    out = format_diagnosis_summary(diag)
    assert "?" in out  # placeholder pra confidence ausente


def test_format_diagnosis_summary_with_string_date():
    """Aceita created_at como string sem quebrar."""
    diag = make_fake_diagnosis(created_at="2026-01-15")
    out = format_diagnosis_summary(diag)
    assert "2026-01-15" in out


# ── index_diagnosis_in_store ──────────────────────────────────────────────────


async def test_index_diagnosis_calls_aput_with_correct_namespace():
    store = AsyncMock()
    diag = make_fake_diagnosis()

    await index_diagnosis_in_store(store, "user-1", diag)

    store.aput.assert_awaited_once()
    call_kwargs = store.aput.call_args.kwargs
    assert call_kwargs["namespace"] == ("user", "user-1", "diagnoses")
    assert call_kwargs["key"] == "diag-uuid-1"
    assert call_kwargs["index"] == ["summary_text"]
    value = call_kwargs["value"]
    assert "summary_text" in value
    assert value["diagnosis_id"] == "diag-uuid-1"
    assert value["disease_id"] == "ferrugem-asiatica"
    assert value["crop_id"] == "crop-soja"
    assert value["confidence"] == pytest.approx(0.942)
    assert value["severity"] == "alta"
    assert value["created_at"] == "2026-05-22T12:00:00+00:00"


async def test_index_diagnosis_skips_when_no_id():
    """Diagnostico sem id nao eh indexado (defensivo)."""
    store = AsyncMock()
    diag = make_fake_diagnosis(id=None)

    await index_diagnosis_in_store(store, "user-1", diag)

    store.aput.assert_not_awaited()


async def test_index_diagnosis_handles_none_created_at():
    store = AsyncMock()
    diag = make_fake_diagnosis(created_at=None)

    await index_diagnosis_in_store(store, "user-1", diag)

    call_kwargs = store.aput.call_args.kwargs
    assert call_kwargs["value"]["created_at"] is None


# ── index_session_summary_in_store ────────────────────────────────────────────


async def test_index_session_summary_calls_aput_correctly():
    store = AsyncMock()
    await index_session_summary_in_store(
        store, "user-1", "session-1", "Conversa sobre ferrugem na soja."
    )

    store.aput.assert_awaited_once()
    call_kwargs = store.aput.call_args.kwargs
    assert call_kwargs["namespace"] == ("user", "user-1", "session_summaries")
    assert call_kwargs["key"] == "session-1"
    assert call_kwargs["value"]["summary_text"] == "Conversa sobre ferrugem na soja."
    assert call_kwargs["value"]["session_id"] == "session-1"
    assert call_kwargs["index"] == ["summary_text"]


async def test_index_session_summary_skips_empty_text():
    store = AsyncMock()
    await index_session_summary_in_store(store, "user-1", "session-1", "")
    store.aput.assert_not_awaited()
