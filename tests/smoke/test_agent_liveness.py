"""Agent liveness (TCC-074) — a prova REAL de que os agentes funcionam.

Diferente dos demais smokes (que mockam `ChatService` e o `diagnosis_graph`),
aqui NADA é mockado: sobe contra Postgres real + chave LLM real e exercita o
chat agent e o diagnosis graph de ponta a ponta.

Toda a suíte está sob `@pytest.mark.live` → **excluída por default** (o
`pyproject.toml` tem `addopts = -m 'not live'`). Para rodar:

    docker compose up -d db          # Postgres
    export OPENAI_API_KEY=sk-...     # (PowerShell: $env:OPENAI_API_KEY = "sk-...")
    uv run pytest -m live -v

Sem a chave → skip limpo. Sem Postgres acessível → skip com instrução.
"""

import io
import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.live

# Erros que indicam "infra não está de pé" (Postgres) — viram skip, não falha.
_INFRA_HINTS = ("connect", "refused", "could not", "postgres", "timeout", "operationalerror")


@pytest.fixture(autouse=True)
async def _reset_singletons():
    """Isola cada teste live: zera os singletons de checkpointer/store (TCC-058)
    e, no teardown, descarta o pool do engine SQLAlchemy.

    pytest-asyncio cria um event loop novo por função; sem descartar o pool, o
    teste seguinte reusaria conexões asyncpg presas ao loop já fechado do teste
    anterior (`RuntimeError: Event loop is closed`). Em produção isso não ocorre
    — uvicorn roda um único loop pro processo todo.
    """
    from app.db.checkpointer import reset_checkpointer_for_tests
    from app.db.store import reset_store_for_tests

    reset_checkpointer_for_tests()
    reset_store_for_tests()
    yield
    from app.db.database import engine

    await engine.dispose()


@pytest.fixture
def require_llm_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY ausente — defina a chave para rodar o liveness do agente")


def _skip_if_infra(exc: Exception) -> None:
    """Converte erro de conexão com Postgres em skip; re-levanta o resto."""
    blob = f"{type(exc).__name__}: {exc}".lower()
    if any(h in blob for h in _INFRA_HINTS):
        pytest.skip(f"Postgres indisponível — rode `docker compose up -d db` ({type(exc).__name__})")
    raise exc


def _no_quota_override():
    """Neutraliza só a quota (o foco é o agente, não o billing); mantém auth real."""
    from unittest.mock import AsyncMock

    from app.core.dependencies import get_usage_service
    from app.main import app

    mock = AsyncMock()
    mock.check_quota = AsyncMock()
    mock.record_usage = AsyncMock()
    app.dependency_overrides[get_usage_service] = lambda: mock


async def _register_real_user(ac: AsyncClient) -> str:
    """Registra um usuário descartável via endpoint real → retorna o JWT."""
    email = f"liveness-{uuid.uuid4().hex[:10]}@example.com"
    r = await ac.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "liveness-pw-123", "full_name": "Liveness Bot"},
    )
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


async def test_chat_agent_real_llm_roundtrip(require_llm_key):
    """POST /chat com LLM de verdade → resposta com texto não-vazio."""
    from app.main import app

    _no_quota_override()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            token = await _register_real_user(ac)
            r = await ac.post(
                "/api/v1/chat",
                data={"messages": "Em uma frase, o que e ferrugem asiatica da soja?", "model": "ensemble"},
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception as exc:  # noqa: BLE001 — infra ausente vira skip
        _skip_if_infra(exc)
        raise
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"].strip(), "o agente retornou content vazio"
    assert body["session_id"]


async def test_diagnosis_graph_real_analyze(require_llm_key):
    """POST /diagnoses/analyze com o graph real → persisted_ids não-vazio.

    Requer catálogo de doenças seedado (`uv run python -m scripts.seed_crops`);
    se não estiver, skipa com instrução em vez de falhar.
    """
    from app.main import app

    _no_quota_override()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            token = await _register_real_user(ac)
            files = {"images": ("folha.jpg", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/jpeg")}
            r = await ac.post(
                "/api/v1/diagnoses/analyze",
                data={"crop_id": "soja", "model": "ensemble"},
                files=files,
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception as exc:  # noqa: BLE001
        blob = f"{type(exc).__name__}: {exc}".lower()
        if "soja" in blob or "seed" in blob or "crop" in blob:
            pytest.skip("catálogo não seedado — rode `uv run python -m scripts.seed_crops`")
        _skip_if_infra(exc)
        raise
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    persisted = r.json()
    assert isinstance(persisted, list) and persisted, "analyze não persistiu nenhum diagnóstico"


# ── TCC-079 / TCC-081: gate de visão + transcrição reais ───────────────────────

# PNG 1x1 válido (base64) — gpt-4o consegue processar; serve pra exercitar o gate.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _silent_wav(seconds: float = 0.4, rate: int = 16000) -> bytes:
    """Gera um WAV mono 16-bit de silêncio (sem dependências externas)."""
    import io as _io
    import wave

    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


async def test_chat_vision_gate_invokes_inspect_image(require_llm_key):
    """POST /chat/stream com imagem → o agente realmente chama inspect_image (gate)."""
    import base64

    from app.main import app

    _no_quota_override()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            token = await _register_real_user(ac)
            files = {"image": ("foto.png", base64.b64decode(_TINY_PNG_B64), "image/png")}
            body = b""
            async with ac.stream(
                "POST",
                "/api/v1/chat/stream",
                data={"messages": "[]", "model": "ensemble"},
                files=files,
                headers={"Authorization": f"Bearer {token}"},
            ) as r:
                assert r.status_code == 200, r.text
                async for chunk in r.aiter_bytes():
                    body += chunk
    except Exception as exc:  # noqa: BLE001
        _skip_if_infra(exc)
        raise
    finally:
        app.dependency_overrides.clear()

    text = body.decode("utf-8", errors="ignore")
    assert "inspect_image" in text, f"agente não chamou o gate de visão; stream={text[:400]}"


async def test_chat_audio_transcription_real(require_llm_key):
    """POST /chat com áudio → pipeline de transcrição real responde 200 + transcript."""
    from app.main import app

    _no_quota_override()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            token = await _register_real_user(ac)
            files = {"audio": ("voice.wav", _silent_wav(), "audio/wav")}
            r = await ac.post(
                "/api/v1/chat",
                data={"messages": "[]", "model": "ensemble"},
                files=files,
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception as exc:  # noqa: BLE001
        _skip_if_infra(exc)
        raise
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    # A transcrição real rodou (silêncio pode dar texto vazio — validamos o tipo/contrato).
    assert "transcript" in r.json()
