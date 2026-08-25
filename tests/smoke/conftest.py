"""Fixtures compartilhadas da suíte de smoke (`tests/smoke/`).

Reusa o padrão de `tests/integration/conftest.py`: `AsyncClient` +
`ASGITransport(app)` e `app.dependency_overrides`. A camada de smoke é
**intencionalmente fina** — garante que cada rota está plugada, exige auth e
devolve schema válido, sem duplicar as asserts profundas de `tests/integration/`.

Ponto importante (TCC-068): `require_quota(feature)` é uma **factory** que
retorna um novo `_dependency` a cada chamada (`app/core/dependencies.py:363`).
Sobrescrever pela chave `require_quota(...)` **não** casa com o objeto registrado
na rota. O jeito robusto é sobrescrever as deps *internas* que o `_dependency`
realmente chama: `get_current_user` e `get_usage_service` (com `check_quota`
no-op). Isso cobre `require_quota`, `require_quota_dual` e rotas que só usam
`get_current_user`.

As env vars obrigatórias (`DATABASE_URL`, `JWT_SECRET_KEY`, ...) já são setadas
pelo `tests/conftest.py` da raiz antes de qualquer import de `app.*`.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_current_user,
    get_current_user_or_api_key,
    get_plan_features,
    get_plan_features_dual,
    get_usage_service,
)
from app.domains.subscriptions.features import ENTERPRISE_FEATURES
from app.main import app
from tests.conftest import make_user_dto


@pytest.fixture
async def smoke_client():
    """Client cru, sem nenhum override — usado nos testes negativos (401/403).

    Limpa `app.dependency_overrides` no teardown (crítico: estado global do app).
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def make_usage_mock() -> AsyncMock:
    """`UsageService` mock — `check_quota` e `record_usage` viram no-op."""
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    return svc


def bypass_auth_overrides() -> None:
    """Aplica os overrides que destravam rotas autenticadas + com quota.

    Sobrescreve `get_current_user` / `get_current_user_or_api_key` (usuário fake),
    `get_usage_service` (quota no-op) e as duas variantes de `plan_features`
    (Enterprise — libera todos os modelos, senão o gate de plano trocaria o
    modelo pedido). Chame no início da fixture-client do arquivo de domínio,
    *antes* de instanciar o `AsyncClient`.
    """
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_current_user_or_api_key] = lambda: make_user_dto()
    app.dependency_overrides[get_usage_service] = make_usage_mock
    app.dependency_overrides[get_plan_features] = lambda: ENTERPRISE_FEATURES
    app.dependency_overrides[get_plan_features_dual] = lambda: ENTERPRISE_FEATURES


@pytest.fixture
def auth_headers(valid_token) -> dict:
    """Header `Bearer` real — reaproveita o fixture `valid_token` da raiz."""
    return {"Authorization": f"Bearer {valid_token}"}
