"""Meta-smoke: garante que todo router está plugado no `app` e que o app sobe.

Falha cedo se alguém esquecer um `include_router` em `app/main.py`, sem precisar
de um teste dedicado por rota. Complementa os smokes por domínio (TCC-069..074).
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

API = "/api/v1"

# Pelo menos uma rota canônica de cada um dos 12 routers + health.
EXPECTED_PATHS = {
    f"{API}/auth/register",
    f"{API}/auth/login",
    f"{API}/auth/me",
    f"{API}/auth/api-keys",
    f"{API}/users/me",
    f"{API}/diagnoses",
    f"{API}/diagnoses/analyze",
    f"{API}/diagnoses/semantic",
    f"{API}/inference",
    f"{API}/chat",
    f"{API}/chat/stream",
    f"{API}/chat/resume",
    f"{API}/chat/interrupts",
    f"{API}/sessions/{{session_id}}/close",
    f"{API}/action-plans/{{disease_id}}",
    f"{API}/subscriptions/plans",
    f"{API}/subscriptions/me",
    f"{API}/talhoes",
    f"{API}/uploads",
    f"{API}/usage/me",
    f"{API}/usage/me/history",
    f"{API}/health",
}


def _registered_paths() -> set[str]:
    return {getattr(r, "path", None) for r in app.routes}


@pytest.mark.parametrize("path", sorted(EXPECTED_PATHS))
def test_route_is_registered(path: str) -> None:
    assert path in _registered_paths(), f"rota {path} não está plugada em app.main"


def test_openapi_schema_builds() -> None:
    schema = app.openapi()
    assert schema["info"]["title"] == "Zé Praga API"
    assert schema.get("paths"), "OpenAPI sem paths — nenhum router exposto"


async def test_health_returns_200() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
