"""Smoke HTTP restrito ao ambiente local isolado do roteiro de UX.

Cria dados sintéticos em uma conta temporária. Não usa LLM, e-mail ou Storage.
"""
import secrets
from uuid import uuid4

import httpx

BASE = "http://127.0.0.1:8000/api/v1"


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=90) as client:
        # Credenciais efêmeras: nunca são impressas ou gravadas em artefatos.
        credentials = {
            "email": f"qa-{uuid4().hex}@example.com",
            "password": secrets.token_urlsafe(24),
            "full_name": "Revisão local de UX",
        }
        registered = client.post("/auth/register", json=credentials)
        assert registered.status_code == 201, registered.status_code
        logged = client.post("/auth/login", json={
            "email": credentials["email"], "password": credentials["password"],
        })
        assert logged.status_code == 200
        client.headers["Authorization"] = "Bearer " + logged.json()["access_token"]
        profile = client.get("/users/me")
        assert profile.status_code == 200
        # Sem assinatura ativa, o backend aplica FREE_FEATURES implicitamente.
        assert profile.json()["plan"] is None
        plan = client.post("/subscriptions/me", json={"plan_name": "pro"})
        assert plan.status_code == 201
        assert client.get("/users/me").json()["plan"]["features"]["export_diagnoses"]
        for index in range(25):
            response = client.post("/diagnoses", json={
                "disease_name": f"Registro QA {index:02}",
                "disease_id": "ferrugem-asiatica",
                "confidence": 0.8,
                "severity": "alta",
                "description": "Registro sintético para teste de paginação.",
                "model_used": "qa-fixture",
            })
            assert response.status_code == 201, response.status_code
        page = client.get("/diagnoses", params={"page": 2, "limit": 12}).json()
        assert page["total"] == 25 and len(page["items"]) == 12
        last_page = client.get("/diagnoses", params={"page": 3, "limit": 12}).json()
        assert len(last_page["items"]) == 1
        filtered = client.get("/diagnoses", params={"search": "QA 24"}).json()
        assert filtered["total"] == 1
        record_id = filtered["items"][0]["id"]
        assert client.get("/diagnoses/" + record_id).status_code == 200
        public = httpx.get(BASE + "/diagnoses/" + record_id)
        assert public.status_code in (401, 403)
        orientations = client.get("/action-plans/ferrugem-asiatica")
        assert orientations.status_code == 200
        assert "especialista" not in orientations.json()["allowed_levels"]
        field = client.post("/talhoes", json={"nome": "Área QA", "hectares": 1})
        assert field.status_code == 201
        assert client.delete("/talhoes/" + field.json()["id"]).status_code == 204
        unconfirmed = client.delete("/diagnoses").json()
        assert "deleted" not in unconfirmed
        assert client.get("/diagnoses").json()["total"] == 25
        confirmed = client.delete("/diagnoses", params={"confirm": "true"}).json()
        assert confirmed["deleted"] == 25
        assert client.get("/diagnoses").json()["total"] == 0
        assert client.get("/chat/interrupts").status_code == 200
    print(
        "OK: cadastro, login, perfil, plano, 25 registros paginados, busca, detalhe, "
        "proteção, orientações, talhões, confirmação de exclusão e interrupts."
    )


if __name__ == "__main__":
    main()
