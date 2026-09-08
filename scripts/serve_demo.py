"""Sobe a API para a execução local com o fluxo **real**.

Uso: ``uv run python -m scripts.serve_demo`` — ou, mais simples, o
``iniciar.ps1`` na raiz do workspace, que sobe banco, API e frontend juntos.

Lê ``backend/.env.local`` (não versionado; copie de ``.env.local.example``).
Esse arquivo é a fonte única da configuração local: banco e imagens na máquina,
OpenAI e Resend reais. As variáveis dele têm precedência sobre o ``.env``, que
guarda a configuração antiga de nuvem.

Diferença para os outros helpers:

- ``scripts.local_dev``  → Postgres local e **sem** LLM (valida contratos REST).
- ``scripts.serve_local`` → ``.env`` como está (exige um Supabase no ar).
- ``scripts.serve_demo``  → o modo de trabalhar e apresentar na máquina.
"""

from __future__ import annotations

import asyncio
import os
import selectors
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_LOCAL = REPO_ROOT / ".env.local"

# Usado só quando não existe .env.local — mantém o script utilizável para quem
# ainda não copiou o example. Sem OpenAI e sem Resend, o chat e o e-mail não
# funcionam; o resto (ONNX, banco, storage, histórico) funciona.
PADRAO_SEM_ENV_LOCAL = {
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/ze_praga",
    "STORAGE_BACKEND": "local",
    "LOCAL_STORAGE_DIR": str(REPO_ROOT / ".local-storage"),
    "PUBLIC_API_URL": "http://127.0.0.1:8000",
    "FRONTEND_URL": "http://127.0.0.1:3101",
    "ALLOWED_ORIGINS": (
        "http://127.0.0.1:3000,http://127.0.0.1:3100,http://127.0.0.1:3101,"
        "http://localhost:3000,http://localhost:3100,http://localhost:3101"
    ),
    "REQUIRE_EMAIL_VERIFICATION": "false",
    "RATE_LIMIT_LOGIN": "100/300",
    "RATE_LIMIT_REGISTER": "100/3600",
    "RATE_LIMIT_EMAIL": "50/3600",
    "RATE_LIMIT_RESET": "100/3600",
}


def carregar_env_local() -> dict[str, str]:
    """Lê ``.env.local`` no formato ``CHAVE=valor``, ignorando comentários."""
    valores: dict[str, str] = {}
    for linha in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        valores[chave.strip()] = valor.strip()
    return valores


def selector_loop() -> asyncio.AbstractEventLoop:
    """psycopg async (checkpointer + Store pgvector) recusa o ProactorEventLoop.

    No Windows o default do asyncio é o Proactor, e o sintoma é traiçoeiro: a
    API sobe, ``/api/v1/health`` responde 200, as rotas REST funcionam — e só o
    chat, a retomada HITL e a memória semântica quebram.
    """
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
    if ENV_LOCAL.exists():
        env = carregar_env_local()
        origem = ".env.local"
    else:
        env = dict(PADRAO_SEM_ENV_LOCAL)
        origem = "padrões embutidos (copie .env.local.example para .env.local)"
    os.environ.update(env)

    # Importado só depois do update: Settings é instanciado no import de app.config.
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    def estado(chave: str, rotulo: str) -> str:
        return f"{rotulo} real" if os.environ.get(chave) else f"{rotulo} desligado"

    onnx = "real" if os.environ.get("INFERENCE_USE_ONNX", "true") == "true" else "mock"

    config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        loop="none",  # não deixa o uvicorn instalar outro loop por cima
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config)

    loop = selector_loop()
    asyncio.set_event_loop(loop)
    print(
        f"Zé Praga · API local em http://{host}:{port}\n"
        f"  config  : {origem}\n"
        f"  banco   : Postgres local (zepraga-ux-local)\n"
        f"  imagens : {os.environ.get('LOCAL_STORAGE_DIR', '(Supabase)')}\n"
        f"  ONNX    : {onnx}"
        f" · {estado('OPENAI_API_KEY', 'agente')}"
        f" · {estado('RESEND_API_KEY', 'e-mail')}\n"
        f"  docs    : http://{host}:{port}/docs",
        flush=True,
    )
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
