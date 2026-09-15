"""Inicializa uma API de teste com banco exclusivamente local.

Uso: uv run python -m scripts.local_dev setup|serve
O frontend demonstra o chat sem serviços externos em npm run start:demo.
Este servidor valida os contratos reais locais, sem credenciais de produção.
"""
import asyncio
import os
import secrets
import selectors
import subprocess
import sys

ENV = {
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/ze_praga",
    "SUPABASE_URL": "http://127.0.0.1:54321",
    "SUPABASE_SERVICE_ROLE_KEY": "local-test-placeholder",
    "JWT_SECRET_KEY": secrets.token_urlsafe(48),
    "OPENAI_API_KEY": "local-test-placeholder",
    "OPENAI_BASE_URL": "http://127.0.0.1:54322/v1",
    "ANTHROPIC_API_KEY": "",
    "TAVILY_API_KEY": "",
    "RESEND_API_KEY": "",
    "CHAT_MODEL": "openai:gpt-4o-mini",
    "VISION_MODEL": "openai:gpt-4o",
    "APP_ENV": "test",
    "CHAT_MAX_RETRIES": "0",
    "CHAT_TIMEOUT_SECONDS": "5",
    "REQUIRE_EMAIL_VERIFICATION": "false",
    "FRONTEND_URL": "http://127.0.0.1:3101",
    "ALLOWED_ORIGINS": "http://127.0.0.1:3100,http://127.0.0.1:3101",
}


def selector_loop() -> asyncio.AbstractEventLoop:
    """Psycopg async exige SelectorEventLoop no Windows."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
    os.environ.update(ENV)
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if mode == "setup":
        # O compose deve ser iniciado com o projeto isolado zepraga-ux-local.
        for args in (
            ["-m", "alembic", "upgrade", "head"],
            ["-m", "scripts.seed_crops"],
            ["-m", "scripts.seed_action_plans"],
            ["-m", "scripts.seed_plan_features"],
        ):
            subprocess.run([sys.executable, *args], check=True)
    elif mode == "serve":
        import uvicorn

        print("API local de teste em 127.0.0.1:8000; LLM, Storage e e-mail externos desativados.")
        uvicorn.run(
            "app.main:app", host="127.0.0.1", port=8000,
            loop="scripts.local_dev:selector_loop",
        )
    else:
        raise SystemExit("Use setup ou serve.")


if __name__ == "__main__":
    main()
