"""Conversor .env -> JSON do ``aws lambda --environment``."""

from pathlib import Path

from scripts.deploy.env_to_lambda import ler_env


def test_ler_env_preserva_valores_com_igual_e_virgula(tmp_path: Path) -> None:
    env = tmp_path / ".env.cloud"
    env.write_text(
        "# comentario\n"
        "DATABASE_URL=postgresql+asyncpg://u:p@h:5432/db?a=1,b=2\n"
        'EMAIL_FROM="Ze Praga <no-reply@x.com>"\n'
        "VAZIA=\n"
        "AWS_REGION=us-east-1\n"
        "\n",
        encoding="utf-8",
    )

    assert ler_env(env) == {
        "DATABASE_URL": "postgresql+asyncpg://u:p@h:5432/db?a=1,b=2",
        "EMAIL_FROM": "Ze Praga <no-reply@x.com>",
    }
