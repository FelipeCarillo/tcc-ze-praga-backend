"""Converte um .env no JSON que o AWS CLI aceita em ``--environment``.

    python scripts/deploy/env_to_lambda.py .env.cloud > lambda-env.json
    aws lambda update-function-configuration --function-name ze-praga-api \\
        --environment file://lambda-env.json

Passar as variáveis inline (``Variables={K=V,...}``) quebra com a vírgula e o
``=`` que aparecem na DATABASE_URL e no EMAIL_FROM — por isso o arquivo JSON.
O ``lambda-env.json`` tem segredo: está no .gitignore, não versione.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# O Lambda reserva estes nomes e recusa a configuração inteira se vierem.
_RESERVADAS = {"AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}


def ler_env(caminho: Path) -> dict[str, str]:
    variaveis: dict[str, str] = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        chave, valor = chave.strip(), valor.strip()
        if len(valor) >= 2 and valor[0] == valor[-1] and valor[0] in "\"'":
            valor = valor[1:-1]
        if valor and chave not in _RESERVADAS:
            variaveis[chave] = valor
    return variaveis


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("uso: python scripts/deploy/env_to_lambda.py .env.cloud > lambda-env.json")
    print(json.dumps({"Variables": ler_env(Path(sys.argv[1]))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
