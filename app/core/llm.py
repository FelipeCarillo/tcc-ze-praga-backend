"""Provider-agnostic chat model factory.

Centraliza ``langchain.chat_models.init_chat_model`` pra que o resto do app nao
amarre em provider especifico (OpenAI, Anthropic, Bedrock, Azure, etc.).

Model ID format: ``"<provider>:<model>"`` — ex:

    openai:gpt-4o-mini
    anthropic:claude-3-5-sonnet-latest
    bedrock_converse:anthropic.claude-3-5-sonnet-20240620-v1:0
    azure_openai:gpt-4o
    google_vertexai:gemini-1.5-pro

Identifiers sem prefixo (legado da config "openai_model" e do
``PlanFeatures.llm_model`` salvos no DB antes da refator multi-provider) sao
tratados como OpenAI por compatibilidade — evita migration forcada do JSONB
``subscription_plans.features``.

Credentials: ``init_chat_model`` nao recebe ``api_key``. Cada provider le sua
env var canonica:

- OpenAI:    ``OPENAI_API_KEY``
- Anthropic: ``ANTHROPIC_API_KEY``
- Bedrock:   ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY`` (+ region)
- Azure:     ``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT``

Em ``app/config.py`` chamamos ``load_dotenv()`` no import pra garantir que as
keys do ``.env`` cheguem em ``os.environ`` (do contrario o ``init_chat_model``
nao acharia).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model

from app.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def _normalize_model_id(model_id: str) -> str:
    """Garante o formato ``"<provider>:<model>"``.

    Identifiers sem ``":"`` sao prefixados com ``"openai:"`` — preserva
    back-compat com valores antigos (ex: ``"gpt-4o-mini"`` em
    ``PlanFeatures.llm_model``) sem precisar reescrever o seed/migration.
    """
    if ":" in model_id:
        return model_id
    return f"openai:{model_id}"


def get_chat_model(model_id: str, **kwargs: Any) -> BaseChatModel:
    """Instancia um chat model agnostico de provider.

    Args:
        model_id: identificador no formato ``"<provider>:<model>"`` (ex:
            ``"anthropic:claude-3-5-sonnet-latest"``). Quando sem prefixo,
            assume ``openai`` por compatibilidade.
        **kwargs: argumentos repassados ao construtor especifico do provider
            (ex: ``temperature=0``, ``max_tokens=2048``).

    Returns:
        ``BaseChatModel`` ja configurado — pronto pra ``ainvoke`` / ``bind_tools``.
    """
    # Defaults de timeout/retries pra bound o pior caso (callers podem
    # sobrescrever via kwargs). init_chat_model repassa esses kwargs ao
    # construtor do provider (ChatOpenAI/ChatAnthropic aceitam ``timeout`` e
    # ``max_retries``).
    kwargs.setdefault("timeout", settings.chat_timeout_seconds)
    kwargs.setdefault("max_retries", settings.chat_max_retries)
    model: BaseChatModel = init_chat_model(_normalize_model_id(model_id), **kwargs)
    return model
