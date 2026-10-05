"""Provider-agnostic chat model factory.

Centraliza ``langchain.chat_models.init_chat_model`` pra que o resto do app nao
amarre em provider especifico (OpenAI, Anthropic, Bedrock, Azure, etc.).

Model ID format: ``"<provider>:<model>"`` — ex:

    openai:gpt-4o-mini
    anthropic:claude-3-5-sonnet-latest
    bedrock_converse:anthropic.claude-3-5-sonnet-20240620-v1:0
    azure_openai:gpt-4o
    google_vertexai:gemini-1.5-pro
    google_genai:gemini-2.5-flash

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
- Gemini:    ``GOOGLE_API_KEY``

Trocar de modelo ou de provider é só mudar ``CHAT_MODEL``/``VISION_MODEL``/
``EMBEDDINGS_MODEL`` no ambiente (e instalar o pacote ``langchain-<provider>``
se for um provider ainda não listado no ``pyproject.toml``).

Em ``app/config.py`` chamamos ``load_dotenv()`` no import pra garantir que as
keys do ``.env`` cheguem em ``os.environ`` (do contrario o ``init_chat_model``
nao acharia).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model

from app.config import settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
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
    # Temperatura só vai ao provider quando configurada: modelos de raciocínio
    # (gpt-5, o-series) rejeitam valor diferente do default.
    if settings.llm_temperature is not None:
        kwargs.setdefault("temperature", settings.llm_temperature)
    model: BaseChatModel = init_chat_model(_normalize_model_id(model_id), **kwargs)
    return model


def get_embeddings(model_id: str | None = None) -> Embeddings:
    """Instancia o modelo de embeddings agnóstico de provider.

    Mesmo formato ``"<provider>:<model>"`` do chat (ex:
    ``openai:text-embedding-3-small``, ``google_genai:gemini-embedding-001``).
    Default: ``settings.embeddings_model``.
    """
    from langchain.embeddings import init_embeddings

    return init_embeddings(_normalize_model_id(model_id or settings.embeddings_model))


def message_text(message: Any) -> str:
    """Texto de uma resposta de LLM, qualquer que seja o provider.

    OpenAI (Chat Completions) devolve ``content`` como string; Anthropic,
    Gemini e a Responses API da OpenAI devolvem lista de blocos, que pode
    incluir blocos de raciocínio e de tool call. Aqui só entram os blocos de
    texto — ler ``content`` como string devolvia ``""`` nesses providers.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)
