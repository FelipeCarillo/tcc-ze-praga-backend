from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Exporta as vars do ``.env`` pra ``os.environ`` no import — necessario pra que
# o ``langchain.chat_models.init_chat_model`` encontre credenciais provider-
# specific (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``AWS_*``, etc.) que ele
# le direto de ``os.environ``, nao via ``Settings``.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str

    # Supabase (Storage only)
    supabase_url: str
    supabase_service_role_key: str

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # Chat models — provider-agnostic IDs no formato ``"<provider>:<model>"``.
    # Suporta openai, anthropic, bedrock_converse, azure_openai, google_vertexai,
    # etc. via ``langchain.chat_models.init_chat_model``.
    chat_model: str = "openai:gpt-4o-mini"
    vision_model: str = "openai:gpt-4o"

    # OpenAI — ainda usado direto pra embeddings (ate hoje LangChain nao tem
    # ``init_embeddings`` agnostico canonico). ``openai_api_key`` permanece
    # legado pra outros caminhos; pra chat use ``OPENAI_API_KEY`` em ``.env``.
    openai_api_key: str | None = None
    openai_embeddings_model: str = "text-embedding-3-small"
    openai_embeddings_dims: int = 1536

    # Agent feature flags — kill-switches deploy-time pras tools V2 dormentes.
    # Default OFF; ative junto com a feature do plano (ex: identify_crop_auto)
    # pra liberar a tool no registry.
    agent_enable_identify_crop: bool = False

    # Tavily (search_web tool — TCC-053)
    tavily_api_key: str | None = None

    # Agent tool flags (TCC-053 / TCC-054)
    agent_enable_search_web: bool = True
    agent_enable_search_scientific: bool = True

    # App
    app_env: str = "development"
    allowed_origins: str = "http://localhost:3000"

    # Agent feature flags
    agent_enable_ask_user: bool = True

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


settings = Settings()  # type: ignore[call-arg]
