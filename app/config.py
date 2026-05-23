from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_embeddings_model: str = "text-embedding-3-small"
    openai_embeddings_dims: int = 1536
    openai_vision_model: str = "gpt-4o"

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
