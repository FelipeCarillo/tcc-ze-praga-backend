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

    # App
    app_env: str = "development"
    allowed_origins: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


settings = Settings()  # type: ignore[call-arg]
