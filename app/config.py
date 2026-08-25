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

    # Limites do cliente LLM — sem isso o SDK pode pendurar/retentar muito além
    # do esperado (causa do timeout de 30s no 1º turno). Bound o pior caso e
    # troca um hang por um erro claro. Overridáveis via env.
    chat_timeout_seconds: int = 60
    chat_max_retries: int = 2

    # OpenAI — ainda usado direto pra embeddings (ate hoje LangChain nao tem
    # ``init_embeddings`` agnostico canonico). ``openai_api_key`` permanece
    # legado pra outros caminhos; pra chat use ``OPENAI_API_KEY`` em ``.env``.
    openai_api_key: str | None = None
    openai_embeddings_model: str = "text-embedding-3-small"
    openai_embeddings_dims: int = 1536

    # Transcrição de áudio (STT) — entrada de voz no chat (TCC-081).
    transcription_model: str = "whisper-1"

    # Anthropic — usado quando ``chat_model``/``vision_model`` apontam pra
    # ``anthropic:...``. Como ``openai_api_key``/``tavily_api_key``, declarado
    # como campo opcional pra que uma ``ANTHROPIC_API_KEY`` no ``.env`` não
    # estoure o ``extra=forbid`` do BaseSettings (langchain ainda lê de os.environ).
    anthropic_api_key: str | None = None

    # Agent feature flags — kill-switches deploy-time pras tools V2 dormentes.
    # Default OFF; ative junto com a feature do plano (ex: identify_crop_auto)
    # pra liberar a tool no registry.
    agent_enable_identify_crop: bool = False

    # Tavily (search_web tool — TCC-053)
    tavily_api_key: str | None = None

    # Agent tool flags (TCC-053 / TCC-054)
    agent_enable_search_web: bool = True
    agent_enable_search_scientific: bool = True

    # ─── E-mail (Resend) — verificação de cadastro ────────────────────────────
    # Sem ``resend_api_key`` o sender cai no NullEmailSender (loga e não envia),
    # mesmo padrão de graceful degradation do InferenceService. Isso mantém
    # dev e testes funcionando sem credencial.
    resend_api_key: str | None = None
    email_from: str = "Zé Praga <onboarding@resend.dev>"
    email_verification_ttl_hours: int = 24

    # TTL do link de redefinição de senha. Bem menor que o de verificação: este
    # token troca a credencial de acesso, então quanto menor a janela, melhor.
    password_reset_ttl_hours: int = 2

    # Gate de cadastro: com ``True`` o usuário nasce inativo e só o link do
    # e-mail o ativa. Default OFF pra não quebrar dev/testes — ligue em produção.
    require_email_verification: bool = False

    # URL pública do frontend — destino do redirect pós-verificação.
    frontend_url: str = "http://localhost:3000"

    # URL pública da própria API — o link do e-mail aponta pra cá, não pro
    # frontend: quem abre é o navegador vindo do cliente de e-mail, e o
    # endpoint já devolve 303 pro frontend. Evita uma página só pra isso.
    public_api_url: str = "http://localhost:8000"

    # App
    app_env: str = "development"
    allowed_origins: str = "http://localhost:3000"

    # Agent feature flags
    # HITL ligado: o ciclo esta fechado ponta a ponta — a tool dispara
    # interrupt(), o checkpointer persiste o snapshot, o frontend renderiza a
    # pergunta (InterruptPrompt) e retoma via POST /chat/resume/stream.
    agent_enable_ask_user: bool = True

    # Inferência ONNX (TCC-023 / ADR-0003) — modelo real treinado no ASDID.
    # Default ON; se o arquivo ou o onnxruntime faltarem, cai no mock
    # automaticamente (graceful fallback no factory get_inference_service).
    inference_use_onnx: bool = True
    inference_onnx_model_path: str = "models/soja_efficientnet_b4.onnx"
    inference_onnx_input_size: int = 380

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


settings = Settings()  # type: ignore[call-arg]
