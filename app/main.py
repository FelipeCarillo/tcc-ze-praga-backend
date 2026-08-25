import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    QuotaExceededError,
    UnauthorizedError,
)
from app.domains.action_plans.router import router as action_plans_router
from app.domains.auth.api_key_router import router as api_keys_router
from app.domains.auth.router import router as auth_router
from app.domains.chat.router import router as chat_router
from app.domains.chat.router import sessions_router as chat_sessions_router
from app.domains.diagnoses.router import router as diagnoses_router
from app.domains.inference.router import router as inference_router
from app.domains.subscriptions.router import router as subscriptions_router
from app.domains.talhoes.router import router as talhoes_router
from app.domains.uploads.router import router as uploads_router
from app.domains.usage.router import router as usage_router
from app.domains.users.router import router as users_router

# Sem isso os loggers de ``app.*`` propagam pra um root sem handler e somem —
# o uvicorn só configura os loggers dele. Valia pro ``logger.exception`` do
# warm-up abaixo e vale pros logs do container no deploy.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Warm-up: inicializa os singletons (roda o setup() de tabelas/índices no
    # Postgres) no boot, não no 1º request do usuário — evita o cold start que
    # estourava o timeout de 30s do chat. Best-effort: se Postgres/OpenAI
    # estiverem indisponíveis no boot, segue lazy no 1º request.
    from app.db.checkpointer import get_checkpointer
    from app.db.store import get_store

    try:
        await get_checkpointer()  # AsyncPostgresSaver.setup()
        await get_store()  # AsyncPostgresStore.setup() + embeddings
    except Exception:  # noqa: BLE001 — boot resiliente
        logging.getLogger(__name__).exception(
            "Warm-up de singletons falhou — seguirá lazy no 1º request"
        )

    yield

    # Lifespan shutdown — fecha singletons (Store + Checkpointer).
    from app.db.checkpointer import close_checkpointer
    from app.db.store import close_store

    await close_store()
    await close_checkpointer()


app = FastAPI(
    title="Zé Praga API",
    description="Core backend service for plant disease diagnosis",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.detail})


@app.exception_handler(UnauthorizedError)
async def unauthorized_handler(request: Request, exc: UnauthorizedError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": exc.detail})


@app.exception_handler(ForbiddenError)
async def forbidden_handler(request: Request, exc: ForbiddenError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": exc.detail})


@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": exc.detail})


@app.exception_handler(QuotaExceededError)
async def quota_exceeded_handler(request: Request, exc: QuotaExceededError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": exc.detail,
            "feature": exc.feature,
            "limit": exc.limit,
            "used": exc.used,
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────

API_PREFIX = "/api/v1"

app.include_router(auth_router, prefix=API_PREFIX)
app.include_router(api_keys_router, prefix=API_PREFIX)
app.include_router(users_router, prefix=API_PREFIX)
app.include_router(diagnoses_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)
app.include_router(chat_router, prefix=API_PREFIX)
app.include_router(chat_sessions_router, prefix=API_PREFIX)
app.include_router(action_plans_router, prefix=API_PREFIX)
app.include_router(subscriptions_router, prefix=API_PREFIX)
app.include_router(talhoes_router, prefix=API_PREFIX)
app.include_router(uploads_router, prefix=API_PREFIX)
app.include_router(usage_router, prefix=API_PREFIX)


@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health() -> dict[str, Any]:
    return {"status": "healthy", "version": "1.0.0"}
