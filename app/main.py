from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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
from app.domains.auth.router import router as auth_router
from app.domains.diagnoses.router import router as diagnoses_router
from app.domains.subscriptions.router import router as subscriptions_router
from app.domains.usage.router import router as usage_router
from app.domains.users.router import router as users_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield  # pragma: no cover


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
app.include_router(users_router, prefix=API_PREFIX)
app.include_router(diagnoses_router, prefix=API_PREFIX)
app.include_router(action_plans_router, prefix=API_PREFIX)
app.include_router(subscriptions_router, prefix=API_PREFIX)
app.include_router(usage_router, prefix=API_PREFIX)


@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health() -> dict:
    return {"status": "healthy", "version": "1.0.0"}
