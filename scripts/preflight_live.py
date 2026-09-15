"""Valida as integrações reais usadas pelo modo local do Zé Praga.

Este comando não troca nenhuma configuração por mock. Ele usa o ``.env``
normal para confirmar que banco Supabase, Storage, ONNX e OpenAI estão
acessíveis antes de abrir a API para uma apresentação.

Uso: ``uv run python -m scripts.preflight_live``
"""

from __future__ import annotations

import asyncio
import io
import selectors
import uuid

from PIL import Image
from sqlalchemy import text


def _ok(message: str) -> None:
    print(f"[ok] {message}", flush=True)


async def _verify_database() -> None:
    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
        crops = (await session.execute(text("SELECT count(*) FROM crops"))).scalar_one()
        diseases = (await session.execute(text("SELECT count(*) FROM diseases"))).scalar_one()
        plans = (
            await session.execute(text("SELECT count(*) FROM subscription_plans"))
        ).scalar_one()

    if not crops or not diseases or not plans:
        raise RuntimeError(
            "Catálogos incompletos no Supabase. Rode as migrations e os seeds antes de subir."
        )
    _ok(f"Supabase Database respondendo ({crops} culturas, {diseases} doenças, {plans} planos)")


async def _verify_storage() -> None:
    from app.db.storage import get_storage_client

    client = get_storage_client()
    buckets = await asyncio.to_thread(client.storage.list_buckets)
    if not any(bucket.name == "uploads" for bucket in buckets):
        raise RuntimeError("Bucket privado 'uploads' não existe no Supabase Storage.")

    path = f"_healthchecks/{uuid.uuid4().hex}.txt"
    bucket = client.storage.from_("uploads")
    try:
        await asyncio.to_thread(
            bucket.upload,
            path,
            b"ze-praga live preflight",
            {"content-type": "text/plain"},
        )
    finally:
        await asyncio.to_thread(bucket.remove, [path])
    _ok("Supabase Storage gravou e removeu um arquivo temporário no bucket privado uploads")


def _sample_image() -> bytes:
    image = Image.new("RGB", (64, 64), color=(54, 128, 54))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


async def _verify_onnx() -> None:
    from app.config import settings
    from app.core.dependencies import _ONNX_MODELS, _get_onnx_classifiers

    if not settings.inference_use_onnx:
        raise RuntimeError("INFERENCE_USE_ONNX precisa estar true no modo local real.")

    classifiers = _get_onnx_classifiers()
    missing = set(_ONNX_MODELS) - set(classifiers)
    if missing:
        raise RuntimeError(f"Modelos ONNX ausentes: {', '.join(sorted(missing))}")

    image_bytes = _sample_image()
    for name, classifier in classifiers.items():
        result = await asyncio.to_thread(classifier.predict, image_bytes, 3)
        if len(result) != 3:
            raise RuntimeError(f"Modelo ONNX {name} retornou top-{len(result)}, esperado top-3.")
    _ok(f"Inferência ONNX real executada nos {len(classifiers)} modelos configurados")


async def _verify_openai() -> None:
    from app.config import settings
    from app.core.llm import get_chat_model
    from langchain_openai import OpenAIEmbeddings

    chat_response = await get_chat_model(settings.chat_model).ainvoke("Responda apenas: OK")
    if not str(chat_response.content).strip():
        raise RuntimeError("O modelo de chat da OpenAI respondeu sem conteúdo.")

    vision_response = await get_chat_model(settings.vision_model).ainvoke("Responda apenas: OK")
    if not str(vision_response.content).strip():
        raise RuntimeError("O modelo de visão da OpenAI respondeu sem conteúdo.")

    embedding = await OpenAIEmbeddings(
        model=settings.openai_embeddings_model,
    ).aembed_query("verificação de prontidão do Zé Praga")
    if len(embedding) != settings.openai_embeddings_dims:
        raise RuntimeError(
            "Dimensão de embedding inesperada: "
            f"{len(embedding)} != {settings.openai_embeddings_dims}."
        )
    _ok("OpenAI respondeu com chat, visão e embeddings reais")


async def _verify_langgraph_postgres() -> None:
    from app.db.checkpointer import close_checkpointer, get_checkpointer
    from app.db.store import close_store, get_store

    try:
        await get_checkpointer()
        await get_store()
    finally:
        await close_store()
        await close_checkpointer()
    _ok("Checkpointer e memória semântica conectaram ao Postgres do Supabase")


async def _main() -> None:
    await _verify_database()
    await _verify_storage()
    await _verify_onnx()
    await _verify_openai()
    await _verify_langgraph_postgres()
    print("\nPronto: o modo local está usando Supabase, ONNX e OpenAI reais.", flush=True)


def main() -> None:
    # psycopg async usado pelo LangGraph não suporta o ProactorEventLoop padrão
    # do Windows; a API tem a mesma configuração em scripts.serve_local.
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
