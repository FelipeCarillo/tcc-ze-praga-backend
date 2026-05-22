"""Memory helpers para indexar diagnoses + summaries no Store (TCC-045).

Estes helpers ficam fora do hot-path do agente — sao chamados pelo
``persist_node`` do diagnosis_graph apos persistir um diagnosis, e pelo
endpoint ``POST /sessions/{id}/close`` apos gerar o summary da sessao.

O texto de summary eh curto (~50 tokens) para minimizar o custo de embedding
e a latencia. O ``store.aput`` ja gera embeddings automaticamente para os
``fields=["summary_text"]`` configurados em ``app/db/store.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from app.domains.diagnoses.dto import DiagnosisDTO


def format_diagnosis_summary(diagnosis: DiagnosisDTO | Any) -> str:
    """Constroi um texto curto (~50 tokens) descrevendo o diagnosis pra embedding.

    Esse texto eh o que vai pro ``summary_text`` indexado no Store e
    serve de chave semantica pra busca futura (``search_my_diagnoses``).

    Args:
        diagnosis: ``DiagnosisDTO`` ou objeto com os atributos
            ``disease_name``, ``disease_id``, ``confidence``, ``severity``
            e ``created_at``.

    Returns:
        String descrevendo o diagnostico de forma compacta.
    """
    disease_name = getattr(diagnosis, "disease_name", "doenca desconhecida")
    disease_id = getattr(diagnosis, "disease_id", "")
    confidence = getattr(diagnosis, "confidence", 0.0)
    severity = getattr(diagnosis, "severity", "")
    created_at = getattr(diagnosis, "created_at", None)

    confidence_pct = (
        f"{float(confidence) * 100:.1f}%"
        if confidence is not None
        else "?"
    )
    date_str = (
        created_at.strftime("%Y-%m-%d")
        if isinstance(created_at, datetime)
        else str(created_at) if created_at else "?"
    )
    return (
        f"Diagnostico de {disease_name} ({disease_id}) "
        f"com {confidence_pct} confianca, severidade {severity}, em {date_str}."
    )


async def index_diagnosis_in_store(
    store: BaseStore,
    user_id: str,
    diagnosis: DiagnosisDTO | Any,
) -> None:
    """Indexa o diagnosis no Store pra busca semantica futura.

    Namespace: ``("user", user_id, "diagnoses")``. A chave eh o
    ``diagnosis.id``, garantindo idempotencia (re-index sobrescreve).

    Args:
        store: ``BaseStore`` (em prod, ``AsyncPostgresStore``).
        user_id: dono do diagnostico.
        diagnosis: DTO ou objeto com ``id``, ``disease_name``,
            ``disease_id``, ``crop_id`` (opcional), ``confidence``,
            ``severity``, ``created_at``.
    """
    summary_text = format_diagnosis_summary(diagnosis)
    diagnosis_id = getattr(diagnosis, "id", None)
    if not diagnosis_id:
        # Defensivo — diagnosis sem id nao pode ser indexado.
        return

    created_at = getattr(diagnosis, "created_at", None)
    created_at_iso = (
        created_at.isoformat()
        if isinstance(created_at, datetime)
        else str(created_at) if created_at else None
    )

    value = {
        "summary_text": summary_text,
        "diagnosis_id": diagnosis_id,
        "crop_id": getattr(diagnosis, "crop_id", None),
        "disease_id": getattr(diagnosis, "disease_id", None),
        "disease_name": getattr(diagnosis, "disease_name", None),
        "confidence": float(getattr(diagnosis, "confidence", 0.0)),
        "severity": getattr(diagnosis, "severity", None),
        "created_at": created_at_iso,
    }

    await store.aput(
        namespace=("user", user_id, "diagnoses"),
        key=diagnosis_id,
        value=value,
        index=["summary_text"],
    )


async def index_session_summary_in_store(
    store: BaseStore,
    user_id: str,
    session_id: str,
    summary_text: str,
) -> None:
    """Indexa um resumo de sessao no Store.

    Namespace: ``("user", user_id, "session_summaries")``. Chave eh o
    ``session_id`` pra permitir re-index ao fechar a mesma sessao
    novamente (idempotente).
    """
    if not summary_text:
        return
    await store.aput(
        namespace=("user", user_id, "session_summaries"),
        key=session_id,
        value={
            "summary_text": summary_text,
            "session_id": session_id,
        },
        index=["summary_text"],
    )
