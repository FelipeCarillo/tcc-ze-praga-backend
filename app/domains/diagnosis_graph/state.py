"""Estado do sub-grafo de diagnostico (TCC-040).

Todos os campos sao opcionais (``total=False``) pra simplificar a montagem
do estado inicial. Os nodes adicionam campos progressivamente:

    Entrada      : user_id, crop_id, image_batch | image_ids, model_id
    load_model   : (no-op — placeholder pra futura inicializacao do modelo)
    run_inference: + predictions
    compose_act..: + action_plans
    persist      : + persisted_ids

``errors`` acumula falhas por imagem se algum node decidir nao abortar.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DiagnosisState(TypedDict, total=False):
    """Estado do sub-grafo de diagnostico (1 invocacao = 1 batch de imagens)."""

    # ── input ─────────────────────────────────────────────────────────────────
    user_id: str
    crop_id: str
    image_batch: list[str]  # base64 ou storage_keys
    image_ids: list[str]
    model_id: str

    # ── opcional (caso o invocador queira propagar mensagens) ────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── progressivos ──────────────────────────────────────────────────────────
    predictions: list[dict]  # [{disease_id, confidence, top_k_logits}, ...]
    diseases: list[dict]  # resolved Disease DTOs
    action_plans: list[dict]  # plans por imagem
    persisted_ids: list[str]  # ids dos diagnoses criados
    errors: list[dict]  # erros por imagem
