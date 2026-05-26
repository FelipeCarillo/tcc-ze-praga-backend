"""Estado do sub-grafo de diagnostico (TCC-040 + TCC-055).

Todos os campos sao opcionais (``total=False``) pra simplificar a montagem
do estado inicial. Os nodes adicionam campos progressivamente:

    Entrada      : user_id, crop_id, image_batch | image_ids, model_id, plan_features
    load_model   : (no-op — placeholder pra futura inicializacao do modelo)
    run_inference: + predictions
    compose_act..: + action_plans            (em paralelo com gather_evidence)
    gather_evid..: + evidence_per_image      (em paralelo com compose_action_plan)
    persist      : + persisted_ids

``errors`` acumula falhas por imagem se algum node decidir nao abortar.

TCC-055 adicionou:
- ``plan_features``: dict serializado das PlanFeatures do plano ativo. Le pelo
  ``gather_evidence_node`` pra decidir quais buscas externas rodar (Free skip,
  Pro=web, Enterprise=web+scientific). Permanece dict (nao PlanFeatures
  pydantic) pra evitar acoplar o sub-grafo ao schema do plano.
- ``evidence_per_image``: list[list[dict]] com index alinhado com ``predictions``.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

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
    plan_features: dict[str, Any]  # TCC-055 — serializado das PlanFeatures (le no gather_evidence)

    # ── opcional (caso o invocador queira propagar mensagens) ────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── progressivos ──────────────────────────────────────────────────────────
    predictions: list[dict[str, Any]]  # [{disease_id, confidence, top_k_logits}, ...]
    diseases: list[dict[str, Any]]  # resolved Disease DTOs
    action_plans: list[dict[str, Any]]  # plans por imagem
    evidence_per_image: list[list[dict[str, Any]]]  # TCC-055 — evidencia externa por predicao
    persisted_ids: list[str]  # ids dos diagnoses criados
    errors: list[dict[str, Any]]  # erros por imagem
