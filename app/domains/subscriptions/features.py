"""PlanFeatures — schema declarativo das capacidades de cada plano (TCC-049).

Cada plano (free/pro/enterprise) tem um conjunto de features que governa:
- Qual LLM model usar (gpt-4o-mini vs gpt-4o)
- Quais modelos de diagnostico estao disponiveis (resnet50, efficientnet, vit, ensemble)
- Quais niveis de plano de acao podem ser retornados (essencial/campo/especialista)
- Quais cultivos sao permitidos (None = todos)
- Flags booleanas pra search web, search scientific, api access, etc.

O ``signature()`` retorna um hash deterministico do conjunto de features pra
permitir cacheamento de grafos LangGraph compilados por feature-set, sem ter
que recompilar a cada request.

As constantes ``FREE_FEATURES``, ``PRO_FEATURES`` e ``ENTERPRISE_FEATURES``
sao a fonte da verdade pro seed (``scripts/seed_plan_features.py``) e pra
migration de backfill (``alembic/versions/0005_add_plan_features.py``).
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


class PlanFeatures(BaseModel):
    """Capacidades de um plano (free/pro/enterprise).

    Args:
        tier_name: "free" | "pro" | "enterprise".
        llm_model: identificador do modelo LLM (ex: ``gpt-4o-mini``).
        diagnosis_models: modelos CNN/ViT permitidos.
        action_plan_levels: niveis de plano de acao permitidos.
        allowed_crops: lista de slugs de cultivos. ``None`` = todos.
        search_web: tool de busca web habilitada.
        search_scientific: tool de busca em artigos cientificos.
        identify_crop_auto: tool de identificacao automatica de cultivo.
        api_access: chaves de API REST.
        export_diagnoses: export PDF/CSV.
        multi_account: contas vinculadas (org).
    """

    tier_name: str
    llm_model: str = "openai:gpt-4o-mini"
    diagnosis_models: list[str] = ["resnet50"]
    action_plan_levels: list[str] = ["essencial"]
    allowed_crops: list[str] | None = None
    search_web: bool = False
    search_scientific: bool = False
    identify_crop_auto: bool = False
    api_access: bool = False
    export_diagnoses: bool = False
    multi_account: bool = False

    def signature(self) -> str:
        """Hash deterministico das features pra cache de grafo.

        Returns:
            Hex string (16 chars) — colisao improvavel pra dezenas de planos.
        """
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


FREE_FEATURES = PlanFeatures(tier_name="free")
PRO_FEATURES = PlanFeatures(
    tier_name="pro",
    llm_model="openai:gpt-4o",
    diagnosis_models=["resnet50", "efficientnet", "vit"],
    action_plan_levels=["essencial", "campo"],
    allowed_crops=["soja"],
    search_web=True,
    export_diagnoses=True,
)
ENTERPRISE_FEATURES = PlanFeatures(
    tier_name="enterprise",
    llm_model="openai:gpt-4o",
    diagnosis_models=["resnet50", "efficientnet", "vit", "ensemble"],
    action_plan_levels=["essencial", "campo", "especialista"],
    allowed_crops=None,
    search_web=True,
    search_scientific=True,
    identify_crop_auto=True,
    api_access=True,
    export_diagnoses=True,
    multi_account=True,
)


# ── Lookup por nome de plano (consumido pelo seed/backfill) ──────────────────

FEATURES_BY_PLAN_NAME: dict[str, PlanFeatures] = {
    "free": FREE_FEATURES,
    "pro": PRO_FEATURES,
    "enterprise": ENTERPRISE_FEATURES,
}
