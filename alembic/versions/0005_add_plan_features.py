"""add features JSONB column to subscription_plans + backfill por tier

Revision ID: 0005_add_plan_features
Revises: 0004_link_diagnoses_to_diseases
Create Date: 2026-05-22 13:00:00.000000

Strategy:
- Add ``features`` JSONB nullable to ``subscription_plans``.
- Backfill the 3 default rows (free/pro/enterprise) with hardcoded JSON that
  matches ``app/domains/subscriptions/features.py``. The JSON is duplicado
  here on purpose — migrations precisam ser self-contained (nao importam
  app.* pra evitar acoplamento em compile-time).
- Column stays nullable: novos planos podem ser criados sem features e o
  seed (``scripts/seed_plan_features.py``) preenche idempotentemente.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "0005_add_plan_features"
down_revision: str | None = "0004_link_diagnoses_to_diseases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Hardcoded JSON pra backfill — espelha PlanFeatures defaults em features.py.
_FREE_JSON = """{
  "tier_name": "free",
  "llm_model": "gpt-4o-mini",
  "diagnosis_models": ["resnet50"],
  "action_plan_levels": ["essencial"],
  "allowed_crops": null,
  "search_web": false,
  "search_scientific": false,
  "identify_crop_auto": false,
  "api_access": false,
  "export_diagnoses": false,
  "multi_account": false
}"""

_PRO_JSON = """{
  "tier_name": "pro",
  "llm_model": "gpt-4o",
  "diagnosis_models": ["resnet50", "efficientnet", "vit"],
  "action_plan_levels": ["essencial", "campo"],
  "allowed_crops": ["soja"],
  "search_web": true,
  "search_scientific": false,
  "identify_crop_auto": false,
  "api_access": false,
  "export_diagnoses": true,
  "multi_account": false
}"""

_ENTERPRISE_JSON = """{
  "tier_name": "enterprise",
  "llm_model": "gpt-4o",
  "diagnosis_models": ["resnet50", "efficientnet", "vit", "ensemble"],
  "action_plan_levels": ["essencial", "campo", "especialista"],
  "allowed_crops": null,
  "search_web": true,
  "search_scientific": true,
  "identify_crop_auto": true,
  "api_access": true,
  "export_diagnoses": true,
  "multi_account": true
}"""


def upgrade() -> None:
    op.add_column(
        "subscription_plans",
        sa.Column("features", JSONB(), nullable=True),
    )

    # Backfill per plan name (idempotente — UPDATE so' altera linhas existentes).
    op.execute(
        f"""
        UPDATE subscription_plans
        SET features = '{_FREE_JSON}'::jsonb
        WHERE name = 'free' AND features IS NULL
        """
    )
    op.execute(
        f"""
        UPDATE subscription_plans
        SET features = '{_PRO_JSON}'::jsonb
        WHERE name = 'pro' AND features IS NULL
        """
    )
    op.execute(
        f"""
        UPDATE subscription_plans
        SET features = '{_ENTERPRISE_JSON}'::jsonb
        WHERE name = 'enterprise' AND features IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("subscription_plans", "features")
