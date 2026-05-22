"""Seed PlanFeatures por nome de plano (TCC-049).

Idempotente: faz UPDATE in-place pra cada plano (free/pro/enterprise) com o
``PlanFeatures`` correspondente de ``app/domains/subscriptions/features.py``.
Se a coluna ``features`` ja' contem o JSON correto (mesma signature), pula.

Diferente da migration ``0005_add_plan_features.py`` (que so' roda 1 vez),
este seed pode ser re-executado a qualquer momento — util quando os defaults
de PlanFeatures evoluem entre sprints.

Usage:
    uv run python -m scripts.seed_plan_features
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.domains.subscriptions.features import FEATURES_BY_PLAN_NAME
from app.models.subscription_plan import SubscriptionPlan


async def seed_plan_features(db: AsyncSession) -> dict[str, str]:
    """Upsert features por nome de plano. Retorna name -> 'created'|'updated'|'skipped'."""
    print("Seeding plan features...")
    result_status: dict[str, str] = {}

    for plan_name, features in FEATURES_BY_PLAN_NAME.items():
        result = await db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.name == plan_name)
        )
        plan = result.scalar_one_or_none()
        if plan is None:
            print(f"  ! Plan '{plan_name}' nao existe no DB — skip (rode seed dos planos antes).")
            result_status[plan_name] = "missing"
            continue

        desired = features.model_dump()
        if plan.features == desired:
            print(f"  = Plan '{plan_name}' features ja atualizadas — skip.")
            result_status[plan_name] = "skipped"
            continue

        plan.features = desired
        print(f"  ~ Plan '{plan_name}': features atualizadas (signature={features.signature()})")
        result_status[plan_name] = "updated"

    await db.commit()
    return result_status


async def main() -> None:
    print("Starting seed_plan_features...\n")
    async with AsyncSessionLocal() as db:
        await seed_plan_features(db)
    print("\nSeed completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
