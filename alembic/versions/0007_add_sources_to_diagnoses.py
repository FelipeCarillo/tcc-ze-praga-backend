"""add sources JSONB column to diagnoses (TCC-056)

Revision ID: 0007_add_sources_to_diagnoses
Revises: 0005_add_plan_features, 0006_add_summary_to_chat_sessions
Create Date: 2026-05-23 16:00:00.000000

Strategy:
- Merge das duas heads (0005_add_plan_features + 0006_add_summary_to_chat_sessions)
  pra resolver branch divergente.
- Add ``sources`` JSONB column nullable com default ``[]`` em ``diagnoses``.
- Existing rows recebem ``'[]'::jsonb`` via server_default no Postgres.
- Schema do payload eh validado em camada de aplicacao
  (``DiagnosisSourceSchema``), nao no DB — JSONB permite evolucao do
  schema sem migration.

Schema dos items (Pydantic ``DiagnosisSourceSchema``):
    {
        "type": "web" | "scientific",
        "url": "...",
        "title": "...",
        "snippet": "..." | null,
        "doi": "..." | null
    }
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "0007_add_sources_to_diagnoses"
down_revision: str | Sequence[str] | None = (
    "0005_add_plan_features",
    "0006_add_summary_to_chat_sessions",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "diagnoses",
        sa.Column(
            "sources",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("diagnoses", "sources")
