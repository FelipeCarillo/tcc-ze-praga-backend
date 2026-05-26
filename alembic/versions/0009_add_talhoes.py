"""add talhoes table (Fase 5B — redesign)

Revision ID: 0009_add_talhoes
Revises: 0008_add_sources_to_diagnoses
Create Date: 2026-05-25 00:00:00.000000

Strategy:
- Linear depois de 0008_add_sources_to_diagnoses.
- Cria a tabela ``talhoes`` (áreas de cultivo cadastradas pelo produtor),
  com FK ``user_id`` -> users.id ON DELETE CASCADE e índice por usuário.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0009_add_talhoes"
down_revision: str | Sequence[str] | None = "0008_add_sources_to_diagnoses"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "talhoes",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("nome", sa.String(), nullable=False),
        sa.Column("apelido", sa.String(), nullable=True),
        sa.Column("hectares", sa.Numeric(8, 2), nullable=True),
        sa.Column(
            "cultura", sa.String(), nullable=False, server_default="soja"
        ),
        sa.Column("data_semeadura", sa.Date(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_talhoes_user_id", "talhoes", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_talhoes_user_id", table_name="talhoes")
    op.drop_table("talhoes")
