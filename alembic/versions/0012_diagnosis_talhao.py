"""liga diagnosticos a talhoes (TCC-093)

Revision ID: 0012_diagnosis_talhao
Revises: 0011_password_reset
Create Date: 2026-10-05 00:00:00.000000

Strategy:
- Linear depois de 0011_password_reset.
- ``diagnoses.talhao_id`` nullable: diagnosticos antigos (e os feitos sem
  escolher talhao) continuam validos e caem no grupo "Sem talhao".
- FK -> talhoes.id ON DELETE SET NULL: apagar um talhao nao apaga o historico
  de laudos dele, so' desfaz o vinculo.
- Indice composto (user_id, talhao_id, created_at) atende o historico agrupado,
  que filtra por usuario, particiona por talhao e ordena pelo mais recente.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012_diagnosis_talhao"
down_revision: str | Sequence[str] | None = "0011_password_reset"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("diagnoses", sa.Column("talhao_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_diagnoses_talhao_id",
        "diagnoses",
        "talhoes",
        ["talhao_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_diagnoses_user_talhao_created",
        "diagnoses",
        ["user_id", "talhao_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_diagnoses_user_talhao_created", table_name="diagnoses")
    op.drop_constraint("fk_diagnoses_talhao_id", "diagnoses", type_="foreignkey")
    op.drop_column("diagnoses", "talhao_id")
