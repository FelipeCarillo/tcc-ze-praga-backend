"""fazendas acima dos talhoes (TCC-096)

Revision ID: 0013_fazendas
Revises: 0012_diagnosis_talhao
Create Date: 2026-10-06 00:00:00.000000

Strategy:
- Linear depois de 0012_diagnosis_talhao.
- Cria ``fazendas`` (por usuario) e ``talhoes.fazenda_id``.
- Backfill: cada usuario que ja tem talhao ganha uma fazenda "Minha fazenda" e
  todos os talhoes dele passam a pertencer a ela. So' depois a coluna vira
  NOT NULL — nenhum talhao fica orfao.
- FK talhoes.fazenda_id -> fazendas.id ON DELETE CASCADE: apagar a fazenda
  apaga os talhoes; os laudos sobrevivem sem talhao (FK da 0012, SET NULL).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013_fazendas"
down_revision: str | Sequence[str] | None = "0012_diagnosis_talhao"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fazendas",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("nome", sa.String(), nullable=False),
        sa.Column("municipio", sa.String(), nullable=True),
        sa.Column("uf", sa.String(length=2), nullable=True),
        sa.Column("hectares", sa.Numeric(10, 2), nullable=True),
        sa.Column("agronomo_nome", sa.String(), nullable=True),
        sa.Column("agronomo_crea", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_fazendas_user_id", "fazendas", ["user_id"])

    op.add_column("talhoes", sa.Column("fazenda_id", sa.String(), nullable=True))

    # Uma "Minha fazenda" por usuario que ja tem talhao. O id usa md5 do
    # user_id para ser deterministico sem depender de extensao (uuid-ossp).
    op.execute(
        """
        INSERT INTO fazendas (id, user_id, nome)
        SELECT DISTINCT 'faz-' || md5(t.user_id), t.user_id, 'Minha fazenda'
        FROM talhoes t
        """
    )
    op.execute("UPDATE talhoes SET fazenda_id = 'faz-' || md5(user_id)")

    op.alter_column("talhoes", "fazenda_id", nullable=False)
    op.create_foreign_key(
        "fk_talhoes_fazenda_id",
        "talhoes",
        "fazendas",
        ["fazenda_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_talhoes_fazenda_id", "talhoes", ["fazenda_id"])


def downgrade() -> None:
    op.drop_index("ix_talhoes_fazenda_id", table_name="talhoes")
    op.drop_constraint("fk_talhoes_fazenda_id", "talhoes", type_="foreignkey")
    op.drop_column("talhoes", "fazenda_id")
    op.drop_index("ix_fazendas_user_id", table_name="fazendas")
    op.drop_table("fazendas")
