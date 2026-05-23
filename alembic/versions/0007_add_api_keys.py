"""add api_keys table + merge heads

Revision ID: 0007_add_api_keys
Revises: 0006_add_summary_to_chat_sessions, 0005_add_plan_features
Create Date: 2026-05-23 10:00:00.000000

Adiciona ``api_keys`` pra usuarios do plano Enterprise gerarem chaves
de programatica (Sprint A5 — TCC-031). O ``key_hash`` eh bcrypt do
plain text retornado uma unica vez na criacao; o ``key_prefix`` (12
chars) eh indexado pra acelerar lookup em ``verify(plain_key)``.

Tambem **merge** das duas heads paralelas que existiam em ``main``
(``0006_add_summary_to_chat_sessions`` da Sprint A2.5 e
``0005_add_plan_features`` da Sprint A4) num unico ponto.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0007_add_api_keys"
down_revision: str | tuple[str, ...] | None = (
    "0006_add_summary_to_chat_sessions",
    "0005_add_plan_features",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("key_hash", sa.String(), nullable=False),
        sa.Column("key_prefix", sa.String(length=12), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("key_hash"),
    )
    op.create_index(op.f("ix_api_keys_user_id"), "api_keys", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_api_keys_key_prefix"), "api_keys", ["key_prefix"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_api_keys_key_prefix"), table_name="api_keys")
    op.drop_index(op.f("ix_api_keys_user_id"), table_name="api_keys")
    op.drop_table("api_keys")
