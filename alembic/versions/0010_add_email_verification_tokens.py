"""add email_verification_tokens table (TCC-090 — gate de cadastro)

Revision ID: 0010_add_email_verification_tokens
Revises: 0009_add_talhoes
Create Date: 2026-08-24 00:00:00.000000

Strategy:
- Linear depois de 0009_add_talhoes.
- Cria ``email_verification_tokens`` guardando o SHA-256 do token (nunca o
  valor cru), com FK ``user_id`` -> users.id ON DELETE CASCADE.
- Não altera ``users``: a coluna ``is_active`` já existe desde 0001 e é ela que
  o login consulta — a verificação só passa a controlá-la.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010_add_email_verification_tokens"
down_revision: str | Sequence[str] | None = "0009_add_talhoes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "email_verification_tokens",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_email_verification_tokens_user_id", "email_verification_tokens", ["user_id"]
    )
    op.create_index(
        "ix_email_verification_tokens_token_hash",
        "email_verification_tokens",
        ["token_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_email_verification_tokens_token_hash", table_name="email_verification_tokens")
    op.drop_index("ix_email_verification_tokens_user_id", table_name="email_verification_tokens")
    op.drop_table("email_verification_tokens")
