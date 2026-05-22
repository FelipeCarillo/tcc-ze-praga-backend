"""add summary_text to chat_sessions

Revision ID: 0006_add_summary_to_chat_sessions
Revises: 0005_enable_pgvector
Create Date: 2026-05-22 14:00:00.000000

Adiciona ``summary_text`` em ``chat_sessions`` pra armazenar o resumo
final da conversa quando o usuario chama ``POST /sessions/{id}/close``.
O mesmo conteudo eh tambem indexado no Store sob namespace
``("user", uid, "session_summaries")`` pra busca semantica futura.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0006_add_summary_to_chat_sessions"
down_revision: str | None = "0005_enable_pgvector"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("summary_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "summary_text")
