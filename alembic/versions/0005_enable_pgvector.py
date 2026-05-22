"""enable pgvector extension

Revision ID: 0005_enable_pgvector
Revises: 0004_link_diagnoses_to_diseases
Create Date: 2026-05-22 13:00:00.000000

Habilita a extensao ``pgvector`` no Postgres pra suportar embeddings em
``langgraph.store.postgres.AsyncPostgresStore``. A criacao das tabelas
``store`` e ``store_vectors`` eh feita pelo ``store.setup()`` do
LangGraph no startup do app (nao via Alembic) — esta migration so
garante que a extensao existe.

No Supabase, a extensao ja vem disponivel em ``CREATE EXTENSION``,
nao precisa habilitar via dashboard.
"""

from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0005_enable_pgvector"
down_revision: str | None = "0004_link_diagnoses_to_diseases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
