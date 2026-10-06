"""liga RLS em todas as tabelas do schema public (seguranca)

Revision ID: 0014_enable_rls
Revises: 0013_fazendas
Create Date: 2026-10-06 00:00:00.000000

Strategy:
- No Supabase, o schema ``public`` e' exposto pela API REST dele (PostgREST)
  e os papeis ``anon``/``authenticated`` tinham todos os grants: com a chave
  publica dava para ler e alterar users (hash de senha), laudos, tokens etc.
- Liga RLS sem politicas: ``anon``/``authenticated`` passam a nao ver nada.
  O backend conecta como dono das tabelas (``postgres``), e dono ignora RLS
  (nao usamos FORCE), entao a API nao muda. O Storage usa a service role,
  que tambem ignora RLS.
- Varre as tabelas existentes (inclui as do checkpointer/store do LangGraph,
  criadas fora do Alembic). Tabelas criadas depois precisam ligar RLS na
  propria migration.
- Aplicada em producao em 06/10/2026 via MCP, junto com o bump da versao.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014_enable_rls"
down_revision: str | Sequence[str] | None = "0013_fazendas"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LOOP = """
DO $$
DECLARE t record;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
    EXECUTE format('ALTER TABLE public.%I {acao} ROW LEVEL SECURITY', t.tablename);
  END LOOP;
END $$;
"""


def upgrade() -> None:
    op.execute(_LOOP.format(acao="ENABLE"))


def downgrade() -> None:
    op.execute(_LOOP.format(acao="DISABLE"))
