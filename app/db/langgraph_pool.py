"""Pool psycopg compartilhado pelo checkpointer e pelo Store do LangGraph.

O ``from_conn_string`` do LangGraph abre **uma** conexão e a segura pela vida
do processo. Se o servidor a derruba — o pooler do Supabase fechando conexão
ociosa, ou o AWS Lambda congelando o processo entre requisições —, toda
chamada seguinte falha com ``the connection is closed`` até a instância
reiniciar. Um pool com ``check`` testa a conexão antes de entregá-la e troca
a quebrada por uma nova, então a falha some sem ninguém perceber.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow
    from psycopg_pool import AsyncConnectionPool

# Poucas conexões por instância: o chat usa checkpointer e Store em série, e o
# pooler do Supabase no plano gratuito tem orçamento curto de conexões.
POOL_MAX_SIZE = 2
# Sem banco no boot, desiste rápido: o warm-up do lifespan é best-effort e o
# Lambda tem 10 s de fase de init.
POOL_OPEN_TIMEOUT_SECONDS = 8.0


async def open_langgraph_pool(
    conn_string: str,
) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    """Abre o pool com os mesmos parâmetros de conexão do ``from_conn_string``."""
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow, dict_row
    from psycopg_pool import AsyncConnectionPool

    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conn_string,
        connection_class=AsyncConnection[DictRow],
        min_size=1,
        max_size=POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,
        open=False,
    )
    await pool.open(wait=True, timeout=POOL_OPEN_TIMEOUT_SECONDS)
    return pool
