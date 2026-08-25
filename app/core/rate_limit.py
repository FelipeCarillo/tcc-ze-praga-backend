"""Rate limit por IP nas rotas de autenticação (TCC-091).

Por que existe: as cotas por plano (``UsageService``) só protegem o que custa
crédito de LLM e só valem para quem já tem conta. ``POST /auth/login`` e
``POST /auth/register`` são anônimos — dava para martelar senha ou criar contas
em série sem nenhum freio.

**Janela deslizante em memória, de propósito.** Um limiter compartilhado exigiria
Redis (dependência e custo novos) ou uma escrita no Postgres por requisição
(caro e barulhento para algo que quase sempre só lê). Como o serviço roda com
``--max-instances 2``, o pior caso é o limite valer o dobro — 2× é ordem de
grandeza certa contra força bruta, que é o que interessa aqui. Se um dia o
serviço escalar de verdade, troque a implementação de ``_Janela``, não os
call sites.

Efeito colateral aceito: reiniciar o container zera as janelas. Para um teto
anti-abuso isso não muda nada.
"""

import asyncio
import time
from collections import defaultdict, deque

from fastapi import Request

from app.core.exceptions import RateLimitedError


def client_ip(request: Request) -> str:
    """IP real do cliente atrás do proxy do Cloud Run.

    O Cloud Run põe a cadeia em ``X-Forwarded-For`` e o **primeiro** item é o
    cliente; ``request.client.host`` traria o IP do balanceador, o que colocaria
    todo mundo no mesmo balde e transformaria o limiter num interruptor geral.
    """
    encaminhado = request.headers.get("x-forwarded-for")
    if encaminhado:
        primeiro = encaminhado.split(",")[0].strip()
        if primeiro:
            return primeiro
    return request.client.host if request.client else "desconhecido"


class _Janela:
    """Janela deslizante: guarda os instantes das batidas recentes por chave."""

    def __init__(self) -> None:
        self._batidas: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def registrar(self, chave: str, limite: int, segundos: int) -> float | None:
        """Registra uma batida.

        Devolve ``None`` quando passou, ou os segundos que faltam para liberar
        quando estourou (para o header ``Retry-After``).
        """
        agora = time.monotonic()
        corte = agora - segundos

        async with self._lock:
            fila = self._batidas[chave]
            while fila and fila[0] <= corte:
                fila.popleft()

            if len(fila) >= limite:
                # A mais antiga da janela é quem define quando abre uma vaga.
                return max(0.0, fila[0] + segundos - agora)

            fila.append(agora)

            # Poda oportunista: sem isso o dicionário cresce para sempre com IPs
            # que bateram uma vez e sumiram — vazamento lento num processo que
            # fica de pé por dias.
            if len(self._batidas) > _MAX_CHAVES:
                self._podar(corte)
            return None

    def _podar(self, corte: float) -> None:
        vazias = [k for k, v in self._batidas.items() if not v or v[-1] <= corte]
        for k in vazias:
            del self._batidas[k]

    def limpar(self) -> None:
        """Zera o estado. Existe para os testes não vazarem contagem entre si."""
        self._batidas.clear()


_MAX_CHAVES = 10_000
_janela = _Janela()


class RateLimit:
    """Dependência do FastAPI que limita batidas por IP numa rota.

    Uso::

        @router.post("/login", dependencies=[Depends(RateLimit(10, 60))])

    O balde é por (rota, IP), então gastar as tentativas de login não consome
    as de cadastro.
    """

    def __init__(self, vezes: int, segundos: int, *, nome: str | None = None) -> None:
        self._vezes = vezes
        self._segundos = segundos
        self._nome = nome

    async def __call__(self, request: Request) -> None:
        escopo = self._nome or request.scope.get("route_path") or request.url.path
        chave = f"{escopo}:{client_ip(request)}"
        faltam = await _janela.registrar(chave, self._vezes, self._segundos)
        if faltam is not None:
            raise RateLimitedError(retry_after=int(faltam) + 1)


def limpar_rate_limit() -> None:
    """Reseta o estado global — usado pelos testes."""
    _janela.limpar()
