"""O grafo do chat monta as tools pelo ``tool_registry`` (nao mais lista fixa).

Antes disso o ``ChatService`` chamava ``build_chat_tools`` e sempre subia com 4
tools, enquanto o registry — com o gate por tier — so' era exercitado em testes.
Estes casos travam o novo contrato: o tool set do grafo varia com o plano.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.chat.service import ChatService
from app.domains.subscriptions.features import (
    ENTERPRISE_FEATURES,
    FREE_FEATURES,
    PRO_FEATURES,
    PlanFeatures,
)


@pytest.fixture
def svc() -> ChatService:
    graph_factory = MagicMock(return_value=MagicMock())
    return ChatService(
        session_repo=AsyncMock(),
        message_repo=AsyncMock(),
        inference_svc=MagicMock(),
        action_plan_svc=AsyncMock(),
        diagnosis_svc=AsyncMock(),
        store_factory=AsyncMock(return_value=MagicMock()),
        diagnosis_graph_factory=graph_factory,
    )


def _names(svc: ChatService, features: PlanFeatures) -> list[str]:
    from app.domains.chat.tool_registry import build_tools

    tools = build_tools(svc._build_tool_factories(), features.model_dump())
    return [t.name for t in tools]


def test_free_nao_ve_tools_pagas(svc: ChatService) -> None:
    names = _names(svc, FREE_FEATURES)
    assert "analyze_image" in names
    assert "inspect_image" in names
    assert "search_my_diagnoses" in names
    for paga in ("search_web", "search_scientific", "compare_diagnoses"):
        assert paga not in names


def test_pro_ganha_search_web(svc: ChatService) -> None:
    names = _names(svc, PRO_FEATURES)
    assert "search_web" in names
    assert "search_scientific" not in names
    assert "compare_diagnoses" not in names


def test_enterprise_ve_tudo(svc: ChatService) -> None:
    names = _names(svc, ENTERPRISE_FEATURES)
    for tool_name in (
        "inspect_image",
        "analyze_image",
        "deep_diagnose",
        "get_disease_info",
        "get_action_plan",
        "search_my_diagnoses",
        "compare_diagnoses",
        "search_web",
        "search_scientific",
    ):
        assert tool_name in names


def test_sem_graph_factory_as_tools_de_subgrafo_ficam_fora() -> None:
    """DI parcial nao pode quebrar o chat — as duas tools somem, o resto sobe."""
    svc = ChatService(
        session_repo=AsyncMock(),
        message_repo=AsyncMock(),
        inference_svc=MagicMock(),
        action_plan_svc=AsyncMock(),
        diagnosis_svc=AsyncMock(),
    )
    factories = svc._build_tool_factories()
    assert "deep_diagnose" not in factories
    assert "compare_diagnoses" not in factories
    assert "analyze_image" in factories

    names = _names(svc, ENTERPRISE_FEATURES)
    assert "deep_diagnose" not in names
    assert "analyze_image" in names


async def test_cache_de_grafo_e_por_assinatura_de_features(
    svc: ChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dois planos com o mesmo tier mas features diferentes nao compartilham grafo.

    O cache antigo era chaveado so' pelo ``tier_name``, o que passou a ser
    errado quando o tool set virou funcao das flags.
    """
    # build_graph instancia o ChatOpenAI de verdade — aqui so' interessa a
    # identidade do objeto cacheado.
    monkeypatch.setattr(
        "app.domains.chat.service.build_graph",
        lambda **kwargs: MagicMock(name="graph"),
    )

    a = PlanFeatures(tier_name="pro", search_web=True)
    b = PlanFeatures(tier_name="pro", search_web=False)
    assert a.signature() != b.signature()

    graph_a = await svc._get_graph(a)
    graph_b = await svc._get_graph(b)
    assert graph_a is not graph_b
    assert await svc._get_graph(a) is graph_a


async def test_tools_bindadas_no_grafo_respeitam_o_plano(
    svc: ChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O que chega em ``build_graph`` e' o subset do plano, nao a lista toda."""
    capturado: dict[str, object] = {}

    def _fake_build_graph(**kwargs: object) -> MagicMock:
        capturado.update(kwargs)
        return MagicMock(name="graph")

    monkeypatch.setattr(
        "app.domains.chat.service.build_graph", _fake_build_graph
    )

    await svc._get_graph(FREE_FEATURES)

    names = {t.name for t in capturado["tools"]}  # type: ignore[union-attr]
    assert "search_web" not in names
    assert {"inspect_image", "analyze_image", "get_action_plan"} <= names
    assert capturado["plan_features"] is FREE_FEATURES


# ── plan_features no estado do grafo ──────────────────────────────────────────
#
# ``get_action_plan`` filtra os níveis por ``plan_features.action_plan_levels``,
# mas o ChatService nunca injetava ``plan_features`` no ChatState — a tool caía
# em ``allowed_levels = None`` e o gate ficava permanentemente desligado, com
# usuário Free recebendo plano de nível especialista.


def _graph_capturando_estado() -> tuple[AsyncMock, dict]:
    from langchain_core.messages import AIMessage, HumanMessage

    capturado: dict = {}

    async def _ainvoke(state, config=None):  # type: ignore[no-untyped-def]
        capturado.update(state)
        return {
            "messages": [HumanMessage(content="u"), AIMessage(content="pronto")],
        }

    graph = AsyncMock()
    graph.ainvoke = AsyncMock(side_effect=_ainvoke)
    return graph, capturado


async def _svc_com_plano(features: PlanFeatures) -> ChatService:
    sub = MagicMock()
    sub.plan.features = features.model_dump()
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=sub)

    session = MagicMock()
    session.id = "sess-1"
    session_repo = AsyncMock()
    session_repo.get_or_create_for_user = AsyncMock(return_value=session)

    return ChatService(
        session_repo=session_repo,
        message_repo=AsyncMock(),
        inference_svc=MagicMock(),
        action_plan_svc=AsyncMock(),
        diagnosis_svc=AsyncMock(),
        sub_repo=sub_repo,
    )


async def test_chat_injeta_plan_features_e_session_no_estado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = await _svc_com_plano(FREE_FEATURES)
    graph, capturado = _graph_capturando_estado()
    monkeypatch.setattr(
        "app.domains.chat.service.build_graph", lambda **kwargs: graph
    )

    await svc.chat(
        user_id="user-1",
        session_id=None,
        message_text="qual o plano de acao?",
        image_bytes=None,
        image_mime=None,
        image_filename=None,
        model_id="ensemble",
    )

    features = capturado["plan_features"]
    assert features.tier_name == "free"
    # É o que liga o gate: Free só enxerga o nível essencial.
    assert features.action_plan_levels == ["essencial"]
    assert capturado["current_session_id"] == "sess-1"


async def test_chat_stream_tambem_injeta_plan_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O streaming é o caminho que a UI usa — não pode ficar sem o gate."""
    svc = await _svc_com_plano(ENTERPRISE_FEATURES)
    capturado: dict = {}

    async def _astream_events(state, **kwargs):  # type: ignore[no-untyped-def]
        capturado.update(state)
        return
        yield  # pragma: no cover — torna a função um async generator

    graph = AsyncMock()
    graph.astream_events = _astream_events
    graph.aget_state = AsyncMock(return_value=MagicMock(tasks=[], values={}))
    graph.ainvoke = AsyncMock(return_value={"messages": []})
    monkeypatch.setattr(
        "app.domains.chat.service.build_graph", lambda **kwargs: graph
    )

    async for _ in svc.chat_stream(
        user_id="user-1",
        session_id=None,
        message_text="oi",
        image_bytes=None,
        image_mime=None,
        image_filename=None,
        model_id="ensemble",
    ):
        pass

    assert capturado["plan_features"].tier_name == "enterprise"
    assert capturado["current_session_id"] == "sess-1"
    # Pre-fetch da memória semântica também só existia no caminho síncrono.
    assert "recent_relevant_diagnoses" in capturado
