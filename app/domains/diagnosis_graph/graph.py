"""Sub-grafo de diagnostico — invocado pelo chatbot_graph ou pelo endpoint REST.

Fluxo (TCC-055 mudou: paralelismo entre compose_action_plan e gather_evidence):

    START -> load_model -> run_inference ┬-> compose_action_plan -┐
                                          └-> gather_evidence -----┴-> persist -> END

A factory ``build_diagnosis_graph`` recebe os 3 services + callables opcionais
pras searches externas e fecha sobre eles via ``functools.partial`` — mesmo
padrao do chatbot_graph. Isso permite que o grafo seja construido fora do
request lifecycle (cacheado por crop_id) e que os testes injetem mocks
facilmente.

LangGraph paralelismo: quando 2 edges saem do mesmo node pra nodes diferentes
e ambos convergem num node downstream, o runtime executa os nodes intermediarios
em paralelo (asyncio.gather). ``persist`` espera os 2 terminarem antes de rodar.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.domains.diagnosis_graph.gather_evidence import (
    SearchCallable,
    gather_evidence_node,
)
from app.domains.diagnosis_graph.nodes import (
    compose_action_plan_node,
    load_model_node,
    persist_node,
    run_inference_node,
)
from app.domains.diagnosis_graph.state import DiagnosisState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore

    from app.domains.action_plans.service import ActionPlanService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService


def build_diagnosis_graph(
    inference_svc: InferenceService,
    action_plan_svc: ActionPlanService,
    diagnosis_svc: DiagnosisService,
    store: BaseStore | None = None,
    tavily_search: SearchCallable | None = None,
    scielo_search: SearchCallable | None = None,
) -> CompiledStateGraph:
    """Compila o sub-grafo de diagnostico com os services injetados.

    Args:
        inference_svc: ``InferenceService`` ja inicializado pro crop alvo.
        action_plan_svc: lookup de planos de acao por disease_id.
        diagnosis_svc: persistencia de Diagnosis rows.
        store: opcional ``BaseStore`` (em prod, ``AsyncPostgresStore``)
            — quando passado, o ``persist_node`` indexa cada diagnostico
            criado no namespace ``("user", uid, "diagnoses")`` pra busca
            semantica via ``search_my_diagnoses`` (TCC-046).
        tavily_search: opcional callable ``(query) -> JSON-str`` pra search_web
            (TCC-055). Quando ``None``, gather_evidence skipa busca web mesmo
            que plan_features ativo.
        scielo_search: opcional callable ``(query) -> JSON-str`` pra
            search_scientific (TCC-055). Idem semantica.

    Returns:
        ``CompiledStateGraph`` pronto pra ``.ainvoke()``.
    """
    workflow: StateGraph = StateGraph(DiagnosisState)

    workflow.add_node(
        "load_model", partial(load_model_node, inference_svc=inference_svc)
    )
    workflow.add_node(
        "run_inference",
        partial(run_inference_node, inference_svc=inference_svc),
    )
    workflow.add_node(
        "compose_action_plan",
        partial(compose_action_plan_node, action_plan_svc=action_plan_svc),
    )
    workflow.add_node(
        "gather_evidence",
        partial(
            gather_evidence_node,
            tavily_search=tavily_search,
            scielo_search=scielo_search,
        ),
    )
    workflow.add_node(
        "persist",
        partial(
            persist_node,
            diagnosis_svc=diagnosis_svc,
            inference_svc=inference_svc,
            store=store,
        ),
    )

    workflow.add_edge(START, "load_model")
    workflow.add_edge("load_model", "run_inference")
    # Fan-out: run_inference dispara compose_action_plan e gather_evidence em paralelo.
    workflow.add_edge("run_inference", "compose_action_plan")
    workflow.add_edge("run_inference", "gather_evidence")
    # Fan-in: persist espera os dois.
    workflow.add_edge("compose_action_plan", "persist")
    workflow.add_edge("gather_evidence", "persist")
    workflow.add_edge("persist", END)

    return workflow.compile()
