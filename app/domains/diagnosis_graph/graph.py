"""Sub-grafo de diagnostico — invocado pelo chatbot_graph ou pelo endpoint REST.

Fluxo:

    START -> load_model -> run_inference -> compose_action_plan -> persist -> END

A factory ``build_diagnosis_graph`` recebe os 3 services por injecao e fecha
sobre eles via ``functools.partial`` — mesmo padrao do chatbot_graph. Isso
permite que o grafo seja construido fora do request lifecycle (cacheado por
crop_id) e que os testes injetem mocks facilmente.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

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
        "persist",
        partial(persist_node, diagnosis_svc=diagnosis_svc, store=store),
    )

    workflow.add_edge(START, "load_model")
    workflow.add_edge("load_model", "run_inference")
    workflow.add_edge("run_inference", "compose_action_plan")
    workflow.add_edge("compose_action_plan", "persist")
    workflow.add_edge("persist", END)

    return workflow.compile()
