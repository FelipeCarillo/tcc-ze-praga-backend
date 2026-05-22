"""Sub-grafo de diagnostico (Sprint A2 / TCC-040).

Compoe um LangGraph compilado que executa o pipeline de diagnostico
end-to-end:

    load_model -> run_inference -> compose_action_plan -> persist

Invocado tanto pelo chatbot_graph (via tool ``deep_diagnose``) quanto pelo
endpoint REST ``POST /api/v1/diagnoses/analyze``. Recebe services via
factory (closures) — mesmo padrao do ``chatbot_graph``.
"""

from app.domains.diagnosis_graph.graph import build_diagnosis_graph
from app.domains.diagnosis_graph.state import DiagnosisState

__all__ = ["build_diagnosis_graph", "DiagnosisState"]
