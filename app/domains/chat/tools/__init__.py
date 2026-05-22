"""Tools do chatbot_graph (Sprint A2 / TCC-041).

Cada tool fica num arquivo separado e expoe uma factory ``build_<name>_tool``
que recebe services por injecao e retorna um ``BaseTool`` (decorated com
``@tool``). As tools usam ``Annotated[ChatState, InjectedState]`` pra resolver
identidade/ambiente/imagens via state — invisivel ao LLM.

O ``ToolRegistry`` (TCC-039) faz a montagem final: quais tools ativar pra um
dado plano/tier do usuario.
"""

from app.domains.chat.tools.deep_diagnose import build_deep_diagnose_tool
from app.domains.chat.tools.get_action_plan import build_get_action_plan_tool
from app.domains.chat.tools.get_disease_info import build_get_disease_info_tool
from app.domains.chat.tools.search_my_diagnoses import (
    build_search_my_diagnoses_tool,
)

__all__ = [
    "build_deep_diagnose_tool",
    "build_get_action_plan_tool",
    "build_get_disease_info_tool",
    "build_search_my_diagnoses_tool",
]
