"""Estado do chatbot_graph + helpers de InjectedState (Sprint A2 / A3).

Este modulo introduz uma versao expandida do ChatState pra suportar o padrao
InjectedState do LangGraph 0.2+ (Annotated[ChatState, InjectedState]) — as
tools resolvem identidade/ambiente/imagens via state em vez de receberem como
arg do LLM.

UploadedFileDTO eh o subset relevante do modelo UploadedFile pro turno atual:
quando o usuario envia 1+ imagens junto da mensagem, elas chegam aqui ja com
ids estaveis. As tools podem resolver por id via ``resolve_image``.

Sprint A3 (TCC-051) adiciona ``plan_features`` ao state — PlanFeatures do
plano ativo, consumido pelas tools pra filtrar comportamento (ex: niveis
permitidos no ``get_action_plan``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.domains.subscriptions.features import PlanFeatures


@dataclass
class UploadedFileDTO:
    """Imagem disponivel no turno atual (subset relevante de UploadedFile model).

    Args:
        id: identificador estavel da imagem (UUID do uploaded_files row).
        original_name: nome original do arquivo enviado pelo usuario.
        mime: content-type (ex: ``image/jpeg``).
        storage_key: chave no storage (ex: ``uploads/<uid>/<sha>.jpg``).
        size_bytes: tamanho do arquivo em bytes.
        b64: conteudo base64 carregado sob demanda — None ate uma tool precisar.
    """

    id: str
    original_name: str
    mime: str
    storage_key: str
    size_bytes: int
    b64: str | None = None


class ChatState(TypedDict, total=False):
    """Estado expandido do chatbot_graph (Sprint A2).

    Convencao InjectedState: tools sao annotated com
    ``Annotated[ChatState, InjectedState]`` e resolvem identidade/ambiente
    /imagens via state — sem expor isso ao LLM.

    Todos os campos sao opcionais (``total=False``) pra simplificar a montagem
    do estado inicial — apenas ``messages``, ``current_user_id`` e
    ``current_session_id`` sao tipicamente preenchidos pela camada de
    transporte (router/service).
    """

    # canonico LangGraph
    messages: Annotated[list[BaseMessage], add_messages]

    # identidade & ambiente (InjectedState — invisivel ao LLM)
    current_user_id: str
    current_session_id: str

    # preferencias do usuario
    selected_model: str  # ex: "ensemble" (NAO escolhido pelo LLM)
    detected_crop_id: str | None  # set por identify_crop V2 ou prefs
    preferred_action_level: str  # "essencial" | "campo" | "especialista"

    # features do plano ativo (Sprint A3 / TCC-051) — PlanFeatures contem
    # llm_model, action_plan_levels, allowed_crops, identify_crop_auto, etc.
    plan_features: PlanFeatures

    # contexto recuperado (Store / DB) — populado em Sprint A2.5
    recent_relevant_diagnoses: list[dict[str, Any]]

    # turno atual
    uploaded_files: list[UploadedFileDTO]

    # progressivo
    diagnoses_in_turn: list[str]  # ids criados neste turno
    pending_interrupt: dict[str, Any] | None


def resolve_image(state: ChatState, image_id: str) -> UploadedFileDTO | None:
    """Resolve uma imagem do turno atual por id.

    Helper pras tools (deep_diagnose etc.) acessarem imagens sem expor a
    estrutura interna do state.

    Args:
        state: estado atual do grafo.
        image_id: id estavel da imagem desejada.

    Returns:
        ``UploadedFileDTO`` se encontrado, ``None`` caso contrario.
    """
    for f in state.get("uploaded_files", []):
        if f.id == image_id:
            return f
    return None
