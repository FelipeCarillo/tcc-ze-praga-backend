"""V2 dormente — identifica cultivo na imagem via gpt-4o vision (TCC-065).

Tool de visao computacional pra preceder o ``deep_diagnose`` quando o usuario
nao declarou o cultivo. Le a primeira imagem do turno (ou a referenciada por
``image_id``), pergunta pro ``gpt-4o`` qual cultivo aparece, e atualiza o
``ChatState.detected_crop_id`` via ``Command(update=...)``.

InjectedState: ``state`` vem do LangGraph (invisivel ao LLM), garantindo que
o crop_id detectado fica disponivel pras proximas tools do mesmo turno.

Flag de feature: a tool so' eh registrada/ativada quando
``settings.agent_enable_identify_crop`` esta on E o plano traz
``identify_crop_auto=True`` (tier Pro+). Default OFF — V2 dormente.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.config import settings
from app.domains.chat.agent_state import ChatState, resolve_image


def build_identify_crop_tool():
    """Factory pra ``identify_crop`` — sem deps externas (LLM e' instanciado dentro).

    A factory permite mockar o LLM em testes via patch direto de ``ChatOpenAI``
    dentro do modulo. O LLM concreto so' eh resolvido quando a tool e' chamada
    (lazy), o que evita tentar conectar com a API em import-time.

    Returns:
        Tool decorada pronta pra ser anexada ao ``llm.bind_tools()``.
    """

    @tool
    async def identify_crop(
        image_id: str,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> Command:
        """Identifica cultivo agricola na imagem via visao computacional (V2 multi-cultivo).

        Use SEMPRE como primeira tool quando usuario envia imagem e cultivo nao
        esta declarado. Apos identificar, crop_id fica no state pras proximas
        tools.
        """
        image = resolve_image(state, image_id)
        if image is None or not image.b64:
            return Command(update={"messages": []})

        allowed = state.get("plan_features", {}).get("allowed_crops")
        if not allowed:
            allowed = ["soja", "milho", "trigo", "cafe", "algodao", "feijao"]

        prompt = (
            f"Identifique qual cultivo agricola aparece nesta imagem. "
            f"Opcoes permitidas: {', '.join(allowed)}. "
            'Responda APENAS JSON: {"crop_id":"<slug>","confidence":<0..1>,"reason":"<breve>"}. '
            "Se confidence < 0.7, use 'desconhecido' como crop_id."
        )

        vision_llm = ChatOpenAI(model=settings.openai_vision_model, temperature=0)
        response = await vision_llm.ainvoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.mime};base64,{image.b64}"
                            },
                        },
                    ]
                )
            ]
        )

        try:
            result = json.loads(response.content)
            crop_id = result.get("crop_id", "desconhecido")
            confidence = float(result.get("confidence", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return Command(update={"messages": []})

        update_state: dict = {}
        if crop_id != "desconhecido" and confidence >= 0.7:
            update_state["detected_crop_id"] = crop_id
        return Command(update=update_state)

    return identify_crop
