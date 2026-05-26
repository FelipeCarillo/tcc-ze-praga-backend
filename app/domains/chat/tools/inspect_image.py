"""Tool ``inspect_image`` — gate de visão (TCC-079).

Lê a imagem do turno (gpt-4o vision via ``settings.vision_model``) e classifica
se é uma foto de planta/folha analisável (cultivo) ou outra coisa (produto,
embalagem, pessoa, etc). O agente chama esta tool **antes** de ``analyze_image``
quando há imagem — se não for planta, recusa educadamente em vez de diagnosticar.

Retorna **JSON-string** (não ``Command``) de propósito: o resultado do gate
precisa ficar visível ao LLM no histórico pra ele decidir o próximo passo.
Espelha o padrão de visão de ``identify_crop.py``.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.config import settings
from app.core.llm import get_chat_model
from app.domains.chat.agent_state import ChatState, resolve_image

_PROMPT = (
    "Você é um triador de imagens agrícolas. Olhe a imagem e classifique. "
    "Responda APENAS JSON, sem comentários nem cercas de código: "
    '{"is_analyzable_plant": <true|false>, '
    '"subject": "folha|planta|produto|pessoa|paisagem|documento|outro", '
    '"reason": "<frase curta em português>"}. '
    "is_analyzable_plant=true só quando for foto de uma planta/folha (idealmente de "
    "cultivo) em que dá pra avaliar sintomas foliares. Produtos, embalagens, rótulos, "
    "pessoas, telas, documentos e paisagens distantes => false. "
    "Na dúvida entre 'folha' e 'planta', use true."
)


def _strip_json_fence(raw: str) -> str:
    """Remove cercas ```json ... ``` que o modelo às vezes adiciona."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    return s.strip()


def build_inspect_image_tool() -> BaseTool:
    """Factory pra ``inspect_image`` — sem deps externas (LLM instanciado lazy)."""

    @tool
    async def inspect_image(
        image_id: str | None = None,
        *,
        state: Annotated[ChatState, InjectedState],
    ) -> str:
        """Olha a imagem enviada e diz se é uma foto de planta/folha analisável.

        Chame SEMPRE como primeira tool quando o usuário enviar uma imagem, antes
        de qualquer diagnóstico. Se ``image_id`` for omitido, usa a primeira imagem
        do turno.

        Returns:
            JSON-string com ``is_analyzable_plant`` (bool), ``subject`` e ``reason``.
        """
        files = state.get("uploaded_files", []) or []
        image = resolve_image(state, image_id) if image_id else (files[0] if files else None)
        if image is None or not image.b64:
            return json.dumps(
                {
                    "is_analyzable_plant": False,
                    "subject": "nenhuma",
                    "reason": "Nenhuma imagem disponível no turno.",
                },
                ensure_ascii=False,
            )

        vision_llm = get_chat_model(settings.vision_model, temperature=0)
        try:
            response = await vision_llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": _PROMPT},
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
            raw = response.content if isinstance(response.content, str) else ""
            data = json.loads(_strip_json_fence(raw))
            return json.dumps(
                {
                    "is_analyzable_plant": bool(data.get("is_analyzable_plant", False)),
                    "subject": str(data.get("subject", "outro")),
                    "reason": str(data.get("reason", "")),
                },
                ensure_ascii=False,
            )
        except Exception:  # noqa: BLE001 — tool sempre retorna string
            # Fallback resiliente: na dúvida, trata como planta pra não travar o usuário.
            return json.dumps(
                {
                    "is_analyzable_plant": True,
                    "subject": "desconhecido",
                    "reason": "Não consegui classificar a imagem com certeza; prosseguindo.",
                },
                ensure_ascii=False,
            )

    return inspect_image
