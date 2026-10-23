"""Tool ``inspect_image`` — gate de visão (TCC-079).

Lê a imagem do turno (gpt-4o vision via ``settings.vision_model``) e classifica
se é uma foto de planta/folha analisável (cultivo) ou outra coisa (produto,
embalagem, pessoa, etc). O agente chama esta tool **antes** de ``analyze_image``
quando há imagem — se não for planta, recusa educadamente em vez de diagnosticar.

Retorna **JSON-string** (não ``Command``) de propósito: o resultado do gate
precisa ficar visível ao LLM no histórico pra ele decidir o próximo passo.
Espelha o padrão de visão de ``identify_crop.py``.

TCC-107: o gate usa só o LLM de visão. O caminho pelo Decisions API (TCC-103)
saiu depois que ele aceitou a foto de uma mão como folha num teste de 10/2026.

TCC-106: foto de tela, print, documento ou foto de outra foto com a folha
visível **é analisável** — o gate marca ``is_reproduction=true`` e devolve uma
``note`` para o agente diagnosticar e, numa frase, recomendar uma foto tirada
direto da folha no campo. Documento/tela sem folha continua recusado.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.config import settings
from app.core.llm import get_chat_model, message_text
from app.domains.chat.agent_state import ChatState, resolve_image

_PROMPT = (
    "Você é um triador de imagens agrícolas. Primeiro descreva o que a imagem "
    "realmente mostra, depois classifique. "
    "Responda APENAS JSON, sem comentários nem cercas de código: "
    '{"what_is_it": "<o que a imagem mostra, com artigo, ex.: uma mão, '
    'um frasco de herbicida, uma folha de soja>", '
    '"subject": "folha|planta|produto|pessoa|paisagem|documento|outro", '
    '"is_analyzable_plant": <true|false>, '
    '"is_reproduction": <true|false>, '
    '"reason": "<frase curta em português>"}. '
    "is_analyzable_plant=true quando aparece uma planta/folha (idealmente de cultivo) "
    "em que dá pra avaliar sintomas foliares — inclusive se ela estiver numa tela, "
    "print, documento, impressão ou foto de outra foto, ou segurada na mão. "
    "Mão, dedos, braço, pele ou rosto SEM folha visível => false, com subject "
    "'pessoa' (cor de pele, palma da mão e linhas da mão não são folha). "
    "Produtos, embalagens, rótulos, objetos verdes, paisagens distantes e "
    "telas/documentos SEM folha visível => false. "
    "is_reproduction=true quando a imagem não foi tirada direto da planta (tela, "
    "monitor, print, documento, impressão, foto de foto). "
    "Na dúvida entre 'folha' e 'planta', use true."
)

# TCC-107: subject que nunca é uma folha analisável vence o booleano — o LLM
# já descreveu "uma mão" e ainda assim marcou is_analyzable_plant=true.
_NON_PLANT_SUBJECTS = frozenset({"produto", "pessoa", "paisagem"})


# TCC-106: o agente repassa a ``note`` ao usuário em uma frase (SYSTEM_PROMPT).
_REPRODUCTION_NOTE = (
    "A folha está numa tela, documento ou foto de foto: dá para analisar, mas "
    "recomende uma foto tirada direto da folha no campo, que tem mais detalhe."
)


def _result(
    *,
    analyzable: bool,
    reproduction: bool,
    subject: str,
    reason: str,
    what_is_it: str = "",
) -> dict[str, object]:
    """Monta a resposta do gate; ``note`` só quando há folha numa reprodução.

    TCC-107: ``what_is_it`` é o que a imagem mostra ("uma mão") — o agente usa
    na recusa ("isso é uma mão, e não uma folha").
    """
    analyzable = analyzable and subject not in _NON_PLANT_SUBJECTS
    data: dict[str, object] = {
        "is_analyzable_plant": analyzable,
        "is_reproduction": reproduction,
        "subject": subject,
        "what_is_it": what_is_it,
        "reason": reason,
    }
    if analyzable and reproduction:
        data["note"] = _REPRODUCTION_NOTE
    return data


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
            JSON-string com ``is_analyzable_plant`` (bool), ``subject``,
            ``what_is_it`` (o que a imagem mostra) e ``reason``.
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

        data_url = f"data:{image.mime};base64,{image.b64}"
        vision_llm = get_chat_model(settings.vision_model)
        try:
            response = await vision_llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": _PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": settings.vision_image_detail,
                                },
                            },
                        ]
                    )
                ]
            )
            raw = message_text(response)
            data = json.loads(_strip_json_fence(raw))
            return json.dumps(
                _result(
                    analyzable=bool(data.get("is_analyzable_plant", False)),
                    reproduction=bool(data.get("is_reproduction", False)),
                    subject=str(data.get("subject", "outro")),
                    reason=str(data.get("reason", "")),
                    what_is_it=str(data.get("what_is_it", "")),
                ),
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
