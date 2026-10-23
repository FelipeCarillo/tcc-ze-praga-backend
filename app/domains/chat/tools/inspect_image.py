"""Tool ``inspect_image`` — gate de visão (TCC-079).

Lê a imagem do turno (gpt-4o vision via ``settings.vision_model``) e classifica
se é uma foto de planta/folha analisável (cultivo) ou outra coisa (produto,
embalagem, pessoa, etc). O agente chama esta tool **antes** de ``analyze_image``
quando há imagem — se não for planta, recusa educadamente em vez de diagnosticar.

Retorna **JSON-string** (não ``Command``) de propósito: o resultado do gate
precisa ficar visível ao LLM no histórico pra ele decidir o próximo passo.
Espelha o padrão de visão de ``identify_crop.py``.

TCC-103: a classificação sai primeiro do Decisions API (pergunta fechada, sem
gerar texto — ~0,4 s contra ~1,7 s do LLM de visão no teste de 17/10/2026).
Se ele falhar ou estiver desligado (``VISION_GATE_DECISIONS=false``), o LLM de
visão responde como antes.

TCC-106: foto de tela, print, documento ou foto de outra foto com a folha
visível **é analisável** — o gate marca ``is_reproduction=true`` e devolve uma
``note`` para o agente diagnosticar e, numa frase, recomendar uma foto tirada
direto da folha no campo. Documento/tela sem folha continua recusado.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.config import settings
from app.core.decisions import DecisionsError, decide, image_input
from app.core.llm import get_chat_model, message_text
from app.domains.chat.agent_state import ChatState, resolve_image

logger = logging.getLogger(__name__)

_PROMPT = (
    "Você é um triador de imagens agrícolas. Olhe a imagem e classifique. "
    "Responda APENAS JSON, sem comentários nem cercas de código: "
    '{"is_analyzable_plant": <true|false>, '
    '"is_reproduction": <true|false>, '
    '"subject": "folha|planta|produto|pessoa|paisagem|documento|outro", '
    '"reason": "<frase curta em português>"}. '
    "is_analyzable_plant=true quando aparece uma planta/folha (idealmente de cultivo) "
    "em que dá pra avaliar sintomas foliares — inclusive se ela estiver numa tela, "
    "print, documento, impressão ou foto de outra foto. Produtos, embalagens, rótulos, "
    "pessoas, paisagens distantes e telas/documentos SEM folha visível => false. "
    "is_reproduction=true quando a imagem não foi tirada direto da planta (tela, "
    "monitor, print, documento, impressão, foto de foto). "
    "Na dúvida entre 'folha' e 'planta', use true."
)


_SUBJECTS = {
    "folha": "folha ou folíolo em close",
    "planta": "planta inteira ou ramo, perto o bastante para ver as folhas",
    "produto": "produto, embalagem, rótulo ou frasco",
    "pessoa": "pessoa ou animal como assunto principal",
    "paisagem": "lavoura ou paisagem vista de longe",
    "documento": "documento, tela, print ou texto",
    "outro": "qualquer outra coisa",
}

_GATE_QUESTIONS: list[dict[str, Any]] = [
    {
        "type": "predicate",
        "name": "is_analyzable_plant",
        "instructions": (
            "Aparece uma planta/folha (idealmente de cultivo) em que dá pra avaliar "
            "sintomas foliares? Conta mesmo se a folha estiver numa tela, print, "
            "documento, impressão ou foto de outra foto. Produtos, embalagens, rótulos, "
            "pessoas, paisagens distantes e telas/documentos sem folha visível não contam."
        ),
    },
    {
        "type": "predicate",
        "name": "is_reproduction",
        "instructions": (
            "A imagem é uma reprodução, e não uma foto tirada direto da planta? "
            "Ex.: foto de tela ou monitor, print, documento, impressão, foto de outra foto."
        ),
    },
    {
        "type": "choice",
        "name": "subject",
        "instructions": "Qual é o assunto principal da imagem?",
        "choices": [{"value": k, "description": v} for k, v in _SUBJECTS.items()],
    },
]


async def _gate_via_decisions(data_url: str) -> dict[str, object]:
    """Classifica a imagem pelo Decisions API (TCC-103)."""
    answers = await decide(
        image_input("Foto enviada por um produtor ao assistente de doenças da soja.", data_url),
        _GATE_QUESTIONS,
    )
    probability = float(answers["is_analyzable_plant"]["probability"])
    reproduction = float(answers["is_reproduction"]["probability"])
    subject = str(answers["subject"]["choice"])
    return _result(
        analyzable=probability >= 0.5,
        reproduction=reproduction >= 0.5,
        subject=subject,
        reason=f"Classificada como {_SUBJECTS.get(subject, subject)}.",
    )


# TCC-106: o agente repassa a ``note`` ao usuário em uma frase (SYSTEM_PROMPT).
_REPRODUCTION_NOTE = (
    "A folha está numa tela, documento ou foto de foto: dá para analisar, mas "
    "recomende uma foto tirada direto da folha no campo, que tem mais detalhe."
)


def _result(
    *, analyzable: bool, reproduction: bool, subject: str, reason: str
) -> dict[str, object]:
    """Monta a resposta do gate; ``note`` só quando há folha numa reprodução."""
    data: dict[str, object] = {
        "is_analyzable_plant": analyzable,
        "is_reproduction": reproduction,
        "subject": subject,
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

        data_url = f"data:{image.mime};base64,{image.b64}"
        if settings.vision_gate_decisions:
            try:
                return json.dumps(await _gate_via_decisions(data_url), ensure_ascii=False)
            except (DecisionsError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Decisions falhou no gate de imagem (%s) — usando o LLM de visão", exc
                )

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
