import json
import random

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from app.core.dependencies import (
    get_diagnosis_service,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.schemas import DiagnosisResponse
from app.domains.diagnoses.service import DiagnosisService
from app.domains.inference.router import _generate_mock_result
from app.domains.diagnoses.schemas import CreateDiagnosisRequest
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, ModelEnum

router = APIRouter(prefix="/chat", tags=["Chat"])

_MODEL_NAMES = {
    "resnet50": "ResNet-50",
    "efficientnet": "EfficientNet-B4",
    "vit": "ViT-B/16",
    "ensemble": "Ensemble",
}

_KEYWORD_RESPONSES: dict[str, str] = {
    "ferrugem": (
        "A **Ferrugem Asiática** (*Phakopsora pachyrhizi*) é a principal doença da soja no Brasil. "
        "Identifique-a por lesões marrom-claras na face inferior das folhas com abundante esporulação. "
        "O controle é feito com fungicidas triazol + estrobilurina. Posso analisar uma imagem da folha se desejar."
    ),
    "mancha": (
        "A **Mancha-Alvo** (*Corynespora cassiicola*) forma lesões circulares com anéis concêntricos. "
        "Use fungicidas sistêmicos (carboxamida + estrobilurina) e faça rotação de culturas para reduzir o inóculo."
    ),
    "antracnose": (
        "A **Antracnose** (*Colletotrichum truncatum*) afeta hastes, vagens e sementes com lesões escuras. "
        "Realize tratamento de sementes e aplique fungicidas foliares em casos severos."
    ),
    "cercosporiose": (
        "A **Cercosporiose** (*Cercospora kikuchii*) causa manchas púrpuras nas sementes e lesões foliares. "
        "O impacto econômico é geralmente baixo, mas utilize sementes certificadas para prevenção."
    ),
    "mildio": (
        "O **Míldio** (*Peronospora manshurica*) forma lesões amarelas na face superior com esporulação "
        "cinza-esbranquiçada na inferior. Evite cultivares suscetíveis e monitore em períodos úmidos."
    ),
    "saudavel": (
        "Ótima notícia! Uma lavoura saudável indica boas práticas de manejo. "
        "Continue monitorando regularmente para detecção precoce de possíveis infecções."
    ),
    "fungicida": (
        "Para o controle de doenças foliares da soja, os fungicidas mais utilizados são as misturas de "
        "**triazol + estrobilurina** e **carboxamida + estrobilurina**. Faça rodízio de mecanismos de ação "
        "para evitar resistência. Consulte sempre um engenheiro agrônomo para dosagem e timing corretos."
    ),
    "monitoramento": (
        "O monitoramento preventivo é fundamental! Inspecione a lavoura a cada 7-10 dias, especialmente "
        "durante os períodos de florescimento e enchimento de grãos. Observe o terço inferior das plantas "
        "onde as doenças costumam iniciar. Use os sistemas de alerta fitossanitário regionais como apoio."
    ),
}

_GREETINGS = {"oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "hey", "hello", "hi"}

_ABOUT_KEYWORDS = {"quem", "voce", "você", "faz", "funciona", "sobre"}

_DEFAULT_RESPONSE = (
    "Sou o **Zé Praga**, assistente especializado em doenças foliares da soja. "
    "Posso ajudá-lo a identificar as 6 principais doenças: Ferrugem Asiática, Mancha-Alvo, "
    "Antracnose, Cercosporiose, Míldio e Planta Saudável. "
    "Envie uma foto da folha para análise ou pergunte sobre sintomas, controle e monitoramento."
)


def _get_text_response(last_message: str) -> str:
    lower = last_message.lower()

    words = set(lower.split())
    if words & _GREETINGS:
        return (
            "Olá! Sou o **Zé Praga**, seu assistente de diagnóstico de doenças foliares da soja. "
            "Posso analisar imagens de folhas ou responder dúvidas sobre as principais doenças. "
            "Como posso ajudar?"
        )

    if _ABOUT_KEYWORDS & words:
        return _DEFAULT_RESPONSE

    for keyword, response in _KEYWORD_RESPONSES.items():
        if keyword in lower:
            return response

    return _DEFAULT_RESPONSE


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    diagnosis: DiagnosisResponse | None = None


@router.post("", response_model=ChatResponse, status_code=200)
async def send_message(
    messages: str = Form(...),
    model: str = Form(default=ModelEnum.ENSEMBLE),
    image: UploadFile | None = File(default=None),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.CHAT)),
    diagnosis_svc: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
) -> ChatResponse:
    await usage_svc.record_usage(
        current_user.id,
        FeatureTypeEnum.CHAT,
        {"model": model, "has_image": image is not None},
    )

    if image is not None:
        result = _generate_mock_result(model, image.filename or "imagem.jpg")
        disease = result["disease"]
        model_label = _MODEL_NAMES.get(model, model)

        body = CreateDiagnosisRequest(
            disease_name=disease["name"],
            disease_id=disease["id"],
            scientific_name=disease["scientific_name"],
            confidence=result["confidence"],
            severity=disease["severity"],
            description=disease["description"],
            model_used=result["model_id"],
            image_url=None,
            image_name=result["image_name"],
            top3=result["top3"],
        )
        saved = await diagnosis_svc.create(current_user.id, body)

        content = (
            f"Analisei a imagem utilizando o modelo **{model_label}** e detectei "
            f"**{disease['name']}**"
            + (f" (*{disease['scientific_name']}*)" if disease["scientific_name"] else "")
            + f" com **{result['confidence'] * 100:.1f}%** de confiança. "
        )
        if disease["id"] != "saudavel":
            content += (
                f"A severidade é classificada como **{disease['severity']}**. "
                "Consulte o plano de ação para recomendações de controle."
            )
        else:
            content += "A folha não apresenta sinais de doença. Continue monitorando regularmente."

        return ChatResponse(role="assistant", content=content, diagnosis=saved)

    try:
        parsed = json.loads(messages)
        last_content = parsed[-1].get("content", "") if parsed else ""
    except (json.JSONDecodeError, (KeyError, IndexError, TypeError)):
        last_content = messages

    return ChatResponse(role="assistant", content=_get_text_response(last_content))
