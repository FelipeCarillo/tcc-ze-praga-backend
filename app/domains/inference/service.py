import random
from typing import ClassVar

from app.domains.diagnoses.schemas import Top3PredictionSchema
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import SeverityEnum


class InferenceService:
    """Mock inference para CNN/ViT. Será trocado por chamada real ao playground depois.

    Centraliza catálogo de doenças e geração de predições — desacopla
    chat/router de inference/router (era import direto da função privada).
    """

    _DISEASES: ClassVar[list[dict]] = [
        {
            "id": "ferrugem-asiatica",
            "name": "Ferrugem Asiática",
            "scientific_name": "Phakopsora pachyrhizi",
            "severity": SeverityEnum.ALTA,
            "description": (
                "Doença fúngica severa que causa lesões amareladas a marrom-escuras na face inferior "
                "das folhas. É a principal doença da soja no Brasil, podendo causar desfolha precoce "
                "e perdas de até 80% na produtividade."
            ),
        },
        {
            "id": "mancha-alvo",
            "name": "Mancha-Alvo",
            "scientific_name": "Corynespora cassiicola",
            "severity": SeverityEnum.MEDIA,
            "description": (
                "Doença fúngica caracterizada por lesões circulares com anéis concêntricos que lembram "
                "um alvo. Causa desfolha prematura e redução na produtividade, especialmente em "
                "cultivares suscetíveis."
            ),
        },
        {
            "id": "antracnose",
            "name": "Antracnose",
            "scientific_name": "Colletotrichum truncatum",
            "severity": SeverityEnum.MEDIA,
            "description": (
                "Doença fúngica que afeta hastes, vagens e sementes, causando lesões escuras e "
                "deprimidas. Pode causar morte de plântulas e apodrecimento de vagens, reduzindo "
                "a qualidade e quantidade dos grãos."
            ),
        },
        {
            "id": "cercosporiose",
            "name": "Cercosporiose",
            "scientific_name": "Cercospora kikuchii",
            "severity": SeverityEnum.BAIXA,
            "description": (
                "Doença fúngica que causa manchas púrpuras nas sementes (mancha púrpura) e lesões "
                "foliares. Reduz a qualidade das sementes e pode causar perdas moderadas na "
                "produtividade quando ocorre em condições favoráveis."
            ),
        },
        {
            "id": "mildio",
            "name": "Míldio",
            "scientific_name": "Peronospora manshurica",
            "severity": SeverityEnum.BAIXA,
            "description": (
                "Doença causada por oomiceto que produz lesões amareladas na face superior das folhas "
                "com esporulação cinza-esbranquiçada na face inferior. Geralmente causa perdas "
                "econômicas moderadas, exceto em condições de alta umidade."
            ),
        },
        {
            "id": "saudavel",
            "name": "Saudável",
            "scientific_name": None,
            "severity": SeverityEnum.NENHUMA,
            "description": (
                "A folha analisada não apresenta sinais visíveis de doenças fúngicas ou bacterianas. "
                "Continue monitorando regularmente para detecção precoce de possíveis infecções."
            ),
        },
    ]

    _MODEL_NAMES: ClassVar[dict[str, str]] = {
        "resnet50": "ResNet-50",
        "efficientnet": "EfficientNet-B4",
        "vit": "ViT-B/16",
        "ensemble": "Ensemble",
    }

    def predict(self, model_id: str, image_name: str) -> InferenceResult:
        primary_idx = random.randint(0, len(self._DISEASES) - 1)
        primary = self._DISEASES[primary_idx]

        base_confidence = 0.70 + random.random() * 0.25
        primary_confidence = round(min(0.99, base_confidence), 3)

        remaining = [d for i, d in enumerate(self._DISEASES) if i != primary_idx]
        random.shuffle(remaining)
        leftover = 1.0 - primary_confidence
        second_conf = round(random.uniform(0.005, leftover * 0.7), 3)
        third_conf = round(max(0.001, leftover - second_conf), 3)

        top3 = [
            Top3PredictionSchema(
                rank=1,
                disease_name=primary["name"],
                disease_id=primary["id"],
                scientific_name=primary["scientific_name"],
                confidence=primary_confidence,
                severity=primary["severity"],
            ),
            Top3PredictionSchema(
                rank=2,
                disease_name=remaining[0]["name"],
                disease_id=remaining[0]["id"],
                scientific_name=remaining[0]["scientific_name"],
                confidence=second_conf,
                severity=remaining[0]["severity"],
            ),
            Top3PredictionSchema(
                rank=3,
                disease_name=remaining[1]["name"],
                disease_id=remaining[1]["id"],
                scientific_name=remaining[1]["scientific_name"],
                confidence=third_conf,
                severity=remaining[1]["severity"],
            ),
        ]

        return InferenceResult(
            disease_id=primary["id"],
            disease_name=primary["name"],
            scientific_name=primary["scientific_name"],
            severity=primary["severity"],
            description=primary["description"],
            confidence=primary_confidence,
            model_id=model_id,
            image_name=image_name,
            top3=top3,
        )

    def get_model_label(self, model_id: str) -> str:
        """Display name pra UI/respostas. Falls back to model_id."""
        return self._MODEL_NAMES.get(model_id, model_id)
