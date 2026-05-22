"""InferenceService — orquestrador do mock de CNN/ViT da soja.

A partir do epic TCC-025 o catalogo de doencas mora no banco (tabela ``diseases``)
e nao mais como ``ClassVar``. O service recebe um snapshot do catalogo
(``diseases: list[DiseaseDTO]``) por injecao na construcao — populado pelo
factory ``get_inference_service`` que usa ``DiseaseRepository.list_by_crop()``.

``predict()`` continua sincrono para nao mudar a assinatura do tool do agent
LangGraph. ``crop_id`` na assinatura permite multi-cultivo no futuro (sprint A2),
mas hoje o catalogo passado ja' filtra so as doencas da soja.
"""

from __future__ import annotations

import random
from typing import ClassVar

from app.domains.diagnoses.schemas import Top3PredictionSchema
from app.domains.inference.repository import DiseaseDTO
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import SeverityEnum


class InferenceService:
    """Mock CNN/ViT que escolhe random uma doenca do catalogo carregado.

    Args:
        diseases: snapshot do catalogo (list[DiseaseDTO]) — injetado pelo
            factory que consulta DiseaseRepository.list_by_crop().
    """

    _MODEL_NAMES: ClassVar[dict[str, str]] = {
        "resnet50": "ResNet-50",
        "efficientnet": "EfficientNet-B4",
        "vit": "ViT-B/16",
        "ensemble": "Ensemble",
    }

    def __init__(self, diseases: list[DiseaseDTO]) -> None:
        if not diseases:
            raise ValueError(
                "InferenceService precisa de pelo menos uma DiseaseDTO no catalogo"
            )
        self._diseases: list[DiseaseDTO] = list(diseases)

    def predict(
        self,
        model_id: str,
        image_name: str,
        crop_id: str | None = None,  # noqa: ARG002 — usado em sprint A2
    ) -> InferenceResult:
        """Gera predicao mockada a partir do catalogo injetado.

        ``crop_id`` reservado pra sprint A2 quando o service consultar
        DiseaseRepository por crop em runtime.
        """
        primary_idx = random.randint(0, len(self._diseases) - 1)
        primary = self._diseases[primary_idx]

        base_confidence = 0.70 + random.random() * 0.25
        primary_confidence = round(min(0.99, base_confidence), 3)

        remaining = [d for i, d in enumerate(self._diseases) if i != primary_idx]
        random.shuffle(remaining)
        leftover = 1.0 - primary_confidence
        second_conf = round(random.uniform(0.005, leftover * 0.7), 3)
        third_conf = round(max(0.001, leftover - second_conf), 3)

        # Defensive: se o catalogo so tem 1 ou 2 itens, fazemos fallback.
        second = remaining[0] if remaining else primary
        third = remaining[1] if len(remaining) > 1 else second

        top3 = [
            Top3PredictionSchema(
                rank=1,
                disease_name=primary.name_pt,
                disease_id=primary.slug,
                scientific_name=primary.scientific_name,
                confidence=primary_confidence,
                severity=SeverityEnum(primary.severity_default),
            ),
            Top3PredictionSchema(
                rank=2,
                disease_name=second.name_pt,
                disease_id=second.slug,
                scientific_name=second.scientific_name,
                confidence=second_conf,
                severity=SeverityEnum(second.severity_default),
            ),
            Top3PredictionSchema(
                rank=3,
                disease_name=third.name_pt,
                disease_id=third.slug,
                scientific_name=third.scientific_name,
                confidence=third_conf,
                severity=SeverityEnum(third.severity_default),
            ),
        ]

        return InferenceResult(
            disease_id=primary.slug,
            disease_name=primary.name_pt,
            scientific_name=primary.scientific_name,
            severity=SeverityEnum(primary.severity_default),
            description=primary.description_md,
            confidence=primary_confidence,
            model_id=model_id,
            image_name=image_name,
            top3=top3,
        )

    def get_model_label(self, model_id: str) -> str:
        """Display name pra UI/respostas. Falls back to model_id."""
        return self._MODEL_NAMES.get(model_id, model_id)

    def get_disease_by_slug(self, slug: str) -> DiseaseDTO | None:
        """Lookup do catalogo carregado por slug — usado pelo agent tool."""
        for disease in self._diseases:
            if disease.slug == slug:
                return disease
        return None

    @property
    def disease_catalog(self) -> list[DiseaseDTO]:
        """Snapshot imutavel do catalogo (usado por testes e agent)."""
        return list(self._diseases)
