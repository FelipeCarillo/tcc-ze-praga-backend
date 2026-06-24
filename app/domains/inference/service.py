"""InferenceService — inferência de doenças foliares da soja.

A partir do TCC-023 (ADR-0003) o service usa um **modelo ONNX real**
(EfficientNet-B4 treinado no ASDID) quando um ``OnnxClassifier`` é injetado e os
bytes da imagem estão disponíveis. Caso contrário — sem classifier, sem bytes, ou
erro de inferência — cai no **mock** (comportamento anterior), garantindo que o
fluxo nunca quebra (graceful degradation).

O catálogo de doenças mora no banco (tabela ``diseases``) e é injetado como
snapshot (``diseases: list[DiseaseDTO]``) pelo factory ``get_inference_service``.
``predict()`` continua síncrono para não mudar a assinatura do tool do agent
LangGraph; ``image_bytes`` é keyword-only e opcional para back-compat.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, ClassVar

from app.domains.diagnoses.schemas import Top3PredictionSchema
from app.domains.inference.repository import DiseaseDTO
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import SeverityEnum

if TYPE_CHECKING:
    from app.domains.inference.onnx_classifier import OnnxClassifier

logger = logging.getLogger(__name__)

# Metadados de fallback para as 6 classes do modelo — usados quando o catálogo
# do banco ainda não tem o slug previsto (ex.: antes de re-rodar o seed que
# troca antracnose -> mancha-olho-de-ra, ADR-0003). Garante rótulo correto
# mesmo sem re-seed.
_FALLBACK_META: dict[str, tuple[str, str | None, str]] = {
    "cercosporiose": ("Cercosporiose", "Cercospora kikuchii", SeverityEnum.BAIXA.value),
    "ferrugem-asiatica": ("Ferrugem Asiática", "Phakopsora pachyrhizi", SeverityEnum.ALTA.value),
    "mancha-alvo": ("Mancha-Alvo", "Corynespora cassiicola", SeverityEnum.MEDIA.value),
    "mancha-olho-de-ra": ("Mancha Olho-de-rã", "Cercospora sojina", SeverityEnum.MEDIA.value),
    "mildio": ("Míldio", "Peronospora manshurica", SeverityEnum.BAIXA.value),
    "saudavel": ("Saudável", None, SeverityEnum.NENHUMA.value),
}


class InferenceService:
    """Inferência de doenças: ONNX real (se disponível) com fallback para mock.

    Args:
        diseases: snapshot do catálogo (list[DiseaseDTO]) — injetado pelo factory
            que consulta DiseaseRepository.list_by_crop().
        classifier: OnnxClassifier opcional. Quando presente e ``image_bytes`` é
            passado a ``predict``, roda inferência real; senão usa o mock.
    """

    _MODEL_NAMES: ClassVar[dict[str, str]] = {
        "resnet50": "ResNet-50",
        "efficientnet": "EfficientNet-B4",
        "vit": "ViT-B/16",
        "ensemble": "Ensemble",
    }

    def __init__(
        self,
        diseases: list[DiseaseDTO],
        classifier: OnnxClassifier | None = None,
    ) -> None:
        if not diseases:
            raise ValueError(
                "InferenceService precisa de pelo menos uma DiseaseDTO no catalogo"
            )
        self._diseases: list[DiseaseDTO] = list(diseases)
        self._classifier = classifier

    def predict(
        self,
        model_id: str,
        image_name: str,
        crop_id: str | None = None,  # noqa: ARG002 — usado em sprint A2
        *,
        image_bytes: bytes | None = None,
    ) -> InferenceResult:
        """Prediz a doença na imagem.

        Usa o modelo ONNX real quando ``self._classifier`` existe e ``image_bytes``
        é fornecido; caso contrário (ou em erro) cai no mock. ``crop_id`` reservado
        pra sprint A2 (catálogo por crop em runtime).
        """
        if self._classifier is not None and image_bytes:
            try:
                return self._predict_onnx(model_id, image_name, image_bytes)
            except Exception:  # noqa: BLE001 — nunca derruba o fluxo; cai no mock
                logger.exception(
                    "Inferência ONNX falhou para %s — usando mock", image_name
                )
        return self._predict_mock(model_id, image_name)

    # ── ONNX real ────────────────────────────────────────────────────────────

    def _resolve_meta(
        self, slug: str
    ) -> tuple[str, str | None, str, str | None]:
        """Resolve (name_pt, scientific_name, severity, description) para um slug.

        Prioriza o catálogo do banco; cai no _FALLBACK_META; por fim usa o slug cru.
        """
        dto = self.get_disease_by_slug(slug)
        if dto is not None:
            return dto.name_pt, dto.scientific_name, dto.severity_default, dto.description_md
        fb = _FALLBACK_META.get(slug)
        if fb is not None:
            name_pt, sci, sev = fb
            return name_pt, sci, sev, None
        return slug, None, SeverityEnum.MEDIA.value, None

    def _predict_onnx(
        self, model_id: str, image_name: str, image_bytes: bytes
    ) -> InferenceResult:
        assert self._classifier is not None
        preds = self._classifier.predict(image_bytes, top_k=3)

        top3: list[Top3PredictionSchema] = []
        for rank, (slug, prob) in enumerate(preds, start=1):
            name_pt, sci, sev, _desc = self._resolve_meta(slug)
            top3.append(
                Top3PredictionSchema(
                    rank=rank,
                    disease_name=name_pt,
                    disease_id=slug,
                    scientific_name=sci,
                    confidence=round(float(prob), 4),
                    severity=SeverityEnum(sev),
                )
            )

        primary_slug, primary_prob = preds[0]
        name_pt, sci, sev, desc = self._resolve_meta(primary_slug)
        return InferenceResult(
            disease_id=primary_slug,
            disease_name=name_pt,
            scientific_name=sci,
            severity=SeverityEnum(sev),
            description=desc,
            confidence=round(float(primary_prob), 4),
            model_id=model_id,
            image_name=image_name,
            top3=top3,
        )

    # ── Mock (fallback) ──────────────────────────────────────────────────────

    def _predict_mock(self, model_id: str, image_name: str) -> InferenceResult:
        """Predição mockada a partir do catálogo injetado (comportamento legado)."""
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
