"""Tests para InferenceService — mock CNN/ViT determinístico via seed.

InferenceService passou a centralizar catálogo + predição que estava em
inference/router.py (chat/router.py importava função privada — coupling
quebrado em TCC-008).
"""

import random

import pytest

from app.domains.inference.schemas import InferenceResult
from app.domains.inference.service import InferenceService
from app.shared.enums import ModelEnum, SeverityEnum


@pytest.fixture
def service() -> InferenceService:
    return InferenceService()


@pytest.fixture(autouse=True)
def _deterministic_random():
    """Seed fixa pra cada teste — predict é estocástico, queremos reproduzibilidade."""
    random.seed(42)
    yield


# ── predict() ─────────────────────────────────────────────────────────────────


def test_predict_returns_inference_result(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "folha.jpg")
    assert isinstance(result, InferenceResult)


def test_predict_preserves_model_id(service: InferenceService):
    result = service.predict("resnet50", "x.jpg")
    assert result.model_id == "resnet50"


def test_predict_preserves_image_name(service: InferenceService):
    result = service.predict(ModelEnum.VIT, "minha_planta.png")
    assert result.image_name == "minha_planta.png"


def test_predict_disease_id_is_in_catalog(service: InferenceService):
    """O disease_id retornado precisa estar no catálogo interno."""
    valid_ids = {d["id"] for d in service._DISEASES}
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert result.disease_id in valid_ids


def test_predict_confidence_is_positive_and_capped(service: InferenceService):
    """confidence > 0 e <= 0.99 (cap definido no service)."""
    for _ in range(20):
        result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
        assert result.confidence > 0.0
        assert result.confidence <= 0.99


def test_predict_severity_is_enum(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert isinstance(result.severity, SeverityEnum)


# ── top3 ──────────────────────────────────────────────────────────────────────


def test_top3_has_three_entries(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert len(result.top3) == 3


def test_top3_ranks_are_1_2_3_in_order(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert [t.rank for t in result.top3] == [1, 2, 3]


def test_top3_first_matches_primary_prediction(service: InferenceService):
    """rank=1 reflete a doença escolhida como primária."""
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    first = result.top3[0]
    assert first.disease_id == result.disease_id
    assert first.disease_name == result.disease_name
    assert first.confidence == result.confidence


def test_top3_all_disease_ids_are_unique(service: InferenceService):
    """Nenhuma doença pode aparecer duas vezes no top3."""
    for _ in range(10):
        result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
        ids = [t.disease_id for t in result.top3]
        assert len(set(ids)) == 3


def test_top3_confidences_are_non_negative(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    for entry in result.top3:
        assert entry.confidence >= 0.0


# ── catálogo de doenças ──────────────────────────────────────────────────────


def test_saudavel_has_no_scientific_name(service: InferenceService):
    """A entrada 'saudavel' do catálogo tem scientific_name None."""
    saudavel = next(d for d in service._DISEASES if d["id"] == "saudavel")
    assert saudavel["scientific_name"] is None
    assert saudavel["severity"] == SeverityEnum.NENHUMA


def test_predict_saudavel_keeps_scientific_name_none(service: InferenceService):
    """Quando o RNG escolhe saudavel, scientific_name no result vem como None."""
    random.seed(0)
    for _ in range(200):
        result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
        if result.disease_id == "saudavel":
            assert result.scientific_name is None
            return
    pytest.fail("Loop nao gerou caso 'saudavel' — ajuste de seed pode ser necessario")


def test_catalog_has_six_diseases(service: InferenceService):
    """Spec do projeto: 6 doenças foliares de soja."""
    assert len(service._DISEASES) == 6


def test_catalog_disease_ids_are_unique(service: InferenceService):
    ids = [d["id"] for d in service._DISEASES]
    assert len(ids) == len(set(ids))


# ── get_model_label ──────────────────────────────────────────────────────────


def test_get_model_label_known_model(service: InferenceService):
    assert service.get_model_label("resnet50") == "ResNet-50"
    assert service.get_model_label("efficientnet") == "EfficientNet-B4"
    assert service.get_model_label("vit") == "ViT-B/16"
    assert service.get_model_label("ensemble") == "Ensemble"


def test_get_model_label_unknown_falls_back(service: InferenceService):
    """Modelo desconhecido — volta o próprio id (sem KeyError)."""
    assert service.get_model_label("modelo-novo") == "modelo-novo"


# ── todos os model_ids enumerados ────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [ModelEnum.RESNET50, ModelEnum.EFFICIENTNET, ModelEnum.VIT, ModelEnum.ENSEMBLE],
)
def test_predict_works_for_every_model_enum(service: InferenceService, model_id):
    result = service.predict(model_id, "leaf.jpg")
    assert result.model_id == model_id
