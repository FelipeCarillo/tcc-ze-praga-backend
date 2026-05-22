"""Tests para InferenceService — mock CNN/ViT determinístico via seed.

Após o epic TCC-025 o catálogo de doenças vem injetado via construtor
(``diseases: list[DiseaseDTO]``), não mais como ClassVar. O service é
"puro" — repos não tocados aqui (cobertura em test_repository.py).
"""

import random

import pytest

from app.domains.inference.repository import DiseaseDTO
from app.domains.inference.schemas import InferenceResult
from app.domains.inference.service import InferenceService
from app.shared.enums import ModelEnum, SeverityEnum


# ── fixtures ─────────────────────────────────────────────────────────────────


def make_disease(
    *,
    id: str = "d-1",
    crop_id: str = "soja-id",
    slug: str = "ferrugem-asiatica",
    name_pt: str = "Ferrugem Asiática",
    scientific_name: str | None = "Phakopsora pachyrhizi",
    severity_default: str = "alta",
    description_md: str | None = "Desc",
    image_url: str | None = None,
) -> DiseaseDTO:
    return DiseaseDTO(
        id=id,
        crop_id=crop_id,
        slug=slug,
        name_pt=name_pt,
        scientific_name=scientific_name,
        severity_default=severity_default,
        description_md=description_md,
        image_url=image_url,
    )


SIX_SOJA_DISEASES = [
    make_disease(
        id="d-1",
        slug="ferrugem-asiatica",
        name_pt="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity_default="alta",
    ),
    make_disease(
        id="d-2",
        slug="mancha-alvo",
        name_pt="Mancha-Alvo",
        scientific_name="Corynespora cassiicola",
        severity_default="media",
    ),
    make_disease(
        id="d-3",
        slug="antracnose",
        name_pt="Antracnose",
        scientific_name="Colletotrichum truncatum",
        severity_default="media",
    ),
    make_disease(
        id="d-4",
        slug="cercosporiose",
        name_pt="Cercosporiose",
        scientific_name="Cercospora kikuchii",
        severity_default="baixa",
    ),
    make_disease(
        id="d-5",
        slug="mildio",
        name_pt="Míldio",
        scientific_name="Peronospora manshurica",
        severity_default="baixa",
    ),
    make_disease(
        id="d-6",
        slug="saudavel",
        name_pt="Saudável",
        scientific_name=None,
        severity_default="nenhuma",
        description_md="Folha saudável",
    ),
]


@pytest.fixture
def service() -> InferenceService:
    return InferenceService(diseases=SIX_SOJA_DISEASES)


@pytest.fixture(autouse=True)
def _deterministic_random():
    random.seed(42)
    yield


# ── construtor ───────────────────────────────────────────────────────────────


def test_service_requires_non_empty_catalog():
    with pytest.raises(ValueError):
        InferenceService(diseases=[])


def test_service_makes_internal_copy_of_catalog():
    """Mutar a list passada nao afeta o catalogo interno."""
    externals = [SIX_SOJA_DISEASES[0]]
    svc = InferenceService(diseases=externals)
    externals.append(SIX_SOJA_DISEASES[1])
    assert len(svc.disease_catalog) == 1


# ── predict() ────────────────────────────────────────────────────────────────


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
    """O disease_id retornado precisa estar no catalogo."""
    valid_slugs = {d.slug for d in SIX_SOJA_DISEASES}
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert result.disease_id in valid_slugs


def test_predict_confidence_is_positive_and_capped(service: InferenceService):
    for _ in range(20):
        result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
        assert result.confidence > 0.0
        assert result.confidence <= 0.99


def test_predict_severity_is_enum(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert isinstance(result.severity, SeverityEnum)


def test_predict_accepts_crop_id_kwarg(service: InferenceService):
    """``crop_id`` esta reservado para sprint A2 — assinatura nao quebra hoje."""
    result = service.predict("ensemble", "x.jpg", crop_id="soja")
    assert isinstance(result, InferenceResult)


# ── top3 ─────────────────────────────────────────────────────────────────────


def test_top3_has_three_entries(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert len(result.top3) == 3


def test_top3_ranks_are_1_2_3_in_order(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    assert [t.rank for t in result.top3] == [1, 2, 3]


def test_top3_first_matches_primary_prediction(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    first = result.top3[0]
    assert first.disease_id == result.disease_id
    assert first.disease_name == result.disease_name
    assert first.confidence == result.confidence


def test_top3_all_disease_ids_are_unique(service: InferenceService):
    for _ in range(10):
        result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
        ids = [t.disease_id for t in result.top3]
        assert len(set(ids)) == 3


def test_top3_confidences_are_non_negative(service: InferenceService):
    result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
    for entry in result.top3:
        assert entry.confidence >= 0.0


# ── catalogo / lookups ───────────────────────────────────────────────────────


def test_get_disease_by_slug_finds(service: InferenceService):
    found = service.get_disease_by_slug("ferrugem-asiatica")
    assert found is not None
    assert found.name_pt == "Ferrugem Asiática"


def test_get_disease_by_slug_returns_none_when_missing(service: InferenceService):
    assert service.get_disease_by_slug("nao-existe") is None


def test_disease_catalog_property_returns_copy(service: InferenceService):
    """Mutar o retorno do property nao afeta o catalogo interno."""
    catalog = service.disease_catalog
    catalog.clear()
    assert len(service.disease_catalog) == 6


def test_saudavel_has_no_scientific_name(service: InferenceService):
    saudavel = service.get_disease_by_slug("saudavel")
    assert saudavel is not None
    assert saudavel.scientific_name is None
    assert SeverityEnum(saudavel.severity_default) == SeverityEnum.NENHUMA


def test_predict_saudavel_keeps_scientific_name_none(service: InferenceService):
    random.seed(0)
    for _ in range(200):
        result = service.predict(ModelEnum.ENSEMBLE, "x.jpg")
        if result.disease_id == "saudavel":
            assert result.scientific_name is None
            return
    pytest.fail("Loop nao gerou caso 'saudavel' — ajuste de seed pode ser necessario")


def test_catalog_has_six_diseases(service: InferenceService):
    assert len(service.disease_catalog) == 6


# ── get_model_label ──────────────────────────────────────────────────────────


def test_get_model_label_known_model(service: InferenceService):
    assert service.get_model_label("resnet50") == "ResNet-50"
    assert service.get_model_label("efficientnet") == "EfficientNet-B4"
    assert service.get_model_label("vit") == "ViT-B/16"
    assert service.get_model_label("ensemble") == "Ensemble"


def test_get_model_label_unknown_falls_back(service: InferenceService):
    assert service.get_model_label("modelo-novo") == "modelo-novo"


# ── parametrize por model_id ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [ModelEnum.RESNET50, ModelEnum.EFFICIENTNET, ModelEnum.VIT, ModelEnum.ENSEMBLE],
)
def test_predict_works_for_every_model_enum(service: InferenceService, model_id):
    result = service.predict(model_id, "leaf.jpg")
    assert result.model_id == model_id
