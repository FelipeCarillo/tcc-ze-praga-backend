"""Tests da troca real de modelo + ensemble (TCC-095).

Cobre ``normalize_model_id`` (aliases chat/REST) e o roteamento de ``predict``:
modelo específico → aquele classifier; ``ensemble`` → média das probabilidades.
Usa fakes (sem onnxruntime) — a inferência ONNX real é exercida nos testes live.
"""

from __future__ import annotations

import pytest

from app.domains.inference.service import (
    EFFICIENTNET_B4,
    ENSEMBLE,
    RESNET50,
    VIT_B16,
    InferenceService,
    normalize_model_id,
)

from .test_service import SIX_SOJA_DISEASES


class FakeClassifier:
    """Classifier determinístico — devolve uma distribuição fixa por slug."""

    def __init__(self, probs: dict[str, float]) -> None:
        self._probs = probs

    def predict_probs(self, image_bytes: bytes) -> dict[str, float]:  # noqa: ARG002
        return dict(self._probs)

    def predict(self, image_bytes: bytes, top_k: int = 3):  # noqa: ARG002
        order = sorted(self._probs.items(), key=lambda kv: kv[1], reverse=True)
        return order[: max(1, top_k)]


# ── normalize_model_id ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ensemble", ENSEMBLE),
        ("efficientnet", EFFICIENTNET_B4),      # alias do chat
        ("efficientnet_b4", EFFICIENTNET_B4),   # id do REST/docs
        ("EfficientNet-B4", EFFICIENTNET_B4),
        ("vit", VIT_B16),                       # alias do chat
        ("vit_b16", VIT_B16),                   # id do REST/docs
        ("ViT-B16", VIT_B16),
        ("resnet50", RESNET50),
        ("resnet-50", RESNET50),
        ("desconhecido", ENSEMBLE),             # fallback
        (None, ENSEMBLE),
        ("", ENSEMBLE),
    ],
)
def test_normalize_model_id(raw, expected):
    assert normalize_model_id(raw) == expected


# ── roteamento por modelo ────────────────────────────────────────────────────


def test_predict_routes_to_selected_model():
    clfs = {
        RESNET50: FakeClassifier({"ferrugem-asiatica": 0.9, "mancha-alvo": 0.1}),
        VIT_B16: FakeClassifier({"mancha-alvo": 0.8, "ferrugem-asiatica": 0.2}),
    }
    svc = InferenceService(diseases=SIX_SOJA_DISEASES, classifiers=clfs)

    r_resnet = svc.predict("resnet50", "x.jpg", image_bytes=b"img")
    r_vit = svc.predict("vit", "x.jpg", image_bytes=b"img")

    assert r_resnet.disease_id == "ferrugem-asiatica"
    assert r_vit.disease_id == "mancha-alvo"
    # modelos diferentes → resultados diferentes (a prova que a troca funciona)
    assert r_resnet.disease_id != r_vit.disease_id


def test_ensemble_averages_probabilities():
    # resnet aposta em ferrugem (0.6); vit aposta em mancha-alvo (0.6).
    # média: ferrugem (0.6+0.2)/2=0.40 vs mancha-alvo (0.2+0.6)/2=0.40 — empate;
    # cercosporiose 0.2/2=0.10 dos dois → 0.10. Usamos pesos que dão vencedor claro.
    clfs = {
        RESNET50: FakeClassifier(
            {"ferrugem-asiatica": 0.7, "mancha-alvo": 0.2, "cercosporiose": 0.1}
        ),
        VIT_B16: FakeClassifier(
            {"ferrugem-asiatica": 0.5, "mancha-alvo": 0.4, "cercosporiose": 0.1}
        ),
    }
    svc = InferenceService(diseases=SIX_SOJA_DISEASES, classifiers=clfs)
    r = svc.predict("ensemble", "x.jpg", image_bytes=b"img")

    assert r.disease_id == "ferrugem-asiatica"
    # confiança = média (0.7+0.5)/2 = 0.60
    assert r.confidence == pytest.approx(0.60, abs=1e-3)
    assert r.model_id == "ensemble"


def test_unknown_model_falls_back_to_ensemble_behavior():
    clfs = {
        EFFICIENTNET_B4: FakeClassifier({"saudavel": 0.95, "mildio": 0.05}),
    }
    svc = InferenceService(diseases=SIX_SOJA_DISEASES, classifiers=clfs)
    # id desconhecido → ensemble → único modelo do registro
    r = svc.predict("xpto", "x.jpg", image_bytes=b"img")
    assert r.disease_id == "saudavel"


def test_no_image_bytes_uses_mock_even_with_classifiers():
    clfs = {EFFICIENTNET_B4: FakeClassifier({"saudavel": 1.0})}
    svc = InferenceService(diseases=SIX_SOJA_DISEASES, classifiers=clfs)
    r = svc.predict("efficientnet", "x.jpg")  # sem image_bytes → mock
    assert r.model_id == "efficientnet"
