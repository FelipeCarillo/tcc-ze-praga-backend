"""Testa o OnnxClassifier real (TCC-023, ADR-0003).

Carrega o modelo ONNX shippado no repo (models/soja_efficientnet_b4.onnx) e roda
uma imagem sintética — valida wiring (carregamento, pré-processamento, softmax,
ordem das classes) sem depender do dataset completo.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.domains.inference.onnx_classifier import OnnxClassifier

MODEL = Path(__file__).resolve().parents[2] / "models" / "soja_efficientnet_b4.onnx"
LABELS = {
    "cercosporiose",
    "ferrugem-asiatica",
    "mancha-alvo",
    "mancha-olho-de-ra",
    "mildio",
    "saudavel",
}

def _real_model_available() -> bool:
    """True só quando o ONNX real está presente.

    Em CI sem Git LFS o arquivo existe, mas é apenas um ponteiro (~130 bytes);
    o modelo real tem dezenas de MB. Sem essa checagem, o onnxruntime falha o
    parse do protobuf no ponteiro. Estes testes de integração então pulam.
    """
    try:
        return MODEL.exists() and MODEL.stat().st_size > 1_000_000
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _real_model_available(),
    reason="modelo ONNX real ausente (ponteiro LFS em CI) — teste de integração pulado",
)


def _synthetic_jpeg() -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_labels_match_training_order() -> None:
    clf = OnnxClassifier.from_path(MODEL, input_size=380)
    assert set(clf.labels) == LABELS
    assert clf.labels[0] == "cercosporiose"  # ordem alfabética do label_map


def test_predict_returns_valid_topk() -> None:
    clf = OnnxClassifier.from_path(MODEL, input_size=380)
    preds = clf.predict(_synthetic_jpeg(), top_k=3)

    assert len(preds) == 3
    slugs = [s for s, _ in preds]
    probs = [p for _, p in preds]

    assert all(s in LABELS for s in slugs)
    assert len(set(slugs)) == 3  # sem duplicatas
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)  # ordenado desc
