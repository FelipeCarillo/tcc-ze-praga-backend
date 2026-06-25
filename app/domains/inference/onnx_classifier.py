"""OnnxClassifier — inferência real de doenças foliares de soja (TCC-023, ADR-0003).

Carrega o modelo ONNX treinado (EfficientNet-B4 no dataset ASDID) e classifica
bytes de imagem nas 6 classes. O pré-processamento replica EXATAMENTE o
``get_val_transforms`` do treino (model-playground): resize bilinear para o
``input_size`` + normalização ImageNet, layout NCHW float32.

Sem dependência de torch/albumentations — apenas ``onnxruntime`` + ``numpy`` +
``Pillow``. A ordem das classes vem do ``.labels.json`` ao lado do ``.onnx``,
que espelha o ``label_map.csv`` gerado por ``splits.py`` (ordem alfabética).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

# Mesmas constantes do treino (model-playground/src/data/transforms.py).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxClassifier:
    """Classificador ONNX de doenças foliares.

    Args:
        model_path: caminho do arquivo ``.onnx``.
        labels: lista de slugs na ordem dos índices de saída do modelo.
        input_size: lado da imagem de entrada (380 para EfficientNet-B4).
    """

    def __init__(self, model_path: str | Path, labels: list[str], input_size: int = 380) -> None:
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._labels = list(labels)
        self._input_size = int(input_size)

    @classmethod
    def from_path(cls, model_path: str | Path, input_size: int = 380) -> OnnxClassifier:
        """Constrói a partir do ``.onnx`` + ``.labels.json`` irmão."""
        model_path = Path(model_path)
        labels_path = model_path.with_suffix(".labels.json")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        return cls(model_path, labels, input_size=input_size)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = (
            Image.open(io.BytesIO(image_bytes))
            .convert("RGB")
            .resize((self._input_size, self._input_size), Image.Resampling.BILINEAR)
        )
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # HWC -> (1, C, H, W)
        return np.ascontiguousarray(x, dtype=np.float32)

    def predict_probs(self, image_bytes: bytes) -> dict[str, float]:
        """Retorna ``{slug: prob}`` com a distribuição completa (softmax).

        Usado pelo ensemble, que precisa do vetor inteiro de cada modelo para
        tirar a média antes de escolher o top-k.
        """
        x = self._preprocess(image_bytes)
        logits = self._session.run([self._output_name], {self._input_name: x})[0][0]
        logits = logits.astype(np.float64)
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        return {self._labels[i]: float(probs[i]) for i in range(len(self._labels))}

    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[tuple[str, float]]:
        """Retorna ``[(slug, prob), ...]`` ordenado por confiança desc (top_k)."""
        probs = self.predict_probs(image_bytes)
        order = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return [(slug, prob) for slug, prob in order[: max(1, top_k)]]
