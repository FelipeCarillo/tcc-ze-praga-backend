"""``run_inference_node`` tem que rodar o ONNX de verdade (TCC-020).

Ate aqui o node chamava ``InferenceService.predict`` **sem** ``image_bytes`` e
o router de ``/diagnoses/analyze`` mandava ``image_batch=[]`` — o caminho
inteiro do sub-grafo caia no mock aleatorio, inclusive o endpoint REST exposto
a chaves de API. Estes testes travam o contrato: os bytes chegam ao service.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

from app.domains.diagnosis_graph.nodes import _decode_image_batch, run_inference_node
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import SeverityEnum


def _result(disease_id: str = "ferrugem-asiatica") -> InferenceResult:
    return InferenceResult(
        disease_id=disease_id,
        disease_name="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença fúngica severa.",
        confidence=0.97,
        model_id="ensemble",
        image_name="folha.jpg",
        top3=[],
    )


def _svc() -> MagicMock:
    svc = MagicMock()
    svc.predict = MagicMock(return_value=_result())
    return svc


async def test_bytes_do_batch_chegam_no_predict() -> None:
    svc = _svc()
    raw = b"fake-jpeg-bytes"
    state = {
        "image_ids": ["img-1"],
        "image_batch": [base64.b64encode(raw).decode("ascii")],
        "model_id": "ensemble",
        "crop_id": "soja",
    }

    out = await run_inference_node(state, inference_svc=svc)

    assert len(out["predictions"]) == 1
    kwargs = svc.predict.call_args.kwargs
    assert kwargs["image_bytes"] == raw
    assert kwargs["model_id"] == "ensemble"
    assert kwargs["image_name"] == "img-1"


async def test_index_do_batch_alinha_com_image_ids() -> None:
    """Batch multi-imagem: cada predict recebe os bytes da SUA imagem."""
    svc = _svc()
    a, b = b"imagem-a", b"imagem-b"
    state = {
        "image_ids": ["img-a", "img-b"],
        "image_batch": [
            base64.b64encode(a).decode("ascii"),
            base64.b64encode(b).decode("ascii"),
        ],
        "model_id": "vit_b16",
    }

    await run_inference_node(state, inference_svc=svc)

    chamadas = [
        (c.kwargs["image_name"], c.kwargs["image_bytes"])
        for c in svc.predict.call_args_list
    ]
    assert chamadas == [("img-a", a), ("img-b", b)]


async def test_batch_ausente_mantem_fallback_mock() -> None:
    """Chamada legada sem ``image_batch``: ``image_bytes=None`` -> mock no service."""
    svc = _svc()
    state = {"image_ids": ["img-1"], "model_id": "ensemble"}

    await run_inference_node(state, inference_svc=svc)

    assert svc.predict.call_args.kwargs["image_bytes"] is None


async def test_batch_menor_que_image_ids_nao_estoura() -> None:
    """Defensivo: sobrar id sem bytes nao pode derrubar o lote inteiro."""
    svc = _svc()
    state = {
        "image_ids": ["img-a", "img-b"],
        "image_batch": [base64.b64encode(b"so-a").decode("ascii")],
    }

    out = await run_inference_node(state, inference_svc=svc)

    assert len(out["predictions"]) == 2
    assert svc.predict.call_args_list[1].kwargs["image_bytes"] is None


def test_decode_tolera_base64_invalido() -> None:
    """Uma imagem corrompida vira ``None`` (mock) e as outras seguem."""
    valido = base64.b64encode(b"ok").decode("ascii")
    assert _decode_image_batch([valido, "!!! nao e base64 !!!", ""]) == [
        b"ok",
        None,
        None,
    ]


def test_decode_de_batch_vazio() -> None:
    assert _decode_image_batch(None) == []
    assert _decode_image_batch([]) == []
