"""Gate de modelo por plano (TCC-051).

``PlanFeatures.diagnosis_models`` existia desde o TCC-049 mas nao era lido em
lugar nenhum — qualquer usuario pedia ``ensemble`` e recebia. Estes testes
fixam o contrato do resolvedor que passou a ser aplicado no chat e nos dois
endpoints REST.
"""

from __future__ import annotations

import pytest

from app.domains.inference.service import resolve_allowed_model
from app.domains.subscriptions.features import (
    ENTERPRISE_FEATURES,
    FREE_FEATURES,
    PRO_FEATURES,
)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("ensemble", "ensemble"),
        ("vit", "vit_b16"),
        ("vit_b16", "vit_b16"),
        ("efficientnet", "efficientnet_b4"),
        ("EfficientNet-B4", "efficientnet_b4"),
        ("resnet50", "resnet50"),
    ],
)
def test_sem_gate_apenas_normaliza(requested: str, expected: str) -> None:
    """Sem lista de permitidos, o resolvedor so' canoniza o vocabulario."""
    model, downgraded = resolve_allowed_model(requested, None)
    assert (model, downgraded) == (expected, False)


def test_free_cai_no_resnet_quando_pede_ensemble() -> None:
    """Free so' tem ResNet-50 — o pedido de ensemble e' rebaixado, nao negado."""
    model, downgraded = resolve_allowed_model(
        "ensemble", FREE_FEATURES.diagnosis_models
    )
    assert model == "resnet50"
    assert downgraded is True


def test_pro_sem_ensemble_cai_no_melhor_permitido() -> None:
    """Pro tem os 3 unicos: o melhor deles pela acuracia e' o EfficientNet-B4."""
    model, downgraded = resolve_allowed_model(
        "ensemble", PRO_FEATURES.diagnosis_models
    )
    assert model == "efficientnet_b4"
    assert downgraded is True


def test_pro_mantem_modelo_permitido() -> None:
    model, downgraded = resolve_allowed_model("vit", PRO_FEATURES.diagnosis_models)
    assert (model, downgraded) == ("vit_b16", False)


def test_enterprise_mantem_ensemble() -> None:
    model, downgraded = resolve_allowed_model(
        "ensemble", ENTERPRISE_FEATURES.diagnosis_models
    )
    assert (model, downgraded) == ("ensemble", False)


def test_vocabulario_do_plano_e_normalizado_dos_dois_lados() -> None:
    """O plano guarda ``efficientnet``; o REST manda ``efficientnet_b4``.

    Sem normalizar os dois lados, um usuario Pro pedindo pelo id do REST seria
    rebaixado por engano.
    """
    model, downgraded = resolve_allowed_model("efficientnet_b4", ["efficientnet"])
    assert (model, downgraded) == ("efficientnet_b4", False)


def test_lista_desconhecida_nao_trava_o_diagnostico() -> None:
    """Plano com modelos que nao normalizam: mantem o pedido em vez de falhar."""
    model, downgraded = resolve_allowed_model("resnet50", [])
    assert (model, downgraded) == ("resnet50", False)


def test_none_vira_ensemble() -> None:
    assert resolve_allowed_model(None, None) == ("ensemble", False)
