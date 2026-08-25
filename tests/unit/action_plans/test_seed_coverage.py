"""Toda classe que o modelo prevê precisa ter plano de ação seedado.

O ADR-0003 trocou o Digipathos pelo ASDID e substituiu antracnose por
mancha-olho-de-rã. O seed de doenças foi atualizado na época; o de planos de
ação, não. O resultado ficou invisível por meses: um plano órfão de antracnose
(doença que não existe mais no catálogo) e **uma das 6 classes do modelo sem
plano nenhum** — o usuário fotografava uma folha com mancha-olho-de-rã, o
diagnóstico saía certo, e a receita vinha como "plano indisponível".

Este teste amarra as duas listas, para que a próxima troca de dataset não
consiga passar despercebida.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.seed_action_plans import ACTION_PLANS

_LABELS_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "soja_efficientnet_b4.labels.json"
)


def _model_classes() -> set[str]:
    return set(json.loads(_LABELS_PATH.read_text(encoding="utf-8")))


def _seeded_disease_ids() -> set[str]:
    return {plan["disease_id"] for plan in ACTION_PLANS}


@pytest.mark.skipif(
    not _LABELS_PATH.exists(),
    reason="labels.json vem via Git LFS — ausente em checkout sem LFS",
)
def test_toda_classe_do_modelo_tem_plano_de_acao() -> None:
    faltando = _model_classes() - _seeded_disease_ids()
    assert not faltando, (
        f"Classes previstas pelo modelo sem plano de ação: {sorted(faltando)}. "
        "O usuário receberia um diagnóstico correto sem receita."
    )


@pytest.mark.skipif(
    not _LABELS_PATH.exists(),
    reason="labels.json vem via Git LFS — ausente em checkout sem LFS",
)
def test_nao_ha_plano_orfao() -> None:
    orfaos = _seeded_disease_ids() - _model_classes()
    assert not orfaos, (
        f"Planos de ação para doenças fora do catálogo: {sorted(orfaos)}. "
        "Provável resquício de troca de dataset (ver ADR-0003)."
    )


def test_todo_plano_tem_os_tres_niveis() -> None:
    """Enterprise paga por 'especialista' — nenhum plano pode faltar com ele."""
    for plan in ACTION_PLANS:
        niveis = set(plan["levels"])
        assert niveis == {"essencial", "campo", "especialista"}, (
            f"{plan['disease_id']} tem níveis {sorted(niveis)}"
        )


def test_nenhum_nivel_vem_vazio() -> None:
    for plan in ACTION_PLANS:
        for nivel, acoes in plan["levels"].items():
            assert acoes, f"{plan['disease_id']}/{nivel} está sem ações"
