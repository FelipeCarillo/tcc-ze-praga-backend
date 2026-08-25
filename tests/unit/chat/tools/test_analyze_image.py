"""Testes da tool analyze_image (TCC-079) — state-aware + persistência."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.tools.analyze_image import build_analyze_image_tool
from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import SeverityEnum

NOW = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)


def _file(file_id: str = "img-1") -> UploadedFileDTO:
    return UploadedFileDTO(
        id=file_id,
        original_name="folha.jpg",
        mime="image/jpeg",
        storage_key="",
        size_bytes=10,
        b64="ZmFrZQ==",
    )


def _inference_result() -> InferenceResult:
    top3 = [
        Top3PredictionSchema(
            rank=1,
            disease_name="Ferrugem Asiática",
            disease_id="ferrugem-asiatica",
            scientific_name="Phakopsora pachyrhizi",
            confidence=0.92,
            severity="alta",
        )
    ]
    return InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="...",
        confidence=0.92,
        model_id="ensemble",
        image_name="folha.jpg",
        top3=top3,
    )


def _diagnosis_response() -> DiagnosisResponse:
    return DiagnosisResponse(
        id="diag-1",
        disease_name="Ferrugem Asiática",
        disease_id="ferrugem-asiatica",
        scientific_name="Phakopsora pachyrhizi",
        confidence=0.92,
        severity="alta",
        description="...",
        model_used="ensemble",
        image_url=None,
        image_name="folha.jpg",
        created_at=NOW,
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name="Ferrugem Asiática",
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.92,
                severity="alta",
            )
        ],
    )


def _services():
    inference = MagicMock()
    inference.predict.return_value = _inference_result()
    inference.disease_catalog = [MagicMock(crop_id="crop-uuid")]
    diagnosis = AsyncMock()
    diagnosis.create.return_value = _diagnosis_response()
    return inference, diagnosis


async def test_analyze_image_persists_and_records_diagnosis() -> None:
    inference, diagnosis = _services()
    tool = build_analyze_image_tool(inference, diagnosis)
    state = {"uploaded_files": [_file()], "current_user_id": "user-1", "selected_model": "ensemble"}

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert isinstance(result, Command)
    assert result.update["diagnoses_in_turn"] == ["diag-1"]
    # Persistiu via diagnosis_svc.create com o user do state.
    diagnosis.create.assert_awaited_once()
    assert diagnosis.create.await_args.args[0] == "user-1"
    inference.predict.assert_called_once()
    # ToolMessage carrega o payload com o diagnosis_id.
    msg = result.update["messages"][0]
    assert isinstance(msg, ToolMessage)
    payload = json.loads(msg.content)
    assert payload["diagnosis_id"] == "diag-1"
    assert payload["disease_id"] == "ferrugem-asiatica"


async def test_analyze_image_no_image_returns_error_without_persisting() -> None:
    inference, diagnosis = _services()
    tool = build_analyze_image_tool(inference, diagnosis)
    state = {"uploaded_files": [], "current_user_id": "user-1"}

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert isinstance(result, Command)
    assert "diagnoses_in_turn" not in result.update
    diagnosis.create.assert_not_awaited()
    inference.predict.assert_not_called()
    payload = json.loads(result.update["messages"][0].content)
    assert "error" in payload


# ── Memória semântica (pgvector) ──────────────────────────────────────────────
#
# O caminho vivo do chat criava o Diagnosis e nunca o indexava no Store: só o
# ``persist_node`` do diagnosis_graph chamava ``index_diagnosis_in_store``, e
# esse node era inalcançável em produção. Resultado: o namespace
# ("user", uid, "diagnoses") ficava sempre vazio e ``search_my_diagnoses``
# nunca achava nada. Estes testes travam a indexação no caminho vivo.


async def test_analyze_image_indexa_diagnostico_no_store() -> None:
    inference, diagnosis = _services()
    store = AsyncMock()
    tool = build_analyze_image_tool(
        inference, diagnosis, store_factory=AsyncMock(return_value=store)
    )
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
    }

    await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    store.aput.assert_awaited_once()
    kwargs = store.aput.await_args.kwargs
    assert kwargs["namespace"] == ("user", "user-1", "diagnoses")
    assert kwargs["key"] == "diag-1"
    assert kwargs["value"]["disease_id"] == "ferrugem-asiatica"


async def test_store_fora_do_ar_nao_derruba_o_diagnostico() -> None:
    """O Diagnosis já está no banco — memória semântica é best-effort."""
    inference, diagnosis = _services()
    tool = build_analyze_image_tool(
        inference,
        diagnosis,
        store_factory=AsyncMock(side_effect=RuntimeError("pgvector offline")),
    )
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
    }

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert result.update["diagnoses_in_turn"] == ["diag-1"]


async def test_sem_store_factory_mantem_comportamento_antigo() -> None:
    inference, diagnosis = _services()
    tool = build_analyze_image_tool(inference, diagnosis)
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
    }

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert result.update["diagnoses_in_turn"] == ["diag-1"]


# ── Gate de modelo por plano ──────────────────────────────────────────────────


async def test_plano_free_rebaixa_ensemble_para_resnet() -> None:
    from app.domains.subscriptions.features import FREE_FEATURES

    inference, diagnosis = _services()
    tool = build_analyze_image_tool(inference, diagnosis)
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
        "plan_features": FREE_FEATURES,
    }

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert inference.predict.call_args.args[0] == "resnet50"
    payload = json.loads(result.update["messages"][0].content)
    assert payload["model_downgraded_to"] == "resnet50"
    assert "note" in payload


async def test_plano_enterprise_mantem_ensemble() -> None:
    from app.domains.subscriptions.features import ENTERPRISE_FEATURES

    inference, diagnosis = _services()
    tool = build_analyze_image_tool(inference, diagnosis)
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
        "plan_features": ENTERPRISE_FEATURES,
    }

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert inference.predict.call_args.args[0] == "ensemble"
    payload = json.loads(result.update["messages"][0].content)
    assert "model_downgraded_to" not in payload


# ── Foto no Storage ───────────────────────────────────────────────────────────
#
# A imagem do chat é efêmera (base64 no estado do turno). Sem subir pro Storage,
# `image_url` ficava None e o histórico nunca tinha miniatura.


async def test_analyze_image_sobe_a_foto_e_grava_a_storage_key() -> None:
    inference, diagnosis = _services()
    upload = AsyncMock()
    upload.upload = AsyncMock(
        return_value=(MagicMock(storage_key="users/user-1/abc-folha.jpg"), False)
    )
    tool = build_analyze_image_tool(inference, diagnosis, upload_svc=upload)
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
    }

    await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    upload.upload.assert_awaited_once()
    kwargs = upload.upload.await_args.kwargs
    assert kwargs["user_id"] == "user-1"
    assert kwargs["original_name"] == "folha.jpg"
    assert kwargs["data"] == b"fake"  # b64 "ZmFrZQ==" decodificado

    body = diagnosis.create.await_args.args[1]
    assert body.image_url == "users/user-1/abc-folha.jpg"


async def test_falha_de_storage_nao_impede_o_diagnostico() -> None:
    """A inferência é o que o usuário veio buscar; a miniatura é acessória."""
    inference, diagnosis = _services()
    upload = AsyncMock()
    upload.upload = AsyncMock(side_effect=RuntimeError("bucket offline"))
    tool = build_analyze_image_tool(inference, diagnosis, upload_svc=upload)
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
    }

    result = await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert result.update["diagnoses_in_turn"] == ["diag-1"]
    assert diagnosis.create.await_args.args[1].image_url is None


async def test_sem_upload_svc_o_diagnostico_fica_sem_imagem() -> None:
    inference, diagnosis = _services()
    tool = build_analyze_image_tool(inference, diagnosis)
    state = {
        "uploaded_files": [_file()],
        "current_user_id": "user-1",
        "selected_model": "ensemble",
    }

    await tool.ainvoke(
        {
            "name": "analyze_image",
            "args": {"image_id": None, "state": state},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert diagnosis.create.await_args.args[1].image_url is None
