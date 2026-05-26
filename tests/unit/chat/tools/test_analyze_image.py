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
