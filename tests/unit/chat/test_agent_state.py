"""Testes do ChatState expandido + UploadedFileDTO + resolve_image (TCC-038)."""

from __future__ import annotations

from app.domains.chat.agent_state import (
    ChatState,
    UploadedFileDTO,
    resolve_image,
)


def _make_file(file_id: str = "img-1") -> UploadedFileDTO:
    return UploadedFileDTO(
        id=file_id,
        original_name=f"{file_id}.jpg",
        mime="image/jpeg",
        storage_key=f"uploads/u1/{file_id}.jpg",
        size_bytes=1024,
    )


def test_uploaded_file_dto_has_b64_default_none() -> None:
    """``b64`` deve ser ``None`` por default — preguicoso por design."""
    f = _make_file()
    assert f.b64 is None
    assert f.id == "img-1"
    assert f.original_name == "img-1.jpg"
    assert f.mime == "image/jpeg"


def test_uploaded_file_dto_can_carry_b64() -> None:
    """Quando uma tool carrega bytes sob demanda, b64 fica disponivel."""
    f = UploadedFileDTO(
        id="img-2",
        original_name="x.png",
        mime="image/png",
        storage_key="uploads/u1/x.png",
        size_bytes=2048,
        b64="aGVsbG8=",
    )
    assert f.b64 == "aGVsbG8="


def test_resolve_image_returns_matching_file() -> None:
    f1 = _make_file("img-a")
    f2 = _make_file("img-b")
    state: ChatState = {"uploaded_files": [f1, f2]}

    assert resolve_image(state, "img-b") is f2
    assert resolve_image(state, "img-a") is f1


def test_resolve_image_returns_none_when_id_not_found() -> None:
    state: ChatState = {"uploaded_files": [_make_file("img-1")]}
    assert resolve_image(state, "img-missing") is None


def test_resolve_image_returns_none_on_empty_list() -> None:
    state: ChatState = {"uploaded_files": []}
    assert resolve_image(state, "img-1") is None


def test_resolve_image_returns_none_when_key_absent() -> None:
    """``uploaded_files`` ausente do dict (total=False) — usa default []."""
    state: ChatState = {}
    assert resolve_image(state, "img-1") is None


def test_chat_state_accepts_partial_keys() -> None:
    """ChatState eh ``total=False`` — campos opcionais nao quebram type-check."""
    state: ChatState = {
        "current_user_id": "u-1",
        "current_session_id": "s-1",
        "selected_model": "ensemble",
    }
    assert state["current_user_id"] == "u-1"
    assert state.get("preferred_action_level") is None
