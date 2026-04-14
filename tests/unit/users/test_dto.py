"""Tests for app/domains/users/dto.py — ensures re-exports are importable."""

from app.domains.users.dto import UserCreateDTO, UserDTO


def test_user_dto_re_export():
    """UserDTO and UserCreateDTO are re-exported from users.dto."""
    assert UserDTO is not None
    assert UserCreateDTO is not None


def test_user_dto_instantiation():
    from datetime import UTC, datetime

    dto = UserDTO(
        id="u1",
        email="a@b.com",
        password_hash="hash",
        full_name="Alice",
        is_active=True,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert dto.id == "u1"
