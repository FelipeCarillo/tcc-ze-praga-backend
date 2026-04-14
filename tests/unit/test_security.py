"""Tests for app/core/security.py."""

import pytest
from jose import jwt

from app.config import settings
from app.core.exceptions import UnauthorizedError
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_returns_string():
    result = hash_password("mypassword123")
    assert isinstance(result, str)
    assert result != "mypassword123"


def test_hash_password_is_bcrypt():
    result = hash_password("secret_pass")
    assert result.startswith("$2b$")


def test_verify_password_correct():
    hashed = hash_password("correct_pass")
    assert verify_password("correct_pass", hashed) is True


def test_verify_password_wrong():
    hashed = hash_password("correct_pass")
    assert verify_password("wrong_pass", hashed) is False


def test_create_access_token_returns_string():
    token = create_access_token("user-id-123")
    assert isinstance(token, str)


def test_create_access_token_has_sub():
    token = create_access_token("user-id-123")
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert payload["sub"] == "user-id-123"


def test_decode_access_token_valid():
    token = create_access_token("user-id-abc")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-id-abc"


def test_decode_access_token_missing_sub():
    # Build a token manually without "sub"
    from datetime import UTC, datetime, timedelta

    payload = {"exp": datetime.now(UTC) + timedelta(hours=1)}
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    with pytest.raises(UnauthorizedError, match="Invalid token payload"):
        decode_access_token(token)


def test_decode_access_token_invalid():
    with pytest.raises(UnauthorizedError, match="Invalid or expired token"):
        decode_access_token("not.a.valid.token")


def test_decode_access_token_wrong_signature():
    token = create_access_token("user-id")
    # Tamper with signature
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + ".invalidsig"
    with pytest.raises(UnauthorizedError, match="Invalid or expired token"):
        decode_access_token(tampered)
