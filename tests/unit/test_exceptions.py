"""Tests for app/core/exceptions.py — all exception constructors."""

import pytest

from app.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    QuotaExceededError,
    UnauthorizedError,
)
from app.shared.enums import FeatureTypeEnum


def test_not_found_without_identifier():
    exc = NotFoundError("User")
    assert exc.detail == "User not found"
    assert str(exc) == "User not found"


def test_not_found_with_identifier():
    exc = NotFoundError("User", "uuid-123")
    assert exc.detail == "User 'uuid-123' not found"


def test_unauthorized_default():
    exc = UnauthorizedError()
    assert exc.detail == "Unauthorized"


def test_unauthorized_custom():
    exc = UnauthorizedError("Token expired")
    assert exc.detail == "Token expired"


def test_forbidden_default():
    exc = ForbiddenError()
    assert exc.detail == "Access forbidden"


def test_forbidden_custom():
    exc = ForbiddenError("No access to resource")
    assert exc.detail == "No access to resource"


def test_conflict_default():
    exc = ConflictError()
    assert exc.detail == "Resource already exists"


def test_conflict_custom():
    exc = ConflictError("Email already in use")
    assert exc.detail == "Email already in use"


def test_quota_exceeded():
    exc = QuotaExceededError(FeatureTypeEnum.INFERENCE, limit=5, used=5)
    assert exc.feature == FeatureTypeEnum.INFERENCE
    assert exc.limit == 5
    assert exc.used == 5
    assert "inference" in exc.detail
    assert "5/5" in exc.detail


def test_quota_exceeded_inherits_exception():
    exc = QuotaExceededError(FeatureTypeEnum.CHAT, limit=10, used=11)
    assert isinstance(exc, Exception)
