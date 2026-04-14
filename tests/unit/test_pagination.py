"""Tests for app/shared/pagination.py — PaginatedResponse.pages."""

from app.shared.pagination import PaginatedResponse


def test_pages_normal():
    r = PaginatedResponse(items=[], total=25, page=1, limit=10)
    assert r.pages == 3


def test_pages_exact():
    r = PaginatedResponse(items=[], total=20, page=1, limit=10)
    assert r.pages == 2


def test_pages_single():
    r = PaginatedResponse(items=[], total=5, page=1, limit=10)
    assert r.pages == 1


def test_pages_zero_total():
    r = PaginatedResponse(items=[], total=0, page=1, limit=10)
    assert r.pages == 0


def test_pages_limit_zero():
    r = PaginatedResponse(items=[], total=50, page=1, limit=0)
    assert r.pages == 0


def test_pages_one_item():
    r = PaginatedResponse(items=["x"], total=1, page=1, limit=20)
    assert r.pages == 1
