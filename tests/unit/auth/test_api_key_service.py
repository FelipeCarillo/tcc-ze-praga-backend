"""Unit tests for ApiKeyService — generate / verify / revoke / list."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import bcrypt
import pytest

from app.domains.auth.api_key_dto import ApiKeyDTO
from app.domains.auth.api_key_service import ApiKeyService


NOW = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)


def _dto(**kwargs) -> ApiKeyDTO:
    defaults = dict(
        id="apik-1",
        user_id="user-1",
        name="my-key",
        key_hash="$2b$12$hashed",
        key_prefix="zp_live_abcd",
        scopes=["diagnoses:analyze"],
        is_active=True,
        last_used_at=None,
        created_at=NOW,
        revoked_at=None,
    )
    return ApiKeyDTO(**{**defaults, **kwargs})


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    return repo


# ── create ────────────────────────────────────────────────────────────────────

async def test_create_returns_dto_and_plain_key(mock_repo):
    persisted = _dto()
    mock_repo.create.return_value = persisted

    svc = ApiKeyService(mock_repo)
    dto, plain = await svc.create("user-1", "my-key")

    assert dto.id == "apik-1"
    assert plain.startswith("zp_live_")
    assert len(plain) > 30  # urlsafe(32) ~> 43 chars + prefix
    mock_repo.create.assert_awaited_once()


async def test_create_persists_bcrypt_hash_not_plain(mock_repo):
    captured = {}

    async def fake_create(data):
        captured["hash"] = data.key_hash
        captured["prefix"] = data.key_prefix
        return _dto(key_hash=data.key_hash, key_prefix=data.key_prefix)

    mock_repo.create.side_effect = fake_create
    svc = ApiKeyService(mock_repo)
    dto, plain = await svc.create("user-1", "my-key")

    # Hash deve ser bcrypt, nao plain text
    assert captured["hash"].startswith("$2b$")
    assert captured["hash"] != plain
    # Prefix deve casar com primeiros 12 chars
    assert captured["prefix"] == plain[:12]
    assert dto.key_prefix == plain[:12]


async def test_create_uses_default_scopes_when_none(mock_repo):
    captured = {}

    async def fake_create(data):
        captured["scopes"] = data.scopes
        return _dto(scopes=data.scopes)

    mock_repo.create.side_effect = fake_create
    svc = ApiKeyService(mock_repo)
    await svc.create("user-1", "my-key", scopes=None)

    assert captured["scopes"] == ["diagnoses:analyze"]


async def test_create_respects_custom_scopes(mock_repo):
    captured = {}

    async def fake_create(data):
        captured["scopes"] = data.scopes
        return _dto(scopes=data.scopes)

    mock_repo.create.side_effect = fake_create
    svc = ApiKeyService(mock_repo)
    await svc.create("user-1", "my-key", scopes=["custom:scope"])

    assert captured["scopes"] == ["custom:scope"]


# ── list ──────────────────────────────────────────────────────────────────────

async def test_list_for_user_delegates_to_repo(mock_repo):
    mock_repo.find_active_by_user.return_value = [_dto(), _dto(id="apik-2")]
    svc = ApiKeyService(mock_repo)
    out = await svc.list_for_user("user-1")
    assert len(out) == 2
    mock_repo.find_active_by_user.assert_awaited_once_with("user-1")


# ── revoke ────────────────────────────────────────────────────────────────────

async def test_revoke_delegates(mock_repo):
    mock_repo.revoke.return_value = True
    svc = ApiKeyService(mock_repo)
    out = await svc.revoke("user-1", "apik-1")
    assert out is True
    mock_repo.revoke.assert_awaited_once_with("apik-1", "user-1")


async def test_revoke_returns_false_when_missing(mock_repo):
    mock_repo.revoke.return_value = False
    svc = ApiKeyService(mock_repo)
    out = await svc.revoke("user-1", "ghost")
    assert out is False


# ── verify ────────────────────────────────────────────────────────────────────

async def test_verify_rejects_short_token(mock_repo):
    svc = ApiKeyService(mock_repo)
    out = await svc.verify("short")
    assert out is None
    mock_repo.find_by_prefix_active.assert_not_called()


async def test_verify_rejects_token_with_bad_prefix(mock_repo):
    svc = ApiKeyService(mock_repo)
    out = await svc.verify("xx_test_aaaaaaaa")
    assert out is None
    mock_repo.find_by_prefix_active.assert_not_called()


async def test_verify_rejects_empty(mock_repo):
    svc = ApiKeyService(mock_repo)
    assert await svc.verify("") is None
    assert await svc.verify(None) is None  # type: ignore[arg-type]


async def test_verify_returns_dto_when_hash_matches(mock_repo):
    plain = "zp_live_abcdEFGH_realtoken_xxxxxxxx"
    real_hash = bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

    candidate = _dto(key_hash=real_hash, key_prefix=plain[:12])
    mock_repo.find_by_prefix_active.return_value = [candidate]

    svc = ApiKeyService(mock_repo)
    out = await svc.verify(plain)
    assert out is not None
    assert out.id == "apik-1"


async def test_verify_returns_none_when_no_match_for_prefix(mock_repo):
    mock_repo.find_by_prefix_active.return_value = []
    svc = ApiKeyService(mock_repo)
    out = await svc.verify("zp_live_AAAAaaaa_garbage")
    assert out is None


async def test_verify_skips_candidate_with_wrong_hash(mock_repo):
    plain = "zp_live_abcdEFGH_realtoken_xxxxxxxx"
    wrong_hash = bcrypt.hashpw(b"different-token", bcrypt.gensalt()).decode()
    candidate = _dto(key_hash=wrong_hash, key_prefix=plain[:12])
    mock_repo.find_by_prefix_active.return_value = [candidate]

    svc = ApiKeyService(mock_repo)
    out = await svc.verify(plain)
    assert out is None


async def test_verify_picks_correct_candidate_among_multiple(mock_repo):
    plain_a = "zp_live_abcdEFGH_token_A_xxxxxxxx"
    plain_b = "zp_live_abcdWXYZ_token_B_xxxxxxxx"  # same 12-char prefix dist
    hash_a = bcrypt.hashpw(plain_a.encode(), bcrypt.gensalt()).decode()
    hash_b = bcrypt.hashpw(plain_b.encode(), bcrypt.gensalt()).decode()

    cand_a = _dto(id="apik-A", key_hash=hash_a, key_prefix=plain_a[:12])
    cand_b = _dto(id="apik-B", key_hash=hash_b, key_prefix=plain_a[:12])
    mock_repo.find_by_prefix_active.return_value = [cand_a, cand_b]

    svc = ApiKeyService(mock_repo)
    out = await svc.verify(plain_a)
    assert out is not None
    assert out.id == "apik-A"


# ── touch_last_used ───────────────────────────────────────────────────────────

async def test_touch_last_used_delegates(mock_repo):
    svc = ApiKeyService(mock_repo)
    await svc.touch_last_used("apik-1")
    mock_repo.touch_last_used.assert_awaited_once_with("apik-1")


# ── helpers ───────────────────────────────────────────────────────────────────

def test_generate_token_has_correct_prefix_and_length():
    token = ApiKeyService._generate_token()
    assert token.startswith("zp_live_")
    assert len(token) >= 40  # urlsafe(32) bytes ≈ 43 base64-url chars


def test_hash_and_check_roundtrip():
    plain = "zp_live_xxxxx_yyyyy_zzzzz"
    h = ApiKeyService._hash(plain)
    assert ApiKeyService._check(plain, h)
    assert not ApiKeyService._check("other", h)


def test_check_handles_invalid_hash_gracefully():
    # bcrypt raises ValueError on malformed hash — should return False, not crash
    assert ApiKeyService._check("anything", "not-a-bcrypt-hash") is False
