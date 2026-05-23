"""Smoke tests for ApiKey model.

Não toca DB — valida metadata SQLAlchemy: colunas, índices, FK, defaults.
"""

from app.models.api_key import ApiKey


def test_api_key_tablename():
    assert ApiKey.__tablename__ == "api_keys"


def test_api_key_required_columns_present():
    cols = {c.name for c in ApiKey.__table__.columns}
    assert {
        "id",
        "user_id",
        "name",
        "key_hash",
        "key_prefix",
        "scopes",
        "is_active",
        "last_used_at",
        "created_at",
        "revoked_at",
    } <= cols


def test_api_key_key_hash_unique():
    key_hash_col = ApiKey.__table__.columns["key_hash"]
    assert key_hash_col.unique is True


def test_api_key_key_prefix_length_12():
    prefix_col = ApiKey.__table__.columns["key_prefix"]
    assert prefix_col.type.length == 12


def test_api_key_user_id_fk_cascade():
    fks = list(ApiKey.__table__.columns["user_id"].foreign_keys)
    assert len(fks) == 1
    assert fks[0].ondelete == "CASCADE"


def test_api_key_user_id_indexed():
    user_id_col = ApiKey.__table__.columns["user_id"]
    assert user_id_col.index is True


def test_api_key_key_prefix_indexed():
    prefix_col = ApiKey.__table__.columns["key_prefix"]
    assert prefix_col.index is True


def test_api_key_scopes_default_factory():
    scopes_col = ApiKey.__table__.columns["scopes"]
    # default eh lambda: ["diagnoses:analyze"]
    default_value = scopes_col.default.arg(None) if callable(scopes_col.default.arg) else scopes_col.default.arg
    assert default_value == ["diagnoses:analyze"]


def test_api_key_is_active_default_true():
    is_active_col = ApiKey.__table__.columns["is_active"]
    assert is_active_col.default.arg is True


def test_api_key_id_default_uuid_callable():
    id_col = ApiKey.__table__.columns["id"]
    assert callable(id_col.default.arg)
    value = id_col.default.arg(None) if id_col.default.is_callable else id_col.default.arg
    # uuid4 string format check
    assert len(value) == 36
    assert value.count("-") == 4


def test_api_key_revoked_at_nullable():
    col = ApiKey.__table__.columns["revoked_at"]
    assert col.nullable is True


def test_api_key_last_used_at_nullable():
    col = ApiKey.__table__.columns["last_used_at"]
    assert col.nullable is True
