"""Smoke tests for Crop/Disease/CropModel/UploadedFile models.

Esses testes não tocam DB — validam metadata SQLAlchemy (tablename, colunas,
relationships back_populates corretos, índices/uniques).
"""

import inspect

from sqlalchemy import inspect as sa_inspect

from app.models.crop import Crop
from app.models.crop_model import CropModel
from app.models.disease import Disease
from app.models.uploaded_file import UploadedFile


# ── Crop ─────────────────────────────────────────────────────────────────────


def test_crop_tablename():
    assert Crop.__tablename__ == "crops"


def test_crop_required_columns_present():
    cols = {c.name for c in Crop.__table__.columns}
    assert {
        "id",
        "slug",
        "name_pt",
        "scientific_name",
        "kingdom",
        "is_active",
        "created_at",
    } <= cols


def test_crop_slug_unique():
    slug_col = Crop.__table__.columns["slug"]
    assert slug_col.unique is True


def test_crop_relationships_exist():
    mapper = sa_inspect(Crop)
    rel_names = {r.key for r in mapper.relationships}
    assert {"diseases", "models"} <= rel_names


def test_crop_disease_relationship_back_populates():
    mapper = sa_inspect(Crop)
    diseases_rel = mapper.relationships["diseases"]
    assert diseases_rel.back_populates == "crop"


def test_crop_models_relationship_back_populates():
    mapper = sa_inspect(Crop)
    models_rel = mapper.relationships["models"]
    assert models_rel.back_populates == "crop"


# ── Disease ──────────────────────────────────────────────────────────────────


def test_disease_tablename():
    assert Disease.__tablename__ == "diseases"


def test_disease_required_columns_present():
    cols = {c.name for c in Disease.__table__.columns}
    assert {
        "id",
        "crop_id",
        "slug",
        "name_pt",
        "scientific_name",
        "severity_default",
        "description_md",
        "image_url",
    } <= cols


def test_disease_has_unique_crop_id_slug_constraint():
    constraint_names = {c.name for c in Disease.__table__.constraints if c.name}
    assert "uq_disease_crop_slug" in constraint_names


def test_disease_crop_relationship_back_populates():
    mapper = sa_inspect(Disease)
    crop_rel = mapper.relationships["crop"]
    assert crop_rel.back_populates == "diseases"


def test_disease_crop_id_fk_cascade():
    fks = list(Disease.__table__.columns["crop_id"].foreign_keys)
    assert len(fks) == 1
    assert fks[0].ondelete == "CASCADE"


# ── CropModel ────────────────────────────────────────────────────────────────


def test_crop_model_tablename():
    assert CropModel.__tablename__ == "crop_models"


def test_crop_model_required_columns_present():
    cols = {c.name for c in CropModel.__table__.columns}
    assert {
        "id",
        "crop_id",
        "version",
        "framework",
        "file_path",
        "image_size",
        "normalization",
        "class_mapping",
        "top_k",
        "deployed_at",
        "is_active",
    } <= cols


def test_crop_model_has_index_on_crop_id_is_active():
    index_names = {idx.name for idx in CropModel.__table__.indexes}
    assert "ix_crop_models_crop_id_is_active" in index_names


def test_crop_model_default_framework_onnx():
    framework_col = CropModel.__table__.columns["framework"]
    assert framework_col.default.arg == "onnx"


def test_crop_model_default_top_k_3():
    top_k_col = CropModel.__table__.columns["top_k"]
    assert top_k_col.default.arg == 3


def test_crop_model_back_populates_crop_models():
    mapper = sa_inspect(CropModel)
    crop_rel = mapper.relationships["crop"]
    assert crop_rel.back_populates == "models"


# ── UploadedFile ─────────────────────────────────────────────────────────────


def test_uploaded_file_tablename():
    assert UploadedFile.__tablename__ == "uploaded_files"


def test_uploaded_file_required_columns_present():
    cols = {c.name for c in UploadedFile.__table__.columns}
    assert {
        "id",
        "session_id",
        "user_id",
        "original_name",
        "mime",
        "storage_key",
        "size_bytes",
        "hash_sha256",
        "uploaded_at",
    } <= cols


def test_uploaded_file_session_id_set_null_on_delete():
    fks = list(UploadedFile.__table__.columns["session_id"].foreign_keys)
    assert len(fks) == 1
    assert fks[0].ondelete == "SET NULL"


def test_uploaded_file_user_id_cascade():
    fks = list(UploadedFile.__table__.columns["user_id"].foreign_keys)
    assert len(fks) == 1
    assert fks[0].ondelete == "CASCADE"


def test_uploaded_file_hash_length_64():
    hash_col = UploadedFile.__table__.columns["hash_sha256"]
    assert hash_col.type.length == 64
