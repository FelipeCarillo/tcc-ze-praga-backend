"""add crops diseases crop_models uploaded_files

Revision ID: 0003_add_crops_and_diseases
Revises: 0002_add_chat_tables
Create Date: 2026-05-22 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0003_add_crops_and_diseases"
down_revision: str | None = "0002_add_chat_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # crops
    op.create_table(
        "crops",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("name_pt", sa.String(), nullable=False),
        sa.Column("scientific_name", sa.String(), nullable=True),
        sa.Column("kingdom", sa.String(), nullable=False, server_default="Plantae"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_crops_slug"),
    )
    op.create_index("ix_crops_slug", "crops", ["slug"], unique=False)

    # diseases
    op.create_table(
        "diseases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("crop_id", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("name_pt", sa.String(), nullable=False),
        sa.Column("scientific_name", sa.String(), nullable=True),
        sa.Column("severity_default", sa.String(), nullable=False),
        sa.Column("description_md", sa.Text(), nullable=True),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["crop_id"], ["crops.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("crop_id", "slug", name="uq_disease_crop_slug"),
    )
    op.create_index("ix_diseases_crop_id", "diseases", ["crop_id"], unique=False)
    op.create_index("ix_diseases_slug", "diseases", ["slug"], unique=False)
    op.create_index(
        "ix_diseases_crop_id_slug", "diseases", ["crop_id", "slug"], unique=False
    )

    # crop_models
    op.create_table(
        "crop_models",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("crop_id", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("framework", sa.String(), nullable=False, server_default="onnx"),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("image_size", sa.Integer(), nullable=False),
        sa.Column("normalization", sa.JSON(), nullable=False),
        sa.Column("class_mapping", sa.JSON(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="3"),
        sa.Column(
            "deployed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["crop_id"], ["crops.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_crop_models_crop_id_is_active",
        "crop_models",
        ["crop_id", "is_active"],
        unique=False,
    )

    # uploaded_files
    op.create_table(
        "uploaded_files",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("original_name", sa.String(), nullable=False),
        sa.Column("mime", sa.String(), nullable=False),
        sa.Column("storage_key", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("hash_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_sessions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_uploaded_files_session_id", "uploaded_files", ["session_id"], unique=False
    )
    op.create_index(
        "ix_uploaded_files_user_id", "uploaded_files", ["user_id"], unique=False
    )
    op.create_index(
        "ix_uploaded_files_hash_sha256",
        "uploaded_files",
        ["hash_sha256"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_uploaded_files_hash_sha256", table_name="uploaded_files")
    op.drop_index("ix_uploaded_files_user_id", table_name="uploaded_files")
    op.drop_index("ix_uploaded_files_session_id", table_name="uploaded_files")
    op.drop_table("uploaded_files")

    op.drop_index("ix_crop_models_crop_id_is_active", table_name="crop_models")
    op.drop_table("crop_models")

    op.drop_index("ix_diseases_crop_id_slug", table_name="diseases")
    op.drop_index("ix_diseases_slug", table_name="diseases")
    op.drop_index("ix_diseases_crop_id", table_name="diseases")
    op.drop_table("diseases")

    op.drop_index("ix_crops_slug", table_name="crops")
    op.drop_table("crops")
