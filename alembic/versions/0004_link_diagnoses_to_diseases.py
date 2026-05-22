"""link diagnoses+action_plans to crops/diseases via FK + backfill

Revision ID: 0004_link_diagnoses_to_diseases
Revises: 0003_add_crops_and_diseases
Create Date: 2026-05-22 00:00:00.000000

Strategy:
- Add ``crop_id`` (FK -> crops.id) nullable to ``diagnoses`` and ``action_plans``.
- Backfill all existing rows with the ``soja`` crop_id (assumes seed_crops ran).
- Alter ``crop_id`` to NOT NULL after backfill.
- Add ``disease_fk_id`` (FK -> diseases.id) nullable to ``diagnoses`` and
  backfill via slug-match against the legacy ``disease_id`` string column.
- ``disease_id`` (string) is kept untouched for backward compatibility — a
  follow-up migration in sprint A2 will drop it.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0004_link_diagnoses_to_diseases"
down_revision: str | None = "0003_add_crops_and_diseases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. Add crop_id (nullable) ────────────────────────────────────────────
    op.add_column(
        "diagnoses",
        sa.Column("crop_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_diagnoses_crop_id",
        "diagnoses",
        "crops",
        ["crop_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.add_column(
        "action_plans",
        sa.Column("crop_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_action_plans_crop_id",
        "action_plans",
        "crops",
        ["crop_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    # ── 2. Backfill crop_id pointing to 'soja' crop ──────────────────────────
    op.execute(
        """
        UPDATE diagnoses
        SET crop_id = (SELECT id FROM crops WHERE slug = 'soja')
        WHERE crop_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE action_plans
        SET crop_id = (SELECT id FROM crops WHERE slug = 'soja')
        WHERE crop_id IS NULL
        """
    )

    # ── 3. Alter NOT NULL ────────────────────────────────────────────────────
    op.alter_column("diagnoses", "crop_id", existing_type=sa.String(), nullable=False)
    op.alter_column(
        "action_plans", "crop_id", existing_type=sa.String(), nullable=False
    )

    # ── 4. Add disease_fk_id + backfill via slug match ───────────────────────
    op.add_column(
        "diagnoses",
        sa.Column("disease_fk_id", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_diagnoses_disease_fk_id",
        "diagnoses",
        "diseases",
        ["disease_fk_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.execute(
        """
        UPDATE diagnoses d
        SET disease_fk_id = (
            SELECT id FROM diseases
            WHERE slug = d.disease_id
            LIMIT 1
        )
        WHERE d.disease_fk_id IS NULL
        """
    )

    # ── 5. Create indexes for FK lookups ─────────────────────────────────────
    op.create_index("ix_diagnoses_crop_id", "diagnoses", ["crop_id"], unique=False)
    op.create_index(
        "ix_diagnoses_disease_fk_id", "diagnoses", ["disease_fk_id"], unique=False
    )
    op.create_index(
        "ix_action_plans_crop_id", "action_plans", ["crop_id"], unique=False
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_action_plans_crop_id", table_name="action_plans")
    op.drop_index("ix_diagnoses_disease_fk_id", table_name="diagnoses")
    op.drop_index("ix_diagnoses_crop_id", table_name="diagnoses")

    # Drop FKs and columns (disease_id string column was preserved so data is intact)
    op.drop_constraint("fk_diagnoses_disease_fk_id", "diagnoses", type_="foreignkey")
    op.drop_column("diagnoses", "disease_fk_id")

    op.drop_constraint("fk_action_plans_crop_id", "action_plans", type_="foreignkey")
    op.drop_column("action_plans", "crop_id")

    op.drop_constraint("fk_diagnoses_crop_id", "diagnoses", type_="foreignkey")
    op.drop_column("diagnoses", "crop_id")
