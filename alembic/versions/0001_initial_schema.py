"""initial schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-05-04 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "subscription_plans",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("chat_daily_limit", sa.Integer(), nullable=True),
        sa.Column("inference_daily_limit", sa.Integer(), nullable=True),
        sa.Column("api_monthly_limit", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "action_plans",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("disease_id", sa.String(), nullable=False),
        sa.Column("level", sa.String(), nullable=False),
        sa.Column("actions", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("disease_id", "level", name="uq_action_plan_disease_level"),
    )
    op.create_index(op.f("ix_action_plans_disease_id"), "action_plans", ["disease_id"], unique=False)

    op.create_table(
        "action_plan_sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("disease_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("detail", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_action_plan_sources_disease_id"),
        "action_plan_sources",
        ["disease_id"],
        unique=False,
    )

    op.create_table(
        "diagnoses",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("disease_name", sa.String(), nullable=False),
        sa.Column("disease_id", sa.String(), nullable=False),
        sa.Column("scientific_name", sa.String(), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("model_used", sa.String(), nullable=False),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.Column("image_name", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "diagnosis_top3",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("diagnosis_id", sa.String(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("disease_name", sa.String(), nullable=False),
        sa.Column("disease_id", sa.String(), nullable=False),
        sa.Column("scientific_name", sa.String(), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=5, scale=3), nullable=False),
        sa.Column("severity", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["diagnosis_id"], ["diagnoses.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "user_subscriptions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("plan_id", sa.String(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["plan_id"], ["subscription_plans.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id"),
    )

    op.create_table(
        "usage_logs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("feature", sa.String(), nullable=False),
        sa.Column(
            "used_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_usage_logs_user_feature_used_at",
        "usage_logs",
        ["user_id", "feature", "used_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_usage_logs_user_feature_used_at", table_name="usage_logs")
    op.drop_table("usage_logs")
    op.drop_table("user_subscriptions")
    op.drop_table("diagnosis_top3")
    op.drop_table("diagnoses")
    op.drop_index(op.f("ix_action_plan_sources_disease_id"), table_name="action_plan_sources")
    op.drop_table("action_plan_sources")
    op.drop_index(op.f("ix_action_plans_disease_id"), table_name="action_plans")
    op.drop_table("action_plans")
    op.drop_table("users")
    op.drop_table("subscription_plans")