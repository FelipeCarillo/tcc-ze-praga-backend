import uuid

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    chat_daily_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_daily_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    api_monthly_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # PlanFeatures serializado (TCC-049). None = sem features definidas (legacy).
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    subscriptions: Mapped[list["UserSubscription"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="plan"
    )
