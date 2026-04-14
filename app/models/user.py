import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    diagnoses: Mapped[list["Diagnosis"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )
    subscription: Mapped["UserSubscription | None"] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    usage_logs: Mapped[list["UsageLog"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )
