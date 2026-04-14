import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    disease_name: Mapped[str] = mapped_column(String, nullable=False)
    disease_id: Mapped[str] = mapped_column(String, nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    model_used: Mapped[str] = mapped_column(String, nullable=False)
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)
    image_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="diagnoses")  # type: ignore[name-defined] # noqa: F821
    top3: Mapped[list["DiagnosisTop3"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="diagnosis",
        cascade="all, delete-orphan",
        order_by="DiagnosisTop3.rank",
    )
