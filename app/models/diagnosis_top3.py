import uuid

from sqlalchemy import ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DiagnosisTop3(Base):
    __tablename__ = "diagnosis_top3"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    diagnosis_id: Mapped[str] = mapped_column(
        String, ForeignKey("diagnoses.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    disease_name: Mapped[str] = mapped_column(String, nullable=False)
    disease_id: Mapped[str] = mapped_column(String, nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False)
    severity: Mapped[str | None] = mapped_column(String, nullable=True)

    diagnosis: Mapped["Diagnosis"] = relationship(back_populates="top3")  # type: ignore[name-defined] # noqa: F821
