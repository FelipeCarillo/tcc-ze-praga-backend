import uuid

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Disease(Base):
    __tablename__ = "diseases"
    __table_args__ = (
        UniqueConstraint("crop_id", "slug", name="uq_disease_crop_slug"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    crop_id: Mapped[str] = mapped_column(
        String, ForeignKey("crops.id", ondelete="CASCADE"), nullable=False, index=True
    )
    slug: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name_pt: Mapped[str] = mapped_column(String, nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String, nullable=True)
    severity_default: Mapped[str] = mapped_column(String, nullable=False)
    description_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)

    crop: Mapped["Crop"] = relationship(back_populates="diseases")  # type: ignore[name-defined] # noqa: F821
