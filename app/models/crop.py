import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Crop(Base):
    __tablename__ = "crops"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    name_pt: Mapped[str] = mapped_column(String, nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String, nullable=True)
    kingdom: Mapped[str] = mapped_column(String, nullable=False, default="Plantae")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    diseases: Mapped[list["Disease"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="crop", cascade="all, delete-orphan"
    )
    models: Mapped[list["CropModel"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="crop", cascade="all, delete-orphan"
    )
