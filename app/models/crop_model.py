import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class CropModel(Base):
    __tablename__ = "crop_models"
    __table_args__ = (
        Index("ix_crop_models_crop_id_is_active", "crop_id", "is_active"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    crop_id: Mapped[str] = mapped_column(
        String, ForeignKey("crops.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String, nullable=False)
    framework: Mapped[str] = mapped_column(String, nullable=False, default="onnx")
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False)
    normalization: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    class_mapping: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    crop: Mapped["Crop"] = relationship(back_populates="models")  # type: ignore[name-defined] # noqa: F821
