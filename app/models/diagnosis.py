import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Diagnosis(Base):
    __tablename__ = "diagnoses"
    # Espelha a migration 0012: historico agrupado filtra por usuario,
    # particiona por talhao e ordena pelo mais recente.
    __table_args__ = (
        Index("ix_diagnoses_user_talhao_created", "user_id", "talhao_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    crop_id: Mapped[str] = mapped_column(
        String, ForeignKey("crops.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    disease_name: Mapped[str] = mapped_column(String, nullable=False)
    # Legacy slug-based identifier — preserved during multi-cultivo migration.
    # New code should prefer ``disease_fk_id`` (FK -> diseases.id).
    disease_id: Mapped[str] = mapped_column(String, nullable=False)
    disease_fk_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("diseases.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    scientific_name: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    model_used: Mapped[str] = mapped_column(String, nullable=False)
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)
    image_name: Mapped[str | None] = mapped_column(String, nullable=True)
    # TCC-093: talhao onde a folha foi fotografada. Nullable — laudos antigos e
    # os feitos sem escolher talhao ficam no grupo "Sem talhao". SET NULL no
    # delete: apagar o talhao nao apaga o historico.
    talhao_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("talhoes.id", ondelete="SET NULL"), nullable=True
    )
    # TCC-056: evidencia externa persistida pelo gather_evidence_node (search_web +
    # search_scientific) em paralelo ao action_plan. Schema dos items eh validado
    # pelo ``DiagnosisSourceSchema`` em camada de aplicacao.
    sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB(),
        nullable=False,
        server_default=text("'[]'::jsonb"),
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="diagnoses")  # type: ignore[name-defined] # noqa: F821
    talhao: Mapped["Talhao | None"] = relationship()  # type: ignore[name-defined] # noqa: F821
    top3: Mapped[list["DiagnosisTop3"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="diagnosis",
        cascade="all, delete-orphan",
        order_by="DiagnosisTop3.rank",
    )
