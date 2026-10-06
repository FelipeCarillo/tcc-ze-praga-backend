import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Fazenda(Base):
    """Fazenda do produtor — agrupa os talhões (TCC-096).

    Todo talhão pertence a uma fazenda (``talhoes.fazenda_id``). O agrônomo
    responsável é opcional e aparece no campo de revisão técnica do relatório.
    """

    __tablename__ = "fazendas"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    nome: Mapped[str] = mapped_column(String, nullable=False)
    municipio: Mapped[str | None] = mapped_column(String, nullable=True)
    uf: Mapped[str | None] = mapped_column(String(2), nullable=True)
    hectares: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    agronomo_nome: Mapped[str | None] = mapped_column(String, nullable=True)
    agronomo_crea: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
