import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Talhao(Base):
    """Talhão (área de cultivo) cadastrado por um produtor.

    Registro simples por usuário — usado no Perfil e, futuramente, para
    agrupar diagnósticos por área.
    """

    __tablename__ = "talhoes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    nome: Mapped[str] = mapped_column(String, nullable=False)
    apelido: Mapped[str | None] = mapped_column(String, nullable=True)
    hectares: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    cultura: Mapped[str] = mapped_column(
        String, nullable=False, default="soja", server_default="soja"
    )
    data_semeadura: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
