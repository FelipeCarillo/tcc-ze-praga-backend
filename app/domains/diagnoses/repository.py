from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.domains.diagnoses.dto import DiagnosisDTO, TalhaoGroupDTO, Top3PredictionDTO
from app.domains.diagnoses.schemas import (
    SEM_TALHAO,
    CreateDiagnosisRequest,
    DiagnosisFilters,
)
from app.models.diagnosis import Diagnosis
from app.models.diagnosis_top3 import DiagnosisTop3
from app.models.fazenda import Fazenda
from app.models.talhao import Talhao


class DiagnosisRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self, user_id: str, data: CreateDiagnosisRequest, *, crop_id: str
    ) -> DiagnosisDTO:
        diagnosis = Diagnosis(
            user_id=user_id,
            crop_id=crop_id,
            disease_name=data.disease_name,
            disease_id=data.disease_id,
            scientific_name=data.scientific_name,
            confidence=data.confidence,
            severity=data.severity,
            description=data.description,
            model_used=data.model_used,
            image_url=data.image_url,
            image_name=data.image_name,
            sources=[s.model_dump() for s in data.sources],
            talhao_id=data.talhao_id,
        )
        self._db.add(diagnosis)
        await self._db.flush()  # get id before adding top3

        for prediction in data.top3:
            self._db.add(
                DiagnosisTop3(
                    diagnosis_id=diagnosis.id,
                    rank=prediction.rank,
                    disease_name=prediction.disease_name,
                    disease_id=prediction.disease_id,
                    scientific_name=prediction.scientific_name,
                    confidence=prediction.confidence,
                    severity=prediction.severity,
                )
            )

        await self._db.commit()
        await self._db.refresh(diagnosis)
        return await self.find_by_id(diagnosis.id, user_id)  # type: ignore[return-value]

    async def find_by_id(self, diagnosis_id: str, user_id: str) -> DiagnosisDTO | None:
        result = await self._db.execute(
            select(Diagnosis)
            .options(joinedload(Diagnosis.top3), joinedload(Diagnosis.talhao))
            .where(Diagnosis.id == diagnosis_id, Diagnosis.user_id == user_id)
        )
        diagnosis = result.unique().scalar_one_or_none()
        return self._to_dto(diagnosis) if diagnosis else None

    async def find_all_by_user(
        self,
        user_id: str,
        filters: DiagnosisFilters,
    ) -> tuple[list[DiagnosisDTO], int]:
        query = (
            select(Diagnosis)
            .options(joinedload(Diagnosis.top3), joinedload(Diagnosis.talhao))
            .where(Diagnosis.user_id == user_id)
        )

        if filters.talhao_id == SEM_TALHAO:
            query = query.where(Diagnosis.talhao_id.is_(None))
        elif filters.talhao_id:
            query = query.where(Diagnosis.talhao_id == filters.talhao_id)
        if filters.severity:
            query = query.where(Diagnosis.severity == filters.severity)
        if filters.search:
            query = query.where(Diagnosis.disease_name.ilike(f"%{filters.search}%"))

        count_result = await self._db.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar_one()

        query = (
            query.order_by(Diagnosis.created_at.desc())
            .offset(filters.offset)
            .limit(filters.limit)
        )
        result = await self._db.execute(query)
        items = [self._to_dto(d) for d in result.unique().scalars().all()]

        return items, total

    async def talhao_belongs_to_user(self, talhao_id: str, user_id: str) -> bool:
        """True se o talhao existe e pertence ao usuario (TCC-093)."""
        result = await self._db.execute(
            select(Talhao.id).where(Talhao.id == talhao_id, Talhao.user_id == user_id)
        )
        return result.scalar_one_or_none() is not None

    async def set_talhao(
        self, diagnosis_id: str, user_id: str, talhao_id: str | None
    ) -> DiagnosisDTO | None:
        """Move o laudo para ``talhao_id`` (None desfaz o vinculo).

        Nao valida a posse do talhao, que fica com o service. Devolve None se
        o laudo nao existe ou nao pertence ao usuario.
        """
        result = await self._db.execute(
            select(Diagnosis).where(
                Diagnosis.id == diagnosis_id, Diagnosis.user_id == user_id
            )
        )
        diagnosis = result.scalar_one_or_none()
        if not diagnosis:
            return None
        diagnosis.talhao_id = talhao_id
        await self._db.commit()
        # Expira o objeto: o relationship ``talhao`` carregado antes do commit
        # apontaria para o talhao antigo.
        self._db.expire(diagnosis)
        return await self.find_by_id(diagnosis_id, user_id)

    async def group_by_talhao(
        self, user_id: str, per_group: int
    ) -> list[TalhaoGroupDTO]:
        """Historico agrupado por talhao (TCC-093).

        Tres consultas, independentes do numero de talhoes:
          1. totais e data mais recente por ``talhao_id`` (GROUP BY);
          2. os ``per_group`` laudos mais recentes de cada grupo, via
             ``row_number()`` particionado por talhao;
          3. os talhoes do usuario, incluindo os ainda sem laudo, que a UI
             mostra vazios para convidar a primeira foto.

        Ordem: grupos com laudo do mais recente para o mais antigo, depois
        talhoes vazios por nome, e "Sem talhao" sempre por ultimo.
        """
        stats_rows = (
            await self._db.execute(
                select(
                    Diagnosis.talhao_id,
                    func.count(Diagnosis.id),
                    func.max(Diagnosis.created_at),
                )
                .where(Diagnosis.user_id == user_id)
                .group_by(Diagnosis.talhao_id)
            )
        ).all()
        stats = {row[0]: (int(row[1]), row[2]) for row in stats_rows}

        ranked = (
            select(
                Diagnosis.id.label("id"),
                func.row_number()
                .over(
                    partition_by=Diagnosis.talhao_id,
                    order_by=(Diagnosis.created_at.desc(), Diagnosis.id.desc()),
                )
                .label("rn"),
            )
            .where(Diagnosis.user_id == user_id)
            .subquery()
        )
        recent_rows = await self._db.execute(
            select(Diagnosis)
            .options(joinedload(Diagnosis.top3), joinedload(Diagnosis.talhao))
            .join(ranked, ranked.c.id == Diagnosis.id)
            .where(ranked.c.rn <= per_group)
            .order_by(Diagnosis.created_at.desc(), Diagnosis.id.desc())
        )
        recent_by_group: dict[str | None, list[DiagnosisDTO]] = {}
        for d in recent_rows.unique().scalars().all():
            recent_by_group.setdefault(d.talhao_id, []).append(self._to_dto(d))

        talhoes = (
            await self._db.execute(
                select(Talhao.id, Talhao.nome, Fazenda.id, Fazenda.nome)
                .join(Fazenda, Fazenda.id == Talhao.fazenda_id)
                .where(Talhao.user_id == user_id)
            )
        ).all()

        groups: list[TalhaoGroupDTO] = []
        for talhao_id, nome, fazenda_id, fazenda_nome in talhoes:
            total, last_at = stats.get(talhao_id, (0, None))
            groups.append(
                TalhaoGroupDTO(
                    talhao_id=talhao_id,
                    talhao_nome=nome,
                    fazenda_id=fazenda_id,
                    fazenda_nome=fazenda_nome,
                    total=total,
                    last_at=last_at,
                    recent=recent_by_group.get(talhao_id, []),
                )
            )
        groups.sort(key=_group_sort_key)
        if None in stats:
            total, last_at = stats[None]
            groups.append(
                TalhaoGroupDTO(
                    talhao_id=None,
                    talhao_nome=None,
                    total=total,
                    last_at=last_at,
                    recent=recent_by_group.get(None, []),
                )
            )
        return groups

    async def delete(self, diagnosis_id: str, user_id: str) -> bool:
        result = await self._db.execute(
            select(Diagnosis).where(
                Diagnosis.id == diagnosis_id, Diagnosis.user_id == user_id
            )
        )
        diagnosis = result.scalar_one_or_none()
        if not diagnosis:
            return False
        await self._db.delete(diagnosis)
        await self._db.commit()
        return True

    async def delete_all_by_user(self, user_id: str) -> int:
        result = await self._db.execute(
            select(Diagnosis).where(Diagnosis.user_id == user_id)
        )
        diagnoses = result.scalars().all()
        count = len(diagnoses)
        for d in diagnoses:
            await self._db.delete(d)
        await self._db.commit()
        return count

    @staticmethod
    def _to_dto(diagnosis: Diagnosis) -> DiagnosisDTO:
        return DiagnosisDTO(
            id=diagnosis.id,
            user_id=diagnosis.user_id,
            disease_name=diagnosis.disease_name,
            disease_id=diagnosis.disease_id,
            scientific_name=diagnosis.scientific_name,
            confidence=float(diagnosis.confidence),
            severity=diagnosis.severity,
            description=diagnosis.description,
            model_used=diagnosis.model_used,
            image_url=diagnosis.image_url,
            image_name=diagnosis.image_name,
            created_at=diagnosis.created_at,
            top3=[
                Top3PredictionDTO(
                    rank=t.rank,
                    disease_name=t.disease_name,
                    disease_id=t.disease_id,
                    scientific_name=t.scientific_name,
                    confidence=float(t.confidence),
                    severity=t.severity,
                )
                for t in diagnosis.top3
            ],
            sources=list(diagnosis.sources or []),
            talhao_id=diagnosis.talhao_id,
            talhao_nome=_talhao_nome(diagnosis),
        )


def _group_sort_key(group: TalhaoGroupDTO) -> tuple[bool, float, str]:
    """Mais recente primeiro; talhoes sem laudo no fim, por nome."""
    recency = -group.last_at.timestamp() if group.last_at else 0.0
    return (group.last_at is None, recency, (group.talhao_nome or "").lower())


def _talhao_nome(diagnosis: Diagnosis) -> str | None:
    """Nome do talhao sem disparar lazy-load (proibido em sessao async).

    Os SELECTs deste repositorio carregam ``talhao`` via joinedload; se algum
    caminho nao carregar, devolve None em vez de falhar com MissingGreenlet.
    """
    if diagnosis.talhao_id is None:
        return None
    talhao = diagnosis.__dict__.get("talhao")
    return talhao.nome if talhao is not None else None
