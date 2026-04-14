from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.domains.diagnoses.dto import DiagnosisDTO, Top3PredictionDTO
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisFilters
from app.models.diagnosis import Diagnosis
from app.models.diagnosis_top3 import DiagnosisTop3


class DiagnosisRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, user_id: str, data: CreateDiagnosisRequest) -> DiagnosisDTO:
        diagnosis = Diagnosis(
            user_id=user_id,
            disease_name=data.disease_name,
            disease_id=data.disease_id,
            scientific_name=data.scientific_name,
            confidence=data.confidence,
            severity=data.severity,
            description=data.description,
            model_used=data.model_used,
            image_url=data.image_url,
            image_name=data.image_name,
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
            .options(joinedload(Diagnosis.top3))
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
            .options(joinedload(Diagnosis.top3))
            .where(Diagnosis.user_id == user_id)
        )

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
        )
