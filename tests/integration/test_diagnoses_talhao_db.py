"""TCC-093 — DiagnosisRepository contra um Postgres real.

Os demais testes mockam o banco; este cobre o que so' o banco garante: a
migration 0012, a FK com ON DELETE SET NULL e a consulta agrupada com
``row_number()`` particionado.

Pulado sem banco. Para rodar, aponte ``ZP_TEST_PG_URL`` para um Postgres com
pgvector e as migrations aplicadas (``alembic upgrade head``), por exemplo:

    ZP_TEST_PG_URL=postgresql+asyncpg://pg:pg@localhost:5432/zp pytest tests/integration/test_diagnoses_talhao_db.py
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.domains.diagnoses.repository import DiagnosisRepository
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisFilters
from app.models.crop import Crop
from app.models.diagnosis import Diagnosis
from app.models.talhao import Talhao
from app.models.user import User
from app.shared.enums import SeverityEnum

PG_URL = os.environ.get("ZP_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="sem ZP_TEST_PG_URL (Postgres real)")


@pytest.fixture
async def session():
    engine = create_async_engine(PG_URL)  # type: ignore[arg-type]
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def mundo(session):
    """Dois usuarios, um crop e tres talhoes (dois do usuario A, um do B)."""
    tag = uuid.uuid4().hex[:8]
    crop = Crop(slug=f"soja-{tag}", name_pt="Soja")
    a = User(email=f"a-{tag}@t.test", password_hash="x")
    b = User(email=f"b-{tag}@t.test", password_hash="x")
    session.add_all([crop, a, b])
    await session.flush()
    sede = Talhao(user_id=a.id, nome="Sede")
    baixada = Talhao(user_id=a.id, nome="Baixada")
    alheio = Talhao(user_id=b.id, nome="Do vizinho")
    session.add_all([sede, baixada, alheio])
    await session.commit()
    return {"crop": crop.id, "a": a.id, "b": b.id, "sede": sede.id,
            "baixada": baixada.id, "alheio": alheio.id}


async def _laudo(repo, session, user, crop, talhao, severity, dias_atras):
    dto = await repo.create(
        user,
        CreateDiagnosisRequest(
            disease_name="Ferrugem",
            disease_id="ferrugem-asiatica",
            confidence=0.9,
            severity=severity,
            model_used="ensemble",
            talhao_id=talhao,
        ),
        crop_id=crop,
    )
    # created_at vem do servidor; recua no tempo para testar a ordenacao.
    row = await session.get(Diagnosis, dto.id)
    row.created_at = datetime.now(UTC) - timedelta(days=dias_atras)
    await session.commit()
    return dto.id


async def test_grupos_por_talhao_com_os_mais_recentes(session, mundo):
    repo = DiagnosisRepository(session)
    a, crop = mundo["a"], mundo["crop"]
    for sev, dias in [(SeverityEnum.NENHUMA, 30), (SeverityEnum.BAIXA, 14),
                      (SeverityEnum.MEDIA, 7), (SeverityEnum.ALTA, 0)]:
        await _laudo(repo, session, a, crop, mundo["sede"], sev, dias)
    await _laudo(repo, session, a, crop, None, SeverityEnum.BAIXA, 3)
    await _laudo(repo, session, mundo["b"], crop, mundo["alheio"], SeverityEnum.ALTA, 1)

    groups = await repo.group_by_talhao(a, per_group=3)

    # Sede (com laudo mais recente) primeiro, Baixada vazia depois, sem talhao no fim.
    assert [g.talhao_nome for g in groups] == ["Sede", "Baixada", None]
    sede, baixada, sem = groups
    assert sede.total == 4
    assert [d.severity for d in sede.recent] == ["alta", "media", "baixa"]
    assert all(d.talhao_nome == "Sede" for d in sede.recent)
    assert baixada.total == 0 and baixada.recent == [] and baixada.last_at is None
    assert sem.total == 1 and sem.recent[0].talhao_id is None
    # Nada do usuario B vaza para A.
    assert all(d.user_id == a for g in groups for d in g.recent)


async def test_filtro_e_troca_de_talhao(session, mundo):
    repo = DiagnosisRepository(session)
    a, crop = mundo["a"], mundo["crop"]
    d1 = await _laudo(repo, session, a, crop, mundo["sede"], SeverityEnum.ALTA, 1)
    d2 = await _laudo(repo, session, a, crop, None, SeverityEnum.BAIXA, 2)

    so_sede, total = await repo.find_all_by_user(a, DiagnosisFilters(talhao_id=mundo["sede"]))
    assert total == 1 and so_sede[0].id == d1
    sem, total = await repo.find_all_by_user(a, DiagnosisFilters(talhao_id="sem-talhao"))
    assert total == 1 and sem[0].id == d2

    movido = await repo.set_talhao(d2, a, mundo["baixada"])
    assert movido is not None and movido.talhao_nome == "Baixada"
    desfeito = await repo.set_talhao(d2, a, None)
    assert desfeito is not None and desfeito.talhao_id is None
    # Laudo de outro usuario nao e' encontrado.
    assert await repo.set_talhao(d1, mundo["b"], None) is None


async def test_posse_do_talhao(session, mundo):
    repo = DiagnosisRepository(session)
    assert await repo.talhao_belongs_to_user(mundo["sede"], mundo["a"])
    assert not await repo.talhao_belongs_to_user(mundo["alheio"], mundo["a"])
    assert not await repo.talhao_belongs_to_user("nao-existe", mundo["a"])


async def test_apagar_talhao_preserva_os_laudos(session, mundo):
    """FK ON DELETE SET NULL: o historico sobrevive, so' perde o vinculo."""
    repo = DiagnosisRepository(session)
    a = mundo["a"]
    d = await _laudo(repo, session, a, mundo["crop"], mundo["baixada"], SeverityEnum.ALTA, 1)
    talhao = await session.get(Talhao, mundo["baixada"])
    await session.delete(talhao)
    await session.commit()
    session.expire_all()

    row = (await session.execute(select(Diagnosis).where(Diagnosis.id == d))).scalar_one()
    assert row.talhao_id is None
