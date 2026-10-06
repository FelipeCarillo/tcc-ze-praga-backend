"""TCC-098 — o agente pergunta e cadastra o talhão, ponta a ponta, num Postgres real.

Grafo LangGraph de verdade (tool registry, ToolNode, ``ask_user`` com
``interrupt()``, checkpointer, ``Command(resume=...)``), repositories e SQL
de verdade. Só o LLM é roteirizado: devolve as chamadas de ferramenta que o
system prompt pede (listar → perguntar → "Novo talhão" → descrever →
cadastrar → analisar). A inferência ONNX é simulada (o modelo não está no
container). Pulado sem ``ZP_TEST_PG_URL``.
"""

import os
import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.domains.chat import agent as agent_module
from app.domains.chat.repository import ChatMessageRepository, ChatSessionRepository
from app.domains.chat.service import ChatService
from app.domains.diagnoses.repository import DiagnosisRepository
from app.domains.diagnoses.schemas import Top3PredictionSchema
from app.domains.diagnoses.service import DiagnosisService
from app.domains.inference.schemas import InferenceResult
from app.models.crop import Crop
from app.models.diagnosis import Diagnosis
from app.models.fazenda import Fazenda
from app.models.talhao import Talhao
from app.models.user import User
from app.shared.enums import SeverityEnum

PG_URL = os.environ.get("ZP_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="sem ZP_TEST_PG_URL (Postgres real)")


class ScriptedLLM(FakeMessagesListChatModel):
    """LLM roteirizado: cada chamada devolve a próxima mensagem do roteiro."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        self.__dict__["seen_tools"] = [t.name for t in tools]
        return self


def _call(name: str, args: dict, n: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{n}", "type": "tool_call"}])


def _inference(crop_id: str) -> MagicMock:
    svc = MagicMock()
    svc.disease_catalog = [MagicMock(crop_id=crop_id)]
    svc.predict.return_value = InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença severa.",
        confidence=0.95,
        model_id="ensemble",
        image_name="folha.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1, disease_name="Ferrugem Asiática", disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi", confidence=0.95, severity="alta",
            )
        ],
    )
    return svc


@pytest.fixture
async def mundo():
    engine = create_async_engine(PG_URL)  # type: ignore[arg-type]
    maker = async_sessionmaker(engine, expire_on_commit=False)
    tag = uuid.uuid4().hex[:8]
    async with maker() as s:
        crop = Crop(slug=f"soja-{tag}", name_pt="Soja")
        user = User(email=f"agente-{tag}@t.test", password_hash="x")
        s.add_all([crop, user])
        await s.flush()
        fazenda = Fazenda(user_id=user.id, nome="Fazenda Boa Vista")
        s.add(fazenda)
        await s.flush()
        sede = Talhao(user_id=user.id, fazenda_id=fazenda.id, nome="Talhão 3", apelido="Sede")
        s.add(sede)
        await s.commit()
        ids = {"crop": crop.id, "user": user.id, "fazenda": fazenda.id, "sede": sede.id}
    yield maker, ids
    await engine.dispose()


async def test_ze_pergunta_cadastra_e_analisa_no_talhao_novo(mundo):
    maker, ids = mundo
    roteiro = [
        _call("list_my_talhoes", {}, 1),
        _call(
            "ask_user",
            {
                "question": "Antes de analisar: em qual talhão você tirou essa foto?",
                "response_kind": "choice",
                "options": ["Talhão 3 · Sede", "Novo talhão", "Pular"],
            },
            2,
        ),
        _call("ask_user", {"question": "Me conta nome, apelido, área e semeadura.", "response_kind": "text"}, 3),
        _call(
            "register_talhao",
            {"nome": "Talhão 9", "apelido": "Rio", "hectares": 40, "data_semeadura": "2026-09-12", "fazenda_id": ids["fazenda"]},
            4,
        ),
        _call("analyze_image", {}, 5),
        AIMessage(content="Cadastrei o Talhão 9 · Rio e a folha indica ferrugem-asiática."),
    ]
    llm = ScriptedLLM(responses=roteiro)
    real_build = agent_module.build_graph

    def build_with_script(**kwargs):
        return real_build(llm=llm, **kwargs)

    saver = MemorySaver()
    async with maker() as db:
        svc = ChatService(
            session_repo=ChatSessionRepository(db),
            message_repo=ChatMessageRepository(db),
            inference_svc=_inference(ids["crop"]),
            action_plan_svc=AsyncMock(),
            diagnosis_svc=DiagnosisService(DiagnosisRepository(db)),
            checkpointer_factory=AsyncMock(return_value=saver),
            db_session_factory=maker,
        )
        with patch("app.domains.chat.service.build_graph", side_effect=build_with_script):
            # 1) Foto sem talhão escolhido: o Zé lista e pergunta.
            r1 = await svc.chat(ids["user"], None, "", b"\xff\xd8\xff", "image/jpeg", "folha.jpg", "ensemble", None)
            assert r1.interrupt is not None and r1.interrupt.response_kind == "choice"
            assert "Novo talhão" in (r1.interrupt.options or [])
            assert {"list_my_talhoes", "use_talhao", "register_talhao", "ask_user"} <= set(llm.__dict__["seen_tools"])

            # 2) "Novo talhão": pergunta a descrição.
            r2 = await svc.resume(ids["user"], r1.session_id, "Novo talhão")
            assert r2.interrupt is not None and r2.interrupt.response_kind == "text"

            # 3) Descrição: cadastra o talhão e analisa a foto nele.
            r3 = await svc.resume(ids["user"], r1.session_id, "Talhão 9, do Rio, 40 hectares, plantei dia 12 de setembro")

    assert r3.interrupt is None
    assert r3.talhao is not None and r3.talhao.created
    assert (r3.talhao.nome, r3.talhao.apelido, r3.talhao.fazenda_nome) == ("Talhão 9", "Rio", "Fazenda Boa Vista")
    assert r3.diagnosis is not None and r3.diagnosis.talhao_id == r3.talhao.id
    assert "Talhão 9" in r3.content

    async with maker() as db:
        novo = (await db.execute(select(Talhao).where(Talhao.id == r3.talhao.id))).scalar_one()
        assert (float(novo.hectares), novo.data_semeadura, novo.fazenda_id) == (40.0, date(2026, 9, 12), ids["fazenda"])
        laudo = (await db.execute(select(Diagnosis).where(Diagnosis.id == r3.diagnosis.id))).scalar_one()
        assert laudo.talhao_id == novo.id
        # O card da conversa no histórico já mostra o laudo e o talhão.
        (card,) = await ChatSessionRepository(db).list_with_preview(ids["user"])
        assert card.diagnosis_count == 1 and card.talhao_nome == "Talhão 9"


async def test_ze_usa_talhao_existente(mundo):
    maker, ids = mundo
    roteiro = [
        _call("list_my_talhoes", {}, 1),
        _call("ask_user", {"question": "Em qual talhão?", "response_kind": "choice", "options": ["Talhão 3 · Sede", "Novo talhão", "Pular"]}, 2),
        _call("use_talhao", {"talhao_id": ids["sede"]}, 3),
        _call("analyze_image", {}, 4),
        AIMessage(content="Ferrugem no Talhão 3 · Sede."),
    ]
    llm = ScriptedLLM(responses=roteiro)
    real_build = agent_module.build_graph
    async with maker() as db:
        svc = ChatService(
            session_repo=ChatSessionRepository(db),
            message_repo=ChatMessageRepository(db),
            inference_svc=_inference(ids["crop"]),
            action_plan_svc=AsyncMock(),
            diagnosis_svc=DiagnosisService(DiagnosisRepository(db)),
            checkpointer_factory=AsyncMock(return_value=MemorySaver()),
            db_session_factory=maker,
        )
        with patch("app.domains.chat.service.build_graph", side_effect=lambda **k: real_build(llm=llm, **k)):
            r1 = await svc.chat(ids["user"], None, "", b"\xff\xd8\xff", "image/jpeg", "folha.jpg", "ensemble", None)
            r2 = await svc.resume(ids["user"], r1.session_id, "Talhão 3 · Sede")
    assert r2.talhao is not None and not r2.talhao.created and r2.talhao.id == ids["sede"]
    assert r2.diagnosis is not None and r2.diagnosis.talhao_id == ids["sede"]


async def test_mesmo_fluxo_em_streaming(mundo):
    """Pelo SSE: interrupt na pergunta e, no resume, os eventos talhao e diagnosis."""
    maker, ids = mundo
    roteiro = [
        _call("list_my_talhoes", {}, 1),
        _call("ask_user", {"question": "Em qual talhão?", "response_kind": "choice", "options": ["Talhão 3 · Sede", "Novo talhão", "Pular"]}, 2),
        _call("register_talhao", {"nome": "Talhão 5", "apelido": "Ponte", "hectares": 22}, 3),
        _call("analyze_image", {}, 4),
        AIMessage(content="Cadastrei o Talhão 5 · Ponte."),
    ]
    llm = ScriptedLLM(responses=roteiro)
    real_build = agent_module.build_graph
    async with maker() as db:
        svc = ChatService(
            session_repo=ChatSessionRepository(db),
            message_repo=ChatMessageRepository(db),
            inference_svc=_inference(ids["crop"]),
            action_plan_svc=AsyncMock(),
            diagnosis_svc=DiagnosisService(DiagnosisRepository(db)),
            checkpointer_factory=AsyncMock(return_value=MemorySaver()),
            db_session_factory=maker,
        )
        with patch("app.domains.chat.service.build_graph", side_effect=lambda **k: real_build(llm=llm, **k)):
            ev1 = [e async for e in svc.chat_stream(ids["user"], None, "", b"\xff\xd8\xff", "image/jpeg", "folha.jpg", "ensemble", None)]
            kinds1 = [e["event"] for e in ev1]
            assert "interrupt" in kinds1 and "diagnosis" not in kinds1
            sid = ev1[-1]["data"]
            ev2 = [e async for e in svc.resume_stream(ids["user"], sid, "Novo talhão")]
    kinds2 = [e["event"] for e in ev2]
    assert {"talhao", "diagnosis"} <= set(kinds2) and kinds2[-1] == "done"
    assert "register_talhao" in [e["data"] for e in ev2 if e["event"] == "tool_call"]
