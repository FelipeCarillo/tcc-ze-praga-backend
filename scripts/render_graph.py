"""Render do grafo LangGraph do Zé Praga em Mermaid + PNG.

Uso (a partir da raiz do backend):
    uv run python scripts/render_graph.py

Saídas:
    - artifacts/graph.mmd   (Mermaid markdown)
    - artifacts/graph.png   (PNG via mermaid.ink, se houver internet)
"""

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.domains.chat.agent import build_graph


def main() -> None:
    # Services não importam pra renderizar o grafo — só pra construir as tools.
    inference_svc = MagicMock()
    action_plan_svc = MagicMock()

    # FakeLLM com bind_tools no-op (igual aos testes) pra não bater na OpenAI.
    class FakeWithTools(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    fake_llm = FakeWithTools(responses=[AIMessage(content="dummy")])

    graph = build_graph(
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        llm=fake_llm,
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    g = graph.get_graph()

    mmd = g.draw_mermaid()
    mmd_path = artifacts / "graph.mmd"
    mmd_path.write_text(mmd, encoding="utf-8")
    print(f"[ok] Mermaid escrito em: {mmd_path}\n")
    print(mmd)

    try:
        png_bytes = g.draw_mermaid_png()
        png_path = artifacts / "graph.png"
        png_path.write_bytes(png_bytes)
        print(f"\n[ok] PNG escrito em: {png_path}  ({len(png_bytes)} bytes)")
    except Exception as exc:  # noqa: BLE001 — best effort
        print(f"\n[skip] PNG não gerado: {exc}")


if __name__ == "__main__":
    main()
