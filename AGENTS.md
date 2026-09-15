# Backend — Zé Praga

Neste workspace, leia `../AGENTS.md` e a auditoria em
`../frontend/docs/AUDITORIA-TCC-2026-09-06.md`. Em clone isolado, use este arquivo
e o README; os arquivos irmãos podem não existir.

Mantenha `router → service → repository → DB`, DTOs tipados e SQLAlchemy async.
Conteúdo de usuário e documentação em português. Preserve cotas, isolamento
por usuário, auth JWT/API key, eventos SSE e retomada de perguntas HITL.
URLs assinadas são resolvidas na leitura; persista storage keys no banco.
`InferenceService.predict()` é síncrono; preserve `image_bytes` keyword-only.
Não confunda fallback mock com inferência real nem `/health` com readiness.
Verifique com `uv run ruff check app/ tests/`, `uv run mypy app/` e
`uv run pytest --cov=app --cov-report=term-missing -q`, em ambiente de teste.
Preserve o mínimo de cobertura configurado e não rode migrations/seeds em produção
durante uma revisão.
