# Zé Praga — Backend

API REST do sistema de diagnóstico de doenças em plantas, desenvolvida como parte do TCC. Este repositório é o **core service** — responsável por autenticação, histórico de diagnósticos, planos de ação e controle de uso. As rotas de chatbot e inferência ML estão em repositório separado.

---

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Linguagem | Python 3.12+ |
| Gerenciador de pacotes | [uv](https://docs.astral.sh/uv/) |
| ORM | [SQLAlchemy 2.x async](https://docs.sqlalchemy.org/en/20/) |
| Driver PostgreSQL | [asyncpg](https://github.com/MagicStack/asyncpg) |
| Migrations | [Alembic](https://alembic.sqlalchemy.org/) |
| Banco de dados | [Supabase](https://supabase.com/) (PostgreSQL) |
| Storage | Supabase Storage (via supabase-py) |
| Autenticação | JWT (python-jose) + bcrypt |
| Testes | pytest + pytest-asyncio + pytest-cov |

---

## Arquitetura

O projeto segue uma arquitetura em camadas com separação estrita de responsabilidades (Clean Code + SOLID):

```
HTTP Request
    │
    ▼
Router          — valida input via Pydantic, injeta dependências
    │
    ▼
Service         — lógica de negócio, orquestra, lança exceções de domínio
    │
    ▼
Repository      — queries SQLAlchemy, retorna DTOs tipados
    │
    ▼
Supabase PostgreSQL
```

### Estrutura de diretórios

```
app/
├── main.py                     # Factory da aplicação, CORS, routers, exception handlers
├── config.py                   # Settings via pydantic-settings
├── core/
│   ├── security.py             # JWT + bcrypt
│   ├── dependencies.py         # Injeção de dependências (get_db, get_current_user, require_quota)
│   └── exceptions.py           # Exceções de domínio
├── db/
│   ├── database.py             # Engine async + session factory
│   ├── base.py                 # Base declarativa SQLAlchemy
│   └── storage.py              # Cliente Supabase (Storage)
├── models/                     # Models SQLAlchemy ORM (8 tabelas)
├── domains/
│   ├── auth/                   # Registro, login, /me
│   ├── users/                  # Perfil do usuário
│   ├── diagnoses/              # Histórico de diagnósticos
│   ├── action_plans/           # Planos de ação por doença
│   ├── subscriptions/          # Planos de assinatura
│   └── usage/                  # Controle de uso e quotas
└── shared/
    ├── enums.py                # Enums compartilhados
    └── pagination.py           # PaginatedResponse genérico
```

Cada domínio segue a estrutura: `router.py · service.py · repository.py · schemas.py · dto.py`

---

## Banco de dados

### Tabelas

| Tabela | Descrição |
|---|---|
| `users` | Usuários da plataforma |
| `diagnoses` | Diagnósticos realizados por usuário |
| `diagnosis_top3` | Top-3 predições por diagnóstico |
| `action_plans` | Planos de ação por doença e nível |
| `action_plan_sources` | Fontes bibliográficas dos planos |
| `subscription_plans` | Planos de assinatura (free, pro, enterprise) |
| `user_subscriptions` | Assinatura ativa do usuário |
| `usage_logs` | Log de uso de features (chat, inference, api) |

---

## Sistema de Quotas

Cada usuário tem limites de uso baseados no seu plano de assinatura:

| Plano | Chat (diário) | Inferência (diária) | API (mensal) |
|---|---|---|---|
| **Free** | 10 | 5 | 0 |
| **Pro** | Ilimitado | Ilimitado | 500 |
| **Enterprise** | Ilimitado | Ilimitado | Ilimitado |

A verificação é feita via dependency `require_quota(feature)` antes das rotas protegidas. Usuários sem assinatura recebem os limites do plano **free** automaticamente.

---

## API

Todas as rotas são prefixadas com `/api/v1`. Documentação interativa disponível em `/docs` após iniciar o servidor.

### Autenticação (`/auth`)

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| POST | `/auth/register` | — | Cadastro de novo usuário |
| POST | `/auth/login` | — | Login, retorna JWT |
| GET | `/auth/me` | JWT | Dados do usuário autenticado |

### Usuários (`/users`)

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/users/me` | JWT | Perfil completo |
| PATCH | `/users/me` | JWT | Atualizar nome ou email |
| DELETE | `/users/me` | JWT | Desativar conta (soft delete) |

### Diagnósticos (`/diagnoses`)

| Método | Rota | Auth | Quota |
|---|---|---|---|
| POST | `/diagnoses` | JWT | INFERENCE |
| GET | `/diagnoses` | JWT | — |
| GET | `/diagnoses/{id}` | JWT | — |
| DELETE | `/diagnoses/{id}` | JWT | — |
| DELETE | `/diagnoses?confirm=true` | JWT | — |

Suporta filtros via query string: `?severity=alta&search=ferrugem&page=1&limit=20`

### Planos de Ação (`/action-plans`)

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/action-plans/{disease_id}` | JWT | Todos os níveis do plano |
| GET | `/action-plans/{disease_id}/{level}` | JWT | Nível específico (essencial, campo, especialista) |

### Assinaturas (`/subscriptions`)

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/subscriptions/plans` | — | Lista planos disponíveis |
| GET | `/subscriptions/me` | JWT | Assinatura atual do usuário |
| POST | `/subscriptions/me` | JWT | Assinar ou trocar de plano |

### Uso (`/usage`)

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/usage/me` | JWT | Resumo de uso atual |
| GET | `/usage/me/history` | JWT | Histórico de uso |

### Health

| Método | Rota | Auth | Descrição |
|---|---|---|---|
| GET | `/health` | — | Status da API |

---

## Instalação e execução

### Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Projeto no [Supabase](https://supabase.com/) com banco PostgreSQL

### 1. Clonar e instalar dependências

```bash
git clone https://github.com/FelipeCarillo/tcc-ze-praga-backend.git
cd tcc-ze-praga-backend
uv sync
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com os valores do seu projeto Supabase:

```dotenv
# Supabase PostgreSQL — connection string direta para SQLAlchemy
# Encontrado em: Dashboard > Project Settings > Database > Connection string > URI
DATABASE_URL=postgresql+asyncpg://postgres:[password]@[host]:5432/postgres

# Supabase — usado apenas para Storage (upload de imagens)
SUPABASE_URL=https://seu-projeto.supabase.co
SUPABASE_SERVICE_ROLE_KEY=sua-service-role-key

# JWT — use uma string longa e aleatória em produção
JWT_SECRET_KEY=troque-por-uma-string-segura
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# App
APP_ENV=development
ALLOWED_ORIGINS=http://localhost:3000
```

### 3. Aplicar migrations

```bash
uv run alembic revision --autogenerate -m "initial"
uv run alembic upgrade head
```

### 4. Popular dados iniciais

```bash
uv run python scripts/seed_action_plans.py
```

O script popula:
- 3 planos de assinatura (free, pro, enterprise)
- Planos de ação completos para as 6 doenças do sistema

### 5. Iniciar o servidor

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Acesse:
- **API:** `http://localhost:8000/api/v1`
- **Docs (Swagger):** `http://localhost:8000/docs`
- **Docs (Redoc):** `http://localhost:8000/redoc`

---

## Testes

```bash
# Rodar todos os testes
uv run pytest

# Com relatório de coverage
uv run pytest --cov=app --cov-report=term-missing

# Relatório HTML
uv run pytest --cov=app --cov-report=html
start htmlcov/index.html
```

**Resultado atual:** 197 testes · 100% de cobertura (linhas + branches)

### Estrutura dos testes

```
tests/
├── conftest.py                  # Fixtures globais
├── unit/                        # Testes unitários com mocks
│   ├── auth/                    # Repository + Service
│   ├── users/
│   ├── subscriptions/
│   ├── usage/
│   ├── diagnoses/
│   ├── action_plans/
│   ├── test_config.py
│   ├── test_security.py
│   ├── test_exceptions.py
│   ├── test_pagination.py
│   ├── test_enums.py
│   └── test_dependencies.py
└── integration/                 # Testes HTTP com AsyncClient
    ├── test_auth_router.py
    ├── test_users_router.py
    ├── test_subscriptions_router.py
    ├── test_usage_router.py
    ├── test_diagnoses_router.py
    ├── test_action_plans_router.py
    ├── test_exception_handlers.py
    └── test_health_router.py
```

---

## Linting

```bash
uv run ruff check app/ tests/
uv run ruff format app/ tests/
```

---

## Repositórios relacionados

- **Frontend:** [tcc-ze-praga-frontend](https://github.com/FelipeCarillo/tcc-ze-praga-frontend) — React
- **ML / Chatbot:** repositório separado com rotas de inferência e chatbot
