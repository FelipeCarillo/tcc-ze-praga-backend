# Deploy do Zé Praga

Guia para colocar o projeto no ar de graça, com cadastro fechado por
verificação de e-mail e um interruptor para desligar tudo quando não estiver
em uso.

## Estado atual (24/08/2026)

Já provisionado:

| Recurso | Identificador | Situação |
|---|---|---|
| Projeto Supabase | `majcxiebwxkakvffnodj` (`ze-praga`, us-east-1, org AgriWave) | ACTIVE_HEALTHY, **18 tabelas migradas até `0010`**, pgvector instalado |
| URL do Supabase | `https://majcxiebwxkakvffnodj.supabase.co` | — |
| API key do Resend | `ze-praga-producao` (id `d99e9655-baff-4831-b3a2-030147c7e54b`) | sending_access |
| Projeto Vercel | `prj_bjAzAmdHbJa0qBn5K0l9I2zYFBls` (`tcc-ze-praga-frontend`) | deploy READY, previews protegidos por Vercel Authentication |

A conexão foi validada de ponta a ponta contra o Supabase: cadastro devolvendo
202, login barrado antes de confirmar, link ativando a conta, login liberado e
link reutilizado sendo recusado.

Falta:

1. **service_role key** — não sai pela API, só pelo painel
   (Project Settings → API).
2. **Bucket de Storage** — criar no painel.
3. **Space do Hugging Face** — criar e empurrar (seção 3). É a única etapa que
   não dá para automatizar: os 507 MB de ONNX vão por `git push` com LFS.
4. **Variáveis na Vercel** — definir no painel (seção 4).
5. **Commitar o frontend** — as telas de verificação de e-mail ainda estão só
   na máquina local; o deploy atual foi feito do `main` do GitHub, sem elas.

> Produção na Vercel **não** pode ser protegida por senha no plano Hobby (só
> previews). Quando o `main` receber push, o deploy de produção nasce com URL
> pública. A defesa real do projeto é o gate de e-mail e o backend desligado.

| Peça | Onde roda | Custo |
|---|---|---|
| Frontend (React) | Vercel | Grátis (Hobby) |
| Backend (FastAPI + ONNX) | Hugging Face Spaces, SDK Docker | Grátis (CPU basic) |
| Banco + Storage | Supabase | Grátis |
| E-mail transacional | Resend | Grátis |

---

## Por que o backend não vai para a Vercel

Três impedimentos, nenhum contornável sem mutilar o projeto:

1. **Tamanho.** `backend/models/` tem ~507 MB de ONNX (só o ViT-B/16 são
   327 MB). O limite de uma função serverless da Vercel é 250 MB
   descompactado.
2. **Streaming.** O chat é SSE de longa duração (`sse-starlette`,
   `POST /chat/resume/stream`). Serverless corta a conexão.
3. **Estado no processo.** O `lifespan` em `app/main.py` inicializa o
   checkpointer e o store do LangGraph no boot justamente para evitar cold
   start no primeiro request. Serverless recria o processo a cada invocação e
   joga esse trabalho fora.

O Hugging Face Spaces resolve os três: container de verdade, processo
persistente e **16 GB de RAM** no tier gratuito — folga para os três modelos e
o ensemble. Como bônus, o Space é um repositório Git com LFS nativo, que é
exatamente como os `.onnx` já estão versionados.

---

## 1. Supabase — banco e storage

1. Crie o projeto em <https://supabase.com/dashboard>. Anote a região e a
   senha do banco.
2. **`DATABASE_URL` — use o pooler, não a conexão direta.**

   O host direto que o painel mostra (`db.<ref>.supabase.co`) resolve **apenas
   em IPv6**. Se o container do Space não tiver egress IPv6 — e normalmente não
   tem — o backend não conecta e morre no boot, sem mensagem que aponte a causa.

   O pooler (Supavisor) responde em IPv4. Use **modo sessão, porta 5432**: ele
   se comporta como conexão direta e mantém prepared statements, de que o
   asyncpg e o checkpointer do LangGraph dependem. O modo transação (6543) os
   desabilita e exigiria `statement_cache_size=0` no asyncpg.

   Repare que o usuário muda: vira `postgres.<ref>`, não `postgres`.

   ```
   postgresql+asyncpg://postgres.<ref>:<senha>@aws-0-<regiao>.pooler.supabase.com:5432/postgres
   ```

   Para o projeto `ze-praga` (verificado funcionando de ponta a ponta):

   ```
   postgresql+asyncpg://postgres.majcxiebwxkakvffnodj:<senha>@aws-0-us-east-1.pooler.supabase.com:5432/postgres
   ```

   O prefixo do host importa: `aws-1-us-east-1` existe e resolve, mas devolve
   `Tenant or user not found` — cada projeto vive em um shard específico.
3. **Project Settings → API**: copie a *Project URL* (`SUPABASE_URL`) e a
   chave *service_role* (`SUPABASE_SERVICE_ROLE_KEY`).
4. **Database → Extensions**: habilite `vector` (pgvector) — a memória
   semântica do agente depende dela.
5. **Storage**: crie o bucket de imagens que o `UploadService` usa.

As migrations rodam sozinhas: o `CMD` do Dockerfile executa
`alembic upgrade head` e os seeds a cada boot do container.

> A migration `0010_add_email_verification_tokens` cria a tabela dos tokens de
> confirmação. É aditiva — não mexe em `users`.

---

## 2. Resend — e-mail de confirmação

1. Crie a conta em <https://resend.com> e gere uma API key
   (**API Keys → Create**). Ela começa com `re_`.
2. Para o remetente há dois caminhos:
   - **Sem domínio próprio:** use `onboarding@resend.dev`. Funciona na hora,
     mas **só entrega no e-mail dono da conta Resend**. Serve para
     demonstração e para a banca, não para usuários de verdade.
   - **Com domínio próprio:** **Domains → Add Domain**, publique os registros
     DNS que o painel indicar e use `no-reply@seudominio.com`. Aí entrega para
     qualquer destinatário.

Sem `RESEND_API_KEY` configurada o backend **não quebra**: cai no
`NullEmailSender`, que registra o link no log em vez de enviar (mesmo padrão de
degradação graciosa do `InferenceService`). Útil em desenvolvimento — o link
aparece no terminal.

---

## 3. Hugging Face Space — backend

1. **New Space** em <https://huggingface.co/new-space>:
   - SDK: **Docker** (blank template)
   - Hardware: **CPU basic** (gratuito)
   - Visibilidade: Public (o cadastro fica fechado pela verificação de e-mail,
     não pela visibilidade do Space)
2. Adicione o Space como um segundo remote do repositório do backend e empurre:

```bash
git remote add space https://huggingface.co/spaces/SEU_USUARIO/ze-praga-api
git push space main
```

O `README.md` já traz o bloco YAML que o Space exige (`sdk: docker`,
`app_port: 8000`) e o `Dockerfile` já roda com UID 1000, que é o que o Spaces
espera. Os `.onnx` sobem por LFS.

3. **Settings → Variables and secrets** — cadastre como *Secret*:

| Variável | Valor |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` do Supabase |
| `SUPABASE_URL` | Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | chave service_role |
| `JWT_SECRET_KEY` | string longa e aleatória — **gere uma nova, não reaproveite a de dev** |
| `OPENAI_API_KEY` | sua chave |
| `RESEND_API_KEY` | `re_...` |
| `EMAIL_FROM` | `Zé Praga <onboarding@resend.dev>` |
| `REQUIRE_EMAIL_VERIFICATION` | `true` — **é isto que fecha o cadastro** |
| `PUBLIC_API_URL` | `https://SEU_USUARIO-ze-praga-api.hf.space` |
| `FRONTEND_URL` | `https://SEU_PROJETO.vercel.app` |
| `ALLOWED_ORIGINS` | `https://SEU_PROJETO.vercel.app` (sem barra no fim) |
| `APP_ENV` | `production` |

Confira em `GET /api/v1/health` quando terminar de subir.

> **Não pule o `REQUIRE_EMAIL_VERIFICATION=true`.** O default é `false` para
> não atrapalhar dev e testes. Com `false` em produção, `POST /auth/register`
> fica aberto e qualquer um cria conta e gasta seu crédito de LLM.

---

## 4. Vercel — frontend

1. **Add New → Project**, importe `tcc-ze-praga-frontend`.
2. Framework: Create React App (detecta sozinho). O `vercel.json` já força
   `CI=false` para que warning de lint não derrube o build.
3. **Environment Variables**:

| Variável | Valor |
|---|---|
| `REACT_APP_API_URL` | `https://SEU_USUARIO-ze-praga-api.hf.space` |
| `REACT_APP_AUTH_MODE` | `api` |
| `REACT_APP_USE_MOCK` | `false` |

4. Deploy. Depois volte ao Space e ajuste `FRONTEND_URL` e `ALLOWED_ORIGINS`
   com o domínio real que a Vercel gerou.

---

## 5. O interruptor

`scripts/deploy/zepraga.ps1` desliga e liga tudo de um comando só.

```bash
cp scripts/deploy/deploy.local.example.json scripts/deploy/deploy.local.json
```

Preencha com o token do Hugging Face, o do Supabase e (opcional) o da Vercel.
O arquivo já está no `.gitignore`.

```powershell
.\scripts\deploy\zepraga.ps1 -Acao status
.\scripts\deploy\zepraga.ps1 -Acao desligar
.\scripts\deploy\zepraga.ps1 -Acao ligar
```

**Desligar** pausa o Space e depois o projeto Supabase — nessa ordem, para que
a API já esteja fora do ar quando o banco sumir. Nada responde, nada gasta.
**Ligar** faz o inverso: banco primeiro, porque a API roda migrations no boot e
precisa do Postgres de pé.

Religar leva ~2 min (o Supabase demora mais que o Space). Se algum passo
falhar, o script imprime o link do painel para fazer no braço — o botão manual
sempre funciona.

Adicione `-IncluirVercel` para pausar também o frontend. Normalmente não é
preciso: sem backend ele é uma página estática que não faz nada.

---

## Camadas de defesa

Ordenadas pelo que efetivamente barra alguém:

1. **Desligado** — Space e Supabase pausados. Superfície zero. É o estado
   padrão fora de demonstrações.
2. **Verificação de e-mail** (`REQUIRE_EMAIL_VERIFICATION=true`) — a conta
   nasce inativa; só o link do e-mail a ativa. Com o remetente sandbox do
   Resend, na prática só o dono da conta Resend consegue se cadastrar.
3. **Cotas por plano** — o `UsageService` já limita chat, inferência e API por
   dia. Quem passa pelo cadastro ainda esbarra no teto do plano Free.
4. **CORS** — `ALLOWED_ORIGINS` restrito ao domínio da Vercel.

### O que ainda não existe

- **Rate limit por IP.** Só `diagnoses` tem alguma limitação. `POST
  /auth/register` e `POST /auth/login` aceitam requisições sem throttle — um
  script pode martelar login. As cotas seguram o custo de LLM, mas não o custo
  de CPU.
- **Reset de senha.** Não há fluxo de "esqueci minha senha". Conta com e-mail
  digitado errado é conta perdida.
- **Teto de gasto na OpenAI.** Configure um *usage limit* no painel da OpenAI.
  É a única proteção real caso algo escape das cotas.
