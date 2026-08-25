# Deploy do Zé Praga

Guia para colocar o projeto no ar de graça, com cadastro fechado por
verificação de e-mail e um interruptor para desligar tudo quando não estiver
em uso.

## Estado atual (25/08/2026)

> **O serviço está DESLIGADO.** O acesso público foi removido — a API responde
> 503. Para ligar: `.\scripts\deploy\zepraga.ps1 -Acao ligar`.

Já provisionado:

| Recurso | Identificador | Situação |
|---|---|---|
| Projeto Supabase | `majcxiebwxkakvffnodj` (`ze-praga`, us-east-1, org AgriWave) | ACTIVE_HEALTHY, **18 tabelas migradas até `0010`**, pgvector instalado |
| URL do Supabase | `https://majcxiebwxkakvffnodj.supabase.co` | — |
| API key do Resend | `ze-praga-producao` (id `d99e9655-baff-4831-b3a2-030147c7e54b`) | sending_access |
| Projeto Vercel | `prj_bjAzAmdHbJa0qBn5K0l9I2zYFBls` (`tcc-ze-praga-frontend`) | deploy READY, previews protegidos por Vercel Authentication |
| Projeto GCP | `ze-praga-tcc`, faturamento `01E8D8-BEE8EB-A15A75` | ativo |
| Serviço Cloud Run | `ze-praga-api` em `us-east1`, revisão `00002-wnf` | **no ar**, `/api/v1/health` respondendo |
| URL da API | `https://ze-praga-api-zwi6li6n7q-ue.a.run.app` | também atende em `https://ze-praga-api-258465616083.us-east1.run.app` |

Validado em produção contra o serviço no ar: cadastro devolvendo 202, login
barrado antes de confirmar, e CORS liberando o domínio da Vercel. Os seeds
rodaram no boot do container — 1 cultura, 6 doenças e 18 planos de ação.

Bucket `uploads` criado e confirmado.

Falta:

1. **Autorizar o Cloud Build no GitHub** — único passo que não sai por API
   (é OAuth no navegador). Veja a seção 6.
2. **Mergear** as branches abertas: `chore/api-url-producao` no frontend e
   `feat/deploy-verificacao-email` no backend.
3. **Remetente do Resend** — veja o aviso abaixo. É o que bloqueia a
   demonstração hoje.

> ### O gargalo da demonstração
>
> O remetente sandbox (`onboarding@resend.dev`) **só entrega no e-mail dono da
> conta Resend**, que é `felipecarillo@outlook.com`. Confirmado em produção: o
> Resend devolveu 403 com essa mensagem exata ao tentar enviar para outro
> endereço.
>
> Consequência prática: hoje **só esse endereço consegue criar conta**. Nem
> outro e-mail seu, nem ninguém da banca. O backend não quebra: o
> `ResendEmailSender` registra o erro e o cadastro segue devolvendo 202. Mas o
> link nunca chega, e a conta fica inativa para sempre.
>
> Saídas, da melhor para a pior:
> 1. Verificar um domínio em resend.com/domains e trocar o `EMAIL_FROM`.
> 2. Usar `felipecarillo@outlook.com` como a conta da demonstração.
> 3. Ativar contas na mão: `UPDATE users SET is_active = true WHERE email = '...'`.

> Produção na Vercel **não** pode ser protegida por senha no plano Hobby (só
> previews). Quando o `main` receber push, o deploy de produção nasce com URL
> pública. A defesa real do projeto é o gate de e-mail e o backend desligado.

| Peça | Onde roda | Custo |
|---|---|---|
| Frontend (React) | Vercel | Grátis (Hobby) |
| Backend (FastAPI + ONNX) | Google Cloud Run (container) | Free tier + ~R$1/mês de imagem |
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

O Cloud Run resolve os três: container de verdade (imagem sem o limite de
250 MB), processo persistente enquanto atende, e sem teto de 29 s como o do
API Gateway da AWS — o `--timeout 3600` cobre SSE longo. Escala a zero, então
parado não custa nada.

O Hugging Face Spaces foi a primeira escolha e caiu: no plano gratuito só o
SDK Gradio estava disponível, e o projeto precisa de Docker.

**Não é literalmente grátis.** O free tier mensal do Cloud Run (2 M requisições,
360.000 GiB-s, 180.000 vCPU-s) cobre a demonstração com folga — a 4 GiB e
2 vCPU dá cerca de 25 h/mês de tempo servindo. Mas a imagem de ~2 GB passa dos
0,5 GB gratuitos do Artifact Registry, o que dá algo perto de **R$ 1 por mês**.

---

## 1. Supabase — banco e storage

1. Crie o projeto em <https://supabase.com/dashboard>. Anote a região e a
   senha do banco.
2. **`DATABASE_URL` — use o pooler, não a conexão direta.**

   O host direto que o painel mostra (`db.<ref>.supabase.co`) resolve **apenas
   em IPv6**. Se o container não tiver egress IPv6 — e o Cloud Run, por padrão,
   não tem — o backend não conecta e morre no boot, sem mensagem que aponte a
   causa.

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

## 3. Google Cloud Run — backend

### Uma vez, no console

1. Crie o projeto em <https://console.cloud.google.com/projectcreate>.
2. Vincule uma **conta de faturamento** (Billing → Link a billing account). O
   Cloud Run exige, mesmo quando o consumo fica dentro do free tier.
3. Instale e autentique o gcloud:

```bash
winget install Google.CloudSDK
```

```bash
gcloud auth login
```

### O deploy

Os segredos ficam em `scripts/deploy/cloudrun.env` (ignorado pelo git). Depois:

```powershell
.\scripts\deploy\cloudrun.ps1 -ProjectId SEU_PROJETO_GCP
```

O script habilita as APIs (`run`, `cloudbuild`, `artifactregistry`), manda o
código pro Cloud Build e publica com estes parâmetros:

| Flag | Valor | Por quê |
|---|---|---|
| `--memory` | `4Gi` | cabe o ensemble dos três ONNX |
| `--cpu` | `2` | inferência é CPU-bound |
| `--concurrency` | `4` | mais que isso e 2 vCPU disputam entre si na inferência |
| `--timeout` | `3600` | o chat é SSE longo; o default de 300 s cortaria |
| `--min-instances` | `0` | escala a zero — parado não custa |
| `--max-instances` | `2` | teto contra abuso e contra surpresa na fatura |

Depois do primeiro deploy o script lê a URL gerada, grava em `cloudrun.env` e
reaplica como `PUBLIC_API_URL`. **Isso não é detalhe:** é a URL que vai dentro
do link do e-mail de confirmação. Errada, ninguém consegue ativar a conta.

Ao final ele bate em `GET /api/v1/health` e mostra a URL para você colar na
Vercel.

> **Não pule o `REQUIRE_EMAIL_VERIFICATION=true`** (já vem assim no
> `cloudrun.env`). O default do código é `false` para não atrapalhar dev e
> testes. Com `false` em produção, `POST /auth/register` fica aberto e qualquer
> um cria conta e gasta seu crédito de LLM.

### Cold start

A imagem tem ~2 GB. Depois de um tempo ociosa, a instância morre e a próxima
chamada paga o custo de subir tudo de novo mais inicializar as sessões ONNX —
pode passar de 30 s. Antes de apresentar para a banca, faça uma chamada em
`/api/v1/health` para aquecer.

Se quiser eliminar isso durante a apresentação, `--min-instances 1` mantém uma
instância viva — mas aí ela é cobrada continuamente e sai do free tier. Ligue
antes, desligue depois.

---

## 6. Deploy automático a cada push na main

O `cloudbuild.yaml` na raiz descreve o pipeline. Já estão prontos: APIs
habilitadas, permissões da conta de build concedidas, bucket de modelos criado
e populado, e a conexão `ze-praga-github` no Cloud Build.

Falta autorizar o Cloud Build no GitHub — OAuth no navegador, sem equivalente
por linha de comando:

```powershell
.\scripts\deploy\trigger.ps1
```

Ele imprime o link de autorização se ainda faltar, e cria o trigger assim que
a conexão estiver pronta.

### A armadilha que esse pipeline resolve

Os três `.onnx` estão no repositório via **Git LFS**. Um `git clone` traz
ponteiros de 134 bytes, não os 485 MB de modelo. Uma imagem construída com
esses ponteiros **sobe saudável**: o `InferenceService` não acha um ONNX válido
e cai no mock por degradação graciosa. O `/health` responde 200, os testes
passam, e a API devolve diagnósticos aleatórios sem nenhum sinal de erro.

Por isso o primeiro passo do build copia os modelos do Cloud Storage
(`gs://ze-praga-tcc-models`) por cima dos ponteiros, e há uma trava: qualquer
`.onnx` com menos de 1 MB derruba o build. Falhar alto é melhor que servir
mock calado.

Isso também evita a cota de banda do GitHub LFS (~1 GB/mês), que dois builds
esgotariam.

Ao treinar um modelo novo, atualize o bucket:

```bash
gcloud storage cp models/* gs://ze-praga-tcc-models/models/
```

### O deploy não liga a aplicação

O trigger publica uma revisão nova, mas **não reabre o acesso público**. Se o
serviço estiver desligado, ele continua respondendo 403 depois do build. Ligar
segue sendo decisão sua, pelo `zepraga.ps1`.

---

## 4. Vercel — frontend

1. **Add New → Project**, importe `tcc-ze-praga-frontend`.
2. Framework: Create React App (detecta sozinho). O `vercel.json` já força
   `CI=false` para que warning de lint não derrube o build.
3. **Environment Variables**:

| Variável | Valor |
|---|---|
| `REACT_APP_API_URL` | URL que o `cloudrun.ps1` imprimiu ao final |
| `REACT_APP_AUTH_MODE` | `api` |
| `REACT_APP_USE_MOCK` | `false` |

4. Deploy. Depois ajuste `FRONTEND_URL` e `ALLOWED_ORIGINS` em
   `scripts/deploy/cloudrun.env` com o domínio real que a Vercel gerou e rode
   o `cloudrun.ps1` de novo (ele atualiza o serviço no lugar).

---

## 5. O interruptor

`scripts/deploy/zepraga.ps1` desliga e liga tudo de um comando só.

```bash
cp scripts/deploy/deploy.local.example.json scripts/deploy/deploy.local.json
```

Preencha com o ID do projeto GCP, o token do Supabase e (opcional) o da Vercel.
O arquivo já está no `.gitignore`.

```powershell
.\scripts\deploy\zepraga.ps1 -Acao status
.\scripts\deploy\zepraga.ps1 -Acao desligar
.\scripts\deploy\zepraga.ps1 -Acao ligar
```

**Desligar** fecha o acesso público do Cloud Run e depois pausa o Supabase —
nessa ordem, para que a API já esteja fora do ar quando o banco sumir.

Fechar o acesso é remover o binding `allUsers` do papel `run.invoker`: a API
passa a responder 403 na hora, e reabrir é um comando. Não dá para usar
`--max-instances=0` — o Cloud Run exige no mínimo 1. Como o serviço já escala a
zero, parado ele não custa nada de qualquer jeito; o que se ganha aqui é fechar
a porta, não economizar. Nada responde, nada gasta.
**Ligar** faz o inverso: banco primeiro, porque a API roda migrations no boot e
precisa do Postgres de pé.

Religar leva ~2 min (o Supabase demora mais que o Cloud Run). Se algum passo
falhar, o script imprime o link do painel para fazer no braço — o botão manual
sempre funciona.

Adicione `-IncluirVercel` para pausar também o frontend. Normalmente não é
preciso: sem backend ele é uma página estática que não faz nada.

---

## Camadas de defesa

Ordenadas pelo que efetivamente barra alguém:

1. **Desligado** — Cloud Run sem acesso público e Supabase pausado. Superfície
   zero. É o estado padrão fora de demonstrações.
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
