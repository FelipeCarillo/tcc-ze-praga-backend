# Deploy enxuto do Zé Praga

Como colocar o projeto inteiro na nuvem gastando o mínimo, sem piorar o que a
pessoa vê no celular. Complementa o [DEPLOY.md](DEPLOY.md), que documenta o
Cloud Run atual, o Supabase, o Resend e o interruptor liga/desliga.

## O que mudou e por quê

| Antes | Agora (perfil enxuto) | Efeito |
|---|---|---|
| Chat por SSE, conexão aberta por resposta | `POST /chat` síncrono; `CHAT_STREAMING_ENABLED=false` desliga o SSE | Mesmo custo de tokens, sem conexão longa. Libera hospedagem serverless (Lambda) e timeout curto. |
| Três ONNX na memória (485 MB, 4 GiB de RAM) | `INFERENCE_MODELS=efficientnet_b4` | 98,77% contra 99,10% do ensemble, com 67 MB. A instância cabe em 2 GiB / 1 vCPU. |
| Gate de visão com imagem em resolução "auto" | `VISION_IMAGE_DETAIL=low` | ~85 tokens por foto no gpt-4o em vez de ~765+. O gate só decide se a foto é de folha. |
| Fallback do streaming rodava o turno **de novo** (`graph.ainvoke`) quando não vinham tokens | Lê a resposta do checkpoint | Corrige bug: chamada dobrada ao LLM e às tools, e até diagnóstico duplicado. |
| Migrations e seeds a cada boot do container | `RUN_MIGRATIONS_ON_BOOT=false` + `docker run <imagem> migrate` | Cold start mais curto; o Lambda não repete seed por instância. |
| Checkpointer e Store com **uma** conexão Postgres pela vida do processo | Pool psycopg com `check` | Antes, conexão derrubada pelo pooler ou processo congelado quebrava todo o chat até reiniciar (reproduzido com `pg_terminate_backend`). Agora se recupera sozinho. |
| Foto do celular enviada inteira (3–12 MB) | Reduzida no aparelho para 1600 px JPEG | Upload várias vezes mais rápido no 4G; cabe no limite de 6 MB do Lambda. |
| Bundle inicial de 336 kB gzip | 242 kB gzip (páginas sob demanda) | Primeira tela mais rápida no celular. |

O que **não** mudou: cotas, verificação de e-mail, histórico, plano de ação,
pergunta do agente (HITL) — agora também pelo `POST /chat/resume` —, áudio e
gate de visão. O SSE continua no código e volta com
`CHAT_STREAMING_ENABLED=true` + `REACT_APP_CHAT_STREAMING=true`.

### A experiência sem streaming

Sem tokens chegando, o indicador do chat avança pelas etapas que o agente
percorre — "Olhando a foto…" → "Rodando o diagnóstico…" → "Consultando
próximos cuidados…" → "Escrevendo a resposta…" — e a resposta aparece inteira,
com o card de diagnóstico. Num diagnóstico por foto, o que importa é o card,
que no streaming também só chegava no fim.

## Onde hospedar

| Opção | Parado | Em uso (demonstração) | Esforço | Veredito |
|---|---|---|---|---|
| **Google Cloud Run** (já está no ar) | R$ 0 | Free tier cobre ~50 h/mês de requisição a 2 GiB / 1 vCPU | Já pronto: `cloudrun.ps1` (perfil enxuto é o padrão) | **Recomendado** |
| **AWS Lambda + Function URL** | US$ 0 | Cota gratuita do Lambda (400 mil GB-s/mês ≈ 55 h a 2 GB) + ECR (~US$ 0,10/GB-mês) | Médio: passo a passo abaixo | Melhor opção **se precisar ser AWS** |
| AWS ECS (Express Mode/Fargate) | Cobra 24 h (tarefa + balanceador) | Dezenas de dólares por mês | Médio | Não compensa para TCC |
| AWS App Runner | — | — | — | Fechado para clientes novos desde 30/04/2026 |
| Vercel (backend) | — | — | — | Inviável: limite de 250 MB e sem processo persistente |

Frontend: continua na **Vercel** (Hobby, grátis). Banco e storage: **Supabase**
(grátis). O que pesa de verdade é o **LLM** (abaixo), não a infraestrutura.

### Custo do LLM por uso

Um turno com foto faz cerca de 4 chamadas ao `gpt-4o-mini` (agente + tools) e
1 ao gate de visão. Estimativa nos preços de tabela da OpenAI: **~US$ 0,003
por foto analisada**, ou ~US$ 3 a cada mil diagnósticos. Para manter isso:

- Deixe os usuários da demonstração no plano **Free**: Pro e Enterprise usam
  `gpt-4o` (o `llm_model` gravado em `subscription_plans.features`), ~16× mais
  caro por token.
- Gate de visão no `gpt-4o` com `detail=low`. Parece contraintuitivo, mas a
  OpenAI multiplica o custo de imagem no `gpt-4o-mini` (~2.833 tokens cobrados
  contra 85 no `gpt-4o`), então para foto o `gpt-4o` "low" sai mais barato.
- **Configure o teto de gasto** em *platform.openai.com → Settings → Limits*.
  Continua sendo a única trava se algo escapar das cotas.

## Opção A — Cloud Run (recomendada)

```powershell
.\scripts\deploy\cloudrun.ps1 -ProjectId ze-praga-tcc            # perfil enxuto
.\scripts\deploy\cloudrun.ps1 -ProjectId ze-praga-tcc -Perfil completo
```

O perfil enxuto aplica `--memory 2Gi --cpu 1 --timeout 300` e, se faltarem no
`cloudrun.env`, `INFERENCE_MODELS=efficientnet_b4`,
`CHAT_STREAMING_ENABLED=false` e `RUN_MIGRATIONS_ON_BOOT=false`.

O `cloudbuild.yaml` (deploy a cada push) agora só baixa os modelos de
`_MODELS` — por padrão `soja_efficientnet_b4` — e tira os demais da imagem.

**Migrations**: com `RUN_MIGRATIONS_ON_BOOT=false`, rode uma vez por deploy que
traga migration nova:

```bash
docker run --rm --env-file .env.cloud <imagem> migrate
```

## Opção B — AWS Lambda + Function URL

A mesma imagem roda no Lambda: o Dockerfile inclui o
[AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter),
que encaminha as requisições para o uvicorn. Só é viável porque o chat é
síncrono: no modo BUFFERED do Lambda não há SSE.

Limites que importam: requisição de até **6 MB** (a foto já sai comprimida do
app) e execução de até 15 min (usaremos 120 s).

> Os comandos abaixo foram montados a partir da documentação da AWS e **não
> foram executados** contra uma conta real nesta revisão. No PowerShell,
> troque a quebra de linha `\` por `` ` ``.

### 1. Imagem no ECR

```bash
# Só o EfficientNet-B4 do LFS; os outros ficam como ponteiro e não carregam.
git lfs pull --include "models/soja_efficientnet_b4.*"

aws ecr create-repository --repository-name ze-praga-api --region us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS \
  --password-stdin <conta>.dkr.ecr.us-east-1.amazonaws.com

# --provenance=false: o Lambda recusa o índice multi-manifest do buildx.
docker buildx build --platform linux/amd64 --provenance=false \
  -t <conta>.dkr.ecr.us-east-1.amazonaws.com/ze-praga-api:latest --push .
```

Região `us-east-1`: a mesma do Supabase, porque a latência do chat é dominada
pelas idas e voltas ao banco.

### 2. Função

```bash
aws iam create-role --role-name ze-praga-lambda \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name ze-praga-lambda \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

cp .env.cloud.example .env.cloud   # preencha
python scripts/deploy/env_to_lambda.py .env.cloud > lambda-env.json

aws lambda create-function --function-name ze-praga-api \
  --package-type Image \
  --code ImageUri=<conta>.dkr.ecr.us-east-1.amazonaws.com/ze-praga-api:latest \
  --role arn:aws:iam::<conta>:role/ze-praga-lambda \
  --memory-size 2048 --timeout 120 --architectures x86_64 \
  --environment file://lambda-env.json
```

2 GB de memória também dão mais CPU: o Lambda distribui vCPU proporcional à
memória, e a inferência ONNX é CPU-bound.

### 3. URL pública

```bash
aws lambda create-function-url-config --function-name ze-praga-api --auth-type NONE

# Desde out/2025, URL nova criada pela CLI exige as DUAS permissões.
aws lambda add-permission --function-name ze-praga-api \
  --statement-id url-publica --action lambda:InvokeFunctionUrl \
  --principal "*" --function-url-auth-type NONE
aws lambda add-permission --function-name ze-praga-api \
  --statement-id url-publica-invoke --action lambda:InvokeFunction \
  --principal "*" --invoked-via-function-url
```

Não configure CORS na Function URL: o FastAPI já responde CORS
(`ALLOWED_ORIGINS`), e os dois juntos duplicam os cabeçalhos.

Com a URL em mãos: coloque-a em `PUBLIC_API_URL` (vai no link do e-mail de
confirmação), gere o JSON de novo e aplique com
`aws lambda update-function-configuration --function-name ze-praga-api --environment file://lambda-env.json`.
Na Vercel, `REACT_APP_API_URL` recebe a mesma URL.

### 4. Migrations, ligar e desligar

```bash
docker run --rm --env-file .env.cloud <imagem> migrate     # uma vez por migration

aws lambda put-function-concurrency --function-name ze-praga-api \
  --reserved-concurrent-executions 0                      # desligar: tudo recebe 429
aws lambda delete-function-concurrency --function-name ze-praga-api   # ligar
```

Contas novas da AWS costumam vir com teto de 10 execuções simultâneas. Isso já
funciona como trava de custo (equivale ao `--max-instances` do Cloud Run).

### Cold start

Depois de um tempo parada, a primeira requisição paga a subida do contêiner,
os imports do LangChain, a sessão ONNX e as conexões com o Supabase: espere
alguns segundos a mais. Antes de apresentar para a banca, chame
`/api/v1/health` para aquecer.

## Frontend

Variáveis na Vercel (já em `.env.production`):

| Variável | Valor |
|---|---|
| `REACT_APP_API_URL` | URL do Cloud Run ou da Function URL |
| `REACT_APP_CHAT_STREAMING` | `false` |
| `REACT_APP_DIAGNOSIS_MODELS` | `efficientnet` — o seletor só oferece o que o servidor carrega |

## Pendências conhecidas

- **Rate limit por processo** (`app/core/rate_limit.py`): no Lambda cada
  instância tem o seu, então o limite efetivo multiplica pelo número de
  instâncias. Para a demonstração, o teto de concorrência basta.
- **Acurácia com foto comprimida**: o classificador redimensiona para 380 px de
  qualquer jeito, mas vale rodar o `evaluate.py` do playground com o test set
  reduzido para 1600 px JPEG 85% e confirmar que os 98,77% se mantêm.
- **Quantização INT8** do EfficientNet-B4 (onnxruntime) reduziria a imagem e a
  RAM mais um pouco, mas exige revalidar a acurácia. Não foi feita.
