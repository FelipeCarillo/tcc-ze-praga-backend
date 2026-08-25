<#
.SYNOPSIS
    Interruptor do Zé Praga em produção — desliga e liga a aplicação inteira.

.DESCRIPTION
    Controla os dois serviços que custam dinheiro e expõem superfície de
    ataque: o serviço no Google Cloud Run (backend + modelos ONNX) e o projeto
    Supabase (banco + storage). O frontend na Vercel é estático — com o
    backend parado ele não faz nada além de mostrar a interface, então não
    precisa ser desligado; use -IncluirVercel se quiser derrubar ele também.

    Desligar o Cloud Run é remover o acesso público (o binding allUsers do
    papel run.invoker): a API passa a responder 403 na hora. Não usamos
    max-instances=0 porque o Cloud Run exige no mínimo 1. Como o serviço já
    escala a zero, parado ele não custa nada de qualquer forma — o que se
    ganha aqui é fechar a porta, não economizar.

    Desligado, o projeto não responde, não gasta cota de LLM e não aceita
    cadastro. Religar leva ~2 min (o Supabase demora mais que o Cloud Run).

.PARAMETER Acao
    desligar | ligar | status

.PARAMETER IncluirVercel
    Também pausa/despausa o projeto da Vercel (o frontend estático).

.EXAMPLE
    .\zepraga.ps1 -Acao status
    .\zepraga.ps1 -Acao desligar
    .\zepraga.ps1 -Acao ligar

.NOTES
    Credenciais: crie "deploy.local.json" nesta mesma pasta (já ignorado pelo
    git) a partir de "deploy.local.example.json". Nunca versione esse arquivo.

    O Cloud Run é controlado via gcloud (precisa estar autenticado); Supabase
    e Vercel via API REST. Se algum passo falhar, o script mostra o link do
    painel para você fazer no braço — o botão manual sempre funciona.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('desligar', 'ligar', 'status')]
    [string]$Acao,

    [switch]$IncluirVercel
)

$ErrorActionPreference = 'Stop'

function Resolve-Gcloud {
    <#
    Acha o gcloud mesmo quando o PATH da sessao esta velho — o instalador do
    Cloud SDK nao atualiza terminais ja abertos, e o sintoma ("gcloud nao
    encontrado" logo depois de instalar) confunde mais do que ajuda.
    #>
    $cmd = Get-Command gcloud -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidatos = @(
        "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:ProgramFiles\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "${env:ProgramFiles(x86)}\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    )
    foreach ($c in $candidatos) { if (Test-Path $c) { return $c } }
    return $null
}

# ── Configuração ──────────────────────────────────────────────────────────────

$configPath = Join-Path $PSScriptRoot 'deploy.local.json'
if (-not (Test-Path $configPath)) {
    Write-Host "Falta o arquivo de credenciais: $configPath" -ForegroundColor Red
    Write-Host "Copie deploy.local.example.json para deploy.local.json e preencha." -ForegroundColor Yellow
    exit 1
}
$cfg = Get-Content $configPath -Raw | ConvertFrom-Json

function Get-Campo([string]$nome) {
    $valor = $cfg.$nome
    if ([string]::IsNullOrWhiteSpace($valor) -or $valor -like '*SEU_*' -or $valor -like '*seu-*') {
        return $null
    }
    return $valor
}

$gcpProject  = Get-Campo 'gcp_project'       # ex: "ze-praga-tcc"
$gcpRegion   = if (Get-Campo 'gcp_region') { Get-Campo 'gcp_region' } else { 'us-east1' }
$gcpService  = if (Get-Campo 'gcp_service') { Get-Campo 'gcp_service' } else { 'ze-praga-api' }
$sbToken     = Get-Campo 'supabase_access_token'
$sbRef       = Get-Campo 'supabase_project_ref'
$vcToken     = Get-Campo 'vercel_token'
$vcProject   = Get-Campo 'vercel_project_id'

# ── Helpers ───────────────────────────────────────────────────────────────────

function Escreve-Passo([string]$texto) { Write-Host "  $texto" -ForegroundColor Cyan }
function Escreve-Ok([string]$texto)    { Write-Host "  OK   $texto" -ForegroundColor Green }
function Escreve-Erro([string]$texto)  { Write-Host "  FALHA $texto" -ForegroundColor Red }
function Escreve-Pulo([string]$texto)  { Write-Host "  --   $texto" -ForegroundColor DarkGray }

function Invoke-Api {
    <#
    Chama a API e devolve @{ ok = $bool; corpo = <objeto|texto> }.
    Nunca lança: um serviço fora do ar não pode impedir o script de tentar
    desligar o outro — meio desligado é pior que desligado inteiro.
    #>
    param(
        [string]$Metodo,
        [string]$Uri,
        [hashtable]$Cabecalhos
    )
    try {
        $resp = Invoke-RestMethod -Method $Metodo -Uri $Uri -Headers $Cabecalhos
        return @{ ok = $true; corpo = $resp }
    } catch {
        return @{ ok = $false; corpo = $_.Exception.Message }
    }
}

# ── Google Cloud Run (backend) ────────────────────────────────────────────────

function Acao-CloudRun([string]$op) {
    if (-not $gcpProject) {
        Escreve-Pulo 'Cloud Run: sem gcp_project em deploy.local.json — pulando.'
        return
    }
    $gcloud = Resolve-Gcloud
    if (-not $gcloud) {
        Escreve-Erro 'gcloud nao encontrado — instale com: winget install Google.CloudSDK'
        return
    }

    $comum = @("--project", $gcpProject, "--region", $gcpRegion)
    $painel = "https://console.cloud.google.com/run/detail/$gcpRegion/$gcpService/security?project=$gcpProject"

    switch ($op) {
        'status' {
            # Presenca do binding allUsers = porta aberta. E o que o desligar
            # remove, entao e a leitura honesta do estado.
            $politica = & $gcloud run services get-iam-policy $gcpService @comum --format json 2>$null | Out-String
            $url = (& $gcloud run services describe $gcpService @comum --format 'value(status.url)' 2>$null)
            if (-not $url) {
                Escreve-Erro "Cloud Run: servico '$gcpService' nao encontrado em $gcpRegion."
                return
            }
            $aberto = $politica -match 'allUsers'
            $estado = if ($aberto) { 'LIGADO (publico)' } else { 'DESLIGADO (403)' }
            Write-Host "  CloudRun: $gcpService -> $estado"
            Write-Host "            $($url.Trim())"
        }
        'desligar' {
            Escreve-Passo "Fechando o acesso publico de $gcpService ..."
            & $gcloud run services remove-iam-policy-binding $gcpService @comum `
                --member=allUsers --role=roles/run.invoker --quiet 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Escreve-Ok 'Cloud Run fechado — a API responde 403 pra todo mundo.'
            } else {
                Escreve-Erro 'Nao consegui remover o binding (talvez ja estivesse fechado).'
                Write-Host "       No braco: $painel" -ForegroundColor Yellow
            }
        }
        'ligar' {
            Escreve-Passo "Reabrindo o acesso publico de $gcpService ..."
            & $gcloud run services add-iam-policy-binding $gcpService @comum `
                --member=allUsers --role=roles/run.invoker --quiet 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Escreve-Ok 'Cloud Run aberto. Primeira chamada tem cold start (imagem grande).'
            } else {
                Escreve-Erro 'Nao consegui adicionar o binding.'
                Write-Host "       No braco: $painel" -ForegroundColor Yellow
            }
        }
    }
}

# ── Supabase (banco + storage) ────────────────────────────────────────────────

function Acao-Supabase([string]$op) {
    if (-not $sbToken -or -not $sbRef) {
        Escreve-Pulo 'Supabase: sem credencial em deploy.local.json — pulando.'
        return
    }
    $headers = @{ Authorization = "Bearer $sbToken" }
    $base = "https://api.supabase.com/v1/projects/$sbRef"

    switch ($op) {
        'status' {
            $r = Invoke-Api -Metodo GET -Uri $base -Cabecalhos $headers
            if ($r.ok) { Write-Host "  Supabase: $sbRef -> $($r.corpo.status)" }
            else { Escreve-Erro "Supabase: $($r.corpo)" }
        }
        'desligar' {
            Escreve-Passo "Pausando o projeto Supabase $sbRef ..."
            $r = Invoke-Api -Metodo POST -Uri "$base/pause" -Cabecalhos $headers
            if ($r.ok) { Escreve-Ok 'Supabase pausado (banco inacessível).' }
            else {
                Escreve-Erro $r.corpo
                Write-Host "       No braço: https://supabase.com/dashboard/project/$sbRef/settings/general -> Pause project" -ForegroundColor Yellow
            }
        }
        'ligar' {
            Escreve-Passo "Restaurando o projeto Supabase $sbRef ..."
            $r = Invoke-Api -Metodo POST -Uri "$base/restore" -Cabecalhos $headers
            if ($r.ok) { Escreve-Ok 'Supabase restaurando (pode levar alguns minutos).' }
            else {
                Escreve-Erro $r.corpo
                Write-Host "       No braço: https://supabase.com/dashboard/project/$sbRef -> Restore project" -ForegroundColor Yellow
            }
        }
    }
}

# ── Vercel (frontend estático — opcional) ─────────────────────────────────────

function Acao-Vercel([string]$op) {
    if (-not $IncluirVercel) { return }
    if (-not $vcToken -or -not $vcProject) {
        Escreve-Pulo 'Vercel: sem credencial em deploy.local.json — pulando.'
        return
    }
    $headers = @{ Authorization = "Bearer $vcToken" }
    $base = "https://api.vercel.com/v9/projects/$vcProject"

    switch ($op) {
        'status'   { $r = Invoke-Api -Metodo GET  -Uri $base            -Cabecalhos $headers
                     if ($r.ok) { Write-Host "  Vercel  : $vcProject -> live=$(-not $r.corpo.paused)" }
                     else { Escreve-Erro "Vercel: $($r.corpo)" } }
        'desligar' { Escreve-Passo 'Pausando o projeto na Vercel ...'
                     $r = Invoke-Api -Metodo POST -Uri "$base/pause"   -Cabecalhos $headers
                     if ($r.ok) { Escreve-Ok 'Vercel pausada.' } else { Escreve-Erro $r.corpo } }
        'ligar'    { Escreve-Passo 'Despausando o projeto na Vercel ...'
                     $r = Invoke-Api -Metodo POST -Uri "$base/unpause" -Cabecalhos $headers
                     if ($r.ok) { Escreve-Ok 'Vercel no ar.' } else { Escreve-Erro $r.corpo } }
    }
}

# ── Execução ──────────────────────────────────────────────────────────────────

Write-Host ''
switch ($Acao) {
    'status' {
        Write-Host 'Zé Praga — situação atual' -ForegroundColor White
        Acao-CloudRun 'status'; Acao-Supabase 'status'; Acao-Vercel 'status'
    }
    'desligar' {
        Write-Host 'Zé Praga — DESLIGANDO' -ForegroundColor Yellow
        # Backend primeiro: com a API fora, ninguém escreve no banco enquanto
        # ele é pausado. A ordem inversa deixaria uma janela de requisições
        # batendo num banco sumindo.
        Acao-CloudRun 'desligar'; Acao-Supabase 'desligar'; Acao-Vercel 'desligar'
        Write-Host ''
        Write-Host 'Projeto desligado. Nada responde, nada gasta.' -ForegroundColor Green
    }
    'ligar' {
        Write-Host 'Zé Praga — LIGANDO' -ForegroundColor Yellow
        # Banco primeiro: a API roda migrations no boot e precisa do Postgres
        # de pé, senão sobe quebrada.
        Acao-Supabase 'ligar'; Acao-CloudRun 'ligar'; Acao-Vercel 'ligar'
        Write-Host ''
        Write-Host 'Subindo. Espere ~2 min e confira com: .\zepraga.ps1 -Acao status' -ForegroundColor Green
    }
}
Write-Host ''
