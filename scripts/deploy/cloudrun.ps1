<#
.SYNOPSIS
    Publica o backend do Zé Praga no Google Cloud Run.

.DESCRIPTION
    Faz o caminho inteiro: habilita as APIs, manda o código pro Cloud Build,
    publica o serviço com as variáveis de ambiente e, no primeiro deploy,
    volta pra preencher PUBLIC_API_URL com a URL que o Cloud Run acabou de
    gerar (o link do e-mail de confirmação aponta pra ela — sem isso a
    verificação de conta nasce quebrada).

    Idempotente: rodar de novo atualiza o serviço no lugar.

.PARAMETER ProjectId
    ID do projeto no GCP. Se omitido, usa o projeto ativo do gcloud.

.PARAMETER Region
    Região do Cloud Run. Default us-east1 — mesma região do Supabase
    (us-east-1), o que encurta o ida-e-volta com o banco, que é o que domina
    a latência.

.EXAMPLE
    .\cloudrun.ps1 -ProjectId meu-projeto-123
    .\cloudrun.ps1 -ProjectId meu-projeto-123 -SkipApiEnable

.NOTES
    Pré-requisitos: gcloud instalado e `gcloud auth login` feito, com uma
    conta de faturamento vinculada ao projeto (o Cloud Run exige, mesmo
    dentro do free tier).

    Os segredos vêm de cloudrun.env, nesta mesma pasta, ignorado pelo git.
#>

[CmdletBinding()]
param(
    [string]$ProjectId,
    [string]$Region = "us-east1",
    [string]$ServiceName = "ze-praga-api",
    [switch]$SkipApiEnable
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

# Sobe pra raiz do backend — o build precisa do contexto inteiro (models/, app/).
$backendRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Push-Location $backendRoot

try {
    # ── Pré-checagens ─────────────────────────────────────────────────────────

    $gcloud = Resolve-Gcloud
    if (-not $gcloud) {
        Write-Host "gcloud não encontrado." -ForegroundColor Red
        Write-Host "Instale com:  winget install Google.CloudSDK" -ForegroundColor Yellow
        Write-Host "Depois:       gcloud auth login" -ForegroundColor Yellow
        exit 1
    }

    $envPath = Join-Path $PSScriptRoot 'cloudrun.env'
    if (-not (Test-Path $envPath)) {
        Write-Host "Falta o arquivo de variáveis: $envPath" -ForegroundColor Red
        exit 1
    }

    if (-not $ProjectId) {
        $ProjectId = (& $gcloud config get-value project 2>$null)
        if ([string]::IsNullOrWhiteSpace($ProjectId) -or $ProjectId -eq '(unset)') {
            Write-Host "Nenhum projeto ativo. Passe -ProjectId ou rode:" -ForegroundColor Red
            Write-Host "  gcloud config set project SEU_PROJETO" -ForegroundColor Yellow
            exit 1
        }
    }

    Write-Host ""
    Write-Host "Projeto : $ProjectId" -ForegroundColor Cyan
    Write-Host "Região  : $Region"    -ForegroundColor Cyan
    Write-Host "Serviço : $ServiceName" -ForegroundColor Cyan
    Write-Host ""

    # ── Lê cloudrun.env ───────────────────────────────────────────────────────

    $vars = [ordered]@{}
    foreach ($line in Get-Content $envPath) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $split = $trimmed.IndexOf('=')
        if ($split -lt 1) { continue }
        $key = $trimmed.Substring(0, $split).Trim()
        $value = $trimmed.Substring($split + 1).Trim()
        if ($value) { $vars[$key] = $value }
    }
    Write-Host "  $($vars.Count) variáveis lidas de cloudrun.env" -ForegroundColor DarkGray

    # ── APIs ──────────────────────────────────────────────────────────────────

    if (-not $SkipApiEnable) {
        Write-Host "  Habilitando APIs (demora na primeira vez)..." -ForegroundColor Cyan
        & $gcloud services enable run.googleapis.com cloudbuild.googleapis.com `
            artifactregistry.googleapis.com --project $ProjectId
        if ($LASTEXITCODE -ne 0) { throw "Falha ao habilitar as APIs." }
    }

    # ── Deploy ────────────────────────────────────────────────────────────────

    # --set-env-vars usa "^|^" como delimitador porque a DATABASE_URL contém
    # vírgulas em potencial e o default do gcloud é vírgula.
    $envArg = "^|^" + (($vars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "|")

    Write-Host "  Buildando e publicando (a primeira vez sobe ~500 MB de ONNX)..." -ForegroundColor Cyan

    & $gcloud run deploy $ServiceName `
        --source . `
        --project $ProjectId `
        --region $Region `
        --platform managed `
        --allow-unauthenticated `
        --memory 4Gi `
        --cpu 2 `
        --concurrency 4 `
        --min-instances 0 `
        --max-instances 2 `
        --timeout 3600 `
        --set-env-vars $envArg

    if ($LASTEXITCODE -ne 0) { throw "Deploy falhou." }

    # ── Fecha o círculo da URL ────────────────────────────────────────────────

    $url = (& $gcloud run services describe $ServiceName --project $ProjectId `
            --region $Region --format 'value(status.url)').Trim()

    Write-Host ""
    Write-Host "  Serviço no ar: $url" -ForegroundColor Green

    if ($vars['PUBLIC_API_URL'] -ne $url) {
        # Primeiro deploy (ou URL mudou): grava no cloudrun.env e reaplica só
        # essa variável. Sem ela o link do e-mail de confirmação aponta pro
        # lugar errado e ninguém consegue ativar a conta.
        Write-Host "  Fixando PUBLIC_API_URL=$url ..." -ForegroundColor Cyan

        $content = Get-Content $envPath
        $content = $content -replace '^PUBLIC_API_URL=.*$', "PUBLIC_API_URL=$url"
        Set-Content -Path $envPath -Value $content -Encoding UTF8

        & $gcloud run services update $ServiceName --project $ProjectId --region $Region `
            --update-env-vars "PUBLIC_API_URL=$url"
        if ($LASTEXITCODE -ne 0) { throw "Falha ao aplicar PUBLIC_API_URL." }
    }

    # ── Saúde ─────────────────────────────────────────────────────────────────

    Write-Host "  Checando /api/v1/health ..." -ForegroundColor Cyan
    try {
        $health = Invoke-RestMethod -Uri "$url/api/v1/health" -TimeoutSec 90
        Write-Host "  OK — $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
    } catch {
        Write-Host "  Health falhou: $_" -ForegroundColor Yellow
        Write-Host "  Veja os logs:  gcloud run services logs read $ServiceName --region $Region --limit 50" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Agora, na Vercel, aponte REACT_APP_API_URL para:" -ForegroundColor White
    Write-Host "  $url" -ForegroundColor Green
    Write-Host ""
}
finally {
    Pop-Location
}
