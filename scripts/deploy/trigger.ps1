<#
.SYNOPSIS
    Liga o deploy automático: push na main -> build -> Cloud Run (TCC-093).

.DESCRIPTION
    Termina a configuração do trigger do Cloud Build. Tudo que dava para fazer
    por API já está feito — APIs habilitadas, permissões concedidas, conexão
    `ze-praga-github` criada e `cloudbuild.yaml` no repositório.

    Falta um passo que só você pode dar: autorizar o Cloud Build no GitHub. É
    um fluxo OAuth no navegador, sem equivalente por linha de comando.

    PASSO MANUAL (uma vez):

      1. Abra o link que o comando abaixo imprime.
      2. Autorize o Google Cloud Build na sua conta do GitHub.
      3. Instale o app no repositório tcc-ze-praga-backend.

         gcloud builds connections describe ze-praga-github --region us-east1 `
           --project ze-praga-tcc --format="value(installationState.actionUri)"

    Depois disso, rode este script. Ele confere se a autorização saiu, liga o
    repositório e cria o trigger.

.PARAMETER Branch
    Branch que dispara o deploy. Default: main.

.EXAMPLE
    .\trigger.ps1
    .\trigger.ps1 -Branch main

.NOTES
    O trigger NÃO reabre o acesso público. Se o serviço estiver desligado pelo
    zepraga.ps1, ele continua respondendo 403 depois do deploy — a nova revisão
    sobe, mas a porta segue fechada. Ligar é decisão sua, sempre.
#>

[CmdletBinding()]
param(
    [string]$Branch = "main",
    [string]$ProjectId = "ze-praga-tcc",
    [string]$Region = "us-east1",
    [string]$Connection = "ze-praga-github",
    [string]$RepoOwner = "FelipeCarillo",
    [string]$RepoName = "tcc-ze-praga-backend",
    [string]$TriggerName = "ze-praga-api-main"
)

$ErrorActionPreference = 'Stop'

function Resolve-Gcloud {
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

$gcloud = Resolve-Gcloud
if (-not $gcloud) {
    Write-Host "gcloud não encontrado. winget install Google.CloudSDK" -ForegroundColor Red
    exit 1
}

$comum = @("--project", $ProjectId, "--region", $Region)

# ── 1. A autorização do GitHub saiu? ──────────────────────────────────────────

Write-Host ""
Write-Host "Conferindo a conexão '$Connection'..." -ForegroundColor Cyan

$estado = (& $gcloud builds connections describe $Connection @comum `
        --format="value(installationState.stage)" 2>$null)

if ($estado -and $estado.Trim() -ne "COMPLETE") {
    $link = (& $gcloud builds connections describe $Connection @comum `
            --format="value(installationState.actionUri)" 2>$null)
    Write-Host ""
    Write-Host "  A conexão ainda está em: $($estado.Trim())" -ForegroundColor Yellow
    Write-Host "  Falta autorizar o Cloud Build no GitHub. Abra:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  $link" -ForegroundColor White
    Write-Host ""
    Write-Host "  Depois de autorizar e instalar o app no repositório, rode este script de novo." -ForegroundColor Yellow
    exit 1
}

Write-Host "  Conexão autorizada." -ForegroundColor Green

# ── 2. Liga o repositório ─────────────────────────────────────────────────────

$existe = (& $gcloud builds repositories describe $RepoName `
        --connection $Connection @comum --format="value(name)" 2>$null)

if (-not $existe) {
    Write-Host "  Ligando o repositório $RepoOwner/$RepoName ..." -ForegroundColor Cyan
    & $gcloud builds repositories create $RepoName `
        --connection $Connection @comum `
        --remote-uri "https://github.com/$RepoOwner/$RepoName.git" --quiet
    if ($LASTEXITCODE -ne 0) { throw "Falha ao ligar o repositório." }
} else {
    Write-Host "  Repositório já ligado." -ForegroundColor DarkGray
}

# ── 3. Cria (ou recria) o trigger ─────────────────────────────────────────────

$repoPath = "projects/$ProjectId/locations/$Region/connections/$Connection/repositories/$RepoName"

$triggerExiste = (& $gcloud builds triggers describe $TriggerName @comum `
        --format="value(name)" 2>$null)

if ($triggerExiste) {
    Write-Host "  Trigger já existe — recriando para aplicar a configuração atual." -ForegroundColor DarkGray
    & $gcloud builds triggers delete $TriggerName @comum --quiet 2>&1 | Out-Null
}

Write-Host "  Criando o trigger para a branch '$Branch' ..." -ForegroundColor Cyan
& $gcloud builds triggers create github $TriggerName `
    @comum `
    --repository=$repoPath `
    --branch-pattern="^$Branch$" `
    --build-config="cloudbuild.yaml" `
    --quiet

if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o trigger." }

Write-Host ""
Write-Host "Pronto. Todo push em '$Branch' agora rebuilda e publica." -ForegroundColor Green
Write-Host ""
Write-Host "Lembre-se: o deploy NÃO reabre o acesso público. Para ligar de verdade:" -ForegroundColor White
Write-Host "  .\zepraga.ps1 -Acao ligar" -ForegroundColor Green
Write-Host ""
Write-Host "Para disparar um build agora, sem esperar um push:" -ForegroundColor White
Write-Host "  gcloud builds triggers run $TriggerName --branch=$Branch --region=$Region --project=$ProjectId" -ForegroundColor Green
Write-Host ""
