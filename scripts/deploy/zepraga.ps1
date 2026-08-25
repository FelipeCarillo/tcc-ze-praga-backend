<#
.SYNOPSIS
    Interruptor do Zé Praga em produção — desliga e liga a aplicação inteira.

.DESCRIPTION
    Controla os dois serviços que custam dinheiro e expõem superfície de
    ataque: o Space do Hugging Face (backend + modelos ONNX) e o projeto
    Supabase (banco + storage). O frontend na Vercel é estático — com o
    backend parado ele não faz nada além de mostrar a interface, então não
    precisa ser desligado; use -IncluirVercel se quiser derrubar ele também.

    Desligado, o projeto não responde, não gasta cota de LLM e não aceita
    cadastro. Religar leva ~2 min (o Supabase demora mais que o Space).

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

    Os endpoints das APIs de gerenciamento podem mudar sem aviso. Se algum
    passo falhar, o script mostra o link do painel para você fazer no braço —
    o botão manual sempre funciona.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('desligar', 'ligar', 'status')]
    [string]$Acao,

    [switch]$IncluirVercel
)

$ErrorActionPreference = 'Stop'

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

$hfToken     = Get-Campo 'hf_token'
$hfSpace     = Get-Campo 'hf_space'          # ex: "felipecarillo/ze-praga-api"
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

# ── Hugging Face Space (backend) ──────────────────────────────────────────────

function Acao-Space([string]$op) {
    if (-not $hfToken -or -not $hfSpace) {
        Escreve-Pulo 'Hugging Face: sem credencial em deploy.local.json — pulando.'
        return
    }
    $headers = @{ Authorization = "Bearer $hfToken" }
    $base = "https://huggingface.co/api/spaces/$hfSpace"

    switch ($op) {
        'status' {
            $r = Invoke-Api -Metodo GET -Uri $base -Cabecalhos $headers
            if ($r.ok) {
                $estagio = $r.corpo.runtime.stage
                Write-Host "  Space   : $hfSpace -> $estagio"
            } else {
                Escreve-Erro "Space: $($r.corpo)"
            }
        }
        'desligar' {
            Escreve-Passo "Pausando o Space $hfSpace ..."
            $r = Invoke-Api -Metodo POST -Uri "$base/pause" -Cabecalhos $headers
            if ($r.ok) { Escreve-Ok 'Space pausado (backend fora do ar).' }
            else {
                Escreve-Erro $r.corpo
                Write-Host "       No braço: https://huggingface.co/spaces/$hfSpace/settings -> Pause Space" -ForegroundColor Yellow
            }
        }
        'ligar' {
            Escreve-Passo "Religando o Space $hfSpace ..."
            $r = Invoke-Api -Metodo POST -Uri "$base/restart" -Cabecalhos $headers
            if ($r.ok) { Escreve-Ok 'Space subindo (leva ~1 min pra ficar de pé).' }
            else {
                Escreve-Erro $r.corpo
                Write-Host "       No braço: https://huggingface.co/spaces/$hfSpace/settings -> Restart Space" -ForegroundColor Yellow
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
        Acao-Space 'status'; Acao-Supabase 'status'; Acao-Vercel 'status'
    }
    'desligar' {
        Write-Host 'Zé Praga — DESLIGANDO' -ForegroundColor Yellow
        # Backend primeiro: com a API fora, ninguém escreve no banco enquanto
        # ele é pausado. A ordem inversa deixaria uma janela de requisições
        # batendo num banco sumindo.
        Acao-Space 'desligar'; Acao-Supabase 'desligar'; Acao-Vercel 'desligar'
        Write-Host ''
        Write-Host 'Projeto desligado. Nada responde, nada gasta.' -ForegroundColor Green
    }
    'ligar' {
        Write-Host 'Zé Praga — LIGANDO' -ForegroundColor Yellow
        # Banco primeiro: a API roda migrations no boot e precisa do Postgres
        # de pé, senão sobe quebrada.
        Acao-Supabase 'ligar'; Acao-Space 'ligar'; Acao-Vercel 'ligar'
        Write-Host ''
        Write-Host 'Subindo. Espere ~2 min e confira com: .\zepraga.ps1 -Acao status' -ForegroundColor Green
    }
}
Write-Host ''
