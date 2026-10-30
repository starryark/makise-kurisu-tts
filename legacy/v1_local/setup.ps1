[CmdletBinding()]
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$FineTuneDir = $PSScriptRoot
$VenvDir = Join-Path $FineTuneDir '.venv-tts'
$Python = Join-Path $VenvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host '[setup] creating .venv-tts (sharing the host Blackwell PyTorch build)'
    py -3.13 -m venv --system-site-packages $VenvDir
}

if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    & $Python -m pip install --requirement (Join-Path $FineTuneDir 'requirements.txt')
}

& $Python (Join-Path $FineTuneDir 'ft00_check.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Preflight failed. For an RTX 5060 Ti, PyTorch must list sm_120 in torch.cuda.get_arch_list().'
}

Write-Host '[setup] ready; run .\prepare.ps1 next'

