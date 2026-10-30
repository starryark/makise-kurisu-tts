[CmdletBinding()]
param(
    [ValidateSet('all', 'latest')]
    [string]$Checkpoint = 'all',
    [int]$Limit = 12,
    [string]$Run = (Join-Path $PSScriptRoot 'runs\kurisu-lora-r32'),
    [switch]$Deterministic
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv-tts\Scripts\python.exe'
$Arguments = @(
    (Join-Path $PSScriptRoot 'ft03_eval.py'),
    '--run', $Run,
    '--ckpt', $Checkpoint,
    '--limit', $Limit
)
if ($Deterministic) { $Arguments += '--deterministic' }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Evaluation exited with code $LASTEXITCODE" }

