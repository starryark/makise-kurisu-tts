[CmdletBinding()]
param(
    [int]$Batch = 16,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv-tts\Scripts\python.exe'
$Arguments = @(
    (Join-Path $PSScriptRoot 'ft01_prepare.py'),
    '--splits', 'train', 'val', 'test',
    '--batch', $Batch
)
if ($Force) { $Arguments += '--force' }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Data preparation exited with code $LASTEXITCODE" }

