[CmdletBinding()]
param(
    [int]$Port = 7860,
    [int]$MaxLoaded = 2,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv-tts\Scripts\python.exe'
& $Python (Join-Path $PSScriptRoot 'ft04_playground.py') `
    --port $Port --max_loaded $MaxLoaded $(if ($NoBrowser) { '--no_browser' })
