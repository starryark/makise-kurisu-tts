[CmdletBinding()]
param(
    [int]$Epochs = 6,
    [string]$Name = 'kurisu-lora-r32',
    [ValidateSet('auto', '1', '2', '4', '6', '8', '12', '16', '24')]
    [string]$BatchSize = 'auto',
    [int]$EffectiveBatch = 32,
    [double]$LearningRate = 0.0001
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv-tts\Scripts\python.exe'
& $Python (Join-Path $PSScriptRoot 'ft00_check.py') --require_prepared
if ($LASTEXITCODE -ne 0) { throw 'Training preflight failed' }

$Arguments = @(
    (Join-Path $PSScriptRoot 'ft02_sft.py'),
    '--name', $Name,
    '--epochs', $Epochs,
    '--batch_size', $BatchSize,
    '--effective_batch', $EffectiveBatch,
    '--lr', $LearningRate,
    '--mode', 'lora',
    '--attn', 'sdpa',
    '--lora_r', '32',
    '--lora_alpha', '64',
    '--keep_last', '3',
    '--eval_every', '200',
    '--tensorboard',
    '--resume', 'auto'
)
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
