param(
    [string]$ManifestPath = "kurisu_qwen3tts\manifests\09_master.jsonl",
    [string]$AudioRoot = "kurisu_qwen3tts\audio\wavs",
    [string]$OutputPath = "hf_dataset_upload"
)

$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path -LiteralPath ".").Path
$manifest = (Resolve-Path -LiteralPath $ManifestPath).Path
$audioRootResolved = (Resolve-Path -LiteralPath $AudioRoot).Path
$outputFull = [System.IO.Path]::GetFullPath((Join-Path $workspace $OutputPath))

if (Test-Path -LiteralPath $outputFull) {
    throw "Output directory already exists: $outputFull"
}

$records = @(
    Get-Content -LiteralPath $manifest -Encoding UTF8 |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.split -in @("train", "val", "test") }
)

if ($records.Count -ne 10469) {
    throw "Expected 10,469 publishable records, found $($records.Count)."
}

$duplicates = @($records | Group-Object id | Where-Object Count -gt 1)
if ($duplicates.Count -gt 0) {
    throw "Duplicate clip IDs found in the publishable manifest."
}

New-Item -ItemType Directory -Path $outputFull | Out-Null

$splitMap = @{
    train = "train"
    val = "validation"
    test = "test"
}

foreach ($split in @("train", "validation", "test")) {
    New-Item -ItemType Directory -Path (Join-Path $outputFull "data\$split") -Force | Out-Null
}

$writers = @{}
try {
    foreach ($split in @("train", "validation", "test")) {
        $metadataPath = Join-Path $outputFull "data\$split\metadata.jsonl"
        $writers[$split] = [System.IO.StreamWriter]::new($metadataPath, $false, [System.Text.UTF8Encoding]::new($false))
    }

    $index = 0
    foreach ($record in $records) {
        $split = $splitMap[[string]$record.split]
        $fileName = "$($record.id).wav"
        $sourceAudio = Join-Path $audioRootResolved $fileName
        $destinationAudio = Join-Path $outputFull "data\$split\$fileName"

        if (-not (Test-Path -LiteralPath $sourceAudio -PathType Leaf)) {
            throw "Missing source audio: $sourceAudio"
        }

        New-Item -ItemType HardLink -Path $destinationAudio -Target $sourceAudio | Out-Null

        $sourceParts = ([string]$record.source_group).Split(":", 2)
        $tags = @(
            [regex]::Matches([string]$record.qwen_text, "\[(gasp|sigh)\]") |
                ForEach-Object { $_.Groups[1].Value } |
                Select-Object -Unique
        )

        $metadata = [ordered]@{
            file_name = $fileName
            transcription = [string]$record.qwen_text
            text = [string]$record.text
            speaker_id = "kurisu"
            speaker_name = "Makise Kurisu"
            character_variant = [string]$record.variant
            source_medium = $sourceParts[0]
            source_series = $sourceParts[1]
            source_group = [string]$record.source_group
            source_quality_tier = [string]$record.tier_src
            duration_seconds = [math]::Round([double]$record.dur, 3)
            expressive_tags = $tags
            from_quarantine = [bool]$record.from_quarantine
            sha256 = [string]$record.sha256
        }

        $writers[$split].WriteLine(($metadata | ConvertTo-Json -Compress -Depth 5))
        $index++
        if (($index % 1000) -eq 0) {
            Write-Host "Staged $index / $($records.Count) clips"
        }
    }
}
finally {
    foreach ($writer in $writers.Values) {
        $writer.Dispose()
    }
}

Write-Host "Staged $($records.Count) labeled audio clips in $outputFull"
