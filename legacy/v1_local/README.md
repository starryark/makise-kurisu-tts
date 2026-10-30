# Local Qwen3-TTS fine-tuning (RTX 5060 Ti 16 GB)

> **Archived: v1.** This is the single-speaker LoRA workflow behind model revision `v1-epoch10`, run on a 16 GB RTX 5060 Ti. In the original layout this folder was `kurisu_qwen3tts/finetune/`, next to the v1 training set built by [`../v1_dataset/`](../v1_dataset/). Its trainer `ft02_sft.py` already has the four upstream fixes. The released v3 model was trained with [`../../train/`](../../train/).

This directory is a complete Windows/PowerShell workflow for single-speaker Makise
Kurisu fine-tuning. It uses the official Qwen3-TTS data/checkpoint formats, with a
memory-safe LoRA trainer and `sdpa` attention because FlashAttention 2 is not
available for native Windows Blackwell builds.

## Current machine profile

- NVIDIA GeForce RTX 5060 Ti, 16 GB, compute capability `sm_120`
- Python 3.13
- PyTorch `2.13.0+cu132` (contains `sm_120` kernels)
- `qwen-tts==0.1.1`, `transformers==4.57.3`
- 1.7B Base model, bf16 frozen weights, fp32 rank-32 LoRA
- full-data worst-case probe: micro-batch 12, about 13.1 GiB peak with automatic
  gradient checkpointing (without it, only micro-batch 1 fits the longest clip)

The Python `sox` package may print a warning that the SoX executable is absent.
Neither codec extraction, training, nor Qwen's neural codec decode uses that binary,
so it is harmless for this workflow.

## Run it

Open PowerShell in this directory:

```powershell
Set-Location '<your checkout>\kurisu_qwen3tts\finetune'

# Verify the already-installed environment (omit -SkipInstall to refresh Python deps).
.\setup.ps1 -SkipInstall

# Resumable codec extraction for train/val/test. Safe to rerun.
.\prepare.ps1

# Six total epochs. Existing state resumes automatically after a reboot/interruption.
.\train.ps1

# Generate the same balanced 12-prompt listening suite from every retained epoch.
.\evaluate.ps1
```

`train.ps1 -Epochs 8` means “continue until eight total epochs,” not “add eight.” To
start a separate experiment, use a new name:

```powershell
.\train.ps1 -Name kurisu-lora-r16-test -Epochs 3
```

The launcher deliberately refuses to reuse an existing run without resume state, and
the trainer refuses fewer than 100 training rows unless
`--allow_small_dataset` is explicitly supplied. These checks prevent a validation-only
smoke file from being mistaken for the real training set.

## Outputs

```text
runs/kurisu-lora-r32/
  state.pt                         resumable optimizer/adapter state
  adapter-epoch-N.safetensors      small LoRA-only artifact
  checkpoint-epoch-N/              merged, directly loadable Qwen model
  train_log.jsonl
  tb/                              TensorBoard logs

eval/kurisu-lora-r32/
  checkpoint-epoch-N/*.wav
  checkpoint-epoch-N/manifest.jsonl
```

Each merged checkpoint loads directly with `Qwen3TTSModel.from_pretrained` and calls
the voice as `speaker="kurisu"` through `generate_custom_voice`.

## Manual commands

The launchers are intentionally thin; all controls remain available:

```powershell
.\.venv-tts\Scripts\python.exe .\ft00_check.py --require_prepared

.\.venv-tts\Scripts\python.exe .\ft02_sft.py `
  --mode lora --init_model_path 1.7B --attn sdpa `
  --lora_r 32 --lora_alpha 64 `
  --batch_size auto --effective_batch 32 `
  --epochs 6 --lr 1e-4 --resume auto --tensorboard

# Every one of the 90 probes, with deterministic greedy decoding:
.\.venv-tts\Scripts\python.exe .\ft03_eval.py `
  --run .\runs\kurisu-lora-r32 --ckpt all --limit 0 --deterministic
```

Do not select the last epoch solely because its loss is lowest. Compare identical
prompt numbers across checkpoint folders for voice identity, onset stability,
speaking-rate drift, naturalness, and whether tagged prompts differ from their
untagged controls.

## Interactive epoch playground (FT-4)

Listen to any checkpoint with your own text through a browser page:

```powershell
.\playground.ps1                    # http://127.0.0.1:7860 (opens automatically)
```

Every complete `runs/<run>/checkpoint-epoch-*` with the `kurisu` speaker is
listed (smoke runs excluded). Two checkpoints stay in VRAM at once (LRU;
`--MaxLoaded` to change); others load on demand in ~7-40 s. Language, max
frames, an optional seed, and greedy decoding let two epochs hear the same
prompt on equal terms. Every generation is appended to
`eval/playground_log.jsonl`. The server binds 127.0.0.1 only.

