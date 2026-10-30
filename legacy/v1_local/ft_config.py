"""Shared configuration for the Kurisu Qwen3-TTS fine-tune.

Everything the fine-tuning stages agree on lives here: where the dataset is, which
base model, and the two settings that decide whether a 16 GB card survives a long
run - the allocator's fragmentation policy and the VRAM cap.

Paths and JSONL helpers are re-exported from ``pipeline/common.py`` so the dataset
build and the fine-tune cannot drift apart on where anything lives.

Nothing here imports torch at module scope. ``vram_guard()`` does, and must be called
before the first CUDA allocation in a process.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent
PROJECT = FINETUNE.parent                       # kurisu_qwen3tts/
sys.path.insert(0, str(PROJECT / "pipeline"))   # one definition of paths/taxonomy

from common import (AUDIO, QWEN, REF_OUT, REPORTS, TARGET_SR, WAVS,  # noqa: E402
                    QWEN_TAGS, read_jsonl, utf8_stdout, write_jsonl)

# Expandable segments is the single most useful allocator setting for a long run on a
# small card, but the CUDA allocator does not implement it on Windows - asking for it
# there only prints a warning. Where it does work it must be set before CUDA is
# touched; where it does not, length-bucketed batches carry the same load by keeping
# the block sizes the allocator sees roughly constant.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# HF weight downloads on Windows are slow without the Rust downloader.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# ---------------------------------------------------------------- models

MODELS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
TOKENIZER_12HZ = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

#: The codec runs at 12.5 Hz; used only to sanity-check prepared code lengths.
CODEC_FPS = 12.5

# ---------------------------------------------------------------- paths

RUNS = FINETUNE / "runs"
EVAL = FINETUNE / "eval"
SPLITS = {
    "train": QWEN / "train_raw.jsonl",
    "val": QWEN / "val_raw.jsonl",
    "test": QWEN / "test_raw.jsonl",
}


def coded_path(split: str) -> Path:
    """Where ``ft01_prepare.py`` writes the JSONL that carries ``audio_codes``."""
    return QWEN / f"{split}_with_codes.jsonl"


def resolve_model(name: str) -> str:
    """Accept ``1.7B`` / ``0.6B`` shorthand, a hub id, or a local checkpoint dir."""
    if name in MODELS:
        return MODELS[name]
    return name


# ---------------------------------------------------------------- gpu

def vram_guard(fraction: float = 0.92, verbose: bool = True) -> str:
    """Cap this process's share of the card and return the device string.

    A hard cap turns "the driver hung and took the desktop with it" into an ordinary
    Python OOM that the caller can catch and halve the batch for. 0.92 leaves ~1.3 GiB
    on a 16 GB card for the display and whatever else is running.
    """
    import torch

    if not torch.cuda.is_available():
        print("[gpu] no CUDA device - running on CPU will be unusably slow")
        return "cpu"
    torch.cuda.init()
    if fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 2 ** 30
    if verbose:
        free, _ = torch.cuda.mem_get_info()
        print(f"[gpu] {props.name}  sm_{props.major}{props.minor}  {total:.1f} GiB total, "
              f"{free / 2 ** 30:.1f} GiB free now, cap {fraction:.0%} = {total * fraction:.1f} GiB")
    return "cuda"


def vram_used() -> tuple[float, float]:
    """(currently allocated, peak allocated) in GiB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 2 ** 30,
            torch.cuda.max_memory_allocated() / 2 ** 30)


def free_vram_gib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.mem_get_info()[0] / 2 ** 30


# ---------------------------------------------------------------- misc

def write_report(name: str, obj: dict) -> Path:
    """Reports land next to the dataset build's own reports, with an ``ft_`` prefix."""
    path = REPORTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {path}")
    return path


def free_disk_gib(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / 2 ** 30


def fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


__all__ = [
    "AUDIO", "QWEN", "REF_OUT", "REPORTS", "TARGET_SR", "WAVS", "QWEN_TAGS",
    "read_jsonl", "write_jsonl", "utf8_stdout",
    "FINETUNE", "PROJECT", "RUNS", "EVAL", "SPLITS", "MODELS", "TOKENIZER_12HZ",
    "CODEC_FPS", "coded_path", "resolve_model", "vram_guard", "vram_used",
    "free_vram_gib", "write_report", "free_disk_gib", "fmt_hms",
]
