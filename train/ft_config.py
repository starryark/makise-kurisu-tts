"""Pod-side stand-in for kurisu_qwen3tts/finetune/ft_config.py.

The trainer (train_sft.py, ported from ft02_sft.py) only needs paths, JSONL helpers
and a few GPU utilities; everything lives under KURISU_WORK on the pod.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
RUNS = WORK / "runs"
REPORTS = WORK / "reports"
QWEN = WORK / "qwen"
REF_OUT = WORK / "build" / "ref.wav"
TARGET_SR = 24000
CODEC_FPS = 12.5
MODELS = {"1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}
TOKENIZER_12HZ = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


def utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def coded_path(split: str) -> Path:
    return QWEN / f"{split}_with_codes.jsonl"


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


def vram_guard(fraction: float = 0.95, verbose: bool = True) -> str:
    import torch

    torch.cuda.init()
    if fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    if verbose:
        p = torch.cuda.get_device_properties(0)
        print(f"[gpu] {p.name}  {p.total_memory / 2 ** 30:.1f} GiB, cap {fraction:.0%}")
    return "cuda"


def vram_used() -> tuple[float, float]:
    import torch

    return torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.max_memory_allocated() / 2 ** 30


def free_vram_gib() -> float:
    import torch

    return torch.cuda.mem_get_info()[0] / 2 ** 30


def write_report(name: str, obj: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {path}")
    return path


def free_disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
