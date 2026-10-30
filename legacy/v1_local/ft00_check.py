"""Preflight check for the local Kurisu Qwen3-TTS fine-tuning environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

import ft_config as C

C.utf8_stdout()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as src:
        return sum(1 for line in src if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Check CUDA, packages, disk and dataset readiness.")
    ap.add_argument("--require_prepared", action="store_true",
                    help="fail unless all three coded JSONLs are essentially complete")
    args = ap.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    print(f"[python] {sys.version.split()[0]}  {sys.executable}")
    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or newer is required")

    try:
        import torch
    except ImportError:
        failures.append("PyTorch is not installed")
        torch = None

    if torch is not None:
        print(f"[torch] {torch.__version__}  CUDA runtime {torch.version.cuda}")
        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is false")
        else:
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            arch = f"sm_{cc[0]}{cc[1]}"
            built = torch.cuda.get_arch_list()
            print(f"[gpu] {name}  {arch}  {total:.1f} GiB")
            if arch not in built:
                failures.append(
                    f"this torch build lacks {arch} kernels (built for: {', '.join(built)})")
            if total < 15:
                warnings.append(f"only {total:.1f} GiB VRAM; use the 0.6B model or lower limits")
            if not torch.cuda.is_bf16_supported():
                failures.append("GPU/PyTorch does not report bfloat16 support")

    for package in ("qwen-tts", "transformers", "accelerate", "soundfile",
                    "safetensors", "tensorboard"):
        try:
            print(f"[package] {package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package: {package}")

    if not C.REF_OUT.exists():
        failures.append(f"missing reference audio: {C.REF_OUT}")
    else:
        info = sf.info(str(C.REF_OUT))
        print(f"[reference] {info.duration:.2f}s  {info.samplerate} Hz  {info.channels}ch")
        if info.samplerate != C.TARGET_SR or info.channels != 1:
            failures.append("reference audio must be 24 kHz mono")

    print("[data]")
    readiness: dict[str, dict[str, int | bool]] = {}
    for split, raw_path in C.SPLITS.items():
        coded_path = C.coded_path(split)
        raw = line_count(raw_path)
        coded = line_count(coded_path)
        # ft01 deliberately rejects sub-300 ms clips, so exact equality is not expected.
        complete = raw > 0 and coded >= int(raw * 0.995)
        readiness[split] = {"raw": raw, "coded": coded, "complete": complete}
        print(f"  {split:>5}: raw {raw:>5}  coded {coded:>5}  "
              f"{'ready' if complete else 'NOT READY'}")
        if raw == 0:
            failures.append(f"missing or empty raw split: {raw_path}")
        if args.require_prepared and not complete:
            failures.append(f"coded {split} split is incomplete; run ft01_prepare.py")

    free = shutil.disk_usage(C.FINETUNE).free / 2**30
    print(f"[disk] {free:.1f} GiB free")
    if free < 15:
        warnings.append("less than 15 GiB free; checkpoints may not fit")

    for warning in warnings:
        print(f"[warn] {warning}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print("[ok] environment is ready" + (" for training" if args.require_prepared else ""))


if __name__ == "__main__":
    main()
