"""Rebuild the whole dataset.

    python run_all.py                 # every stage
    python run_all.py --from 06       # resume at a stage
    python run_all.py --only 09 10    # just these
    python run_all.py --list

Stage 07 needs the isolated aligner interpreter (qwen-asr pins transformers 4.57.6,
which would break the vocalburst models). Point ALIGNER_PYTHON at it, e.g.

    set ALIGNER_PYTHON=...\\venv_aligner\\Scripts\\python.exe

Every stage caps itself at CUDA_FRACTION (default 0.80) of VRAM so a concurrent
GPU workload on the same card is not starved.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("00", "s00_taxonomy.py", "dump taxonomy to yaml", False),
    ("01", "s01_inventory.py", "select clips, normalise transcripts, hash", False),
    ("02", "s02_audio.py", "24 kHz mono + per-source loudness alignment", False),
    ("03", "s03_classify_isolated.py", "vocalburst on isolated + control", True),
    ("04", "s04_secondary.py", "CED second opinion", True),
    ("05", "s05_calibrate.py", "per-family thresholds from the control set", False),
    ("06", "s06_locate.py", "locate bursts inside spoken lines", True),
    ("07", "s07_align.py", "forced alignment (ALIGNER_PYTHON)", True),
    ("08", "s08_events.py", "name + place in-speech bursts", True),
    ("09", "s09_decide.py", "one decision per clip, master manifest", False),
    ("10", "s10_export.py", "splits, Qwen jsonl, probe set, report", False),
    ("11", "s11_ref_check.py", "verify ref.wav is this voice", True),
    ("12", "make_readme.py", "regenerate README from the reports", False),
    ("13", "verify_dataset.py", "audit the export", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, f, d, gpu in STAGES:
            print(f"  {n}  {'gpu' if gpu else '   '}  {f:26s} {d}")
        return 0

    todo = STAGES
    if args.only:
        todo = [s for s in STAGES if s[0] in set(args.only)]
    elif args.start:
        todo = [s for s in STAGES if s[0] >= args.start]

    aligner = os.environ.get("ALIGNER_PYTHON", "")
    for num, script, desc, uses_gpu in todo:
        interp = aligner if script == "s07_align.py" and aligner else sys.executable
        if script == "s07_align.py" and not aligner:
            print(f"\n### {num} {script} SKIPPED - set ALIGNER_PYTHON to the venv "
                  f"interpreter to run forced alignment")
            continue
        print(f"\n{'=' * 70}\n### {num}  {desc}\n{'=' * 70}", flush=True)
        t = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.setdefault("CUDA_FRACTION", "0.80")
        r = subprocess.run([interp, str(HERE / script)], cwd=HERE, env=env)
        print(f"--- {script} finished in {time.time() - t:.1f}s (exit {r.returncode})")
        if r.returncode != 0 and script not in ("verify_dataset.py", "s11_ref_check.py"):
            print(f"!!! stage {num} failed, stopping")
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
