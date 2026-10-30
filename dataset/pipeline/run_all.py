"""Run the whole combined-dataset build, in order, resumably.

    python run_all.py              # every stage that has no output yet
    python run_all.py --from k05   # from a stage onwards
    python run_all.py --only k09   # one stage
    python run_all.py --force      # ignore existing outputs

Stages that need the GPU are marked; on a shared card they are the slow ones.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from kcommon import MANIFESTS, REPORTS, utf8_stdout

utf8_stdout()

HERE = Path(__file__).resolve().parent

STAGES = [
    ("k01", "k01_inventory.py", "01_inventory.jsonl", False),
    ("k02", "k02_audio.py", "02_audio.jsonl", False),
    ("k03", "k03_features.py", "03_features.jsonl", False),
    ("k04", "k04_speaker.py", "04_speaker.jsonl", True),
    ("k05", "k05_locate.py", "05_locator.jsonl", True),
    ("k06", "k06_align.py", "06_alignment.jsonl", True),
    ("k07", "k07_nvv.py", "07_nvv.jsonl", True),
    ("k08", "k08_calibrate.py", "08_nvv_policy.json", False),
    ("k09", "k09_decide.py", "09_master.jsonl", False),
    ("k10", "k10_split.py", "10_split.jsonl", False),
    ("k11", "k11_export.py", "11_published.jsonl", False),
    ("k12", "k12_verify.py", None, False),
    ("k13", "k13_readme.py", None, False),
    ("k15", "k15_tagging_doc.py", None, False),
]


def free_vram_gib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        total, used = (float(x) for x in out.split(","))
        return (total - used) / 1024.0
    except Exception:
        return 99.0


def wait_for_vram(need_gib: float, timeout_s: float) -> bool:
    """Block until the card has room. This box shares its GPU with other work."""
    t0 = time.time()
    warned = False
    while time.time() - t0 < timeout_s:
        free = free_vram_gib()
        if free >= need_gib:
            if warned:
                print(f"[gpu] {free:.1f} GiB free, continuing", flush=True)
            return True
        if not warned:
            print(f"[gpu] waiting for {need_gib:.1f} GiB (only {free:.1f} free) ...",
                  flush=True)
            warned = True
        time.sleep(30)
    print(f"[gpu] gave up waiting after {timeout_s / 60:.0f} min; running anyway",
          flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--need-vram", type=float, default=0.0,
                    help="GiB of free VRAM to wait for before each GPU stage")
    ap.add_argument("--vram-timeout-min", type=float, default=240.0)
    a = ap.parse_args()

    started = a.start is None
    for name, script, output, gpu in STAGES:
        if a.only and name != a.only:
            continue
        if not a.only:
            if not started and name != a.start:
                continue
            started = True
        out = MANIFESTS / output if output else None
        if out is not None and out.exists() and not a.force and not a.only:
            print(f"[skip] {name} ({output} exists)")
            continue
        print(f"\n=== {name} {script}{'  [gpu]' if gpu else ''} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", str(HERE / script)], cwd=HERE)
        print(f"=== {name} finished rc={rc} in {(time.time() - t0) / 60:.1f} min ===",
              flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
