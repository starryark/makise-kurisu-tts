"""Stage 14 - push the staged repository to the Hugging Face Hub.

Refuses to run unless stage 12 verified the exact directory being uploaded, and
prints what will change before doing anything.  ``--dry-run`` stops there.

``delete_patterns`` is what makes this an update rather than a merge: the clip ids
are new, so without it the previous 10,469 files would survive alongside the new
ones and the dataset would contain two builds at once.

    python k14_push.py --dry-run
    python k14_push.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

from kcommon import HF_OUT, LEGACY, REPORTS, utf8_stdout

utf8_stdout()

REPO_ID = "starrydark/Kurisu_Voice"
REPO_TYPE = "dataset"
TOKEN_FILE = LEGACY / "kurisu_qwen3tts" / "hf token.txt"


def token() -> str:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    raise SystemExit("no HF token: set HF_TOKEN or provide the token file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--message", default=None)
    a = ap.parse_args()

    ver = REPORTS / "12_verify.json"
    if not ver.exists():
        raise SystemExit("run k12_verify.py first")
    v = json.loads(ver.read_text(encoding="utf-8"))
    if v["failures"]:
        raise SystemExit(f"verification reported {len(v['failures'])} failure(s); "
                         "refusing to push")

    local = sorted(p for p in HF_OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in local)
    print(f"local: {len(local):,} files, {size / 2**30:.2f} GiB")

    from huggingface_hub import HfApi
    api = HfApi(token=token())
    who = api.whoami()
    print(f"account: {who.get('name')}")

    try:
        info = api.repo_info(a.repo, repo_type=REPO_TYPE)
        remote = [s.rfilename for s in info.siblings]
        print(f"remote: {len(remote):,} files currently in {a.repo}")
        by = collections.Counter(f.split("/")[1] if f.startswith("data/") else "(root)"
                                 for f in remote)
        print(f"        {dict(by)}")
    except Exception as e:
        print(f"remote: could not read {a.repo} ({type(e).__name__}); it may be new")
        remote = []

    exp = json.loads((REPORTS / "11_export.json").read_text(encoding="utf-8"))
    msg = a.message or (
        f"Multilingual rebuild: {exp['clips']:,} clips / {exp['hours']} h in "
        f"{len(exp['by_language'])} languages, calibrated expressive tags, "
        f"restored per-source dynamics, leakage-safe splits")

    print("\nthis will REPLACE the remote contents (delete_patterns='**')")
    print(f"commit message: {msg}")
    if a.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    api.upload_folder(
        folder_path=str(HF_OUT),
        repo_id=a.repo,
        repo_type=REPO_TYPE,
        delete_patterns=["**"],
        commit_message=msg,
    )
    info = api.repo_info(a.repo, repo_type=REPO_TYPE)
    print(f"\npushed. remote now has {len(info.siblings):,} files")
    print(f"https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
