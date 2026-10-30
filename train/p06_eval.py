"""Generate the probe suite from each checkpoint and score it.

  python p06_eval.py --name <suite> --ckpt real --ckpt clone:<base> --ckpt <dir> ...

  real            the held-out test clips themselves (ceiling for SIM, floor for CER)
  clone:<model>   the Base model cloning the voice in ICL mode from build/ref.wav
                  (what you get with no fine-tune at all)
  <dir or hub id> a CustomVoice checkpoint, called as speaker "kurisu"

Decoding is the upstream default (sampling, T=0.9, top-k 50, rep. penalty 1.05) with
a fixed seed per batch, so two checkpoints hear the same randomness. Greedy decoding
is NOT used: codec LMs loop under greedy search, which is what made the previous
evaluation's clips all hit the length cap.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import metrics

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
CODE = Path(__file__).resolve().parent
N_TEST = 24
# Overridable from the CLI for confirmation suites: a disjoint slice of held-out
# lines and different sampling seeds, scored by the same rule.
OPTS = {"n_test": N_TEST, "test_offset": 0, "seed_base": 1000, "novel": True}
BATCH = 12


def tag_of(ckpt: str) -> str:
    """Directory name under eval/<suite>/ for a --ckpt value."""
    return "real" if ckpt == "real" else ckpt.rstrip("/").replace(":", "_").replace("/", "__")[-80:]


def heldout_pool() -> list[dict]:
    """Untagged 2.5-9 s test-split lines in a fixed shuffled order; suites take slices."""
    test = [json.loads(l) for l in (WORK / "qwen" / "test_with_codes.jsonl").read_text(encoding="utf-8").splitlines() if l]
    test = [t for t in test if "[" not in t["text"] and 2.5 <= len(t["audio_codes"]) / 12.5 <= 9.0]
    random.Random(7).shuffle(test)
    return test


def probe_rows() -> list[dict]:
    rows = [json.loads(l) for l in (CODE / "probes.jsonl").read_text(encoding="utf-8").splitlines() if l]
    test = heldout_pool()
    if not OPTS["novel"]:
        rows = []
    lo = OPTS["test_offset"]
    for i, t in enumerate(test[lo:lo + OPTS["n_test"]]):
        rows.append({"pid": f"ja_test_{i:02d}", "kind": "ja_heldout", "lang": "ja",
                     "text": t["text"], "real": t["audio"]})
    return rows


def ref_centroid() -> np.ndarray:
    cache = WORK / "eval" / "ref_centroid.npy"
    if cache.exists():
        return np.load(cache)
    test = [json.loads(l) for l in (WORK / "qwen" / "test_with_codes.jsonl").read_text(encoding="utf-8").splitlines() if l]
    held = {t["audio"] for t in heldout_pool()[:N_TEST]}  # the main suite's held-out lines
    paths = [t["audio"] for t in test if t["audio"] not in held and len(t["audio_codes"]) >= 25]
    c = metrics.centroid(paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, c)
    print(f"[eval] reference centroid from {len(paths)} real test clips")
    return c


def generate(ckpt: str, rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    if ckpt == "real":
        for r in rows:
            if r.get("real"):
                f = f"{r['pid']}.wav"
                wav, sr = sf.read(r["real"])
                sf.write(str(out / f), wav, sr)
                manifest.append({**r, "file": f})
    else:
        from qwen_tts import Qwen3TTSModel
        clone = ckpt.startswith("clone:")
        model = Qwen3TTSModel.from_pretrained(ckpt.split(":", 1)[1] if clone else ckpt,
                                              device_map="cuda:0", dtype=torch.bfloat16,
                                              attn_implementation="sdpa")
        prompt = None
        if clone:
            ref = json.loads((WORK / "build" / "ref.json").read_text(encoding="utf-8"))
            prompt = model.create_voice_clone_prompt(ref_audio=ref["audio"], ref_text=ref["text"])
        for b in range(0, len(rows), BATCH):
            chunk = rows[b:b + BATCH]
            torch.manual_seed(OPTS["seed_base"] + b)
            texts = [r["text"] for r in chunk]
            kw = dict(max_new_tokens=400)
            if clone:
                wavs, sr = model.generate_voice_clone(text=texts, language=["Auto"] * len(chunk),
                                                      voice_clone_prompt=prompt * len(chunk), **kw)
            else:
                wavs, sr = model.generate_custom_voice(text=texts, speaker=["kurisu"] * len(chunk),
                                                       language=["Auto"] * len(chunk), **kw)
            for r, w in zip(chunk, wavs):
                f = f"{r['pid']}.wav"
                sf.write(str(out / f), np.asarray(w, dtype=np.float32), sr)
                manifest.append({**r, "file": f})
        del model
        torch.cuda.empty_cache()
    (out / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
                                        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--n_test", type=int, default=N_TEST)
    ap.add_argument("--test_offset", type=int, default=0)
    ap.add_argument("--seed_base", type=int, default=1000)
    ap.add_argument("--no_probes", action="store_true", help="held-out test lines only")
    args = ap.parse_args()

    OPTS.update(n_test=args.n_test, test_offset=args.test_offset, seed_base=args.seed_base,
                novel=not args.no_probes)
    centroid = ref_centroid()
    rows = probe_rows()
    suite = WORK / "eval" / args.name
    table = {}
    for ckpt in args.ckpt:
        out = suite / tag_of(ckpt)
        if not (args.skip_existing and (out / "summary.json").exists()):
            generate(ckpt, rows, out)
            summary = metrics.score_dir(out, centroid)
        else:
            summary = json.loads((out / "summary.json").read_text())
        table[ckpt] = summary
        print(f"[eval] {ckpt}: {json.dumps(summary.get('all'))}", flush=True)
    prev = suite / "table.json"
    merged = json.loads(prev.read_text()) if prev.exists() else {}
    merged.update(table)
    prev.write_text(json.dumps(merged, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
