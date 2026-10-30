"""Stage 07 - forced alignment of the existing transcript.

Qwen3-ForcedAligner-0.6B. The transcript is never replaced: it is the lexical ground
truth (supplement s10), and alignment only says *where* each unit of it lands. Two
things come out of that:

1. token spans, so an accepted tag can be inserted at the right lexical position;
2. the **lexically unexplained** regions - stretches of audio no transcript unit
   covers. A real non-lexical vocalisation inside a spoken line has to live in one
   of those. That is the precision gate stage 08 applies to the locator's events,
   and it is what stops a burst being invented on top of a word.

MUST be run with the isolated venv interpreter - qwen-asr pins transformers 4.57.6
and would break the vocalburst models in the main environment:

    <scratchpad>/venv_aligner/Scripts/python.exe s07_align.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
import torch

from common import MANIFESTS, REPORTS, read_jsonl, utf8_stdout, write_jsonl

utf8_stdout()

FRACTION = float(os.environ.get("CUDA_FRACTION", "0.80"))
BATCH = int(os.environ.get("ALIGN_BATCH", "4"))
LIMIT = int(os.environ.get("ALIGN_LIMIT", "0"))
MIN_GAP_S = 0.12          # shorter than this is not a vocalisation, it is a stop closure
ENERGY_FLOOR_DB = -45.0   # a gap only counts as unexplained if something is audible in it


def energetic(x: np.ndarray, sr: int, s: float, e: float) -> tuple[bool, float]:
    a, b = max(0, int(s * sr)), min(len(x), int(e * sr))
    if b - a < sr // 100:
        return False, -120.0
    rms = float(np.sqrt(np.mean(x[a:b] ** 2) + 1e-20))
    db = 20 * float(np.log10(max(rms, 1e-12)))
    return db > ENERGY_FLOOR_DB, round(db, 2)


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(FRACTION, 0)
        print(f"[gpu] cap {FRACTION:.0%}  batch {BATCH}")

    from qwen_asr import Qwen3ForcedAligner

    rows = [r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")
            if r["speech_status"] in ("speech", "onomatopoeia_candidate") and r["text"]]
    if LIMIT:
        rows = rows[:LIMIT]
    print(f"aligning {len(rows)} clips ...")

    model = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0")

    out, failed = [], 0
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        try:
            res = model.align(audio=[r["audio_abs"] for r in chunk],
                              text=[r["text"] for r in chunk],
                              language="Japanese")
        except Exception:
            res = []
            for r in chunk:
                try:
                    res.append(model.align(audio=r["audio_abs"], text=r["text"],
                                           language="Japanese")[0])
                except Exception as ex:
                    res.append(None)
                    print(f"   ! align failed {r['id']}: {type(ex).__name__}")

        for r, a in zip(chunk, res):
            if a is None:
                failed += 1
                out.append({"id": r["id"], "dur": r["dur"], "ok": False,
                            "tokens": [], "gaps": []})
                continue
            toks = [{"text": it.text,
                     "start_s": round(float(it.start_time), 3),
                     "end_s": round(float(it.end_time), 3)} for it in a.items]
            x, sr = sf.read(r["audio_abs"], dtype="float32", always_2d=False)
            if x.ndim == 2:
                x = x.mean(axis=1)

            # lexically unexplained regions: head, inter-token, tail
            spans = [(t["start_s"], t["end_s"]) for t in toks]
            gaps = []
            cursor = 0.0
            for s, e in spans:
                if s - cursor >= MIN_GAP_S:
                    gaps.append((cursor, s, "head" if cursor == 0.0 else "internal"))
                cursor = max(cursor, e)
            if r["dur"] - cursor >= MIN_GAP_S:
                gaps.append((cursor, r["dur"], "tail" if spans else "whole"))

            gap_rows = []
            for s, e, kind in gaps:
                hot, db = energetic(x, sr, s, e)
                gap_rows.append({"start_s": round(s, 3), "end_s": round(e, 3),
                                 "kind": kind, "rms_dbfs": db, "energetic": hot})
            covered = sum(e - s for s, e in spans)
            out.append({
                "id": r["id"], "dur": r["dur"], "ok": True,
                "tokens": toks, "gaps": gap_rows,
                "n_tokens": len(toks),
                "coverage": round(covered / r["dur"], 3) if r["dur"] else 0.0,
                "first_token_start_s": toks[0]["start_s"] if toks else None,
                "last_token_end_s": toks[-1]["end_s"] if toks else None,
            })

        if (i // BATCH) % 50 == 0:
            done = min(i + BATCH, len(rows))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"   {done}/{len(rows)}  {rate:.1f} clip/s  "
                  f"eta {(len(rows) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)

    write_jsonl(MANIFESTS / "07_alignment.jsonl", out)

    ok = [o for o in out if o["ok"]]
    energetic_gaps = [g for o in ok for g in o["gaps"] if g["energetic"]]
    report = {
        "model": "Qwen/Qwen3-ForcedAligner-0.6B",
        "language": "Japanese",
        "clips": len(out), "aligned": len(ok), "failed": failed,
        "min_gap_s": MIN_GAP_S, "energy_floor_dbfs": ENERGY_FLOOR_DB,
        "coverage": {
            "median": round(float(np.median([o["coverage"] for o in ok])), 3) if ok else None,
            "p05": round(float(np.percentile([o["coverage"] for o in ok], 5)), 3) if ok else None,
        },
        "clips_with_energetic_unexplained_gap": len(
            {g_id for g_id, o in ((o["id"], o) for o in ok)
             if any(g["energetic"] for g in o["gaps"])}),
        "energetic_gaps": len(energetic_gaps),
        "energetic_gap_by_kind": {
            k: sum(1 for g in energetic_gaps if g["kind"] == k)
            for k in ("head", "internal", "tail", "whole")},
        "energetic_gap_duration_s": {
            "median": round(float(np.median(
                [g["end_s"] - g["start_s"] for g in energetic_gaps])), 3)
            if energetic_gaps else None,
        },
    }
    (REPORTS / "07_alignment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
