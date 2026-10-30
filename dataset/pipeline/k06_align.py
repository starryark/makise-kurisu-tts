"""Stage 06 - forced alignment, in six languages.

``Qwen/Qwen3-ForcedAligner-0.6B``.  The transcript is never replaced: it is the
lexical ground truth (supplement s10) and alignment only says *where* each unit of
it lands.  Two products come out of that:

1. token spans, so an accepted tag can be inserted at the right lexical position
   instead of being dumped at the front of the line;
2. **lexically unexplained regions** - stretches of audio no transcript unit
   covers.  A genuine non-lexical vocalisation inside a spoken line has to live in
   one of those, and requiring it is what stops the classifier inventing a sigh on
   top of a word that is already written down.

Alignment covers every clip that has text; set ALIGN_SHORTLIST_ONLY=1 to restrict
it to the locator shortlist.
"""
from __future__ import annotations

import collections
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from kcommon import MANIFESTS, PROJECT, REPORTS, read_jsonl, utf8_stdout, write_json, write_jsonl

utf8_stdout()

BATCH = int(os.environ.get("ALIGN_BATCH", "4"))
MIN_GAP_S = 0.12          # shorter than this is a stop closure, not a vocalisation
ENERGY_FLOOR_DB = -45.0

ALIGNER_LANGUAGE = {
    "ja": "Japanese", "en": "English", "zh": "Chinese",
    "de": "German", "fr": "French", "es": "Spanish",
}


def energetic(x: np.ndarray, sr: int, s: float, e: float) -> tuple[bool, float]:
    a, b = max(0, int(s * sr)), min(len(x), int(e * sr))
    if b - a < sr // 100:
        return False, -120.0
    rms = float(np.sqrt(np.mean(x[a:b] ** 2) + 1e-20))
    db = 20.0 * float(np.log10(max(rms, 1e-12)))
    return db > ENERGY_FLOOR_DB, round(db, 2)


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.init()
        frac = float(os.environ.get("CUDA_FRACTION", "0.85"))
        if frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(frac, 0)

    from qwen_asr import Qwen3ForcedAligner

    audio = {r["id"]: r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")}
    shortlist = {o["id"] for o in read_jsonl(MANIFESTS / "05_locator.jsonl")
                 if o["events"]}
    rows = [r for r in audio.values()
            if r["speech_status"] == "speech" and r["text"]]
    if os.environ.get("ALIGN_SHORTLIST_ONLY") == "1":
        rows = [r for r in rows if r["id"] in shortlist]
    rows.sort(key=lambda r: (r["language"], -r["duration"]))
    print(f"aligning {len(rows)} of {len(audio)} clips "
          f"({len(shortlist)} of them shortlisted by the locator)")

    model = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0")

    out, failed = [], 0
    t0 = time.time()
    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)

    done = 0
    for lang, group in by_lang.items():
        name = ALIGNER_LANGUAGE[lang]
        for i in range(0, len(group), BATCH):
            chunk = group[i:i + BATCH]
            paths = [str(PROJECT / r["audio"]) for r in chunk]
            try:
                res = model.align(audio=paths, text=[r["text"] for r in chunk],
                                  language=name)
            except Exception:
                res = []
                for r, p in zip(chunk, paths):
                    try:
                        res.append(model.align(audio=p, text=r["text"],
                                               language=name)[0])
                    except Exception as ex:
                        res.append(None)
                        print(f"   ! align failed {r['id']}: {type(ex).__name__}")
            for r, a in zip(chunk, res):
                if a is None:
                    failed += 1
                    out.append({"id": r["id"], "language": lang,
                                "duration": r["duration"], "ok": False,
                                "tokens": [], "gaps": []})
                    continue
                toks = [{"text": it.text,
                         "start_s": round(float(it.start_time), 3),
                         "end_s": round(float(it.end_time), 3)} for it in a.items]
                x, sr = sf.read(PROJECT / r["audio"], dtype="float32",
                                always_2d=False)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                spans = [(t["start_s"], t["end_s"]) for t in toks]
                gaps, cursor = [], 0.0
                for s, e in spans:
                    if s - cursor >= MIN_GAP_S:
                        gaps.append((cursor, s, "head" if cursor == 0.0 else "internal"))
                    cursor = max(cursor, e)
                if r["duration"] - cursor >= MIN_GAP_S:
                    gaps.append((cursor, r["duration"], "tail" if spans else "whole"))
                gap_rows = []
                for s, e, kind in gaps:
                    hot, db = energetic(x, sr, s, e)
                    gap_rows.append({"start_s": round(s, 3), "end_s": round(e, 3),
                                     "kind": kind, "rms_dbfs": db, "energetic": hot})
                covered = sum(e - s for s, e in spans)
                out.append({
                    "id": r["id"], "language": lang, "duration": r["duration"],
                    "ok": True, "tokens": toks, "gaps": gap_rows,
                    "n_tokens": len(toks),
                    "coverage": round(covered / r["duration"], 3) if r["duration"] else 0.0,
                })
            done += len(chunk)
            if (done // BATCH) % 60 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"   {done}/{len(rows)} {rate:.1f} clip/s "
                      f"eta {(len(rows) - done) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)

    write_jsonl(MANIFESTS / "06_alignment.jsonl", out)
    ok = [o for o in out if o["ok"]]
    hot = [g for o in ok for g in o["gaps"] if g["energetic"]]
    write_json(REPORTS / "06_alignment.json", {
        "model": "Qwen/Qwen3-ForcedAligner-0.6B",
        "clips_aligned": len(ok), "failed": failed,
        "min_gap_s": MIN_GAP_S, "energy_floor_dbfs": ENERGY_FLOOR_DB,
        "by_language": {l: sum(1 for o in ok if o["language"] == l)
                        for l in sorted({o["language"] for o in ok})},
        "coverage_median": round(float(np.median([o["coverage"] for o in ok])), 3) if ok else None,
        "clips_with_energetic_unexplained_gap": sum(
            1 for o in ok if any(g["energetic"] for g in o["gaps"])),
        "energetic_gaps_by_kind": {
            k: sum(1 for g in hot if g["kind"] == k)
            for k in ("head", "internal", "tail", "whole")},
    })
    print(json.dumps(json.loads((REPORTS / "06_alignment.json").read_text("utf-8")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
