"""Stage 05 - where in each clip does a non-verbal burst plausibly happen.

``laion/vocalburst-locator`` v2 over every clip in the corpus, at the v2
post-processing values (threshold 0.50 / merge 0.10 / min-dur 0.10) rather than
the v1-era defaults still hard-coded in the upstream inference script.

The locator alone is not evidence.  On acted Japanese it fires mostly on short
lexical interjections that are already written in the transcript, and on the
European dubs it fires on emphatic onsets.  Its output is a *shortlist*: stage 06
aligns only the clips that have at least one located region, and stage 07 asks
the classifier to name only the regions that the aligner says nothing lexical
covers.  Running it first is purely an economy - it turns an alignment job over
fourteen thousand clips into one over a few thousand.
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf

import gpu
from kcommon import MANIFESTS, PROJECT, REPORTS, read_jsonl, utf8_stdout, write_json, write_jsonl
from vb_models import VocalBurstLocator, to_16k

utf8_stdout()

BATCH = 8
MAX_S = 30.0


def main() -> None:
    gpu.configure()
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    loc = VocalBurstLocator()
    out = []

    def run(chunk):
        waves, meta = [], []
        for r in chunk:
            x, sr = sf.read(PROJECT / r["audio"], dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            waves.append(to_16k(x, sr)[: int(MAX_S * 16000)])
            meta.append(r)
        probs = loc.frame_probs(waves)
        res = []
        for r, p in zip(meta, probs):
            ev = loc.events(p, min(r["duration"], MAX_S))
            res.append({"id": r["id"], "duration": r["duration"],
                        "events": ev,
                        "max_prob": round(float(p.max()), 4),
                        "burst_frames": int((p > 0.5).sum())})
        return res

    out = gpu.run_batched(rows, run, batch=BATCH, label="locate", progress_every=800)
    write_jsonl(MANIFESTS / "05_locator.jsonl", out)

    n_ev = sum(len(o["events"]) for o in out)
    with_ev = [o for o in out if o["events"]]
    durs = [e["end_s"] - e["start_s"] for o in out for e in o["events"]]
    write_json(REPORTS / "05_locator.json", {
        "model": "laion/vocalburst-locator (model_v2.pt)",
        "postprocessing": {"threshold": 0.50, "merge_gap_s": 0.10, "min_dur_s": 0.10},
        "clips": len(out),
        "clips_with_event": len(with_ev),
        "events": n_ev,
        "event_duration_s": {
            "median": round(float(np.median(durs)), 3) if durs else None,
            "p95": round(float(np.percentile(durs, 95)), 3) if durs else None},
        "note": "a located region is a shortlist entry, not a label",
    })
    print(json.dumps(json.loads((REPORTS / "05_locator.json").read_text("utf-8")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
