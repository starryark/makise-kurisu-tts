"""Stage 06 - locate vocal bursts inside the lexical clips.

laion/vocalburst-locator, model_v2.pt, v2 post-processing (threshold 0.50 /
merge 0.10 / min-dur 0.10). Frame probabilities are kept, not just the extracted
events, so stage 08 can re-threshold without another GPU pass.

Only clips that carry lexical text are scanned - the purely non-lexical ones were
already named whole-clip in stage 03.

Writes manifests/06_locator.jsonl and reports/06_locator.json.
"""
from __future__ import annotations

import collections
import json

import numpy as np
import soundfile as sf

import gpu
from common import MANIFESTS, REPORTS, read_jsonl, utf8_stdout, write_jsonl
from vb_models import VocalBurstLocator, to_16k

utf8_stdout()

THRESHOLD, MERGE_GAP, MIN_DUR = 0.50, 0.10, 0.10


def main() -> None:
    gpu.configure()
    rows = [r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")
            if r["speech_status"] in ("speech", "onomatopoeia_candidate")]
    print(f"locating bursts in {len(rows)} lexical clips "
          f"(v2 ckpt, t={THRESHOLD} merge={MERGE_GAP} min_dur={MIN_DUR})")

    loc = VocalBurstLocator()
    out = []

    def run(chunk):
        waves = []
        for r in chunk:
            x, sr = sf.read(r["audio_abs"], dtype="float32", always_2d=False)
            if x.ndim == 2:
                x = x.mean(axis=1)
            waves.append(to_16k(x, sr))
        P = loc.frame_probs(waves)
        res = []
        for r, p in zip(chunk, P):
            n_frames = max(1, int(round(r["dur"] / loc.CLIP_S * loc.FRAMES)))
            inside = p[:n_frames]
            res.append({
                "id": r["id"],
                "dur": r["dur"],
                "events": loc.events(p, r["dur"], THRESHOLD, MERGE_GAP, MIN_DUR),
                "frame_prob_max": round(float(inside.max()), 4),
                "frame_prob_mean": round(float(inside.mean()), 4),
                # 20 ms frames, quantised to 3 dp - enough to re-threshold offline
                "frame_probs": [round(float(v), 3) for v in inside],
            })
        return res

    out = gpu.run_batched(rows, run, batch=2, label="locate", progress_every=1000)
    write_jsonl(MANIFESTS / "06_locator.jsonl", out)

    n_ev = collections.Counter(len(o["events"]) for o in out)
    durs = [e["end_s"] - e["start_s"] for o in out for e in o["events"]]
    scores = [e["score"] for o in out for e in o["events"]]
    report = {
        "model": "laion/vocalburst-locator",
        "checkpoint": loc.checkpoint,
        "post_processing": {"threshold": THRESHOLD, "merge_gap_s": MERGE_GAP,
                            "min_duration_s": MIN_DUR},
        "clips_scanned": len(out),
        "clips_with_events": sum(1 for o in out if o["events"]),
        "events_per_clip": {str(k): v for k, v in sorted(n_ev.items())},
        "total_events": len(durs),
        "event_duration_s": {
            "median": round(float(np.median(durs)), 3) if durs else None,
            "p05": round(float(np.percentile(durs, 5)), 3) if durs else None,
            "p95": round(float(np.percentile(durs, 95)), 3) if durs else None,
        },
        "event_score": {
            "median": round(float(np.median(scores)), 3) if scores else None,
            "max": round(float(np.max(scores)), 3) if scores else None,
        },
    }
    (REPORTS / "06_locator.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
