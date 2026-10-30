"""Stage 04 - independent second opinion on the isolated / onomatopoeia clips.

mispeech/ced-base (AudioSet-527) votes on the same clips stage 03 classified. It is
a different architecture trained on a different taxonomy and different data, so its
agreement is real evidence rather than a correlated re-run (supplement s12).

The negative control goes through it too: the point is to find out which families
survive *both* models on ordinary speech, and that is what sets the auto-accept
policy in stage 07.

Updates manifests/03_isolated_classified.jsonl in place (adds a "secondary" block)
and writes reports/04_secondary.json.
"""
from __future__ import annotations

import collections
import json

import numpy as np
import soundfile as sf

import gpu
from common import MANIFESTS, REPORTS, read_jsonl, utf8_stdout, write_jsonl
from vb_models import CedSecondary, to_16k

utf8_stdout()


def load16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return to_16k(x, sr)


def main() -> None:
    gpu.configure()
    audio = {r["id"]: r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")}
    rows = list(read_jsonl(MANIFESTS / "03_isolated_classified.jsonl"))
    print(f"second opinion on {len(rows)} clips (mispeech/ced-base)")

    ced = CedSecondary()

    def run(chunk):
        return ced.scores([load16k(audio[r["id"]]["audio_abs"]) for r in chunk])

    res = gpu.run_batched(rows, run, batch=2, label="ced", progress_every=100)

    for r, s in zip(rows, res):
        prim = r["vocalburst"]["family"]
        r["secondary"] = {
            "model": "mispeech/ced-base",
            **s,
            "agrees_top1": prim == s["family"],
            "agrees_top2": CedSecondary.agrees(prim, s, rank=2),
        }
    write_jsonl(MANIFESTS / "03_isolated_classified.jsonl", rows)

    # ---------------- report: which families survive BOTH models?
    iso = [r for r in rows if r["speech_status"] == "isolated_nvv_candidate"]
    ono = [r for r in rows if r["speech_status"] == "onomatopoeia_candidate"]
    ctl = [r for r in rows if r["is_control"]]

    def survivors(sel, t: float, need_secondary: bool):
        c = collections.Counter()
        for r in sel:
            v = r["vocalburst"]
            if v["family_score"] < t:
                continue
            if need_secondary and not r["secondary"]["agrees_top2"]:
                continue
            c[v["family"]] += 1
        return dict(c.most_common())

    report = {
        "model": "mispeech/ced-base",
        "clips": len(rows),
        "agreement": {
            "isolated_top1": f"{sum(r['secondary']['agrees_top1'] for r in iso)}/{len(iso)}",
            "isolated_top2": f"{sum(r['secondary']['agrees_top2'] for r in iso)}/{len(iso)}",
            "onomatopoeia_top2": f"{sum(r['secondary']['agrees_top2'] for r in ono)}/{len(ono)}",
        },
        "control_speech_score_median": round(float(np.median(
            [r["secondary"]["speech_score"] for r in ctl])), 4),
        "isolated_speech_score_median": round(float(np.median(
            [r["secondary"]["speech_score"] for r in iso])), 4),
        "surviving_families": {
            "isolated_t0.70_primary_only": survivors(iso, 0.70, False),
            "isolated_t0.70_both_models": survivors(iso, 0.70, True),
            "isolated_t0.50_both_models": survivors(iso, 0.50, True),
            "CONTROL_t0.70_primary_only": survivors(ctl, 0.70, False),
            "CONTROL_t0.70_both_models": survivors(ctl, 0.70, True),
            "CONTROL_t0.50_both_models": survivors(ctl, 0.50, True),
        },
    }
    (REPORTS / "04_secondary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
