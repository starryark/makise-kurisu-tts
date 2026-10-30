"""Stage 03 - classify the isolated / onomatopoeia clips, plus a negative control.

Three populations go through laion/vocalburst-classifier-single:

1. ``isolated_nvv_candidate`` - transcript carries no lexical content at all
   (``……``, ``♪〜``). These are the quarantined short expressive takes.
2. ``onomatopoeia_candidate`` - the whole transcript is a written vocalisation
   (``ふふ``, ``はあ……``). Audio has to agree with that prior before anything is
   rewritten (supplement s3: never infer an NVV from the text alone).
3. a random sample of ordinary speech - the negative control. It measures how
   often the classifier hallucinates a burst on plain acted Japanese, which is
   what the auto-accept thresholds in stage 07 are calibrated against.

Writes manifests/03_isolated_classified.jsonl and reports/03_classifier.json.
"""
from __future__ import annotations

import collections
import json
import random

import numpy as np
import soundfile as sf

import gpu
from common import (FAMILY_TO_QWEN, MANIFESTS, REPORTS, read_jsonl, utf8_stdout,
                    write_jsonl)
from vb_models import VocalBurstClassifier, to_16k

utf8_stdout()

N_CONTROL = 400
SEED = 20260821


def load16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return to_16k(x, sr)


def main() -> None:
    gpu.configure()
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    by_status = collections.defaultdict(list)
    for r in rows:
        by_status[r["speech_status"]].append(r)

    rng = random.Random(SEED)
    control = rng.sample(by_status["speech"], min(N_CONTROL, len(by_status["speech"])))
    for r in control:
        r["_control"] = True

    targets = (by_status["isolated_nvv_candidate"]
               + by_status["onomatopoeia_candidate"]
               + control)
    print(f"classifying {len(targets)} clips "
          f"({len(by_status['isolated_nvv_candidate'])} isolated, "
          f"{len(by_status['onomatopoeia_candidate'])} onomatopoeia, "
          f"{len(control)} negative control)")

    clf = VocalBurstClassifier()

    def run(chunk):
        waves = [load16k(r["audio_abs"]) for r in chunk]
        return clf.classify(waves)

    results = gpu.run_batched(targets, run, label="classify", progress_every=100)

    out = []
    for r, res in zip(targets, results):
        fam = res["family"]
        out.append({
            "id": r["id"],
            "speech_status": r["speech_status"],
            "is_control": bool(r.get("_control")),
            "dur": r["dur"],
            "text": r["text"],
            "text_prior": r.get("text_prior"),
            "text_prior_rule": r.get("text_prior_rule"),
            "sha256": r["sha256"],
            "vocalburst": {
                "model": "laion/vocalburst-classifier-single",
                "revision": clf.revision,
                "scope": "whole_clip",
                **res,
            },
            "qwen_tag_candidate": FAMILY_TO_QWEN.get(fam) if fam else None,
        })
    write_jsonl(MANIFESTS / "03_isolated_classified.jsonl", out)

    # ---------------- report
    def dist(sel):
        c = collections.Counter()
        for o in sel:
            v = o["vocalburst"]
            c[v["family"] if v["p_no_burst"] < 0.5 else "no_burst(gated)"] += 1
        return dict(c.most_common())

    iso = [o for o in out if o["speech_status"] == "isolated_nvv_candidate"]
    ono = [o for o in out if o["speech_status"] == "onomatopoeia_candidate"]
    ctl = [o for o in out if o["is_control"]]

    def conf_bands(sel):
        b = collections.Counter()
        for o in sel:
            s = o["vocalburst"]["family_score"]
            b["ge_0.85" if s >= 0.85 else "0.70-0.85" if s >= 0.70
              else "0.50-0.70" if s >= 0.50 else "lt_0.50"] += 1
        return dict(b)

    # prior/audio agreement on the onomatopoeia set
    agree = sum(1 for o in ono
                if o["text_prior"] and o["vocalburst"]["family"] in o["text_prior"])
    report = {
        "model": "laion/vocalburst-classifier-single",
        "revision": clf.revision,
        "counts": {"isolated": len(iso), "onomatopoeia": len(ono), "control": len(ctl)},
        "isolated": {
            "family_distribution": dist(iso),
            "confidence_bands": conf_bands(iso),
            "median_p_no_burst": round(float(np.median(
                [o["vocalburst"]["p_no_burst"] for o in iso])), 4),
        },
        "onomatopoeia": {
            "family_distribution": dist(ono),
            "prior_audio_agreement": f"{agree}/{len(ono)}",
        },
        "negative_control_ordinary_speech": {
            "family_distribution": dist(ctl),
            "median_p_no_burst": round(float(np.median(
                [o["vocalburst"]["p_no_burst"] for o in ctl])), 4),
            "p_no_burst_ge_0.5": sum(1 for o in ctl if o["vocalburst"]["p_no_burst"] >= 0.5),
            "would_pass_0.85_family_gate": sum(
                1 for o in ctl if o["vocalburst"]["family_score"] >= 0.85),
            "would_pass_0.70_family_gate": sum(
                1 for o in ctl if o["vocalburst"]["family_score"] >= 0.70),
        },
    }
    (REPORTS / "03_classifier.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
