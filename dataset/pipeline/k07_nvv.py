"""Stage 07 - name the sounds.

``laion/vocalburst-classifier-single`` is the primary namer, with fine-class
probabilities aggregated into families *before* anything is ranked (supplement
s7) so that a clip reading ``Contented Sigh 0.31 / Wistful Sigh 0.28 / Relief
Sigh 0.21`` is correctly a confident ``sigh`` and not a weak one.
``mispeech/ced-base`` votes independently on the same audio as a second opinion.

Four kinds of observation are produced, and the difference between them is what
makes stage 08 able to measure anything:

``whole_clip``    a clip whose script line carries no lexical content at all --
                  ``……`` in the game scenario, an empty subtitle in the anime.
                  The take *is* the sound, so the whole file is classified.
                  Every clip in the game quarantine pool is one of these.
``in_speech``     a region the locator flagged that the aligner says no
                  transcript unit reaches, cropped with context padding.
``gold``          a clip a human reviewer annotated with an event tag.  These are
                  the positives stage 08 calibrates on.
``control``       ordinary speech clips, sampled per language, that no human
                  marked.  These measure how often each family is hallucinated on
                  plain acted dialogue, which is the number that actually decides
                  where a threshold goes.
"""
from __future__ import annotations

import collections
import json
import random

import numpy as np
import soundfile as sf

import gpu
from kcommon import (MANIFESTS, PROJECT, REPORTS, TAG_TO_FAMILY, read_jsonl,
                     utf8_stdout, write_json, write_jsonl)
from vb_models import CedSecondary, VocalBurstClassifier, to_16k

utf8_stdout()

SEED = 20260905
CONTROL_PER_LANG = 400
OVERLAP_MIN = 0.50        # of the located event, how much must fall in a gap
CONTEXT_PAD_S = 0.08      # supplement s9
MIN_EVENT_S = 0.12
BATCH = 16


def crop(x: np.ndarray, sr: int, s: float, e: float, pad: float) -> np.ndarray:
    a = max(0, int((s - pad) * sr))
    b = min(len(x), int((e + pad) * sr))
    return x[a:b] if b > a else x[:1]


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> None:
    gpu.configure()
    rng = random.Random(SEED)
    rows = {r["id"]: r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")}
    loc = {o["id"]: o for o in read_jsonl(MANIFESTS / "05_locator.jsonl")}
    ali = {o["id"]: o for o in read_jsonl(MANIFESTS / "06_alignment.jsonl")}

    obs: list[dict] = []

    # ---- whole-clip candidates ---------------------------------------------
    for r in rows.values():
        if r["speech_status"] == "nonlexical_candidate" or r["from_quarantine"]:
            obs.append({"id": r["id"], "scope": "whole_clip", "language": r["language"],
                        "start_s": 0.0, "end_s": r["duration"], "pad": 0.0})

    # ---- human gold: every clip a reviewer gave an event tag ----------------
    gold_ids = set()
    for r in rows.values():
        for e in r["human_events"]:
            gold_ids.add(r["id"])
    for cid in sorted(gold_ids):
        r = rows[cid]
        if r["speech_status"] == "nonlexical_candidate" or r["from_quarantine"]:
            continue                       # already covered as a whole-clip candidate
        obs.append({"id": cid, "scope": "gold", "language": r["language"],
                    "start_s": 0.0, "end_s": r["duration"], "pad": 0.0,
                    "gold_tags": sorted({e["tag"] for e in r["human_events"]})})

    # ---- in-speech regions: located AND lexically unexplained AND audible ---
    for cid, L in loc.items():
        A = ali.get(cid)
        if not A or not A.get("ok"):
            continue
        r = rows[cid]
        hot = [g for g in A["gaps"] if g["energetic"]]
        if not hot:
            continue
        for ev in L["events"]:
            if ev["end_s"] - ev["start_s"] < MIN_EVENT_S:
                continue
            span = ev["end_s"] - ev["start_s"]
            best, bg = 0.0, None
            for g in hot:
                o = overlap(ev["start_s"], ev["end_s"], g["start_s"], g["end_s"])
                if o > best:
                    best, bg = o, g
            if bg is None or best / span < OVERLAP_MIN:
                continue
            s = max(ev["start_s"], bg["start_s"])
            e = min(ev["end_s"], bg["end_s"])
            if e - s < MIN_EVENT_S:
                continue
            obs.append({"id": cid, "scope": "in_speech", "language": r["language"],
                        "start_s": round(s, 3), "end_s": round(e, 3),
                        "pad": CONTEXT_PAD_S, "gap_kind": bg["kind"],
                        "locator_score": ev["score"], "gap_rms_dbfs": bg["rms_dbfs"]})

    # ---- negative control ---------------------------------------------------
    taken = {o["id"] for o in obs}
    by_lang = collections.defaultdict(list)
    for r in rows.values():
        if (r["id"] not in taken and r["speech_status"] == "speech"
                and r["style"] == "speech" and r["duration"] >= 0.8
                and not r["human_notes"]["raw"]):
            by_lang[r["language"]].append(r["id"])
    for lang, ids in by_lang.items():
        rng.shuffle(ids)
        for cid in ids[:CONTROL_PER_LANG]:
            obs.append({"id": cid, "scope": "control", "language": lang,
                        "start_s": 0.0, "end_s": rows[cid]["duration"], "pad": 0.0})

    print(f"observations: {len(obs)} "
          f"({collections.Counter(o['scope'] for o in obs)})")

    cls = VocalBurstClassifier()
    ced = CedSecondary()

    cache: dict[str, tuple[np.ndarray, int]] = {}

    def load(cid: str):
        if cid not in cache:
            if len(cache) > 400:
                cache.clear()
            x, sr = sf.read(PROJECT / rows[cid]["audio"], dtype="float32",
                            always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            cache[cid] = (x, sr)
        return cache[cid]

    def run(chunk):
        waves = []
        for o in chunk:
            x, sr = load(o["id"])
            w = crop(x, sr, o["start_s"], o["end_s"], o["pad"])
            waves.append(to_16k(w, sr)[: 16000 * 30])
        vb = cls.classify(waves)
        cd = ced.scores(waves)
        res = []
        for o, v, c in zip(chunk, vb, cd):
            m = dict(o)
            m["vocalburst"] = v
            m["secondary"] = {**c, "agrees_top2": CedSecondary.agrees(v["family"], c)}
            res.append(m)
        return res

    obs.sort(key=lambda o: o["id"])          # keeps the audio cache warm
    out = gpu.run_batched(obs, run, batch=BATCH, label="classify", progress_every=400)
    write_jsonl(MANIFESTS / "07_nvv.jsonl", out)

    def fam_hist(scope):
        return dict(collections.Counter(
            o["vocalburst"]["family"] for o in out if o["scope"] == scope).most_common())

    write_json(REPORTS / "07_nvv.json", {
        "primary": "laion/vocalburst-classifier-single (family-aggregated)",
        "secondary": "mispeech/ced-base",
        "observations": len(out),
        "by_scope": dict(collections.Counter(o["scope"] for o in out)),
        "control_per_language": CONTROL_PER_LANG,
        "family_top1_by_scope": {s: fam_hist(s)
                                 for s in ("whole_clip", "gold", "in_speech", "control")},
        "in_speech_gate": {"locator_gap_overlap_min": OVERLAP_MIN,
                           "min_event_s": MIN_EVENT_S,
                           "context_pad_s": CONTEXT_PAD_S},
    })
    print(json.dumps(json.loads((REPORTS / "07_nvv.json").read_text("utf-8")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
