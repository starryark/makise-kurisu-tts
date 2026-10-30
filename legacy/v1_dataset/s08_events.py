"""Stage 08 - name and place the vocal bursts found inside spoken lines.

A located event only becomes a candidate if it lands where the transcript does not
reach. Two independent signals must agree:

    locator says "a burst is here"        (laion/vocalburst-locator v2)
  AND aligner says "no transcript unit covers this"  (Qwen3-ForcedAligner)
  AND there is actually audible energy there

That intersection is the whole point: on this corpus the locator on its own fires
mostly on short Japanese lexical interjections (え, うん, あっ) that are already in
the transcript, and tagging those would both duplicate the text and teach the model
to bolt a sound onto a word it is already saying.

Surviving regions are cropped (tight + context, supplement s9), named by the
vocalburst classifier, checked against CED, and gated with the per-family policy
calibrated in stage 05. Everything else goes to the review queue.

Writes manifests/08_events.jsonl and reports/08_events.json.
"""
from __future__ import annotations

import collections
import json
import re

import numpy as np
import soundfile as sf

import gpu
from common import (MANIFESTS, REPORTS, read_jsonl, utf8_stdout, write_jsonl)
from vb_models import CedSecondary, VocalBurstClassifier, to_16k

utf8_stdout()

OVERLAP_MIN = 0.50        # fraction of the located event that must fall in a gap
CONTEXT_PAD_S = 0.08      # supplement s9: 50-150 ms of context around the tight crop
MIN_EVENT_S = 0.12
PUNCT = "…。、！？ 　―─—．・“”"


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def token_offsets(text: str, tokens: list[dict]) -> list[tuple[int, int]]:
    """Character span of each aligned token inside the normalised transcript."""
    offs, cur = [], 0
    for t in tokens:
        s = t["text"]
        i = text.find(s, cur) if s else -1
        if i < 0:
            offs.append((cur, cur))
        else:
            offs.append((i, i + len(s)))
            cur = i + len(s)
    return offs


def insertion_index(text: str, offs: list[tuple[int, int]], kind: str,
                    after_token: int | None) -> int:
    """Where the tag string goes. Punctuation is kept, never replaced (supplement s3)."""
    if kind == "head" or after_token is None:
        return 0
    end = offs[after_token][1] if after_token < len(offs) else len(text)
    while end < len(text) and text[end] in PUNCT:   # keep '……' attached to the words
        end += 1
    return end


def main() -> None:
    gpu.configure()
    audio = {r["id"]: r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")}
    loc = {o["id"]: o for o in read_jsonl(MANIFESTS / "06_locator.jsonl")}
    ali = {o["id"]: o for o in read_jsonl(MANIFESTS / "07_alignment.jsonl")}
    policy = json.loads((REPORTS / "05_calibration.json").read_text(encoding="utf-8"))["policy"]

    # ---------------- build candidates
    candidates, gap_only = [], []
    for cid, L in loc.items():
        A = ali.get(cid)
        if not A or not A.get("ok"):
            continue
        gaps = [g for g in A["gaps"] if g["energetic"]]
        offs = token_offsets(audio[cid]["text"], A["tokens"])
        used_gap = set()
        for ev in L["events"]:
            best, best_ov = None, 0.0
            for gi, g in enumerate(gaps):
                ov = overlap(ev["start_s"], ev["end_s"], g["start_s"], g["end_s"])
                if ov > best_ov:
                    best, best_ov = gi, ov
            ev_len = ev["end_s"] - ev["start_s"]
            if best is None or ev_len <= 0 or best_ov / ev_len < OVERLAP_MIN:
                continue
            g = gaps[best]
            used_gap.add(best)
            s = max(ev["start_s"], g["start_s"])
            e = min(ev["end_s"], g["end_s"])
            if e - s < MIN_EVENT_S:
                continue
            after = None
            if g["kind"] != "head":
                after = max((i for i, (a, b) in enumerate(
                    [(t["start_s"], t["end_s"]) for t in A["tokens"]]) if b <= g["start_s"] + 1e-3),
                    default=None)
            candidates.append({
                "id": cid, "start_s": round(s, 3), "end_s": round(e, 3),
                "locator_score": ev["score"], "gap_kind": g["kind"],
                "gap_rms_dbfs": g["rms_dbfs"], "gap_overlap": round(best_ov / ev_len, 3),
                "after_token": after,
                "insert_at": insertion_index(audio[cid]["text"], offs, g["kind"], after),
            })
        for gi, g in enumerate(gaps):
            if gi not in used_gap and g["end_s"] - g["start_s"] >= 0.25:
                gap_only.append({"id": cid, **g})

    print(f"{len(candidates)} located-and-unexplained candidates "
          f"in {len({c['id'] for c in candidates})} clips; "
          f"{len(gap_only)} unexplained-but-not-located regions logged for review")

    if not candidates:
        write_jsonl(MANIFESTS / "08_events.jsonl", [])
        (REPORTS / "08_events.json").write_text(json.dumps(
            {"candidates": 0, "note": "no region satisfied locator AND alignment gap"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # ---------------- crop, name, second opinion
    clf = VocalBurstClassifier()

    def crop(c, pad: float):
        r = audio[c["id"]]
        x, sr = sf.read(r["audio_abs"], dtype="float32", always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        a = max(0, int((c["start_s"] - pad) * sr))
        b = min(len(x), int((c["end_s"] + pad) * sr))
        return to_16k(x[a:b], sr)

    def run_clf(chunk):
        return clf.classify([crop(c, CONTEXT_PAD_S) for c in chunk])

    prim = gpu.run_batched(candidates, run_clf, batch=4, label="classify-crop",
                           progress_every=200)
    del clf
    gpu.free()

    ced = CedSecondary()

    def run_ced(chunk):
        return ced.scores([crop(c, CONTEXT_PAD_S) for c in chunk])

    sec = gpu.run_batched(candidates, run_ced, batch=2, label="ced-crop", progress_every=200)

    out = []
    for c, p, s in zip(candidates, prim, sec):
        fam = p["family"]
        pol = policy.get(fam, {}).get("auto_accept")
        agrees = CedSecondary.agrees(fam, s, rank=2)
        auto = bool(pol and p["family_score"] >= pol["family_score_min"]
                    and p["family_margin"] >= pol["family_margin_min"]
                    and agrees)
        out.append({
            **c,
            "text": audio[c["id"]]["text"],
            "vocalburst": {"model": "laion/vocalburst-classifier-single",
                           "scope": "event_crop", **p},
            "secondary": {"model": "mispeech/ced-base", **s,
                          "agrees_top2": agrees},
            "decision": {
                "family": fam,
                "status": "auto_accept" if auto else "uncertain",
                "reason": (f"policy {fam} t>={pol['family_score_min']} met, CED agrees"
                           if auto else
                           f"no policy for {fam}" if not pol else
                           "below calibrated threshold or CED disagrees"),
            },
        })
    write_jsonl(MANIFESTS / "08_events.jsonl", out)
    write_jsonl(MANIFESTS / "08_gap_only_regions.jsonl", gap_only)

    acc = [o for o in out if o["decision"]["status"] == "auto_accept"]
    report = {
        "gate": {"locator_and_alignment_gap_overlap_min": OVERLAP_MIN,
                 "context_pad_s": CONTEXT_PAD_S, "min_event_s": MIN_EVENT_S},
        "located_events_total": sum(len(o["events"]) for o in loc.values()),
        "candidates_after_gap_gate": len(out),
        "candidates_by_gap_kind": dict(collections.Counter(o["gap_kind"] for o in out)),
        "auto_accepted": len(acc),
        "auto_accepted_by_family": dict(collections.Counter(
            o["decision"]["family"] for o in acc)),
        "uncertain": len(out) - len(acc),
        "uncertain_by_family": dict(collections.Counter(
            o["decision"]["family"] for o in out
            if o["decision"]["status"] != "auto_accept").most_common()),
        "gap_only_regions_for_review": len(gap_only),
    }
    (REPORTS / "08_events.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
