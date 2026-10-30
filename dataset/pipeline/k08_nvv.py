"""Non-verbal-vocalisation calibration, split by scope.

A whole-clip verdict and an in-speech verdict are not the same measurement and
must not share a threshold.

``whole_clip``
    The take *is* the sound -- the script line is only ``……``.  The classifier
    sees a burst and nothing else, so its confidence is structurally high, and
    the honest negative control is an ordinary speech clip judged the same way:
    whole file, no crop.

``in_speech``
    A crop from inside a spoken line that the locator flagged and the aligner
    could not explain lexically.  Confidence is structurally lower -- the crop is
    short and neighbouring phonemes bleed in through the context padding -- so a
    whole-clip threshold applied here rejects nearly everything.  The right
    negative control is a crop from a clip a reviewer *did* look at and left
    unmarked: same cropping, same pipeline, known answer.

Two calibration modes, because the evidence is not evenly distributed
--------------------------------------------------------------------
The reviewers annotated 125 clips with an event inside a spoken line but only 18
whose whole take is the sound.  So:

* where a scope has enough gold positives, the rule is chosen for **precision**:
  the highest-recall operating point whose measured precision clears the floor;
* where it does not, the rule is chosen for **specificity** instead: the
  highest-recall operating point whose *false-positive rate on the negative
  control* stays under a hard cap, with the second model required to agree.
  That is a weaker claim and it is labelled as such in the report -- it bounds
  how often the family is hallucinated on material that certainly does not
  contain it, which is the quantity that actually matters when the alternative is
  discarding the clip.

A family with neither enough positives nor a specificity point that passes
anything gets no rule at all, and is never assigned automatically.
"""
from __future__ import annotations

import collections

import numpy as np

PRECISION_FLOOR = 0.85
MIN_RECALL = 0.10
MIN_POSITIVES = 8
MIN_NEGATIVES = 30
MAX_CONTROL_FPR = 0.005          # <= 0.5% of negative-control clips may fire
PRESENCE_PRECISION_FLOOR = 0.90

SCORE_GRID = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)]
MARGIN_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def _fires(o, family, smin, mmin, need_second):
    v = o["vocalburst"]
    if v["family"] != family or v["family_score"] < smin:
        return False
    if v["family_margin"] < mmin:
        return False
    if need_second and not o["secondary"]["agrees_top2"]:
        return False
    return True


def _sweep_precision(pos, neg, family):
    best = None
    for need_second in (True, False):
        for smin in SCORE_GRID:
            for mmin in MARGIN_GRID:
                tp = sum(1 for o in pos if _fires(o, family, smin, mmin, need_second))
                fp = sum(1 for o in neg if _fires(o, family, smin, mmin, need_second))
                if tp == 0:
                    continue
                prec, rec = tp / (tp + fp), tp / len(pos)
                if prec < PRECISION_FLOOR or rec < MIN_RECALL:
                    continue
                if best is None or (rec, prec) > (best["recall"], best["precision"]):
                    best = {"mode": "precision", "family_score_min": smin,
                            "family_margin_min": mmin, "secondary_required": need_second,
                            "precision": round(prec, 3), "recall": round(rec, 3),
                            "true_positives": tp, "false_positives": fp,
                            "positives": len(pos), "negatives": len(neg)}
    return best


def _sweep_specificity(pos, neg, candidates, family):
    """Bound the hallucination rate instead of measuring precision."""
    if not neg:
        return None
    cap = max(1, int(MAX_CONTROL_FPR * len(neg)))
    best = None
    for smin in SCORE_GRID:
        for mmin in MARGIN_GRID:
            need_second = True                      # cheap insurance, always on here
            fp = sum(1 for o in neg if _fires(o, family, smin, mmin, need_second))
            if fp > cap:
                continue
            passed = sum(1 for o in candidates
                         if _fires(o, family, smin, mmin, need_second))
            if passed == 0:
                continue
            tp = sum(1 for o in pos if _fires(o, family, smin, mmin, need_second))
            key = (tp, passed, -smin)
            if best is None or key > best["_key"]:
                best = {"_key": key, "mode": "specificity",
                        "family_score_min": smin, "family_margin_min": mmin,
                        "secondary_required": need_second,
                        "control_false_positives": fp, "negatives": len(neg),
                        "control_fpr": round(fp / len(neg), 5),
                        "control_fpr_cap": MAX_CONTROL_FPR,
                        "gold_positives": len(pos), "gold_recovered": tp,
                        "candidates_passing": passed}
    if best:
        best.pop("_key")
    return best


def calibrate(obs: list[dict], rows: dict) -> dict:
    from kcommon import TAG_TO_FAMILY

    gold_iso = collections.defaultdict(set)
    gold_in = collections.defaultdict(set)
    for r in rows.values():
        fams = {TAG_TO_FAMILY[e["tag"]] for e in r["human_events"]
                if e["tag"] in TAG_TO_FAMILY}
        if not fams:
            continue
        (gold_iso if not r["text"] else gold_in)[r["id"]] |= fams

    reviewed_clean = {r["id"] for r in rows.values()
                      if r["human_reviewed"] and not r["human_events"]}

    whole = [o for o in obs if o["scope"] in ("whole_clip", "gold")]
    control = [o for o in obs if o["scope"] == "control"]
    in_speech = [o for o in obs if o["scope"] == "in_speech"]

    whole_candidates = [o for o in whole if o["scope"] == "whole_clip"]
    ins_neg = [o for o in in_speech if o["id"] in reviewed_clean]
    ins_cand = [o for o in in_speech if o["id"] not in reviewed_clean]

    iso_pos = collections.defaultdict(list)
    for o in whole:
        for f in gold_iso.get(o["id"], ()):
            iso_pos[f].append(o)
    ins_pos = collections.defaultdict(list)
    for o in in_speech:
        for f in gold_in.get(o["id"], ()):
            ins_pos[f].append(o)

    families = sorted({o["vocalburst"]["family"] for o in obs
                       if o["vocalburst"]["family"]}
                      | {f for s in gold_iso.values() for f in s}
                      | {f for s in gold_in.values() for f in s})

    policy = {"whole_clip": {}, "in_speech": {}}
    evidence = {}
    for fam in families:
        if fam in ("no_burst", "other"):
            continue
        ip, sp = iso_pos.get(fam, []), ins_pos.get(fam, [])

        rule_iso = None
        if len(ip) >= MIN_POSITIVES and len(control) >= MIN_NEGATIVES:
            rule_iso = _sweep_precision(ip, control, fam)
        if rule_iso is None and len(control) >= MIN_NEGATIVES:
            rule_iso = _sweep_specificity(ip, control, whole_candidates, fam)

        rule_ins = None
        if len(sp) >= MIN_POSITIVES and len(ins_neg) >= MIN_NEGATIVES:
            rule_ins = _sweep_precision(sp, ins_neg, fam)
        if rule_ins is None and len(ins_neg) >= MIN_NEGATIVES:
            rule_ins = _sweep_specificity(sp, ins_neg, ins_cand, fam)

        if rule_iso:
            policy["whole_clip"][fam] = {**rule_iso, "source": "whole_clip evidence"}
        if rule_ins:
            policy["in_speech"][fam] = {**rule_ins, "source": "in_speech evidence"}

        evidence[fam] = {
            "whole_clip_positives": len(ip),
            "in_speech_positives": len(sp),
            "control_clips": len(control),
            "control_top1_hits": sum(1 for o in control
                                     if o["vocalburst"]["family"] == fam),
            "in_speech_negatives": len(ins_neg),
            "in_speech_negative_top1_hits": sum(
                1 for o in ins_neg if o["vocalburst"]["family"] == fam),
            "whole_clip_candidates_top1": sum(
                1 for o in whole_candidates if o["vocalburst"]["family"] == fam),
            "in_speech_candidates_top1": sum(
                1 for o in ins_cand if o["vocalburst"]["family"] == fam),
            "whole_clip_rule": (rule_iso or {}).get("mode", "none"),
            "in_speech_rule": (rule_ins or {}).get("mode", "none"),
        }

    # ---- presence-only gate --------------------------------------------------
    pres_pos = [o for o in whole if o["id"] in gold_iso]
    presence = None
    for nb in [round(x, 2) for x in np.arange(0.02, 0.71, 0.02)]:
        for sp_ in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.01):
            tp = sum(1 for o in pres_pos
                     if o["vocalburst"]["p_no_burst"] <= nb
                     and o["secondary"]["speech_score"] <= sp_)
            fp = sum(1 for o in control
                     if o["vocalburst"]["p_no_burst"] <= nb
                     and o["secondary"]["speech_score"] <= sp_)
            if tp == 0:
                continue
            prec, rec = tp / (tp + fp), tp / max(len(pres_pos), 1)
            if prec >= PRESENCE_PRECISION_FLOOR and rec >= MIN_RECALL:
                if presence is None or rec > presence["recall"]:
                    presence = {"p_no_burst_max": nb, "speech_score_max": sp_,
                                "precision": round(prec, 3), "recall": round(rec, 3),
                                "true_positives": tp, "false_positives": fp,
                                "positives": len(pres_pos), "negatives": len(control)}
    if presence is None and control:
        # No precision floor reachable with 18 positives: fall back to bounding
        # the false-positive rate on ordinary speech, and say so.
        cap = max(1, int(MAX_CONTROL_FPR * len(control)))
        for nb in [round(x, 2) for x in np.arange(0.02, 0.71, 0.02)]:
            for sp_ in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.01):
                fp = sum(1 for o in control
                         if o["vocalburst"]["p_no_burst"] <= nb
                         and o["secondary"]["speech_score"] <= sp_)
                if fp > cap:
                    continue
                tp = sum(1 for o in pres_pos
                         if o["vocalburst"]["p_no_burst"] <= nb
                         and o["secondary"]["speech_score"] <= sp_)
                passed = sum(1 for o in whole_candidates
                             if o["vocalburst"]["p_no_burst"] <= nb
                             and o["secondary"]["speech_score"] <= sp_)
                if passed == 0:
                    continue
                cand = {"mode": "specificity", "p_no_burst_max": nb,
                        "speech_score_max": sp_, "control_false_positives": fp,
                        "control_fpr": round(fp / len(control), 5),
                        "control_fpr_cap": MAX_CONTROL_FPR,
                        "gold_positives": len(pres_pos), "gold_recovered": tp,
                        "candidates_passing": passed, "negatives": len(control),
                        "precision": None, "recall": round(tp / max(len(pres_pos), 1), 3)}
                if presence is None or (cand["gold_recovered"], cand["candidates_passing"]) > (
                        presence["gold_recovered"], presence["candidates_passing"]):
                    presence = cand
    if presence is not None:
        presence.setdefault("mode", "precision")

    return {
        "policy": policy,
        "evidence": evidence,
        "presence_gate": presence,
        "precision_floor": PRECISION_FLOOR,
        "presence_precision_floor": PRESENCE_PRECISION_FLOOR,
        "max_control_fpr": MAX_CONTROL_FPR,
        "min_recall": MIN_RECALL,
        "min_positives_for_precision_mode": MIN_POSITIVES,
        "gold_isolated_clips": len(gold_iso),
        "gold_in_speech_clips": len(gold_in),
        "reviewed_unmarked_clips": len(reviewed_clean),
        "whole_clip_candidates": len(whole_candidates),
        "in_speech_candidates": len(ins_cand),
    }
