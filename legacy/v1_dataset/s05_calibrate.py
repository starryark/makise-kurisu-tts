"""Stage 05 - threshold calibration for the isolated-clip decision.

The supplement asks for class-specific thresholds calibrated on the character rather
than a guessed universal 0.70 (s27/s28). There is no human-listened gold set here, so
the calibration uses the strongest evidence available without one:

  positives  = the isolated-NVV candidates (transcript is purely non-lexical, so a
               vocalisation is what the clip must contain)
  negatives  = the 400 ordinary-speech control clips (a whole spoken line should not
               be named as one dominant burst)

That gives a real, measurable false-positive rate per family per threshold. It is a
proxy, not ground truth - it cannot tell a sigh from a gasp, only "burst" from
"ordinary speech" - so every family it clears is still listed for human spot-checking
in reports/05_listening_queue.csv.

Writes reports/05_threshold_sweep.csv, reports/05_calibration.json and the queue.
"""
from __future__ import annotations

import collections
import csv
import json

from common import (EASY_FAMILIES, HARD_FAMILIES, MANIFESTS, REPORTS, read_jsonl,
                    utf8_stdout)

utf8_stdout()

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
MARGINS = [0.0, 0.10, 0.20, 0.30]


def main() -> None:
    rows = list(read_jsonl(MANIFESTS / "03_isolated_classified.jsonl"))
    pos = [r for r in rows if r["speech_status"] == "isolated_nvv_candidate"]
    neg = [r for r in rows if r["is_control"]]

    def passes(r, t, m, need_sec):
        v = r["vocalburst"]
        if v["family_score"] < t or v["family_margin"] < m:
            return False
        if need_sec and not r["secondary"]["agrees_top2"]:
            return False
        return True

    sweep_rows = []
    for need_sec in (False, True):
        for t in THRESHOLDS:
            for m in MARGINS:
                kept = [r for r in pos if passes(r, t, m, need_sec)]
                fp = [r for r in neg if passes(r, t, m, need_sec)]
                sweep_rows.append({
                    "secondary_required": int(need_sec),
                    "family_score_min": t,
                    "family_margin_min": m,
                    "isolated_kept": len(kept),
                    "isolated_coverage": round(len(kept) / len(pos), 4),
                    "control_false_positives": len(fp),
                    "control_fp_rate": round(len(fp) / len(neg), 4),
                    "kept_families": "|".join(
                        f"{k}:{v}" for k, v in
                        collections.Counter(r["vocalburst"]["family"] for r in kept).most_common()),
                })
    with open(REPORTS / "05_threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader()
        w.writerows(sweep_rows)

    # ---- per-family FP rate on ordinary speech (the number that decides the policy)
    per_family = {}
    fams = {r["vocalburst"]["family"] for r in pos} | {r["vocalburst"]["family"] for r in neg}
    for fam in sorted(f for f in fams if f):
        entry = {}
        for need_sec in (False, True):
            for t in (0.50, 0.70, 0.85):
                k = f"{'both' if need_sec else 'primary'}_t{t}"
                entry[k] = {
                    "isolated": sum(1 for r in pos
                                    if r["vocalburst"]["family"] == fam
                                    and passes(r, t, 0.0, need_sec)),
                    "control_fp": sum(1 for r in neg
                                      if r["vocalburst"]["family"] == fam
                                      and passes(r, t, 0.0, need_sec)),
                }
        per_family[fam] = entry

    # ---- the policy this evidence supports
    #
    # A family only earns an auto-accept rule if there is enough of it to have
    # measured anything. "0 control false positives" is meaningless for a family the
    # classifier never proposed - that is absence of evidence, not evidence of
    # safety - so MIN_SUPPORT candidates are required before a rule is granted.
    MIN_SUPPORT = 5
    support = collections.Counter(r["vocalburst"]["family"] for r in pos)
    policy = {}
    for fam, e in per_family.items():
        if fam in ("no_burst", "other", None):
            continue
        fp_both_050 = e["both_t0.5"]["control_fp"]
        fp_both_070 = e["both_t0.7"]["control_fp"]
        n = support[fam]
        if fam in HARD_FAMILIES:
            policy[fam] = {"auto_accept": None, "support": n,
                           "reason": "hard family - always reviewed (supplement s16)"}
        elif n < MIN_SUPPORT:
            policy[fam] = {"auto_accept": None, "support": n,
                           "reason": f"only {n} candidates - too few to calibrate, reviewed"}
        elif fp_both_070 == 0 and fp_both_050 <= 2 and fam in EASY_FAMILIES:
            policy[fam] = {"auto_accept": {"family_score_min": 0.50,
                                           "family_margin_min": 0.10,
                                           "secondary_required": True},
                           "support": n,
                           "reason": f"0 control FP at t0.70 both-model, "
                                     f"{fp_both_050} at t0.50, n={n}"}
        elif fp_both_070 == 0:
            policy[fam] = {"auto_accept": {"family_score_min": 0.70,
                                           "family_margin_min": 0.20,
                                           "secondary_required": True},
                           "support": n,
                           "reason": f"0 control FP at t0.70 both-model, n={n}"}
        else:
            policy[fam] = {"auto_accept": None, "support": n,
                           "reason": f"{fp_both_070} control FP survive t0.70 both-model"}

    out = {
        "positives_isolated": len(pos),
        "negatives_ordinary_speech": len(neg),
        "note": ("Proxy calibration - separates 'burst' from 'ordinary speech'. "
                 "It does not verify the family name. Human spot-check required "
                 "before training (see 05_listening_queue.csv)."),
        "per_family": per_family,
        "policy": policy,
    }
    (REPORTS / "05_calibration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- listening queue: the clips a human should hear first
    queue = []
    for r in pos:
        v = r["vocalburst"]
        fam = v["family"]
        pol = policy.get(fam, {}).get("auto_accept")
        auto = bool(pol and v["family_score"] >= pol["family_score_min"]
                    and v["family_margin"] >= pol["family_margin_min"]
                    and r["secondary"]["agrees_top2"])
        queue.append({
            "id": r["id"], "dur": r["dur"], "text": r["text"],
            "family": fam, "family_score": v["family_score"],
            "family_margin": v["family_margin"],
            "fine_top1": v["top_classes"][0]["label"] if v["top_classes"] else "",
            "ced_family": r["secondary"]["family"],
            "ced_score": r["secondary"]["family_score"],
            "ced_agrees_top2": int(r["secondary"]["agrees_top2"]),
            "would_auto_accept": int(auto),
            "priority": (0 if fam in HARD_FAMILIES else
                         1 if not auto else 2),
        })
    queue.sort(key=lambda q: (q["priority"], -q["family_score"]))
    with open(REPORTS / "05_listening_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(queue[0]))
        w.writeheader()
        w.writerows(queue)

    print(json.dumps({"policy": policy,
                      "auto_accept_would_keep": sum(q["would_auto_accept"] for q in queue),
                      "of_isolated": len(queue)}, ensure_ascii=False, indent=2))
    print("\nbest operating points (isolated coverage vs control FP):")
    for r in sweep_rows:
        if r["secondary_required"] and r["family_margin_min"] == 0.10:
            print(f"  t={r['family_score_min']:.2f} both-models  keep {r['isolated_kept']:3d}"
                  f"/{len(pos)}  controlFP {r['control_false_positives']:2d}/{len(neg)}"
                  f"  {r['kept_families']}")


if __name__ == "__main__":
    main()
