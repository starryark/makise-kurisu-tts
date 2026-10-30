"""Stage 08 - measure every detector against the human gold set, then decide
where its threshold goes.  Nothing is labelled here; thresholds are.

The gold set was not commissioned for this build.  It already existed: the manual
review of the six-language anime corpus wrote bracket annotations straight into
the transcripts -- 132 ``[digital]``, 47 ``[gasp]``, 33 ``[breath]``, 25
``[whisper]``, 18 ``[yell]``, 123 ``[noise]`` and so on across 1,478 reviewed
clips.  Those are human judgements on this exact character in this exact
material, which is the "character-specific gold set" the labelling supplement
calls the highest-value step in the process (s30).  This stage spends it.

Two families of detector are calibrated, and they are calibrated differently
because the evidence is different.

**Non-verbal vocalisations** are named by a model trained for exactly that job,
so what has to be measured is how often it names one where there is none.
Positives come from the gold clips; negatives come from several hundred ordinary
speech clips per language that no reviewer marked.  A family only gets an
auto-accept rule if some (family_score, family_margin, second-model-agrees)
operating point reaches the precision floor, and the rule is the point with the
best recall among those that do.

**Delivery, playback and contamination** -- whisper, yell, digital, echo, and
non-character noise -- have no off-the-shelf model, so a small logistic
regression is fitted per language on the deterministic features from stage 03 and
scored leave-one-out, so the reported precision is out of sample.  A language
whose fit cannot clear the floor does not get automatic propagation for that tag;
its human labels still ship.  That is why ``[digital]`` propagates in Spanish and
German but not in Japanese: the Japanese original marks Amadeus with a thin,
bass-cut EQ that the reviewers heard as narrative context rather than an audible
band-limit, and the fit says so out loud instead of guessing.

Only the manually reviewed subset can be used for these, because that is the only
place where the *absence* of a tag means a human looked and decided no.
"""
from __future__ import annotations

import collections
import json
import pickle

import numpy as np

from kcommon import (MANIFESTS, REPORTS, TAG_TO_FAMILY, read_jsonl, utf8_stdout,
                     write_json)

utf8_stdout()

PRECISION_FLOOR = 0.85
MIN_RECALL = 0.10
MIN_POSITIVES = 8
MIN_NEGATIVES = 30

ALL_LANGS = ["ja", "en", "zh", "de", "fr", "es"]

#: How each detector may pool languages when it is fitted.
#
# ``[digital]`` is the one tag whose acoustic signature is genuinely
# language-specific: the European dubs render Amadeus through an audible
# low-pass, the Japanese original through a thin bass-cut EQ, and pooling the two
# would fit a model whose sign is wrong on half the corpus.  So it gets a
# per-language fit, plus one pooled fit over the four dubs that share a direction
# and can therefore lend each other examples.
#
# Whispering, shouting, room reverberation and stray non-character sound are
# physiological or physical rather than linguistic, and the level-relative
# features are already normalised per (language, medium) cell, so those pool
# across everything -- which is the only way [yell] (18 gold clips) and [echo]
# (29) get enough positives to be fitted at all.
#
# Each entry is (name, languages to FIT on, languages to APPLY to).  Those differ
# in exactly one place: the Chinese reviewers never used ``[digital]`` even though
# the Chinese Amadeus clips are audibly band-limited, so Chinese has no positives
# to fit on and would also poison the fit if its unlabelled clips were used as
# negatives.  It is therefore fitted out of and predicted into, and stage 09
# checks the result against a label the detector never saw -- whether the clips it
# fires on are the Amadeus ones.
GROUPING = {
    "[digital]": [("dubs", ["en", "de", "fr", "es"], ["en", "de", "fr", "es", "zh"])],
    "[whisper]": [("all", ALL_LANGS, ALL_LANGS)],
    "[yell]": [("all", ALL_LANGS, ALL_LANGS)],
    "[echo]": [("all", ALL_LANGS, ALL_LANGS)],
    "noise": [("all", ALL_LANGS, ALL_LANGS)],
}

#: name -> (where the human label lives, its key, feature list, ships in transcript)
DETECTORS = {
    "[digital]": ("conditions", "[digital]",
                  ["log_roll995", "hf_ratio_db", "lf_ratio_db", "spectral_tilt_db",
                   "spectral_centroid_hz", "roll95_hz"], True),
    "[whisper]": ("conditions", "[whisper]",
                  ["voiced_ratio", "hnr_db", "rel_active_rms_db", "spectral_flatness",
                   "hf_ratio_db", "f0_range_st"], True),
    "[yell]": ("conditions", "[yell]",
               ["rel_active_rms_db", "rel_f0_median_st", "f0_range_st",
                "spectral_tilt_db", "hnr_db", "units_per_active_s"], True),
    "[echo]": ("conditions", "[echo]",
               ["decay_db_per_s", "late_energy_ratio_db", "snr_est_db",
                "spectral_flatness", "noise_floor_dbfs"], True),
    "noise": ("contamination", "noise",
              ["snr_est_db", "noise_floor_dbfs", "spectral_flatness",
               "hf_ratio_db", "voiced_ratio", "decay_db_per_s"], False),
}


# ----------------------------------------------------------------- NVV gates
#
# Lives in k08_nvv.py: whole-clip and in-speech evidence are different
# measurements and get separate gates.

from k08_nvv import calibrate as nvv_calibration          # noqa: E402


# --------------------------------------------------- condition-tag detectors

def build_condition_features(rows: dict, feats: dict) -> dict:
    """Derived features, some of them relative to the clip's own language/medium cell."""
    cells = collections.defaultdict(list)
    for cid, r in rows.items():
        cells[(r["language"], r["medium"])].append(cid)
    out = {}
    for _cell, ids in cells.items():
        arms = np.array([rows[c]["active_rms_dbfs"] for c in ids], dtype=float)
        med_rms = float(np.median(arms))
        f0 = np.array([feats[c]["f0_median_hz"] or np.nan for c in ids], dtype=float)
        med_f0 = float(np.nanmedian(f0)) if np.isfinite(f0).any() else 220.0
        for c in ids:
            f = feats[c]
            out[c] = {
                "log_roll995": float(np.log10(max(f["roll995_hz"], 100.0))),
                "roll95_hz": float(f["roll95_hz"]),
                "hf_ratio_db": float(f["hf_ratio_db"]),
                "lf_ratio_db": float(f["lf_ratio_db"]),
                "spectral_tilt_db": float(f["spectral_tilt_db"]),
                "spectral_centroid_hz": float(f["spectral_centroid_hz"]),
                "spectral_flatness": float(f["spectral_flatness"]),
                "voiced_ratio": float(f["voiced_ratio"]),
                "hnr_db": float(f["hnr_db"] if f["hnr_db"] is not None else 0.0),
                "f0_range_st": float(f["f0_range_st"] or 0.0),
                "units_per_active_s": float(f["units_per_active_s"]),
                "rel_active_rms_db": float(rows[c]["active_rms_dbfs"] - med_rms),
                "rel_f0_median_st": float(
                    12.0 * np.log2((f["f0_median_hz"] or med_f0) / med_f0)),
                "decay_db_per_s": float(f["decay_db_per_s"]
                                        if f["decay_db_per_s"] is not None else 0.0),
                "late_energy_ratio_db": float(f["late_energy_ratio_db"]
                                              if f["late_energy_ratio_db"] is not None else 0.0),
                "snr_est_db": float(f["snr_est_db"] if f["snr_est_db"] is not None else 40.0),
                "noise_floor_dbfs": float(f["noise_floor_dbfs"]
                                          if f["noise_floor_dbfs"] is not None else -70.0),
            }
    return out


def fit_detector(name, langs, rows, cf, scope):
    """Leave-one-out logistic regression on the reviewed clips of ``langs``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    field, key, names, ships = DETECTORS[name]
    langs = set(langs)
    ids = [cid for cid, r in rows.items()
           if r["language"] in langs and r["human_reviewed"] and r["style"] == "speech"]
    y = np.array([1 if key in rows[c]["human_notes"][field] else 0 for c in ids])         if ids else np.zeros(0, dtype=int)
    res = {"scope": scope, "languages": sorted(langs), "reviewed_clips": len(ids),
           "positives": int(y.sum()), "usable": False, "ships_as_tag": ships}
    if y.sum() < MIN_POSITIVES or (len(y) - y.sum()) < MIN_NEGATIVES:
        res["reason"] = "not enough reviewed positives or negatives"
        return res, None

    X = np.asarray([[cf[c][n] for n in names] for c in ids], dtype=float)

    def new_clf():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000,
                                                class_weight="balanced"))

    oof = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), dtype=bool)
        m[i] = False
        if y[m].sum() < 3:
            continue
        c = new_clf()
        c.fit(X[m], y[m])
        oof[i] = c.predict_proba(X[i:i + 1])[0, 1]
    try:
        res["loo_auc"] = round(float(roc_auc_score(y, oof)), 3)
    except Exception:
        res["loo_auc"] = None

    best = None
    for thr in np.arange(0.50, 0.995, 0.005):
        fired = oof >= thr
        tp = int((fired & (y == 1)).sum())
        fp = int((fired & (y == 0)).sum())
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / y.sum()
        if prec >= PRECISION_FLOOR and rec >= MIN_RECALL:
            if best is None or rec > best["recall"]:
                best = {"threshold": round(float(thr), 3),
                        "precision": round(prec, 3), "recall": round(rec, 3),
                        "true_positives": tp, "false_positives": fp}
    if best is None:
        res["reason"] = (f"no operating point reached precision >= {PRECISION_FLOOR} "
                         f"with recall >= {MIN_RECALL} (LOO)")
        return res, None

    clf = new_clf()
    clf.fit(X, y)
    coef = clf.named_steps["logisticregression"].coef_[0]
    res.update({"usable": True, "features": names,
                "coefficients": {n: round(float(c), 3) for n, c in zip(names, coef)},
                "gold_prevalence": round(float(y.mean()), 4), **best})
    return res, (clf, names, best["threshold"])


def main() -> None:
    rows = {r["id"]: r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")}
    feats = {f["id"]: f for f in read_jsonl(MANIFESTS / "03_features.jsonl")}
    obs = list(read_jsonl(MANIFESTS / "07_nvv.jsonl"))

    nvv = nvv_calibration(obs, rows)

    cf = build_condition_features(rows, feats)
    cond, models, chosen = {}, {}, {}
    for name in DETECTORS:
        cond[name] = {"per_language": {}, "pooled": {}}
        for lang in ALL_LANGS:
            res, model = fit_detector(name, [lang], rows, cf, f"language:{lang}")
            cond[name]["per_language"][lang] = res
            if model:
                models[f"{name}|lang:{lang}"] = model
        for gname, fit_langs, _apply in GROUPING.get(name, []):
            res, model = fit_detector(name, fit_langs, rows, cf, f"pooled:{gname}")
            cond[name]["pooled"][gname] = res
            if model:
                models[f"{name}|pool:{gname}"] = model

        # one model per language: its own fit when that clears the floor,
        # otherwise the pooled fit it belongs to, with the better recall winning
        for lang in ALL_LANGS:
            cands = []
            own = cond[name]["per_language"][lang]
            if own.get("usable"):
                cands.append((own["recall"], f"lang:{lang}", own))
            for gname, fit_langs, apply_langs in GROUPING.get(name, []):
                g = cond[name]["pooled"].get(gname) or {}
                if g.get("usable") and lang in apply_langs:
                    cands.append((g["recall"], f"pool:{gname}", g))
            if not cands:
                continue
            cands.sort(reverse=True)
            _rec, key, res = cands[0]
            chosen[f"{name}|{lang}"] = {
                "model_key": f"{name}|{key}", "scope": res["scope"],
                "threshold": res["threshold"], "precision": res["precision"],
                "recall": res["recall"], "loo_auc": res.get("loo_auc"),
                "gold_prevalence": res["gold_prevalence"],
                "fitted_on": res["languages"],
                "out_of_group": lang not in res["languages"],
                "ships_as_tag": res["ships_as_tag"]}

    with open(MANIFESTS / "08_condition_models.pkl", "wb") as fh:
        pickle.dump({"models": models, "chosen": chosen}, fh)
    with open(MANIFESTS / "08_condition_features.pkl", "wb") as fh:
        pickle.dump(cf, fh)
    write_json(MANIFESTS / "08_nvv_policy.json", nvv)

    report = {
        "precision_floor": PRECISION_FLOOR,
        "min_recall": MIN_RECALL,
        "gold_set": {
            "reviewed_clips": sum(1 for r in rows.values() if r["human_reviewed"]),
            "event_tag_instances": dict(collections.Counter(
                e["tag"] for r in rows.values() for e in r["human_events"]).most_common()),
            "condition_tag_clips": dict(collections.Counter(
                c for r in rows.values()
                for c in r["human_notes"]["conditions"]).most_common()),
            "contamination_clips": dict(collections.Counter(
                c for r in rows.values()
                for c in r["human_notes"]["contamination"]).most_common()),
        },
        "nvv": nvv,
        "detectors": cond,
        "detector_selected_per_language": chosen,
        "detectors_usable": {
            t: sorted({k.split("|")[1] for k in chosen if k.startswith(t + "|")})
            for t in DETECTORS},
    }
    write_json(REPORTS / "08_calibration.json", report)
    print(json.dumps({"nvv_policy": nvv["policy"],
                      "nvv_evidence": nvv["evidence"],
                      "detectors_usable": report["detectors_usable"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
