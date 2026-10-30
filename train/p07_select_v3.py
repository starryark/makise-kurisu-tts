"""Apply SELECTION_RULE.md Amendment A (A1-A8) to eval/<suite> (written by p06_eval_v3.py).

  python p07_select_v3.py --suite v3 [--cand <key> ...] [--seed_bases 26000] [--e1 E1a,E1b]
                          [--B 4000] [--allow_partial] [--force_a1] [--check_a1]

Modes:
  screen   no --cand: every table key except "real" and every starrydark/Kurisu_Qwen3_TTS* key.
  final    --cand given, no --seed_bases: each candidate needs S >= 4 seed bases (A2/A8).
  re-test  --seed_bases 26000 --cand <winner>: the final pick at the fresh base (A2). A1 and the
           A3 null still come from the unfiltered v2 rows; A7 is skipped (S < 2, it is judged on
           the S >= 2 run).

Reads, per checkpoint directory eval/<suite>/<tag>/: scored.jsonl, ltas.npz,
scored_ced.jsonl, scored_accent.jsonl, cap.json; suite-level eval/real_ltas.npz,
eval/confirm_dedup.json, eval/e0/validity.json and eval/archive_v2_confirm/scored.jsonl
(v2's archived step-240 confirm run, for A1). Prints a per-gate verdict table and writes
eval/<suite>/selection.json.

Gates (numbers fixed in the rule; readings registered 2026-10-28):
  A1  reference = exactly the table key starrydark/Kurisu_Qwen3_TTS@92a08362... . Its seed-6000
      Auto confirm clips must reproduce the archived v2 run: >= 90% of the 148 clips within
      0.02 s on duration (ASR-text identity reported). Otherwise A1 FAILS, both CER means are
      printed and nothing is ranked (exit 1) unless --force_a1. The old CER check (0.0297 +-
      --a1_tol) is reported as harness_ok, secondary. E1a/E1b (--e1) give the training-seed
      noise: |paired mean E1a - E1b| per metric.
  A2  shipped mode vs the reference's shipped mode, the 148 confirm texts, at the requested seed
      bases (error if v2 or the candidate lacks one, unless --allow_partial; error if any text
      is missing). Two-stage paired bootstrap (texts resampled; each text's seeds resampled
      within it - registered reading). One-sided 90%: upper bound dCER <= +0.010, lower bound
      dSIM >= -0.015, runaway == 0. Language-trained candidates: mean dCER(Auto) <= +0.010.
      De-duplicated sub-score (texts without a train twin) reported alongside.
  A3  same rows: per-text (mean over seeds) 1/3-octave LTAS difference, per-band median over
      texts. RMS <= 1.0 dB, max |band| <= 2.0 dB, median per-text tilt difference within
      +-1.0 dB. Each threshold is replaced by the v2 seed-6000 vs seed-11000 null 95th
      percentile if that is larger (null per-text differences scaled by 1/sqrt(S) to match an
      S-seed average - registered reading; sign-flip + text bootstrap).
  A4  per language, never pooled; metrics valid per eval/e0/validity.json. Fail rule
      (registered): fails only if zh M1 (lid_ratio, valid per A4) is worse than v2 by more
      than the E1 spread, paired mean over common texts and seeds; every other accent metric
      is report-only; without --e1 A4 cannot fail ("pending E1"). zh gain claimed only if: a
      valid metric's one-sided 90% bound shows less accent, the improvement exceeds the E1
      spread (E1 required), shared-character SER not worse (mean dSER_shared <= 0) and zh SIM
      drop <= 0.03. English: report only, never claimed.
  A5  per extra speaker: projection of (row - kurisu, same text/seed, per-band median) onto
      (real row - real kurisu median LTAS) >= 0.5, and the row's JA CER <= 0.07. phone's
      projection is in-sample: reported only; phone keeps its CER guard.
  A6  per event token: mean CER(tagged) - CER(twin) <= 0.05, runaway rate not above the
      twins', CED detection rate at 5% FPR (threshold from this checkpoint's untagged
      shipped-mode confirm outputs). Reported; tokens stay experimental.
  A7  per [bg_<class>]: >= 30 prompts (generations: 15 texts x S - registered reading),
      detection >= 50% at 5% FPR, and A2 + A3 pass.
  A8  candidates passing A2-A5 are ranked by rule steps 1-3 on the paired means:
      runaway == 0, lowest mean dCER, ties within 0.01 -> highest mean dSIM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
SELECT_KINDS = ("ja_heldout", "ja_novel")
SELECT_N = 148
V2_REPO = "starrydark/Kurisu_Qwen3_TTS"
V2_PIN = V2_REPO + "@92a08362cb2c95f7dfc8a7ecf4cc00db625f9d57"
A1_CER, A1_TOL = 0.0297, 0.003
A1_BASE, A1_DUR_TOL, A1_MATCH = 6000, 0.02, 0.90
A2_CER, A2_SIM, A2_AUTO = 0.010, -0.015, 0.010
A3_RMS, A3_MAX, A3_TILT = 1.0, 2.0, 1.0
A4_SIM_DROP = -0.03
A4_M1 = "lid_ratio"
A5_PROJ, A5_CER = 0.5, 0.07
A5_REPORT_ONLY = ("phone",)
A6_DCER = 0.05
A7_MIN_PROMPTS, A7_DET = 30, 0.5
FINAL_MIN_S = 4
FPR = 0.05
TIE = 0.01
Q = 90                      # one-sided 90% bound
M23 = ("ser", "ser_shared", "ter", "ter_strict")


def fail(msg: str):
    raise SystemExit(f"p07: ERROR: {msg}")


# ----------------------------------------------------------------- loading
def tag_of(ckpt: str) -> str:
    return "real" if ckpt == "real" else ckpt.rstrip("/").replace(":", "_").replace("/", "__")[-80:]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_ckpt(d: Path, seed_filter: set | None = None) -> dict:
    rows = read_jsonl(d / "scored.jsonl")
    if seed_filter:
        rows = [r for r in rows if r.get("seed_base") in seed_filter or r.get("mode") == "real"]
    ltas = {}
    if (d / "ltas.npz").exists():
        z = np.load(d / "ltas.npz")
        ltas = {str(p): (L, float(t)) for p, L, t in zip(z["pid"], z["ltas"], z["tilt"])}
    cap = json.loads((d / "cap.json").read_text(encoding="utf-8")) if (d / "cap.json").exists() else {}
    return {"rows": rows, "by_pid": {r["pid"]: r for r in rows}, "ltas": ltas,
            "ced": {r["pid"]: r["ced"] for r in read_jsonl(d / "scored_ced.jsonl")},
            "accent": {r["pid"]: r for r in read_jsonl(d / "scored_accent.jsonl")},
            "cap": cap, "shipped": cap.get("shipped_mode", "auto"), "dir": str(d)}


def ref_key_of(table: dict, ref: str) -> str:
    """The v2 reference: exactly one table key equal to the pinned id (also by directory)."""
    if ref != V2_PIN:
        fail(f"--ref must be the registered reference {V2_PIN} (got {ref})")
    hits = [k for k in table if k == ref or tag_of(k) == tag_of(ref)]
    if len(hits) != 1:
        fail(f"reference {ref} matches {len(hits)} table keys {hits}; v2 keys present: "
             f"{[k for k in table if k.startswith(V2_REPO)]}")
    return hits[0]


def resolve_key(table: dict, name: str, what: str) -> str:
    """Exact table key, else the unique key containing `name`; error on 0 or > 1."""
    if name in table:
        return name
    hits = [k for k in table if name in k]
    if len(hits) != 1:
        fail(f"{what} '{name}' matches {len(hits)} table keys: {hits}")
    return hits[0]


def is_runaway(r: dict) -> bool:
    expected = r["dur"] / max(r["dur_ratio"], 1e-6)
    return r["dur"] >= 31.0 or (r["dur_ratio"] > 5.0 and r["dur"] - expected > 6.5)


def is_select(r: dict) -> bool:
    return r.get("group") == "confirm" and r.get("kind") in SELECT_KINDS


def select_bases(ck: dict, mode: str) -> list:
    return sorted({r.get("seed_base") for r in ck["rows"] if is_select(r) and r.get("mode") == mode})


def coverage(cand: dict, ref: dict, name: str, req: list | None, allow_partial: bool) -> list:
    """Seed bases used for this candidate's A2/A3. Requested bases = --seed_bases, else the
    candidate's own. Error if the candidate or v2 lacks a requested base (unless
    allow_partial) or if any of the 148 select texts is missing at a used base."""
    cb, rb = select_bases(cand, cand["shipped"]), select_bases(ref, ref["shipped"])
    req = sorted(req) if req else cb
    if not req:
        fail(f"{name}: no confirm select rows in shipped mode '{cand['shipped']}'")
    lost = sorted(set(req) - (set(cb) & set(rb)))
    if lost:
        msg = f"{name}: seed bases {lost} requested but missing (candidate has {cb}, v2 has {rb})"
        if not allow_partial:
            fail(msg + "; generate them, or pass --allow_partial")
        print(f"!! {msg}; --allow_partial: using the common bases")
    use = sorted(set(req) & set(cb) & set(rb))
    if not use:
        fail(f"{name}: no common seed bases with v2")
    tids = sorted({r["tid"] for ck in (cand, ref) for r in ck["rows"] if is_select(r)})
    if len(tids) != SELECT_N:
        fail(f"{name}: {len(tids)} confirm select texts found, expected {SELECT_N}")
    modes = [cand["shipped"]] + (["auto"] if cand["shipped"] != "auto" else [])
    missing = []
    for ck, who, ms in ((cand, name, modes), (ref, "v2", [ref["shipped"]])):
        have = {(r["tid"], r.get("mode"), r.get("seed_base")) for r in ck["rows"] if is_select(r)}
        missing += [f"{who}: {t}~{m}~s{b}" for m in ms for b in use for t in tids if (t, m, b) not in have]
    if missing:
        fail(f"{len(missing)} confirm clips missing (texts are never dropped silently):\n  " + "\n  ".join(missing))
    return use


# ----------------------------------------------------------------- statistics
def matrix(cand: dict, ref: dict, value, cand_key, ref_key, seeds: list | None = None,
           strict: bool = False) -> tuple[np.ndarray, list, list]:
    """Paired differences D[text, seed] = value(cand row) - value(ref row).
    seeds: the seed bases to use (default: those both have). Texts not complete at every seed
    are dropped - unless strict, which raises listing them instead.
    cand_key / ref_key(row) -> (tid, seed_base) or None to exclude the row."""
    c, r = {}, {}
    for row in cand["rows"]:
        k = cand_key(row)
        if k is not None:
            v = value(cand, row)
            if v is not None:
                c[k] = v
    for row in ref["rows"]:
        k = ref_key(row)
        if k is not None:
            v = value(ref, row)
            if v is not None:
                r[k] = v
    seeds = sorted({s for _, s in c} & {s for _, s in r}) if seeds is None else list(seeds)
    alltids = sorted({t for t, _ in c} | {t for t, _ in r})
    tids = [t for t in alltids if all((t, s) in c and (t, s) in r for s in seeds)]
    if strict and len(tids) < len(alltids):
        miss = [f"{t}~s{s} ({'candidate' if (t, s) not in c else 'v2'})"
                for t in alltids for s in seeds if (t, s) not in c or (t, s) not in r]
        fail(f"{len(miss)} paired values missing:\n  " + "\n  ".join(miss))
    if not seeds or not tids:
        return np.zeros((0, 0)), [], seeds
    D = np.array([[np.asarray(c[(t, s)]) - np.asarray(r[(t, s)]) for s in seeds] for t in tids], dtype=float)
    return D, tids, seeds


def boot_mean(D: np.ndarray, B: int = 4000, seed: int = 0) -> np.ndarray:
    """Two-stage bootstrap of the grand mean of D[text, seed]: resample texts, then
    resample seeds within each resampled text."""
    rng = np.random.default_rng(seed)
    T, S = D.shape
    out = np.empty(B)
    for i in range(0, B, 500):
        b = min(500, B - i)
        it = rng.integers(0, T, (b, T))
        isd = rng.integers(0, S, (b, T, S))
        out[i:i + b] = D[it[..., None], isd].mean(axis=(1, 2))
    return out


def bound(D: np.ndarray, upper: bool, B: int = 4000, seed: int = 0) -> tuple[float, float]:
    """(mean, one-sided 90% bound): upper bound if upper else lower bound."""
    dist = boot_mean(D, B, seed)
    return float(D.mean()), float(np.percentile(dist, Q if upper else 100 - Q))


def ltas_stats(dL: np.ndarray, dT: np.ndarray) -> dict:
    """dL [texts, bands] per-text LTAS differences; dT [texts] tilt differences."""
    m = np.median(dL, 0)
    return {"rms": float(np.sqrt(np.mean(m ** 2))), "max": float(np.max(np.abs(m))),
            "tilt": float(np.median(dT)), "bands": [round(float(x), 2) for x in m]}


def ltas_null(nL: np.ndarray, nT: np.ndarray, s_cand: int, B: int = 2000, seed: int = 1) -> dict:
    """95th percentiles of the A3 statistics under H0 from the reference's own seed-A minus
    seed-B per-text differences, scaled to an s_cand-seed average; sign-flip + bootstrap."""
    rng = np.random.default_rng(seed)
    k = 1.0 / math.sqrt(max(s_cand, 1))
    T = len(nL)
    rms, mx, tl = [], [], []
    for _ in range(B):
        it = rng.integers(0, T, T)
        sg = rng.choice([-1.0, 1.0], T)
        m = np.median(sg[:, None] * nL[it] * k, 0)
        rms.append(np.sqrt(np.mean(m ** 2)))
        mx.append(np.max(np.abs(m)))
        tl.append(abs(np.median(sg * nT[it] * k)))
    return {"rms_p95": float(np.percentile(rms, 95)), "max_p95": float(np.percentile(mx, 95)),
            "tilt_p95": float(np.percentile(tl, 95)), "scaled_for_S": s_cand}


# ----------------------------------------------------------------- gates
def key_confirm(mode: str, tids: set | None = None):
    def f(r):
        if is_select(r) and r.get("mode") == mode and (tids is None or r["tid"] in tids):
            return r["tid"], r.get("seed_base")
        return None
    return f


def key_accent(lang: str):
    return lambda r: (r["tid"], r.get("seed_base")) if r.get("group") == "accent" and r.get("lang") == lang else None


def val_metric(m):
    return lambda ck, r: r.get(m)


def acc_value(m):
    """Accent metric of a row; M2/M3 metrics only where the reference is valid (m23_valid)."""
    def f(ck, r):
        a = ck["accent"].get(r["pid"])
        if a is None or (m in M23 and not a.get("m23_valid")):
            return None
        return a.get(m)
    return f


def load_archive(path: Path) -> dict:
    """Archived v2 confirm run (pids ja_test_XX / ja_novel_XX = v3 tids), select kinds only."""
    return {r["pid"]: r for r in read_jsonl(path) if r.get("kind") in SELECT_KINDS}


def a1(ref_all: dict, archive: dict, tol: float, e1: tuple | None) -> dict:
    """Per-clip reproduction of v2's archived confirm run by the v2 rows at seed base 6000,
    Auto (always the unfiltered reference rows)."""
    regen = {r["tid"]: r for r in ref_all["rows"] if is_select(r) and r.get("mode") == "auto"
             and r.get("seed_base") == A1_BASE}
    out = {"archive_n": len(archive), "regen_n": len(regen), "dur_tol_s": A1_DUR_TOL, "min_match": A1_MATCH}
    common = [p for p in archive if p in regen]
    if archive:
        dur_ok = [p for p in common if abs(float(regen[p]["dur"]) - float(archive[p]["dur"])) <= A1_DUR_TOL]
        asr_ok = [p for p in common if (regen[p].get("asr") or "").strip() == (archive[p].get("asr") or "").strip()]
        out.update(dur_match=round(len(dur_ok) / len(archive), 4), asr_identical=round(len(asr_ok) / len(archive), 4),
                   missing_in_regen=sorted(set(archive) - set(regen)),
                   cer_archive=round(float(np.mean([archive[p]["cer"] for p in archive])), 4),
                   cer_regen=round(float(np.mean([regen[p]["cer"] for p in common])), 4) if common else None)
        out["ok"] = out["dur_match"] >= A1_MATCH
        if not out["ok"]:
            out["mismatched"] = sorted(set(archive) - set(dur_ok))
    else:
        out.update(ok=False, reason="archive eval/archive_v2_confirm/scored.jsonl missing or empty")
        out["cer_regen"] = round(float(np.mean([r["cer"] for r in regen.values()])), 4) if regen else None
    out["ref_cer_6000"] = out.get("cer_regen")
    out["harness_ok"] = out["ref_cer_6000"] is not None and abs(out["ref_cer_6000"] - A1_CER) <= tol
    out["harness_note"] = f"secondary: CER {out['ref_cer_6000']} vs {A1_CER} +- {tol}"
    for b in sorted({r.get("seed_base") for r in ref_all["rows"] if r.get("seed_base")}):
        s = [r for r in ref_all["rows"] if is_select(r) and r.get("mode") == ref_all["shipped"] and r.get("seed_base") == b]
        if s:
            out[f"ref_s{b}"] = {"cer": round(float(np.mean([r["cer"] for r in s])), 4),
                                "sim": round(float(np.mean([r["sim"] for r in s])), 4), "n": len(s)}
    if e1:
        a, b = e1
        noise = {}
        for m in ("cer", "sim"):
            D, tids, seeds = matrix(a, b, val_metric(m), key_confirm(a["shipped"]), key_confirm(b["shipped"]))
            if D.size:
                noise[m] = round(abs(float(D.mean())), 4)
        out["training_seed_noise"] = noise
    return out


def a2(cand: dict, ref: dict, dedup: dict, B: int, use: list) -> dict:
    out = {"shipped_mode": cand["shipped"]}
    keep = None
    if dedup:
        keep = {pid for pid, v in dedup.items() if not v.get("twin")}
    for label, tids in (("all", None), ("dedup", keep)):
        if label == "dedup" and tids is None:
            out[label] = {"status": "n/a (eval/confirm_dedup.json missing)"}
            continue
        res = {}
        for m, upper, lim in (("cer", True, A2_CER), ("sim", False, A2_SIM)):
            D, T, S = matrix(cand, ref, val_metric(m), key_confirm(cand["shipped"], tids),
                             key_confirm(ref["shipped"], tids), seeds=use, strict=True)
            if not D.size:
                res[m] = {"status": "n/a (no common seeds/texts)"}
                continue
            mean, bd = bound(D, upper, B)
            ok = bd <= lim if upper else bd >= lim
            res[m] = {"mean": round(mean, 4), "bound90": round(bd, 4), "limit": lim, "pass": bool(ok),
                      "n_texts": len(T), "seed_bases": S}
        out[label] = res
    sel = [r for r in cand["rows"] if is_select(r) and r.get("mode") == cand["shipped"]]
    out["runaway"] = int(sum(is_runaway(r) for r in sel))
    out["auto_screen"] = None
    if cand["shipped"] != "auto":
        D, T, S = matrix(cand, ref, val_metric("cer"), key_confirm("auto"), key_confirm(ref["shipped"]),
                         seeds=use, strict=True)
        out["auto_screen"] = {"mean_dcer": round(float(D.mean()), 4) if D.size else None,
                              "pass": bool(D.size and D.mean() <= A2_AUTO), "n_texts": len(T)}
    a = out["all"]
    out["pass"] = bool(a.get("cer", {}).get("pass") and a.get("sim", {}).get("pass") and out["runaway"] == 0
                       and (out["auto_screen"] is None or out["auto_screen"]["pass"]))
    return out


def a3(cand: dict, ref: dict, ref_all: dict, B: int, use: list) -> dict:
    """ref: the (possibly seed-filtered) reference rows paired with the candidate; ref_all: the
    unfiltered reference rows, whose seed-6000 vs seed-11000 difference is the null."""
    def lt(ck, r):
        x = ck["ltas"].get(r["pid"])
        return None if x is None else np.append(x[0], x[1])     # bands + tilt as last element
    D, T, S = matrix(cand, ref, lt, key_confirm(cand["shipped"]), key_confirm(ref["shipped"]),
                     seeds=use, strict=True)
    if not D.size:
        return {"status": "n/a", "pass": False}
    per_text = D.mean(1)                                        # [texts, bands+1]
    st = ltas_stats(per_text[:, :-1], per_text[:, -1])
    rseeds = sorted({r.get("seed_base") for r in ref_all["rows"] if r.get("seed_base")})
    thr = {"rms": A3_RMS, "max": A3_MAX, "tilt": A3_TILT}
    null = None
    if len(rseeds) >= 2:
        s1, s2 = rseeds[:2]
        kc = key_confirm(ref_all["shipped"])
        a = {r["tid"]: lt(ref_all, r) for r in ref_all["rows"] if kc(r) and r["seed_base"] == s1}
        b = {r["tid"]: lt(ref_all, r) for r in ref_all["rows"] if kc(r) and r["seed_base"] == s2}
        common = sorted(t for t in set(a) & set(b) if a[t] is not None and b[t] is not None)
        if common:
            N = np.stack([a[t] - b[t] for t in common])
            null = ltas_null(N[:, :-1], N[:, -1], len(S))
            null["seed_bases"] = [s1, s2]
            thr = {"rms": max(A3_RMS, null["rms_p95"]), "max": max(A3_MAX, null["max_p95"]),
                   "tilt": max(A3_TILT, null["tilt_p95"])}
    ok = st["rms"] <= thr["rms"] and st["max"] <= thr["max"] and abs(st["tilt"]) <= thr["tilt"]
    return {**{k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()},
            "thresholds": {k: round(v, 3) for k, v in thr.items()}, "null": null,
            "n_texts": len(T), "seed_bases": S, "pass": bool(ok)}


def a4(cand: dict, ref: dict, validity: dict, e1_acc: dict | None, B: int) -> dict:
    out = {"pass": True, "claims": []}
    for lang in ("zh", "en"):
        res = {}
        key = key_accent(lang)
        for m, v in (validity.get(lang) or {}).items():
            sgn = 1 if v.get("sign", "+").startswith("+") else -1
            D, T, S = matrix(cand, ref, acc_value(m), key, key)
            if not D.size:
                res[m] = {"valid": bool(v.get("valid")), "sign": sgn, "status": "n/a"}
                continue
            dist = boot_mean(D, B)
            mean = float(D.mean())
            # bounds on the "more accent" side: sgn*delta upper bound < 0 => less accent
            hi = float(np.percentile(sgn * dist, Q))
            lo = float(np.percentile(sgn * dist, 100 - Q))
            r = {"valid": bool(v.get("valid")), "sign": sgn, "mean": round(mean, 4),
                 "accent_dir_bounds90": [round(lo, 4), round(hi, 4)],
                 "less_accent_sig": hi < 0, "more_accent_sig": lo > 0, "n_texts": len(T), "seed_bases": S}
            r["e1_noise"] = (e1_acc or {}).get(lang, {}).get(m)
            res[m] = r
        sim_D, _, _ = matrix(cand, ref, val_metric("sim"), key, key)
        cer = [r["cer"] for r in cand["rows"] if key(r)]
        res["_guards"] = {"mean_dsim": round(float(sim_D.mean()), 4) if sim_D.size else None,
                          "cer_mean": round(float(np.mean(cer)), 4) if cer else None,
                          "runaway": int(sum(is_runaway(r) for r in cand["rows"] if key(r)))}
        if lang == "zh":
            sim_ok = bool(sim_D.size and sim_D.mean() >= A4_SIM_DROP)
            ss = res.get("ser_shared", {})
            ser_ok = ss.get("mean") is not None and ss["mean"] <= 0
            for m, r in res.items():
                if m.startswith("_") or not r.get("valid") or "mean" not in r:
                    continue
                beyond = r["e1_noise"] is not None and r["sign"] * r["mean"] < -r["e1_noise"]
                if r["less_accent_sig"] and beyond and ser_ok and sim_ok:
                    out["claims"].append(f"zh:{m}")
            res["_guards"].update(sim_ok=sim_ok, ser_shared_not_worse=bool(ser_ok))
            # registered fail rule: zh M1 worse than v2 by more than the E1 spread
            m1, v1 = res.get(A4_M1) or {}, (validity.get("zh") or {}).get(A4_M1) or {}
            if not v1.get("valid"):
                out["gate"] = "M1 not valid per A4 (e0/validity.json): A4 cannot fail"
            elif "mean" not in m1:
                out["pass"], out["gate"] = False, "FAIL: no paired zh M1 values (accent rows missing)"
            elif m1["e1_noise"] is None:
                out["gate"] = "pending E1 (run with --e1): A4 cannot fail"
            else:
                worse = m1["sign"] * m1["mean"] > m1["e1_noise"]
                out["pass"] = not worse
                out["gate"] = (f"{'FAIL' if worse else 'ok'}: zh M1 paired mean {m1['mean']:+.4f} "
                               f"vs E1 spread {m1['e1_noise']} ({'more' if worse else 'not more'} accented)")
        else:
            res["_note"] = "no valid English accent metric is claimed (A4)"
        out[lang] = res
    if not validity:
        out["note"] = "eval/e0/validity.json missing - no metric validated, no claim possible"
    return out


def a5(cand: dict, real: dict) -> dict:
    out = {"pass": True, "rows": {}}
    spk = [s for s in cand["cap"].get("speakers", []) if s != "kurisu"]
    if not spk:
        out["status"] = "n/a (no extra speakers)"
        return out
    kur = {(r["tid"], r["seed_base"]): r["pid"] for r in cand["rows"]
           if r.get("group") == "confirm" and r.get("mode") == cand["shipped"]}
    for s in spk:
        rows = [r for r in cand["rows"] if r.get("group") == "cond" and r.get("speaker") == s]
        per = {}
        for r in rows:
            k = kur.get((r["tid"], r["seed_base"]))
            if k and r["pid"] in cand["ltas"] and k in cand["ltas"]:
                per.setdefault(r["tid"], []).append(cand["ltas"][r["pid"]][0] - cand["ltas"][k][0])
        res = {"n_rows": len(rows), "cer_mean": round(float(np.mean([r["cer"] for r in rows])), 4) if rows else None}
        if per:
            dm = np.median(np.stack([np.mean(v, 0) for v in per.values()]), 0)
            res["model_diff_db"] = [round(float(x), 2) for x in dm]
            if s in real and "kurisu" in real:
                dr = real[s]["median_ltas"] - real["kurisu"]["median_ltas"]
                res["real_diff_db"] = [round(float(x), 2) for x in dr]
                res["projection"] = round(float(dm @ dr / (dr @ dr)), 3) if float(dr @ dr) > 0 else None
        cer_ok = res["cer_mean"] is not None and res["cer_mean"] <= A5_CER
        proj_ok = res.get("projection") is not None and res["projection"] >= A5_PROJ
        res.update(cer_ok=bool(cer_ok), projection_ok=bool(proj_ok))
        if s in A5_REPORT_ONLY:
            res["pass"] = bool(cer_ok)
            res["note"] = "in-sample reference takes: projection reported only (A5, A9/E5); CER guard applies"
        else:
            res["pass"] = bool(proj_ok and cer_ok)
        out["pass"] &= res["pass"]
        out["rows"][s] = res
    return out


def _detect(cand: dict, tok: str, rows: list[dict]) -> dict:
    import ced_tagger
    neg = [ced_tagger.class_score(cand["ced"][r["pid"]], tok) for r in cand["rows"]
           if is_select(r) and r.get("mode") == cand["shipped"] and r["pid"] in cand["ced"]]
    thr = ced_tagger.fpr_threshold(neg, FPR)
    sc = [ced_tagger.class_score(cand["ced"][r["pid"]], tok) for r in rows if r["pid"] in cand["ced"]]
    sc = [x for x in sc if x is not None]
    k = int(sum(x > thr for x in sc))
    return {"threshold": round(thr, 5) if thr == thr else None, "n_neg": len(neg), "n": len(sc),
            "rate": round(k / len(sc), 3) if sc else None, "ci95": ced_tagger.wilson(k, len(sc))}


def a6(cand: dict) -> dict:
    out = {}
    twins = {(r["tid"], r["seed_base"]): r for r in cand["rows"] if r.get("kind") == "tag_twin"}
    for tok in cand["cap"].get("tag_tokens", []):
        rows = [r for r in cand["rows"] if r.get("group") == "tags" and r.get("token") == tok]
        pairs = [(r, twins[(r["twin"], r["seed_base"])]) for r in rows if (r["twin"], r["seed_base"]) in twins]
        dcer = float(np.mean([a["cer"] - b["cer"] for a, b in pairs])) if pairs else None
        ra_t = float(np.mean([is_runaway(r) for r in rows])) if rows else None
        ra_w = float(np.mean([is_runaway(r) for r in twins.values()])) if twins else None
        det = _detect(cand, tok, rows)
        tw = _detect(cand, tok, list(twins.values()))
        out[tok] = {"n": len(rows), "dcer": None if dcer is None else round(dcer, 4),
                    "dcer_ok": dcer is not None and dcer <= A6_DCER,
                    "runaway_rate": ra_t, "twin_runaway_rate": ra_w,
                    "runaway_ok": ra_t is not None and ra_w is not None and ra_t <= ra_w,
                    "detection": det, "twin_detection": tw["rate"], "status": "experimental (reported)"}
    return out


def a7(cand: dict, a2_pass: bool, a3_pass: bool) -> dict:
    """n counts generations (15 texts x S seed bases): the registered reading of '>= 30 prompts'."""
    out = {}
    for tok in cand["cap"].get("sfx_tokens", []):
        rows = [r for r in cand["rows"] if r.get("group") == "sfx" and r.get("token") == tok]
        det = _detect(cand, tok, rows)
        enough = det["n"] >= A7_MIN_PROMPTS
        ships = bool(enough and det["rate"] is not None and det["rate"] >= A7_DET and a2_pass and a3_pass)
        out[tok] = {**det, "enough_prompts": enough, "ships": ships}
    return out


def rank(results: dict) -> list:
    """A8 / rule steps 1-3 on candidates passing A2-A5: runaway == 0, lowest mean dCER,
    ties within 0.01 -> highest mean dSIM."""
    ok = {k: v for k, v in results.items() if v["A2"]["pass"] and v["A3"]["pass"] and v["A4"]["pass"]
          and v["A5"]["pass"] and v["A2"]["runaway"] == 0}
    if not ok:
        return []
    best = min(v["A2"]["all"]["cer"]["mean"] for v in ok.values())
    tied = [k for k, v in ok.items() if v["A2"]["all"]["cer"]["mean"] <= best + TIE]
    rest = sorted((k for k in ok if k not in tied), key=lambda k: ok[k]["A2"]["all"]["cer"]["mean"])
    tied.sort(key=lambda k: -ok[k]["A2"]["all"]["sim"]["mean"])
    return tied + rest


def e1_accent_noise(e1: tuple, validity: dict) -> dict:
    """|paired mean E1a - E1b| per language and accent metric (same rows/filters as A4)."""
    out = {}
    for lang in ("zh", "en"):
        for m in (validity.get(lang) or {}):
            D, _, _ = matrix(e1[0], e1[1], acc_value(m), key_accent(lang), key_accent(lang))
            if D.size:
                out.setdefault(lang, {})[m] = round(abs(float(D.mean())), 4)
    return out


def load_suite_inputs() -> tuple[dict, dict, dict]:
    dedup_p = WORK / "eval" / "confirm_dedup.json"
    dedup = json.loads(dedup_p.read_text(encoding="utf-8")) if dedup_p.exists() else {}
    val_p = WORK / "eval" / "e0" / "validity.json"
    validity = json.loads(val_p.read_text(encoding="utf-8")) if val_p.exists() else {}
    real_p = WORK / "eval" / "real_ltas.npz"
    real = np.load(real_p, allow_pickle=True)["data"].item() if real_p.exists() else {}
    return dedup, validity, real


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="v3")
    ap.add_argument("--ref", default=V2_PIN, help="table key of v2 (must be the pinned id)")
    ap.add_argument("--cand", action="append", default=None,
                    help="table keys (final mode: S >= 4 each); default all but real / v2 keys")
    ap.add_argument("--seed_bases", default=None, help="re-test: restrict to these seed bases (26000)")
    ap.add_argument("--e1", default=None, help="E1a,E1b: exact keys or unique substrings (training-seed noise)")
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--a1_tol", type=float, default=A1_TOL, help="secondary A1 CER tolerance (harness_ok)")
    ap.add_argument("--allow_partial", action="store_true", help="allow v2/candidate lacking requested seed bases")
    ap.add_argument("--force_a1", action="store_true", help="rank even if A1 fails (reported)")
    ap.add_argument("--check_a1", action="store_true", help="only the A1 harness check; exit 1 on failure")
    args = ap.parse_args()
    suite = WORK / "eval" / args.suite
    table = json.loads((suite / "table.json").read_text(encoding="utf-8"))
    sf_ = sorted({int(s) for s in args.seed_bases.split(",")}) if args.seed_bases else None
    mode = "retest" if sf_ else "final" if args.cand else "screen"
    ref_key = ref_key_of(table, args.ref)
    ref_all = load_ckpt(suite / tag_of(ref_key))
    ref = load_ckpt(suite / tag_of(ref_key), set(sf_)) if sf_ else ref_all
    if args.cand:
        bad = [k for k in args.cand if k == "real" or k.startswith(V2_REPO)]
        if bad:
            fail(f"--cand may not name the reference / real / any {V2_REPO}* key: {bad}")
        cands = [resolve_key(table, k, "--cand") for k in args.cand]
    else:
        cands = [k for k in table if k != "real" and not k.startswith(V2_REPO)]
    dedup, validity, real = load_suite_inputs()

    e1 = e1_acc = None
    if args.e1:
        names = [n for n in args.e1.split(",") if n]
        if len(names) != 2:
            fail(f"--e1 needs two runs (E1a,E1b), got {names}")
        found = [resolve_key(table, n, "--e1") for n in names]
        if found[0] == found[1]:
            fail(f"--e1 names the same key twice: {found}")
        e1 = tuple(load_ckpt(suite / tag_of(k)) for k in found)     # E1 noise: all their seed bases
        e1_acc = e1_accent_noise(e1, validity)

    archive = load_archive(WORK / "eval" / "archive_v2_confirm" / "scored.jsonl")
    res_a1 = a1(ref_all, archive, args.a1_tol, e1)
    a1_line = (f"A1 harness: v2@{A1_BASE} dur match {res_a1.get('dur_match')} (>= {A1_MATCH} within "
               f"{A1_DUR_TOL}s; ASR identical {res_a1.get('asr_identical')}) -> {'OK' if res_a1['ok'] else 'FAIL'}; "
               f"{res_a1['harness_note']} -> {'ok' if res_a1['harness_ok'] else 'off'}")
    if not res_a1["ok"]:
        print(f"!! A1 FAILED: {res_a1['reason'] + '; ' if res_a1.get('reason') else ''}CER regenerated {res_a1.get('cer_regen')} vs archived "
              f"{res_a1.get('cer_archive')}; dur match {res_a1.get('dur_match')}. Debug the harness first.")
    if args.check_a1:
        print(a1_line)
        raise SystemExit(0 if res_a1["ok"] else 1)
    if not res_a1["ok"] and not args.force_a1:
        (suite / "selection.json").write_text(json.dumps({"suite": args.suite, "ref": ref_key, "A1": res_a1,
                                                          "aborted": "A1 failed (no --force_a1)"},
                                                         indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        print(a1_line)
        raise SystemExit(1)

    results = {}
    for k in cands:
        ck = load_ckpt(suite / tag_of(k), set(sf_) if sf_ else None)
        use = coverage(ck, ref, k, sf_, args.allow_partial)
        if mode == "final" and len(use) < FINAL_MIN_S:
            fail(f"{k}: final mode needs S >= {FINAL_MIN_S} seed bases, has {use}")
        r2 = a2(ck, ref, dedup, args.B, use)
        r3 = a3(ck, ref, ref_all, args.B, use)
        res = {"seed_bases": use, "A2": r2, "A3": r3, "A4": a4(ck, ref, validity, e1_acc, args.B),
               "A5": a5(ck, real), "A6": a6(ck)}
        if len(use) < 2:
            res["A7"], res["A7_note"] = {}, f"skipped: S={len(use)} < 2 (re-test); A7 is judged on the S >= 2 run"
        else:
            res["A7"] = a7(ck, r2["pass"], r3["pass"])
        results[k] = res
    order = rank(results)
    out = {"suite": args.suite, "mode": mode, "ref": ref_key, "seed_filter": sf_, "A1": res_a1,
           "e1_accent_noise": e1_acc, "results": results, "ranking": order, "winner": order[0] if order else None,
           "notes": ["A4 fail rule (registered): zh M1 worse than v2 by more than the E1 spread; other metrics report-only",
                     "A5 passes trivially for checkpoints without extra speakers; phone projection report-only",
                     "A7 prompts = generations (15 texts x S), registered",
                     "Step 4 (zero-shot clone guard) is not applied: A8 uses steps 1-3"]}
    if not res_a1["ok"]:
        out["notes"].append("A1 FAILED - ranked only because of --force_a1")
    (suite / "selection.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    def f(x, d=4):
        return "-" if x is None else f"{x:+.{d}f}" if isinstance(x, float) else str(x)
    print(f"\n{a1_line}\ntraining-seed noise {res_a1.get('training_seed_noise')}; accent {e1_acc}; mode {mode}")
    print(f"{'candidate':42s} {'A2 dCER(ub90)':>15s} {'dSIM(lb90)':>12s} {'run':>4s} {'auto':>6s} {'A2':>4s} "
          f"{'A3 rms/max/tilt':>18s} {'A3':>4s} {'A4':>4s} {'A5':>4s}")
    for k, v in results.items():
        c, s = v["A2"]["all"].get("cer", {}), v["A2"]["all"].get("sim", {})
        a3r = v["A3"]
        auto = v["A2"]["auto_screen"]
        print(f"{k[-42:]:42s} {f(c.get('mean'))}({f(c.get('bound90'))}) {f(s.get('bound90'))} "
              f"{v['A2']['runaway']:>4d} {('-' if auto is None else 'ok' if auto['pass'] else 'FAIL'):>6s} "
              f"{'ok' if v['A2']['pass'] else 'FAIL':>4s} "
              f"{a3r.get('rms', 0):.2f}/{a3r.get('max', 0):.2f}/{a3r.get('tilt', 0):+.2f}".rjust(18) +
              f" {'ok' if a3r['pass'] else 'FAIL':>4s} {'ok' if v['A4']['pass'] else 'FAIL':>4s} "
              f"{'ok' if v['A5']['pass'] else 'FAIL':>4s}   S={len(v['seed_bases'])}")
        print(f"    A4: {v['A4'].get('gate')}")
        for spk, r in v["A5"].get("rows", {}).items():
            print(f"    A5 {spk}: projection {r.get('projection')} cer {r.get('cer_mean')} -> "
                  f"{'ok' if r['pass'] else 'FAIL'}{' (projection report-only)' if spk in A5_REPORT_ONLY else ''}")
        for tok, t in v["A6"].items():
            print(f"    A6 {tok}: dCER {t['dcer']}  runaway {t['runaway_rate']} vs twin {t['twin_runaway_rate']}  "
                  f"detect {t['detection']['rate']} {t['detection']['ci95']} (twins {t['twin_detection']})  experimental")
        if v.get("A7_note"):
            print(f"    A7 {v['A7_note']}")
        for tok, t in v["A7"].items():
            print(f"    A7 {tok}: n {t['n']}  detect {t['rate']} {t['ci95']}  -> {'ships' if t['ships'] else 'cut'}")
        if v["A4"]["claims"]:
            print(f"    A4 accent claims (pending blind rating): {v['A4']['claims']}")
    print(f"\nranking (A8, steps 1-3 on passing candidates): {order}")
    print(f"wrote {suite / 'selection.json'}")


if __name__ == "__main__":
    main()
