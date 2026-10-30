"""A9 (adoption rules) and A10 (E0 gate) of SELECTION_RULE.md, applied mechanically.

  python p12_adopt.py --suite v3 --e1 E1a,E1b [--e2 K] [--e3 K] [--e3f K] [--e4 K --e4_parent K]
                      [--e5 K] [--e6 K] [--e7 K] [--B 4000] [--allow_partial] [--force_a1]

Run keys are table.json keys of eval/<suite> (exact, or a substring matching exactly one key).
Reads eval/<suite>/table.json and each run's scored files (via p07_select_v3), eval/e0
(validity.json, table.json, B/D/F scored_accent.jsonl) and the A1 archive; writes
eval/<suite>/adoption.json and prints ADOPT / REJECT / SKIP per run with reasons.

Definitions (registered 2026-10-28):
  screen      A2 at the run's seed bases (S = 2 for ablations) vs v2 at the same bases, as in
              p07: CER bound <= +0.010, SIM bound >= -0.015, runaway == 0, and mean Auto dCER
              <= +0.010 (language-trained runs: the Auto confirm rows; others: shipped = Auto).
  E1          per (text, seed) mean of E1a and E1b; E1 spread = |paired mean E1a - E1b| on the
              same metric and rows.
  zh M1       lid_ratio = log P(ja) - log P(zh) on accent rows, lang zh (higher = more Japanese),
              valid per A4 (eval/e0/validity.json). "Improves on X by more than the E1 spread":
              paired mean (run - X) < -spread.
  zh SER      shared-character syllable error rate (ser_shared, m23_valid rows): "not worse" =
              paired mean (run - E1) <= 0.
  zh SIM      scored.jsonl sim on the accent zh rows: "drops <= 0.03" = paired mean >= -0.03.
Rules: E2/E3 adopt if screen + M1 improves on E1 + SER not worse (both pass -> E3); E3f adopts
over E3 if M1 improves on E3 + screen; E4 only if the A10 gate is met, adopts if M1 improves on
its parent + zh SIM drop <= 0.03 + screen; E5 adopts if A3 + A5 >= 0.5 for amadeus and
amadeus_anime + every row's JA CER <= 0.07 (phone projection reported only); E6 adopts if every
token's A6 dCER <= 0.05 and runaways do not increase + screen; E7 adopts per class (A7 >= 50% at
5% FPR, >= 30 generations, A2 and A3). Nothing adopted -> final = E1 recipe, v3 not published.
A10: M1(B) <= (M1(F) + M1(D)) / 2, seed-averaged zh M1 over texts common to B, D, F; M1 valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import p07_select_v3 as p07

A9_DSIM = -0.03


def paired(run: dict, refs: list, value, allow_partial: bool, what: str) -> tuple:
    """(paired mean of value(run) - mean(value(ref) for ref in refs), n) over the (text, seed)
    pairs of accent zh rows that all have; error if refs miss any of the run's pairs."""
    key = p07.key_accent("zh")
    tabs = []
    for ck in [run] + refs:
        t = {}
        for r in ck["rows"]:
            k = key(r)
            if k is not None:
                v = value(ck, r)
                if v is not None:
                    t[k] = float(v)
        tabs.append(t)
    common = set(tabs[0]).intersection(*tabs[1:])
    lost = sorted(set(tabs[0]) - common)
    if lost:
        msg = f"{what}: {len(lost)} of the run's zh (text, seed) values missing in the comparator, e.g. {lost[:5]}"
        if not allow_partial:
            p07.fail(msg + " (or pass --allow_partial)")
        print(f"!! {msg}; --allow_partial: using the common pairs")
    if not common:
        return None, 0
    return float(np.mean([tabs[0][k] - np.mean([t[k] for t in tabs[1:]]) for k in common])), len(common)


def a10(e0: Path, validity: dict) -> dict:
    v = (validity.get("zh") or {}).get(p07.A4_M1) or {}
    per = {}
    for c in ("B", "D", "F"):
        p = e0 / c / "scored_accent.jsonl"
        if not p.exists():
            return {"gate": False, "M1_valid": bool(v.get("valid")), "reason": f"E0 condition {c} missing ({p})"}
        by: dict = {}
        for r in p07.read_jsonl(p):
            if r.get("lang") == "zh" and r.get(p07.A4_M1) is not None:
                by.setdefault(r.get("tid") or r["pid"].rsplit("_s", 1)[0], []).append(float(r[p07.A4_M1]))
        per[c] = {t: float(np.mean(x)) for t, x in by.items()}
    common = sorted(set(per["B"]) & set(per["D"]) & set(per["F"]))
    if not common:
        return {"gate": False, "M1_valid": bool(v.get("valid")), "reason": "no zh texts common to B, D, F"}
    m = {c: float(np.mean([per[c][t] for t in common])) for c in per}
    mid = (m["F"] + m["D"]) / 2
    ok = bool(v.get("valid")) and m["B"] <= mid
    out = {"M1": {c: round(x, 4) for c, x in m.items()}, "midpoint_FD": round(mid, 4), "n_texts": len(common),
           "M1_valid": bool(v.get("valid")), "gate": ok,
           "reason": ("pseudo-Mandarin route (E4) runs" if ok else "M1 not valid per A4: route skipped"
                      if not v.get("valid") else "M1(B) not at least halfway from F to D: route skipped")}
    vm = v.get("means") or {}
    if vm:
        out["validity_means_check"] = {c: vm.get(c) for c in ("B", "D", "F")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="v3")
    ap.add_argument("--e1", required=True, help="E1a,E1b")
    for k in ("e2", "e3", "e3f", "e4", "e4_parent", "e5", "e6", "e7"):
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--force_a1", action="store_true")
    a = ap.parse_args()
    suite = p07.WORK / "eval" / a.suite
    table = json.loads((suite / "table.json").read_text(encoding="utf-8"))
    ref_key = p07.ref_key_of(table, p07.V2_PIN)
    ref = p07.load_ckpt(suite / p07.tag_of(ref_key))
    dedup, validity, real = p07.load_suite_inputs()
    res_a1 = p07.a1(ref, p07.load_archive(p07.WORK / "eval" / "archive_v2_confirm" / "scored.jsonl"), p07.A1_TOL, None)
    if not res_a1["ok"]:
        print(f"!! A1 FAILED (dur match {res_a1.get('dur_match')}, CER regenerated {res_a1.get('cer_regen')} vs "
              f"archived {res_a1.get('cer_archive')}). Debug the harness first.")
        if not a.force_a1:
            raise SystemExit(1)

    keys = {}
    for lab, name in (("E2", a.e2), ("E3", a.e3), ("E3f", a.e3f), ("E4", a.e4), ("E4_parent", a.e4_parent),
                      ("E5", a.e5), ("E6", a.e6), ("E7", a.e7)):
        if name:
            keys[lab] = p07.resolve_key(table, name, f"--{lab.lower()}")
    names = [n for n in a.e1.split(",") if n]
    if len(names) != 2:
        p07.fail(f"--e1 needs two runs (E1a,E1b), got {names}")
    e1k = [p07.resolve_key(table, n, "--e1") for n in names]
    if e1k[0] == e1k[1]:
        p07.fail(f"--e1 names the same key twice: {e1k}")
    cache: dict = {}

    def ck(k):
        if k not in cache:
            cache[k] = p07.load_ckpt(suite / p07.tag_of(k))
        return cache[k]
    e1a, e1b = ck(e1k[0]), ck(e1k[1])
    m1v = (validity.get("zh") or {}).get(p07.A4_M1) or {}
    m1_valid = bool(m1v.get("valid"))
    sgn = 1 if m1v.get("sign", "+").startswith("+") else -1
    M1, SER = p07.acc_value(p07.A4_M1), p07.acc_value("ser_shared")
    sp_m1, _ = paired(e1a, [e1b], M1, a.allow_partial, "E1 spread (M1)")
    sp_ser, _ = paired(e1a, [e1b], SER, a.allow_partial, "E1 spread (SER)")
    spread = {"m1": None if sp_m1 is None else abs(sp_m1), "ser_shared": None if sp_ser is None else abs(sp_ser)}
    gate10 = a10(p07.WORK / "eval" / "e0", validity)
    screens: dict = {}

    def screen(k):
        if k not in screens:
            c = ck(k)
            use = p07.coverage(c, ref, k, None, a.allow_partial)
            r2, r3 = p07.a2(c, ref, dedup, a.B, use), p07.a3(c, ref, ref, a.B, use)
            auto = r2["auto_screen"]["mean_dcer"] if r2["auto_screen"] else r2["all"]["cer"]["mean"]
            ok = bool(r2["pass"] and auto is not None and auto <= p07.A2_AUTO)
            screens[k] = {"use": use, "A2": r2, "A3": r3, "auto_dcer": auto, "ok": ok,
                          "why": (f"A2 screen {'passes' if ok else 'FAILS'} (S={len(use)}: dCER {r2['all']['cer'].get('mean')} "
                                  f"ub90 {r2['all']['cer'].get('bound90')}, dSIM lb90 {r2['all']['sim'].get('bound90')}, "
                                  f"runaway {r2['runaway']}, Auto dCER {auto})")}
        return screens[k]

    def m1_vs(k, refs, label):
        d, n = paired(ck(k), refs, M1, a.allow_partial, f"{k} M1 vs {label}")
        ok = m1_valid and spread["m1"] is not None and d is not None and sgn * d < -spread["m1"]
        why = ("zh M1 not valid per A4" if not m1_valid else "E1 M1 spread unavailable" if spread["m1"] is None
               else f"zh M1 vs {label} {d:+.4f} (n {n}) {'beyond' if ok else 'not beyond'} -spread {spread['m1']:.4f}"
               if d is not None else f"no paired zh M1 vs {label}")
        return ok, d, why

    out: dict = {}

    def verdict(lab, v, reasons, **kw):
        out[lab] = {"key": keys.get(lab), "verdict": v, "reasons": reasons, **kw}

    for lab in ("E2", "E3"):
        if lab not in keys:
            verdict(lab, "SKIP", ["run not given"])
            continue
        s = screen(keys[lab])
        ok_m1, d, why = m1_vs(keys[lab], [e1a, e1b], "E1")
        ds, _ = paired(ck(keys[lab]), [e1a, e1b], SER, a.allow_partial, f"{lab} SER vs E1")
        ser_ok = ds is not None and ds <= 0
        reasons = [s["why"], why, f"zh shared-char SER vs E1 {ds if ds is None else round(ds, 4)} "
                                  f"({'not worse' if ser_ok else 'WORSE or n/a'})"]
        verdict(lab, "ADOPT" if s["ok"] and ok_m1 and ser_ok else "REJECT", reasons, dM1=d, dSER_shared=ds)
    if out["E2"]["verdict"] == "ADOPT" and out["E3"]["verdict"] == "ADOPT":
        out["E2"]["verdict"] = "REJECT"
        out["E2"]["reasons"].append("passes, but E3 is preferred when both pass (A9)")
    if "E3f" not in keys:
        verdict("E3f", "SKIP", ["run not given"])
    elif "E3" not in keys:
        verdict("E3f", "SKIP", ["E3 not given: E3f is judged against E3"])
    else:
        s = screen(keys["E3f"])
        ok_m1, d, why = m1_vs(keys["E3f"], [ck(keys["E3"])], "E3")
        reasons = [s["why"], why]
        v = "ADOPT" if s["ok"] and ok_m1 else "REJECT"
        if v == "ADOPT" and out["E3"]["verdict"] != "ADOPT":
            reasons.append("note: E3 itself was not adopted; E3f adopts over E3 only (vs E1: "
                           f"{m1_vs(keys['E3f'], [e1a, e1b], 'E1')[2]}) - lead to confirm")
        verdict("E3f", v, reasons, dM1=d)
        if v == "ADOPT" and out["E3"]["verdict"] == "ADOPT":
            out["E3"]["verdict"] = "REJECT"
            out["E3"]["reasons"].append("superseded by E3f (A9)")
    if not gate10["gate"]:
        verdict("E4", "SKIP", [f"A10 gate not met: {gate10['reason']} - E4 must not run"])
    elif "E4" not in keys:
        verdict("E4", "SKIP", ["A10 gate met, but run not given"])
    elif "E4_parent" not in keys:
        p07.fail("--e4 needs --e4_parent")
    else:
        s = screen(keys["E4"])
        par = ck(keys["E4_parent"])
        ok_m1, d, why = m1_vs(keys["E4"], [par], "parent")
        dsim, _ = paired(ck(keys["E4"]), [par], lambda c, r: r.get("sim"), a.allow_partial, "E4 SIM vs parent")
        sim_ok = dsim is not None and dsim >= A9_DSIM
        verdict("E4", "ADOPT" if s["ok"] and ok_m1 and sim_ok else "REJECT",
                [s["why"], why, f"zh SIM vs parent {dsim if dsim is None else round(dsim, 4)} "
                                f"({'drop <= 0.03' if sim_ok else 'drops > 0.03 or n/a'})"], dM1=d, dSIM_zh=dsim)
    if "E5" not in keys:
        verdict("E5", "SKIP", ["run not given"])
    else:
        s = screen(keys["E5"])
        r5 = p07.a5(ck(keys["E5"]), real)
        rows = r5.get("rows", {})
        proj = {x: (rows.get(x) or {}).get("projection") for x in ("amadeus", "amadeus_anime")}
        proj_ok = all(v is not None and v >= p07.A5_PROJ for v in proj.values())
        cer_ok = bool(rows) and all(v.get("cer_ok") for v in rows.values())
        reasons = [f"A3 {'passes' if s['A3']['pass'] else 'FAILS'} (rms/max/tilt {s['A3'].get('rms')}/"
                   f"{s['A3'].get('max')}/{s['A3'].get('tilt')})",
                   f"A5 projections {proj} ({'>= 0.5' if proj_ok else 'below 0.5 or missing'})",
                   f"row JA CER {({x: v.get('cer_mean') for x, v in rows.items()})} "
                   f"({'all <= 0.07' if cer_ok else 'a row > 0.07 or missing'})",
                   f"phone projection {(rows.get('phone') or {}).get('projection')} (in-sample, reported only)"]
        verdict("E5", "ADOPT" if s["A3"]["pass"] and proj_ok and cer_ok else "REJECT", reasons)
    if "E6" not in keys:
        verdict("E6", "SKIP", ["run not given"])
    else:
        s = screen(keys["E6"])
        t = p07.a6(ck(keys["E6"]))
        tok_ok = bool(t) and all(v["dcer_ok"] and v["runaway_ok"] for v in t.values())
        verdict("E6", "ADOPT" if s["ok"] and tok_ok else "REJECT",
                [s["why"]] + ([f"{k}: dCER {v['dcer']} ({'ok' if v['dcer_ok'] else '> 0.05'}), runaway "
                               f"{v['runaway_rate']} vs twins {v['twin_runaway_rate']} "
                               f"({'ok' if v['runaway_ok'] else 'increase'})" for k, v in t.items()]
                              or ["no tag tokens in this checkpoint"]) + ["tags ship as experimental regardless"])
    if "E7" not in keys:
        verdict("E7", "SKIP", ["run not given"])
    else:
        s = screen(keys["E7"])
        if len(s["use"]) < 2:
            p07.fail(f"E7 needs S >= 2 (30 generations per class), has {s['use']}")
        t = p07.a7(ck(keys["E7"]), s["A2"]["pass"], s["A3"]["pass"])
        ships = [k for k, v in t.items() if v["ships"]]
        verdict("E7", "ADOPT" if ships else "REJECT",
                [f"A2 {'passes' if s['A2']['pass'] else 'FAILS'}, A3 {'passes' if s['A3']['pass'] else 'FAILS'}"] +
                [f"{k}: n {v['n']} detect {v['rate']} ({'ships' if v['ships'] else 'cut'})" for k, v in t.items()],
                classes_adopted=ships)
    adopted = [lab for lab, v in out.items() if v["verdict"] == "ADOPT"]
    summary = {"suite": a.suite, "ref": ref_key, "A1_ok": res_a1["ok"], "e1": e1k, "e1_spread": spread,
               "m1_valid": m1_valid, "A10": gate10, "runs": out, "adopted": adopted,
               "conclusion": (f"final = E1 recipe + {adopted}" if adopted else
                              "no adopted change: final = E1 recipe; v3 is NOT published (A9)")}
    (suite / "adoption.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                         encoding="utf-8")
    print(f"A10 (E0 gate): {'MET' if gate10['gate'] else 'NOT MET'} - {gate10['reason']}; "
          f"M1 {gate10.get('M1')} midpoint(F,D) {gate10.get('midpoint_FD')}")
    print(f"E1 spread: {spread}; zh M1 valid per A4: {m1_valid}")
    for lab, v in out.items():
        print(f"{lab:4s} {v['verdict']:6s} {v['key'] or ''}")
        for r in v["reasons"]:
            print(f"       - {r}")
    print(f"\n{summary['conclusion']}\nwrote {suite / 'adoption.json'}")


if __name__ == "__main__":
    main()
