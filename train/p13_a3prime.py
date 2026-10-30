"""A3' (SELECTION_RULE.md, Amendment C): did a device EQ leak into the default voice?

The default-voice shift is the candidate's `kurisu`-row LTAS minus v2's LTAS, paired on
the confirm texts and seeds in shipped mode. Its median over texts is projected onto
each real device direction: real amadeus takes minus real kurisu takes, and real phone
takes minus real kurisu takes. A5 does the same, so 1.0 means "the full device shift".
The threshold, fixed before any A3' value was computed, is <= 0.10 in both directions.

  python p13_a3prime.py --cand runs/E1a/ --cand runs/E3/ ...   (substrings of table keys)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import p07_select_v3 as P

V2 = "starrydark/Kurisu_Qwen3_TTS@92a08362cb2c95f7dfc8a7ecf4cc00db625f9d57"
DIRECTIONS = ("amadeus", "phone")
THRESHOLD = 0.10


def a3prime(cand: dict, ref: dict, real: dict) -> dict:
    def lt(ck, r):
        x = ck["ltas"].get(r["pid"])
        return None if x is None else np.asarray(x[0])
    D, tids, seeds = P.matrix(cand, ref, lt, P.key_confirm(cand["shipped"]),
                              P.key_confirm(ref["shipped"]), strict=True)
    if not D.size:
        return {"pass": False, "status": "n/a"}
    shift = np.median(D.mean(1), 0)                       # [bands], median over texts
    out = {"n_texts": len(tids), "seed_bases": seeds, "projections": {}}
    for s in DIRECTIONS:
        dr = real[s]["median_ltas"] - real["kurisu"]["median_ltas"]
        out["projections"][s] = round(float(shift @ dr / (dr @ dr)), 4)
    worst = max(out["projections"].values())
    out["max_band"] = {"db": round(float(np.max(np.abs(shift))), 3), "band": int(np.argmax(np.abs(shift)))}
    out["pass"] = bool(worst <= THRESHOLD)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", required=True)
    args = ap.parse_args()
    suite = P.WORK / "eval" / "v3"
    table = json.loads((suite / "table.json").read_text(encoding="utf-8"))
    _, _, real = P.load_suite_inputs()
    ref_key = P.ref_key_of(table, V2)
    for name in args.cand:
        key = P.resolve_key(table, name, "--cand")
        cand = P.load_ckpt(suite / P.tag_of(key))
        bases = sorted({r.get("seed_base") for r in cand["rows"] if r.get("seed_base")})
        ref = P.load_ckpt(suite / P.tag_of(ref_key), set(bases))
        res = a3prime(cand, ref, real)
        print(f"A3' {name}: {'PASS' if res['pass'] else 'FAIL'} {json.dumps(res)}")


if __name__ == "__main__":
    main()
