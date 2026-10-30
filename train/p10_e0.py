"""E0: where does the Japanese accent in Kurisu's Mandarin/English come from?
(+ does the x-vector alone carry the Amadeus / phone device EQ?)

Inference only - no training. Every condition reads the same probe sentences with the
same seeds (one torch seed per batch and seed), so differences come from model/prompt:

  A    Base, x-vector only, Kurisu vector (v2's row 3000), language Auto
  B    Base, x-vector only, Kurisu vector, language Chinese/English
  C    Base, ICL from a Japanese Kurisu take (+ transcript), Chinese/English  <- graded: accented
  D    Base, x-vector only, native reference (Chinese / English dub take)    <- native anchor
  E    v2 (published), speaker kurisu, Auto
  F    v2, speaker kurisu, Chinese/English
  G25/G50/G75  v2 weights blended toward Base (alpha = share of v2: 0.25/0.5/0.75),
       Chinese/English; all shared float weights blended, v2's speaker row 3000 kept
       <- dose-response G25 -> G50 -> G75 -> F
  H    CustomVoice ono_anna, language Japanese on zh/en text. INFORMATIONAL ONLY: a
       wrong-language anchor (it reads hanzi as Japanese), not an accent anchor.
  J    CustomVoice ono_anna (Qwen's own Japanese voice), Chinese/English    <- graded
  R    real native recordings (Chinese / English dub), different texts      <- native, unpaired
  X_<row>  tilt probe: Base, x-vector only, the mean speaker vector of
       build_v3/ref_<row>.txt (train_sft.speaker_embedding, i.e. exactly the trainer's
       locked vector) for kurisu / amadeus / amadeus_anime / phone, on 36 JA confirm texts.
       If X_amadeus - X_kurisu already reproduces the real Amadeus - Kurisu LTAS
       difference, the x-vector carries the EQ by itself.

Outputs eval/e0/<cond>/{*.wav, manifest.jsonl, scored*.jsonl}, then
  eval/e0/table.json     per condition and language: CER/SIM (metrics, v3 text norm) and
                         accent aggregates (accent_metrics), never pooled across languages
  eval/e0/validity.json  A4 metric validity per language and metric:
                         d(D, C) > 1 on per-text means (oriented: C more Japanese),
                         monotone more Japanese over G25 -> G50 -> G75 -> F (condition means),
                         seed-to-seed per-text Pearson r >= 0.5 (Fisher-z mean over the
                         graded TTS conditions C D E F G25 G50 G75 J)
  eval/e0/tilt_probe.json  X_<row> vs X_kurisu: median paired LTAS / tilt difference and its
                         projection onto the real-take difference (A5 statistic)
Decoding: upstream defaults, non-streaming. Seeds: --seeds (default 1,2; >= 2 needed for
reliability). Pin v2 with KURISU_V2_REV=<full commit sha>.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
CODE = Path(__file__).resolve().parent
E0 = WORK / "eval" / "e0"
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CV = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
V2 = os.environ.get("KURISU_V2", "starrydark/Kurisu_Qwen3_TTS")
V2_REV = os.environ.get("KURISU_V2_REV") or None
BATCH = 12
LANG = {"zh": "Chinese", "en": "English", "ja": "Japanese"}
XROWS = ("kurisu", "amadeus", "amadeus_anime", "phone")
GRADED = ("C", "D", "E", "F", "G25", "G50", "G75", "J")
DOSE = ("G25", "G50", "G75", "F")
# metric -> +1 if higher = more Japanese-accented, -1 if lower = more accented
METRICS = {"zh": {"lid_ratio": 1, "p_target": -1, "ser": 1, "ser_shared": 1, "ter": 1, "ter_strict": 1},
           "en": {"lid_ratio": 1, "p_target": -1}}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def probes() -> list[dict]:
    rows = []
    for lang, name in (("zh", "probes_zh_v3.jsonl"), ("en", "probes_en_v3.jsonl")):
        for r in read_jsonl(CODE / name):
            rows.append({**r, "tid": r["pid"], "lang": lang})
    return rows


def xvec_rows() -> list[dict]:
    """The 36 JA texts of p06_eval_v3's cond rows (confirm batches 0, 4, 5)."""
    import p06_eval_v3 as ev
    suite = ev.suite_rows()
    return [{**r, "tid": r["pid"]} for b in ev.COND_BATCHES for r in suite[b:b + BATCH]]


def v2_path() -> str:
    from huggingface_hub import snapshot_download
    if V2_REV is None:
        print(f"[e0] WARNING: {V2} not pinned (set KURISU_V2_REV); using the current main")
    return snapshot_download(V2, revision=V2_REV)


def load(model_id: str):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0", dtype=torch.bfloat16,
                                         attn_implementation="sdpa")


def run_cond(cond: str, rows: list[dict], seeds: list[int], fn) -> None:
    """All seeds x languages x batches of one condition; the manifest is written once at
    the end, so a crashed condition is regenerated rather than half-scored."""
    out = E0 / cond
    if (out / "manifest.jsonl").exists():
        print(f"[e0] {cond}: exists, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    man = []
    for seed in seeds:
        for lang in sorted({r["lang"] for r in rows}):
            sel = [r for r in rows if r["lang"] == lang]
            for b in range(0, len(sel), BATCH):
                chunk = sel[b:b + BATCH]
                torch.manual_seed(seed * 100003 + b)
                wavs, sr = fn(chunk, lang)
                for r, w in zip(chunk, wavs):
                    f = f"{r['tid']}_s{seed}.wav"
                    sf.write(str(out / f), np.asarray(w, dtype=np.float32), sr)
                    man.append({**r, "pid": f"{r['tid']}_s{seed}", "file": f, "cond": cond, "seed": seed})
    tmp = out / "manifest.jsonl.tmp"
    tmp.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man), encoding="utf-8")
    tmp.replace(out / "manifest.jsonl")
    print(f"[e0] {cond}: done ({len(man)} clips)", flush=True)


def kurisu_vector(v2_dir: str) -> torch.Tensor:
    """v2's locked speaker row: exactly the conditioning the fine-tune was trained with."""
    from safetensors import safe_open
    with safe_open(str(Path(v2_dir) / "model.safetensors"), "pt") as f:
        return f.get_slice("talker.model.codec_embedding.weight")[3000:3001][0].float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--only", default=None, help="comma list of conditions to run (X = all X_<row>)")
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) < 2:
        print("[e0] WARNING: one seed - A4 seed-to-seed reliability cannot be computed")
    only = set(args.only.split(",")) if args.only else None
    rows = probes()
    refs = json.loads((CODE / "e0_refs_src.json").read_text(encoding="utf-8"))
    hf = WORK / "src" / "hf"            # raw dataset files, as downloaded by bootstrap
    for k in ("ja", "zh", "en"):
        refs[k]["audio"] = str(hf / refs[k]["path"])
    for r in refs["native_pool"]:
        r["audio"] = str(hf / r["path"])
    kw = dict(max_new_tokens=400, non_streaming_mode=True)

    def want(c):
        return (only is None or c in only or (c.startswith("X_") and "X" in only)) and not args.score_only

    v2_dir = v2_path() if any(want(c) for c in ("A", "B", "E", "F", "G25", "G50", "G75")) else None
    xconds = [f"X_{r}" for r in XROWS if (WORK / "build_v3" / f"ref_{r}.txt").exists()]
    base_sd = None
    if any(want(c) for c in ["A", "B", "C", "D"] + xconds):
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        base = load(BASE)

        def clone(prompt, lang_of):
            def fn(chunk, lang):
                return base.generate_voice_clone(
                    text=[r["text"] for r in chunk], language=[lang_of(lang)] * len(chunk),
                    voice_clone_prompt=[prompt(lang)] * len(chunk), **kw)
            return fn

        def xvec(vec):
            return VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec,
                                        x_vector_only_mode=True, icl_mode=False)
        if any(want(c) for c in "AB"):
            xk = xvec(kurisu_vector(v2_dir).to("cuda:0").to(torch.bfloat16))
            if want("A"):
                run_cond("A", rows, seeds, clone(lambda l: xk, lambda l: "Auto"))
            if want("B"):
                run_cond("B", rows, seeds, clone(lambda l: xk, lambda l: LANG[l]))
        if want("C"):
            icl = base.create_voice_clone_prompt(ref_audio=refs["ja"]["audio"], ref_text=refs["ja"]["text"])[0]
            run_cond("C", rows, seeds, clone(lambda l: icl, lambda l: LANG[l]))
        if want("D"):
            native = {l: base.create_voice_clone_prompt(ref_audio=refs[l]["audio"], x_vector_only_mode=True)[0]
                      for l in ("zh", "en")}
            run_cond("D", rows, seeds, clone(lambda l: native[l], lambda l: LANG[l]))
        if any(want(c) for c in xconds):
            from train_sft import ref_paths, speaker_embedding   # the trainer's own averaging
            jrows = xvec_rows()
            for c in xconds:
                if want(c):
                    name = c[2:]
                    vec = speaker_embedding(base.model, name, ref_paths(WORK / "build_v3" / f"ref_{name}.txt"),
                                            "cuda:0", torch.bfloat16)[0]
                    item = xvec(vec)
                    run_cond(c, jrows, seeds, clone(lambda l, it=item: it, lambda l: "Auto"))
        if any(want(c) for c in ("G25", "G50", "G75")):
            base_sd = {k: v.detach().to("cpu") for k, v in base.model.state_dict().items()}
        del base
        torch.cuda.empty_cache()

    if any(want(c) for c in ("E", "F", "G25", "G50", "G75")):
        v2 = load(v2_dir)

        def custom(lang_of, speaker="kurisu"):
            def fn(chunk, lang):
                return v2.generate_custom_voice(
                    text=[r["text"] for r in chunk], speaker=[speaker] * len(chunk),
                    language=[lang_of(lang)] * len(chunk), **kw)
            return fn
        if want("E"):
            run_cond("E", rows, seeds, custom(lambda l: "Auto"))
        if want("F"):
            run_cond("F", rows, seeds, custom(lambda l: LANG[l]))
        if any(want(c) for c in ("G25", "G50", "G75")):
            if base_sd is None:
                b = load(BASE)
                base_sd = {k: v.detach().to("cpu") for k, v in b.model.state_dict().items()}
                del b
                torch.cuda.empty_cache()
            v2_sd = {k: v.detach().to("cpu").clone() for k, v in v2.model.state_dict().items()}
            spk_key = "talker.model.codec_embedding.weight"
            for alpha in (0.25, 0.5, 0.75):
                cond = f"G{int(alpha * 100)}"
                if not want(cond):
                    continue
                with torch.no_grad():
                    for k, p in v2.model.state_dict().items():
                        if k in base_sd and base_sd[k].shape == p.shape and p.is_floating_point():
                            mix = base_sd[k].float() + alpha * (v2_sd[k].float() - base_sd[k].float())
                            if k == spk_key:
                                mix[3000] = v2_sd[k][3000].float()
                            p.copy_(mix.to(p.dtype))
                run_cond(cond, rows, seeds, custom(lambda l: LANG[l]))
        del v2
        torch.cuda.empty_cache()

    if want("H") or want("J"):
        cv = load(CV)

        def anna(lang_of):
            def fn(chunk, lang):
                return cv.generate_custom_voice(
                    text=[r["text"] for r in chunk], speaker=["ono_anna"] * len(chunk),
                    language=[lang_of(lang)] * len(chunk), **kw)
            return fn
        if want("H"):
            run_cond("H", rows, seeds, anna(lambda l: "Japanese"))
        if want("J"):
            run_cond("J", rows, seeds, anna(lambda l: LANG[l]))
        del cv
        torch.cuda.empty_cache()

    if want("R"):
        out = E0 / "R"
        if not (out / "manifest.jsonl").exists():
            out.mkdir(parents=True, exist_ok=True)
            man = []
            for i, r in enumerate(refs["native_pool"]):
                f = f"real_{r['lang']}_{i:03d}.wav"
                wav, sr = sf.read(r["audio"])
                sf.write(str(out / f), wav, sr)
                man.append({"pid": f"real_{r['lang']}_{i:03d}", "tid": f"real_{r['lang']}_{i:03d}",
                            "text": r["text"], "lang": r["lang"], "file": f, "cond": "R", "seed": 0})
            (out / "manifest.jsonl").write_text(
                "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man), encoding="utf-8")

    score_all()


# ----------------------------------------------------------------- scoring / analysis
def score_all() -> None:
    import accent_metrics
    import metrics

    import p06_eval_v3
    centroid = p06_eval_v3.centroid()      # v2's centroid definition, as in the v3 suite
    table = {}
    for d in sorted(E0.iterdir()):
        if not d.is_dir() or not (d / "manifest.jsonl").exists():
            continue
        if not (d / "scored.jsonl").exists():
            metrics.score_dir(d, centroid, text_norm="v3")
        scored = read_jsonl(d / "scored.jsonl") if (d / "scored.jsonl").exists() else []
        entry = {"lang": {}}
        for lang in sorted({r["lang"] for r in scored}):
            entry["lang"][lang] = {"asr_sim": metrics.aggregate([r for r in scored if r["lang"] == lang])}
        if any(r["lang"] in ("zh", "en") for r in scored):
            if not (d / "scored_accent.jsonl").exists():
                accent_metrics.score_dir(d, d / "manifest.jsonl")
            acc = read_jsonl(d / "scored_accent.jsonl")
            for lang in ("zh", "en"):
                sel = [r for r in acc if r["lang"] == lang]
                if sel:
                    entry["lang"].setdefault(lang, {})["accent"] = accent_metrics.aggregate(sel)
        if d.name.startswith("X_"):
            spectra = {}
            for r in scored:
                b, t = p06_eval_v3.ltas13(str(d / r["file"]))
                spectra[r["pid"]] = (b, t)
            np.savez(d / "ltas.npz", pid=np.array(list(spectra)), ltas=np.stack([v[0] for v in spectra.values()]),
                     tilt=np.array([v[1] for v in spectra.values()]), fc=p06_eval_v3.FC)
            entry["tilt_med"] = round(float(np.median([v[1] for v in spectra.values()])), 2)
        if d.name == "H":
            entry["note"] = "wrong-language anchor (reads hanzi as Japanese); informational, not used for validity"
        table[d.name] = entry
        print(f"[e0] {d.name}: " + json.dumps({l: {k: (v.get('cer_mean') if k == 'asr_sim' else
                                                       {m: (v.get(m) or {}).get('mean') for m in ('lid_ratio', 'ser_shared')})
                                                   for k, v in e.items()} for l, e in entry["lang"].items()},
                                                 ensure_ascii=False), flush=True)
    (E0 / "table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    validity = accent_validity()
    (E0 / "validity.json").write_text(json.dumps(validity, indent=2, ensure_ascii=False), encoding="utf-8")
    for lang, ms in validity.items():
        for m, v in ms.items():
            print(f"[e0] A4 {lang}:{m:11s} d(D,C) {v['d_DC']}  monotone {v['monotone']}  "
                  f"reliability {v['reliability_mean']}  -> {'VALID' if v['valid'] else 'invalid'}")
    tp = tilt_probe()
    (E0 / "tilt_probe.json").write_text(json.dumps(tp, indent=2), encoding="utf-8")
    for row, v in tp.get("rows", {}).items():
        print(f"[e0] tilt probe {row}: projection {v['projection']}  tilt diff {v['tilt_diff_med']} dB")


def _per_text(rows: list[dict], metric: str) -> tuple[dict, dict]:
    """{tid: mean over seeds}, {seed: {tid: value}}."""
    by_seed: dict = {}
    for r in rows:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by_seed.setdefault(r["seed"], {})[r["tid"]] = float(v)
    tids = sorted({t for s in by_seed.values() for t in s})
    means = {t: float(np.mean([s[t] for s in by_seed.values() if t in s])) for t in tids}
    return means, by_seed


def _rows_for(cond: str, lang: str) -> list[dict]:
    p = E0 / cond / "scored_accent.jsonl"
    if not p.exists():
        return []
    rows = [r for r in read_jsonl(p) if r["lang"] == lang]
    if lang == "zh":                       # M2/M3 values only where the reference is valid
        for r in rows:
            if not r.get("m23_valid"):
                for k in ("ser", "ser_shared", "ter", "ter_strict"):
                    r[k] = None
    for r in rows:
        r.setdefault("tid", r.get("pid", "").rsplit("_s", 1)[0])
    return rows


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def seed_reliability(by_seed: dict) -> float | None:
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return None
    rs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            common = sorted(set(by_seed[seeds[i]]) & set(by_seed[seeds[j]]))
            if len(common) >= 5:
                x = np.array([by_seed[seeds[i]][t] for t in common])
                y = np.array([by_seed[seeds[j]][t] for t in common])
                if x.std() > 0 and y.std() > 0:
                    rs.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(rs)) if rs else None


def fisher_mean(rs) -> float | None:
    rs = [min(max(r, -0.999), 0.999) for r in rs if r is not None]
    return float(np.tanh(np.mean(np.arctanh(rs)))) if rs else None


def accent_validity() -> dict:
    """A4: per language and metric. d(D, C) on per-text means (sign-oriented so > 0 means
    C looks more Japanese); paired dz reported too. Monotone: condition means strictly
    more Japanese over G25 -> G50 -> G75 -> F (increasing for "+" metrics such as M1 and
    shared-character SER, decreasing for p_target); the reverse is rejected. Reliability: mean
    seed-to-seed per-text Pearson r within each graded condition, Fisher-z averaged."""
    out = {}
    for lang, mets in METRICS.items():
        out[lang] = {}
        for m, sgn in mets.items():
            per = {}
            for c in GRADED + ("H", "A", "B", "R"):
                rows = _rows_for(c, lang)
                if rows:
                    per[c] = _per_text(rows, m)
            res = {"sign": "+ higher = more Japanese" if sgn > 0 else "- lower = more Japanese",
                   "means": {c: round(float(np.mean(list(v[0].values()))), 4) for c, v in per.items() if v[0]}}
            d_dc = dz = None
            if "C" in per and "D" in per:
                d_dc = sgn * cohens_d(list(per["C"][0].values()), list(per["D"][0].values()))
                common = sorted(set(per["C"][0]) & set(per["D"][0]))
                if len(common) > 2:
                    diff = sgn * (np.array([per["C"][0][t] for t in common]) - np.array([per["D"][0][t] for t in common]))
                    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else None
            dose = [res["means"].get(c) for c in DOSE]
            mono, direction = None, None
            if all(v is not None for v in dose):
                inc = all(a < b for a, b in zip(dose, dose[1:]))
                dec = all(a > b for a, b in zip(dose, dose[1:]))
                fwd = (inc and sgn > 0) or (dec and sgn < 0)
                direction = ("more Japanese with more v2 weight" if fwd
                             else "less Japanese with more v2 weight" if (inc or dec) else "none")
                # registered: only the more-Japanese direction counts (M1 and shared-character SER
                # must increase G25 < G50 < G75 < F); the reverse direction is rejected
                mono = fwd
            rel = {c: seed_reliability(per[c][1]) for c in GRADED if c in per}
            rel_mean = fisher_mean(list(rel.values()))
            info = {}
            if "H" in per and "D" in per:
                info["d_DH_wrong_language"] = round(sgn * cohens_d(list(per["H"][0].values()), list(per["D"][0].values())), 3)
            if "R" in per and "C" in per:
                info["d_RC_unpaired"] = round(sgn * cohens_d(list(per["C"][0].values()), list(per["R"][0].values())), 3)
            res.update(d_DC=None if d_dc is None or math.isnan(d_dc) else round(d_dc, 3),
                       dz_DC=None if dz is None else round(dz, 3), monotone=mono, direction=direction,
                       reliability={c: None if v is None else round(v, 3) for c, v in rel.items()},
                       reliability_mean=None if rel_mean is None else round(rel_mean, 3),
                       informational=info)
            res["valid"] = bool(res["d_DC"] is not None and res["d_DC"] > 1 and mono
                                and rel_mean is not None and rel_mean >= 0.5)
            if lang == "en":
                res["note"] = "SELECTION_RULE A4: no English accent claim is made regardless"
            out[lang][m] = res
    return out


def tilt_probe() -> dict:
    """X_<row> vs X_kurisu (same texts and seeds): median over texts of the per-text
    (mean over seeds) LTAS difference, its projection onto the real-take difference
    (real <row> - real kurisu median LTAS; A5's statistic) and the median tilt difference."""
    import p06_eval_v3
    real = p06_eval_v3.real_speaker_ltas()

    def load(cond):
        p = E0 / cond / "ltas.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return {str(pid): (L, t) for pid, L, t in zip(z["pid"], z["ltas"], z["tilt"])}
    ref = load("X_kurisu")
    res = {"rows": {}, "real_available": sorted(real)}
    if ref is None:
        return res
    for row in XROWS[1:]:
        cur = load(f"X_{row}")
        if cur is None:
            continue
        tids = sorted({p.rsplit("_s", 1)[0] for p in cur})
        dl, dt = [], []
        for t in tids:
            pairs = [(cur[p], ref[p]) for p in cur if p.rsplit("_s", 1)[0] == t and p in ref]
            if pairs:
                dl.append(np.mean([a[0] - b[0] for a, b in pairs], 0))
                dt.append(np.mean([a[1] - b[1] for a, b in pairs]))
        if not dl:
            continue
        dm = np.median(np.stack(dl), 0)
        entry = {"n_texts": len(dl), "median_ltas_diff_db": [round(float(x), 2) for x in dm],
                 "tilt_diff_med": round(float(np.median(dt)), 2), "projection": None}
        if row in real and "kurisu" in real:
            dr = real[row]["median_ltas"] - real["kurisu"]["median_ltas"]
            entry["real_diff_db"] = [round(float(x), 2) for x in dr]
            entry["projection"] = round(float(dm @ dr / (dr @ dr)), 3) if float(dr @ dr) > 0 else None
        res["rows"][row] = entry
    return res


if __name__ == "__main__":
    main()
