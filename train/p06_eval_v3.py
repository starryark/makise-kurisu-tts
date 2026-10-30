"""v3 evaluation (SELECTION_RULE.md, Amendment A): generate + score one suite per checkpoint.

  python p06_eval_v3.py --name v3 --ckpt real --ckpt starrydark/Kurisu_Qwen3_TTS[@<rev>] \
      [--ckpt <dir> ...] [--seeds 2 | --seed_bases 6000,11000] [--skip_existing]

Sampling seeds (A1/A2). Every text is generated once per seed base B (default
6000, 11000, 16000, 21000, ... for --seeds 1, 2, 3, 4, ...). Within a base, each group is
cut into batches of 12 in a fixed order and batch b is seeded torch.manual_seed(B + off + b):
  confirm  off 0     the 176 rows of suite_confirm.jsonl in v2's order. With B = 6000 and
                     language Auto this is byte-for-byte v2's confirm run (same batches,
                     same seeds), which A1 uses to check the harness (CER 0.0297).
  accent   off 1000  probes_zh_v3 + probes_en_v3, explicit Chinese / English.
  cond     off 0     amadeus / amadeus_anime / phone rows on confirm batches 0, 4, 5 (12 novel
                     + 24 held-out JA texts), same batch composition and seeds as confirm,
                     so each one is paired with the kurisu confirm output of the same text,
                     seed and batch neighbours (A5).
  tags     off 3000  15 short JA texts x each event token x {initial, final}, + untagged twins
  sfx      off 4000  15 texts x each [bg_<class>] token (prefix) -> 15 x S prompts per class
Modes (A2): the shipped mode is language "Japanese" for checkpoints trained with
language ids (kurisu_v3.json "lang_trained": true), "Auto" otherwise. Language-trained
checkpoints also get the confirm suite in Auto (the A2 Auto screen). cond / tags / sfx
use the shipped mode. Row ids: pid = "<tid>~<mode>~s<B>" (mode auto | ja | x = explicit).

Scores, per checkpoint directory:
  scored.jsonl         metrics.score_dir(text_norm="v3"): CER (numbers spelled on both
                       sides), ECAPA SIM to v2's real-Kurisu centroid, runaway; + tilt_db
  ltas.npz             1/3-octave LTAS 160 Hz-10 kHz (nfft 4096, active frames) + tilt (A3/A5)
  scored_ced.jsonl     CED AudioSet label probabilities (A6/A7) on tags / sfx rows and on the
                       shipped-mode untagged confirm outputs (the 5%-FPR negatives)
  scored_accent.jsonl  accent_metrics on zh/en rows (A4)
  summary.json, cap.json
Suite-level caches under eval/: ref_centroid_v2def.npy (+ ids), real_ltas.npz (median LTAS
of real takes per speaker, A5), confirm_dedup.json (A2 de-duplicated sub-score).
p07_select_v3.py applies the gates.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

import metrics

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
CODE = Path(__file__).resolve().parent
BATCH = 12
SEED_BASE0, SEED_STEP = 6000, 5000
OFFSET = {"confirm": 0, "cond": 0, "accent": 1000, "tags": 3000, "sfx": 4000}
COND_BATCHES = (0, 48, 60)            # confirm rows 0-11 (ja_novel) and 48-71 (ja_test_00-23)
SELECT_KINDS = ("ja_heldout", "ja_novel")
LANG_NAME = {"ja": "Japanese", "zh": "Chinese", "en": "English"}
TAG_TEXTS = [
    "そう、分かったわ。", "別に、あんたのためじゃないから。", "もう、知らない。",
    "それは仮説にすぎないわ。", "ちょっと待って、今のは何？", "ありがとう、岡部。",
    "まったく、どうしてこうなるのよ。", "実験は成功したみたいね。", "私は科学者よ。",
    "あの時のこと、覚えてる？", "冗談でしょ？", "少し休憩しましょう。",
    "そんなの、ありえない。", "今日はもう帰るわ。", "大丈夫、心配しないで。",
]
# A3: 1/3-octave centres 1000 * 2^(k/3), nominal 160 Hz (k=-8) .. 10 kHz (k=10)
FC = 1000.0 * 2.0 ** (np.arange(-8, 11) / 3.0)
NFFT, HOP = 4096, 512


def tag_of(ckpt: str) -> str:
    return "real" if ckpt == "real" else ckpt.rstrip("/").replace(":", "_").replace("/", "__")[-80:]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def seed_bases(seeds: int, explicit: str | None) -> list[int]:
    if explicit:
        return [int(s) for s in explicit.split(",") if s.strip()]
    return [SEED_BASE0 + SEED_STEP * i for i in range(seeds)]


# ----------------------------------------------------------------- checkpoints
def _split_rev(ckpt: str) -> tuple[str, str | None]:
    if not Path(ckpt).exists() and "@" in ckpt:
        repo, rev = ckpt.rsplit("@", 1)
        return repo, rev
    return ckpt, None


def resolve(ckpt: str, weights: bool = True) -> Path:
    """Local dir for a checkpoint; hub ids may carry a pinned revision: repo@<commit>."""
    if Path(ckpt).exists():
        return Path(ckpt)
    from huggingface_hub import snapshot_download
    repo, rev = _split_rev(ckpt)
    kw = {} if weights else {"allow_patterns": ["*.json", "*.txt"]}
    return Path(snapshot_download(repo, revision=rev, **kw))


def capabilities(ckpt: str) -> dict:
    """Speakers, tag tokens and language training of a checkpoint.

    Tokens: added_tokens.json entries starting with "[" (atomic tokens), plus a
    "tokens" list in kurisu_v3.json if the queue writes one (tags trained as sub-words).
    """
    if ckpt == "real":
        return {"speakers": ["kurisu"], "tag_tokens": [], "sfx_tokens": [], "lang_trained": False}
    path = resolve(ckpt, weights=False)
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    speakers = list(cfg.get("talker_config", {}).get("spk_id", {}))
    tokens = set()
    added = path / "added_tokens.json"
    if added.exists():
        tokens |= {t for t in json.loads(added.read_text(encoding="utf-8")) if t.startswith("[")}
    meta_p = path / "kurisu_v3.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    tokens |= {t for t in meta.get("tokens", []) if t.startswith("[")}
    return {"speakers": speakers,
            "tag_tokens": sorted(t for t in tokens if not t.startswith("[bg_")),
            "sfx_tokens": sorted(t for t in tokens if t.startswith("[bg_")),
            "lang_trained": bool(meta.get("lang_trained", False))}


def modes_for(cap: dict, override: str | None = None) -> tuple[str, list[str]]:
    """(shipped mode, confirm modes). Shipped = "ja" (language Japanese) if trained with
    language ids, else "auto"; language-trained checkpoints also get the Auto screen."""
    shipped = "ja" if cap["lang_trained"] else "auto"
    modes = [m.strip() for m in override.split(",")] if override else \
        ([shipped, "auto"] if shipped == "ja" else [shipped])
    return shipped, modes


def _language(lang: str, mode: str) -> str:
    return "Auto" if mode == "auto" else LANG_NAME[lang]


# ----------------------------------------------------------------- plan
def suite_rows() -> list[dict]:
    return read_jsonl(CODE / "suite_confirm.jsonl")


def build_plan(cap: dict, bases: list[int], modes: list[str], shipped: str) -> list[dict]:
    """Batches [{group, seed, speaker, rows}] - every row carries its own language."""
    suite = suite_rows()
    plan: list[dict] = []

    def add(group: str, base: int, rows: list[dict], speaker: str = "kurisu",
            starts: tuple | None = None, src: list[dict] | None = None):
        if starts is None:
            for b in range(0, len(rows), BATCH):
                assert b < 1000 or group == "confirm"
                plan.append({"group": group, "seed": base + OFFSET[group] + b, "speaker": speaker,
                             "rows": rows[b:b + BATCH]})
        else:                      # cond: reuse confirm batch b's composition and seed
            for b, chunk in zip(starts, rows):
                plan.append({"group": group, "seed": base + OFFSET[group] + b, "speaker": speaker,
                             "rows": chunk})

    for base in bases:
        for mode in modes:
            rows = [{**r, "tid": r["pid"], "pid": f"{r['pid']}~{mode}~s{base}", "group": "confirm",
                     "speaker": "kurisu", "mode": mode, "seed_base": base,
                     "language": _language(r["lang"], mode)} for r in suite]
            add("confirm", base, rows)
        acc = []
        for lang, name in (("zh", "probes_zh_v3.jsonl"), ("en", "probes_en_v3.jsonl")):
            for r in read_jsonl(CODE / name):
                tid = f"acc_{r['pid']}"
                acc.append({**r, "tid": tid, "pid": f"{tid}~x~s{base}", "lang": lang,
                            "kind": f"accent_{lang}", "group": "accent", "speaker": "kurisu",
                            "mode": "x", "seed_base": base, "language": LANG_NAME[lang]})
        add("accent", base, acc)
        for spk in [s for s in cap["speakers"] if s != "kurisu"]:
            chunks = []
            for b in COND_BATCHES:
                chunk = suite[b:b + BATCH]
                assert all(r["lang"] == "ja" for r in chunk), "cond batches must be all-JA"
                chunks.append([{**r, "tid": r["pid"], "pid": f"cond~{spk}~{r['pid']}~{shipped}~s{base}",
                                "kind": f"cond_{spk}", "group": "cond", "speaker": spk,
                                "mode": shipped, "seed_base": base, "language": _language("ja", shipped),
                                "real": None} for r in chunk])
            add("cond", base, chunks, speaker=spk, starts=COND_BATCHES)
        tags = []
        for tok in cap["tag_tokens"]:
            name = tok.strip("[]")
            for i, text in enumerate(TAG_TEXTS):
                for pos, t in (("i", tok + text), ("f", text + tok)):
                    tid = f"tag_{name}_{pos}{i:02d}"
                    tags.append({"tid": tid, "text": t, "lang": "ja", "kind": f"tag_{name}",
                                 "token": tok, "twin": f"twin_{i:02d}"})
        if cap["tag_tokens"] or cap["sfx_tokens"]:
            tags += [{"tid": f"twin_{i:02d}", "text": text, "lang": "ja", "kind": "tag_twin"}
                     for i, text in enumerate(TAG_TEXTS)]
        tags = [{**r, "pid": f"{r['tid']}~{shipped}~s{base}", "group": "tags", "speaker": "kurisu",
                 "mode": shipped, "seed_base": base, "language": _language("ja", shipped)} for r in tags]
        add("tags", base, tags)
        sfx = []
        for tok in cap["sfx_tokens"]:
            cls = tok.strip("[]")[3:]
            for i, text in enumerate(TAG_TEXTS):
                tid = f"sfx_{cls}_{i:02d}"
                sfx.append({"tid": tid, "pid": f"{tid}~{shipped}~s{base}", "text": tok + text,
                            "lang": "ja", "kind": f"sfx_{cls}", "token": tok, "twin": f"twin_{i:02d}",
                            "group": "sfx", "speaker": "kurisu", "mode": shipped, "seed_base": base,
                            "language": _language("ja", shipped)})
        add("sfx", base, sfx)
    pids = [r["pid"] for p in plan for r in p["rows"]]
    assert len(pids) == len(set(pids)), "duplicate pids in plan"
    return plan


def real_plan() -> list[dict]:
    rows = [{**r, "tid": r["pid"], "group": "confirm", "speaker": "kurisu", "mode": "real",
             "seed_base": 0, "language": "Auto"} for r in suite_rows() if r.get("real")]
    return [{"group": "confirm", "seed": 0, "speaker": "kurisu", "rows": rows}]


# ----------------------------------------------------------------- generation
def generate(ckpt: str, plan: list[dict], out: Path) -> None:
    """Resumable: a batch whose rows are all already in the manifest (with their wav) is
    skipped; any other batch is regenerated whole, so seeds and batch composition hold."""
    out.mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.jsonl"
    done = {m["pid"]: m for m in (read_jsonl(man_p) if man_p.exists() else []) if (out / m["file"]).is_file()}
    todo = [p for p in plan if not all(r["pid"] in done for r in p["rows"])]
    if ckpt == "real":
        for p in todo:
            for r in p["rows"]:
                if r.get("real") and Path(r["real"]).exists():
                    wav, sr = sf.read(r["real"])
                    sf.write(str(out / f"{r['pid']}.wav"), wav, sr)
                    done[r["pid"]] = {**r, "file": f"{r['pid']}.wav"}
    elif todo:
        import torch
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(resolve(ckpt)), device_map="cuda:0",
                                              dtype=torch.bfloat16, attn_implementation="sdpa")
        with open(man_p, "a", encoding="utf-8") as fh:
            for k, p in enumerate(todo):
                torch.manual_seed(p["seed"])
                wavs, sr = model.generate_custom_voice(
                    text=[r["text"] for r in p["rows"]], speaker=[p["speaker"]] * len(p["rows"]),
                    language=[r["language"] for r in p["rows"]], max_new_tokens=400,
                    non_streaming_mode=True)
                for r, w in zip(p["rows"], wavs):
                    f = f"{r['pid']}.wav"
                    sf.write(str(out / f), np.asarray(w, dtype=np.float32), sr)
                    done[r["pid"]] = {**r, "file": f}
                    fh.write(json.dumps(done[r["pid"]], ensure_ascii=False) + "\n")
                fh.flush()
                if k % 20 == 0:
                    print(f"[eval] {k + 1}/{len(todo)} batches", flush=True)
        del model
        torch.cuda.empty_cache()
    order = [r["pid"] for p in plan for r in p["rows"]]
    seen = set(order)
    write_jsonl(man_p, [done[pid] for pid in order if pid in done] +
                [m for pid, m in done.items() if pid not in seen])   # other seed bases stay scored


# ----------------------------------------------------------------- LTAS (A3 / A5)
def ltas13(path: str) -> tuple[np.ndarray, float]:
    """1/3-octave LTAS in dB (19 bands, nominal 160 Hz-10 kHz, each band the mean power of
    its FFT bins, normalised to the total over the bands), over active frames (energy
    above the clip's median frame), nfft 4096 / hop 512 Hann. Tilt: mean power 2-8 kHz
    minus mean power 100 Hz-1 kHz, in dB."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = wav.mean(-1) if wav.ndim > 1 else wav
    if len(wav) < NFFT:
        wav = np.pad(wav, (0, NFFT - len(wav)))
    frames = np.lib.stride_tricks.sliding_window_view(wav, NFFT)[::HOP] * np.hanning(NFFT)
    spec = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    e = spec.sum(-1)
    spec = spec[e > np.percentile(e, 50)] if len(e) > 4 else spec
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    p = spec.mean(0) + 1e-20
    bands = np.array([p[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].mean() for c in FC])
    bands = 10 * np.log10(bands / bands.sum())
    tilt = 10 * np.log10(p[(f >= 2000) & (f < 8000)].mean()) - 10 * np.log10(p[(f >= 100) & (f < 1000)].mean())
    return bands.astype(np.float32), float(tilt)


def real_speaker_ltas() -> dict:
    """Median LTAS of real takes per speaker (A5): validation+test takes, topped up from
    train when fewer than 30, untagged, >= 1.5 s, at most 200 (seeded sample)."""
    cache = WORK / "eval" / "real_ltas.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["data"].item()
    kept_p = WORK / "build_v3" / "kept.jsonl"
    if not kept_p.exists():
        print("[eval] no build_v3/kept.jsonl - A5 real-take LTAS unavailable")
        return {}
    kept = [k for k in read_jsonl(kept_p) if "[" not in k["text"] and k["dur"] >= 1.5]
    data = {}
    for spk in sorted({k["speaker"] for k in kept}):
        mine = sorted((k for k in kept if k["speaker"] == spk), key=lambda k: k["id"])
        takes = [k for k in mine if k["split"] in ("validation", "val", "test")]
        if len(takes) < 30:
            takes += [k for k in mine if k["split"] == "train"][: 200 - len(takes)]
        if len(takes) > 200:
            takes = random.Random(0).sample(takes, 200)
        spectra = [ltas13(k["audio"]) for k in takes if Path(k["audio"]).exists()]
        if not spectra:
            continue
        L = np.stack([s[0] for s in spectra])
        data[spk] = {"median_ltas": np.median(L, 0), "tilt_med": float(np.median([s[1] for s in spectra])),
                     "n": len(spectra), "ids": [k["id"] for k in takes]}
        print(f"[eval] real LTAS {spk}: {len(spectra)} takes")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, data=np.array(data, dtype=object), fc=FC)
    return data


# ----------------------------------------------------------------- centroid / de-dup
def centroid() -> np.ndarray:
    """v2's definition (p06_eval.ref_centroid): ECAPA centroid of v2's test-split takes,
    minus the 24 main-suite held-out lines, >= 25 codec frames (2.0 s). No tag filter.
    Audio: v2's own build if present (qwen/test_with_codes.jsonl), else build_v3 wavs."""
    cache = WORK / "eval" / "ref_centroid_v2def.npy"
    if cache.exists():
        return np.load(cache)
    v2_test = (CODE / "v2_test_ids.txt").read_text().split()
    main_ids = set((CODE / "suite_main_heldout_ids.txt").read_text().split())
    v2_src = WORK / "qwen" / "test_with_codes.jsonl"
    used, missing = [], []
    if v2_src.exists():
        for t in read_jsonl(v2_src):
            if Path(t["audio"]).stem not in main_ids and len(t["audio_codes"]) >= 25:
                used.append((Path(t["audio"]).stem, t["audio"]))
        source = str(v2_src)
    else:
        kept_p = WORK / "build_v3" / "kept.jsonl"
        kept = {k["id"]: k for k in read_jsonl(kept_p)} if kept_p.exists() else {}
        for i in v2_test:
            if i in main_ids:
                continue
            k = kept.get(i)
            path = Path(k["audio"]) if k else WORK / "build_v3" / "wavs" / f"{i}.wav"
            if not path.exists():
                missing.append(i)
                continue
            dur = k["dur"] if k else sf.info(str(path)).duration
            if dur * 12.5 >= 25:
                used.append((i, str(path)))
        source = "build_v3"
    c = metrics.centroid([p for _, p in used])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, c)
    (cache.with_suffix(".json")).write_text(json.dumps(
        {"source": source, "n": len(used), "ids": [i for i, _ in used], "missing": missing}, indent=1))
    print(f"[eval] centroid (v2 definition) from {len(used)} real test clips ({source}); "
          f"{len(missing)} v2 test ids missing")
    return c


def _norm_dedup(t: str) -> str:
    t = unicodedata.normalize("NFKC", re.sub(r"\[[^\]]*\]", "", t)).lower()
    return re.sub(r"[\W_]+", "", t)


def confirm_dedup(threshold: float = 0.85) -> dict:
    """A2 de-duplicated sub-score: confirm JA texts with a word-for-word or >= 0.85
    similar (difflib ratio on punctuation-free NFKC text) twin among build_v3 train texts."""
    cache = WORK / "eval" / "confirm_dedup.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    kept_p = WORK / "build_v3" / "kept.jsonl"
    if not kept_p.exists():
        print("[eval] no build_v3/kept.jsonl - de-dup sub-score unavailable")
        return {}
    train = {}
    for k in read_jsonl(kept_p):
        if k["split"] == "train":
            for t in (k.get("text"), k.get("text_plain")):
                if t and _norm_dedup(t):
                    train.setdefault(_norm_dedup(t), k["id"])
    out = dedup_against([r for r in suite_rows() if r["kind"] in SELECT_KINDS], train, threshold)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] de-dup: {sum(v['twin'] for v in out.values())}/{len(out)} confirm texts have a train twin")
    return out


def dedup_against(rows: list[dict], train: dict, threshold: float = 0.85) -> dict:
    keys = list(train)
    lens = np.array([len(k) for k in keys])
    out = {}
    for r in rows:
        q = _norm_dedup(r["text"])
        best, match = (1.0, q) if q in train else (0.0, None)
        if best < 1.0 and q:
            # ratio <= 2*min(len)/(sum len): skip candidates that cannot reach the threshold
            ok = np.nonzero(2 * np.minimum(lens, len(q)) / (lens + len(q)) >= threshold)[0]
            sm = difflib.SequenceMatcher(None, q, autojunk=False)
            for i in ok:
                sm.set_seq1(keys[i])
                if sm.real_quick_ratio() >= max(threshold, best) and sm.quick_ratio() >= max(threshold, best):
                    rt = sm.ratio()
                    if rt > best:
                        best, match = rt, keys[i]
        out[r["pid"]] = {"twin": bool(best >= threshold), "ratio": round(best, 3),
                         "train_id": train.get(match) if match else None}
    return out


# ----------------------------------------------------------------- scoring
def score(out: Path, centroid_vec: np.ndarray, cap: dict, shipped: str, use_ced: bool = True) -> dict:
    summary = metrics.score_dir(out, centroid_vec, text_norm="v3")
    rows = read_jsonl(out / "scored.jsonl") if (out / "scored.jsonl").exists() else []
    if not rows:
        summary = {"n": 0}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    spectra, tilts = [], []
    for r in rows:
        b, t = ltas13(str(out / r["file"]))
        spectra.append(b)
        tilts.append(t)
        r["tilt_db"] = round(t, 2)
    np.savez(out / "ltas.npz", pid=np.array([r["pid"] for r in rows]), ltas=np.stack(spectra),
             tilt=np.array(tilts, dtype=np.float32), fc=FC)
    write_jsonl(out / "scored.jsonl", rows)

    tokens = cap.get("tag_tokens", []) + cap.get("sfx_tokens", [])
    if use_ced and tokens:
        import ced_tagger
        sel = [r for r in rows if r["group"] in ("tags", "sfx") or
               (r["group"] == "confirm" and r["kind"] in SELECT_KINDS and r.get("mode") == shipped)]
        ced = ced_tagger.CED()
        names = ced_tagger.all_names(tokens)
        probs = ced.score([str(out / r["file"]) for r in sel], names)
        write_jsonl(out / "scored_ced.jsonl", [{"pid": r["pid"], "ced": p} for r, p in zip(sel, probs)])
        del ced

    xl = [r for r in rows if r["lang"] in ("zh", "en")]
    if xl:
        import accent_metrics
        write_jsonl(out / "manifest_accent.jsonl", xl)
        summary["accent"] = accent_metrics.score_dir(out, out / "manifest_accent.jsonl")

    # the pooled metrics "select" mixes modes and seeds; replace it with the shipped mode
    sel_rows = [r for r in rows if r["group"] == "confirm" and r["kind"] in SELECT_KINDS]
    shipped_rows = [r for r in sel_rows if r.get("mode") in (shipped, "real")]
    summary["select"] = metrics.aggregate(shipped_rows)
    summary["select_by"] = {f"{m}:s{b}": metrics.aggregate([r for r in sel_rows if r.get("mode") == m
                                                           and r.get("seed_base") == b])
                            for m, b in sorted({(r.get("mode"), r.get("seed_base")) for r in sel_rows})}
    for kind in sorted({r["kind"] for r in rows}):
        k = summary.setdefault(f"kind:{kind}", {})
        k["tilt_med"] = round(float(np.median([r["tilt_db"] for r in rows if r["kind"] == kind])), 2)
    summary.update(cap=cap, shipped_mode=shipped,
                   seed_bases=sorted({r["seed_base"] for r in rows if r.get("seed_base")}),
                   modes=sorted({r["mode"] for r in rows if r.get("mode")}))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def covered(summary_p: Path, bases: list[int], modes: list[str]) -> bool:
    if not summary_p.exists():
        return False
    s = json.loads(summary_p.read_text(encoding="utf-8"))
    return set(bases) <= set(s.get("seed_bases", [])) and set(modes) <= set(s.get("modes", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="'real', a checkpoint dir, or a hub id (optionally repo@<commit>)")
    ap.add_argument("--seeds", type=int, default=2, help="sampling seeds per text (A2: 2 ablations, 4 final)")
    ap.add_argument("--seed_bases", default=None, help="explicit comma list, e.g. 21000 for the fresh re-test")
    ap.add_argument("--modes", default=None, help="override confirm modes, e.g. auto,ja")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip a checkpoint whose summary already covers these seed bases and modes")
    ap.add_argument("--no_ced", action="store_true")
    args = ap.parse_args()
    bases = seed_bases(args.seeds, args.seed_bases)
    c = centroid()
    real_speaker_ltas()
    confirm_dedup()
    suite = WORK / "eval" / args.name
    table = {}
    for ckpt in args.ckpt:
        out = suite / tag_of(ckpt)
        cap = capabilities(ckpt)
        shipped, modes = ("real", ["real"]) if ckpt == "real" else modes_for(cap, args.modes)
        req_bases = [0] if ckpt == "real" else bases
        if args.skip_existing and (ckpt == "real" and (out / "summary.json").exists()
                                   or covered(out / "summary.json", req_bases, modes)):
            table[ckpt] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            continue
        plan = real_plan() if ckpt == "real" else build_plan(cap, bases, modes, shipped)
        n = sum(len(p["rows"]) for p in plan)
        print(f"[eval] {ckpt}: {n} rows in {len(plan)} batches, seed bases {req_bases}, modes {modes} "
              f"(shipped {shipped}), cap {json.dumps(cap)}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cap.json").write_text(json.dumps({**cap, "ckpt": ckpt, "shipped_mode": shipped,
                                                   "modes": modes}, indent=1), encoding="utf-8")
        generate(ckpt, plan, out)
        table[ckpt] = score(out, c, cap, shipped, use_ced=not args.no_ced)
        print(f"[eval] {ckpt}: select {json.dumps(table[ckpt].get('select'))}", flush=True)
    prev = suite / "table.json"
    merged = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    merged.update(table)
    prev.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
