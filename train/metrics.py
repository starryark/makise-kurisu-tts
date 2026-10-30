"""Objective scores for generated speech, model-agnostic.

score_dir(gen_dir) reads gen_dir/manifest.jsonl rows {file, text, lang} and adds:

  cer        reading-space CER (ja) or char CER (zh) / WER (en) of Qwen3-ASR vs the
             intended text - intelligibility and "said what it was asked to say"
  sim        cosine of an ECAPA embedding against the centroid of real held-out
             Kurisu clips - identity (only meaningful within one comparison; raw
             cosines between young female voices are all high)
  utmos      UTMOSv2 predicted MOS - naturalness proxy
  dur_ratio  duration over the expected duration at the corpus speaking rate;
             `runaway` (see is_runaway) is calibrated on real held-out takes

Used to rank checkpoints; the final call is still by ear.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
_models: dict = {}


def _asr():
    if "asr" not in _models:
        from qwen_asr import Qwen3ASRModel
        _models["asr"] = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", max_inference_batch_size=32, max_new_tokens=256)
    return _models["asr"]


def _ecapa():
    if "spk" not in _models:
        from speechbrain.inference.speaker import EncoderClassifier
        _models["spk"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(WORK / "cache" / "ecapa"), run_opts={"device": "cuda:0"})
    return _models["spk"]


def _utmos():
    # UTMOSv2's CPU feature path runs at <3 clips/s on the training pods, so it is
    # opt-in (METRICS_UTMOS=1) and used for the final candidates only.
    if os.environ.get("METRICS_UTMOS") != "1":
        return None
    if "utmos" not in _models:
        try:
            import utmosv2
            try:
                _models["utmos"] = utmosv2.create_model(pretrained=True, device="cuda:0")
            except TypeError:
                _models["utmos"] = utmosv2.create_model(pretrained=True)
        except Exception as exc:  # optional dependency
            print(f"[metrics] UTMOSv2 unavailable: {exc}")
            _models["utmos"] = None
    return _models["utmos"]


def embed(paths: list[str]) -> np.ndarray:
    import torchaudio
    enc = _ecapa()
    out = []
    for p in paths:
        wav, sr = torchaudio.load(p)
        wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, 16000)
        with torch.no_grad():
            e = enc.encode_batch(wav.to("cuda:0")).squeeze().float().cpu().numpy()
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.stack(out)


def centroid(paths: list[str]) -> np.ndarray:
    c = embed(paths).mean(0)
    return c / np.linalg.norm(c)


_EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}


def _en_int(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        return _EN_TENS[n // 10] + ("" if n % 10 == 0 else " " + _EN_ONES[n % 10])
    for unit, word in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= unit:
            rest = n % unit
            return _en_int(n // unit) + " " + word + ("" if rest == 0 else " " + _en_int(rest))
    return str(n)


def _en_year(n: int) -> str:
    if 2000 <= n <= 2009:
        return _en_int(n)
    hi, lo = divmod(n, 100)
    return _en_int(hi) + (" hundred" if lo == 0 else " oh " + _EN_ONES[lo] if lo < 10 else " " + _en_int(lo))


def _en_ordinal(n: int) -> str:
    w = _en_int(n).split()
    last = w[-1]
    w[-1] = _EN_ORD.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(w)


def en_spell_numbers(text: str) -> str:
    """Digits -> words, applied to reference and ASR output alike (v3 scoring), so a
    number counts as read correctly whichever form the ASR writes it in.
    3:30 -> three thirty, 1.5 -> one point five, 60% -> sixty percent, 15th -> fifteenth,
    2026 -> twenty twenty six (4-digit 1100-2099), 42 -> forty two."""
    # comma-grouped numbers are cardinals, never years
    t = re.sub(r"\d{1,3}(?:,\d{3})+(?![\d.])", lambda m: _en_int(int(m.group(0).replace(",", ""))), text)
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: _en_int(int(m.group(1))) + (
        "" if m.group(2) == "00" else " oh " + _EN_ONES[int(m.group(2))] if m.group(2)[0] == "0"
        else " " + _en_int(int(m.group(2)))), t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: _en_ordinal(int(m.group(1))), t)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: _en_int(int(m.group(1))) + " point " +
               " ".join(_EN_ONES[int(c)] for c in m.group(2)), t)
    t = re.sub(r"\d+", lambda m: _en_year(int(m.group(0))) if len(m.group(0)) == 4 and
               1100 <= int(m.group(0)) <= 2099 else _en_int(int(m.group(0))), t)
    return t


def _norm_text(text: str, lang: str, mode: str = "v2") -> str:
    """mode "v2" is v2's scoring exactly. mode "v3" spells numbers on both sides (zh via
    accent_metrics.normalize_zh: 3点20分 -> 三点二十分; en via en_spell_numbers), so
    number syllables are scored instead of dropped (zh) or mismatched (en)."""
    text = unicodedata.normalize("NFKC", re.sub(r"\[[a-z_]+\]", "", text)).lower()
    if lang == "ja":
        from p03_build import reading
        return reading(text)
    if lang == "zh":
        if mode == "v3":
            from accent_metrics import normalize_zh
            text = "".join(normalize_zh(text)[0])
        return "".join(re.findall(r"[一-鿿]", text))
    if mode == "v3":
        text = en_spell_numbers(text)
    return " ".join(re.findall(r"[a-z0-9']+", text))


def _err(ref: str, hyp: str, lang: str) -> float:
    from p03_build import cer
    if lang == "en":
        ref_w, hyp_w = ref.split(), hyp.split()
        # word-level edit distance via the same routine on token ids
        vocab = {w: chr(0x4e00 + i) for i, w in enumerate(sorted(set(ref_w + hyp_w)))}
        return cer("".join(vocab[w] for w in ref_w), "".join(vocab[w] for w in hyp_w))
    return cer(ref, hyp)


LANG_NAME = {"ja": "Japanese", "en": "English", "zh": "Chinese"}
UNITS_PER_S = {"ja": 7.9, "zh": 4.5, "en": 2.6}  # real-speech medians; ja from the corpus


def score_dir(gen_dir: Path, ref_centroid: np.ndarray, text_norm: str = "v2") -> dict:
    """text_norm: "v2" (v2's p06_eval, unchanged) or "v3" (p06_eval_v3: numbers spelled
    on both sides, see _norm_text). With no scorable rows, v2 returns {"n": 0} and writes
    nothing; v3 also writes an empty scored.jsonl so callers can read it."""
    rows = [json.loads(l) for l in (gen_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    rows = [r for r in rows if r.get("file") and (gen_dir / r["file"]).is_file()]
    if not rows:
        if text_norm != "v2":
            (gen_dir / "scored.jsonl").write_text("", encoding="utf-8")
        return {"n": 0}
    paths = [str(gen_dir / r["file"]) for r in rows]
    asr = _asr()
    for lang in {r["lang"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        for s in range(0, len(idx), 32):
            chunk = idx[s:s + 32]
            res = asr.transcribe([paths[i] for i in chunk], language=LANG_NAME[lang])
            for i, h in zip(chunk, res):
                rows[i]["asr"] = h.text.strip()
    embs = embed(paths)
    utmos = _utmos()
    import soundfile as sf
    for i, r in enumerate(rows):
        ref = _norm_text(r["text"], r["lang"], text_norm)
        hyp = _norm_text(r.get("asr", ""), r["lang"], text_norm)
        r["cer"] = round(_err(ref, hyp, r["lang"]), 4)
        r["sim"] = round(float(embs[i] @ ref_centroid), 4)
        info = sf.info(paths[i])
        r["dur"] = round(info.duration, 3)
        units = len(ref) if r["lang"] != "en" else len(ref.split())
        r["dur_ratio"] = round(r["dur"] / max(units / UNITS_PER_S[r["lang"]], 0.5), 3)
        if utmos is not None:
            try:
                r["utmos"] = round(float(utmos.predict(input_path=paths[i])), 3)
            except Exception:
                r["utmos"] = None
    (gen_dir / "scored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return summarize(gen_dir, rows)


def is_runaway(r: dict) -> bool:
    """Calibrated on 536 real held-out/val takes (acted speech is slow and pause-heavy:
    duration ratio p99 4.9, excess p99 6.4 s): hitting the 400-frame generation cap,
    or both >5x the expected duration and >6.5 s over it. Real audio passes ~99%."""
    expected = r["dur"] / max(r["dur_ratio"], 1e-6)
    return r["dur"] >= 31.0 or (r["dur_ratio"] > 5.0 and r["dur"] - expected > 6.5)


def aggregate(sel: list[dict]) -> dict:
    """CER / SIM / runaway summary of scored rows (the `select` block's format)."""
    if not sel:
        return {}
    out = {"n": len(sel),
           "cer_mean": round(float(np.mean([r["cer"] for r in sel])), 4),
           "cer_gt20": round(float(np.mean([r["cer"] > 0.2 for r in sel])), 3),
           "sim_mean": round(float(np.mean([r["sim"] for r in sel])), 4),
           "runaway": int(sum(is_runaway(r) for r in sel)),
           "dur_ratio_med": round(float(np.median([r["dur_ratio"] for r in sel])), 3)}
    u = [r["utmos"] for r in sel if r.get("utmos") is not None]
    if u:
        out["utmos_mean"] = round(float(np.mean(u)), 3)
    return out


def summarize(gen_dir: Path, rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [json.loads(l) for l in (gen_dir / "scored.jsonl").read_text(encoding="utf-8").splitlines() if l]
    agg = aggregate

    # "select" is the pre-registered checkpoint criterion: untagged Japanese only
    # (held-out test lines + novel sentences). Tagged probes are diagnostics.
    summary = {"select": agg([r for r in rows if r.get("kind") in ("ja_heldout", "ja_novel")]),
               "all": agg(rows)}
    for lang in sorted({r["lang"] for r in rows}):
        summary[lang] = agg([r for r in rows if r["lang"] == lang])
    for kind in sorted({r.get("kind") for r in rows if r.get("kind")}):
        summary[f"kind:{kind}"] = agg([r for r in rows if r.get("kind") == kind])
    (gen_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
