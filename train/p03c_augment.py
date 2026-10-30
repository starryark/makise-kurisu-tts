"""v3 augmentation: (A) phone-channel copies and (B) background-SFX mixes of clean game Kurisu.

(A) Phone (speaker row `phone`).  The show's phone voice is a stylised device EQ, not a
    300-3400 Hz telephone band (research/labeling_sota_2026-10.md §1.4). The curve is measured
    here from the 28 real `[digital]` clips:
    - device_response = median LTAS(phone) - median LTAS(same-series anime Kurisu), reported;
    - applied_response = median LTAS(phone) - median LTAS(the selected clean pool), calibrated
      in the band domain and on the jittered filtered copies, so the augmented median lands
      on the real phone median (spectral shape). The inputs are game clips, which differ
      from anime by up to ~3.7 dB, so the device curve alone would miss.
    ~400 diverse clean game clips get a minimum-phase FIR of that curve with per-clip
    jitter (tilt, band-edge warp). Compression/saturation/noise are added only if the
    measured non-linear statistics call for it (rule in `nonlinear_decision`). A share
    (default 15%) goes through a true narrowband chain (300-3400 Hz, 8 kHz, 8-bit mu-law).
    Validation is the per-band median LTAS shape error against the real phone clips, plus an
    out-of-sample check on unseen clean clips and the bootstrap SD of the 28-clip target.
(B) Background SFX (speaker `kurisu`, text prefixed `[bg_<class>]`).  CapSpeech-style mixing of
    CC0 / CC-BY FSD50K clips (licence checked per clip; voice/music co-labels excluded; library
    sources the AST tagger cannot recognise as their class are dropped, per-class floors in SFX)
    under clean game Kurisu, with continuous ambiences looped and events placed in pauses. Each source is used at most N times
    and each speech clip at most once. Some source sounds per class are held out
    (other uploaders) for evaluation mixes only.

Every augmented clip is tagged and its clean source stays in training untouched, so the untagged
default stays clean. Sources come from the train split only. The ref_audio clip, tagged and
contaminated clips are never used. The phone and SFX speech pools are disjoint.

Outputs under --out (default $KURISU_WORK/augment): wavs/{phone,sfx,sfx_eval}/*.wav,
augment_rows.jsonl, phone_curve.json, sfx_attribution.json, sfx_eval_rows.jsonl, report.json.

Pod:   python p03c_augment.py --meta $KURISU_WORK/src/ja_meta.jsonl --wav_dir $KURISU_WORK/build_v3/wavs            --kept $KURISU_WORK/build_v3/kept.jsonl
Local: python p03c_augment.py --meta <hf_dataset_upload>/data --out <dir>
       (a directory --meta is read like p01_fetch does. Without --wav_dir, raw wavs are trimmed and
       levelled with p03_build.normalise_audio first, so they look like build output.)
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p03_build import (LEAD_PAD, PEAK_MAX, SR, TAIL_PAD, TARGET_LUFS, normalise_audio,  # noqa: E402
                       speech_bounds, train_text)
LEAD, TAIL = int(LEAD_PAD * SR), int(TAIL_PAD * SR)   # the build's fixed silent pads

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
REF_ID = "game_ja_SGRE_kurisu_01686"   # the single clean ref_audio of every row; never augmented
BRACKET = re.compile(r"\[[^\]]*\]")

# 1/6-octave bands, 62.5 Hz .. 10.08 kHz (centres 1 kHz * 2^(k/6)); validation uses 150 Hz - 8 kHz.
FC = 1000.0 * 2.0 ** (np.arange(-24, 21) / 6.0)
F_LO, F_HI = FC * 2 ** (-1 / 12), FC * 2 ** (1 / 12)
VAL = (FC >= 150) & (FC <= 8000)
NFFT, N_TAPS = 32768, 4096          # FIR design grid (0.73 Hz) and length (171 ms)


# ============================================================== shared helpers

def log(msg: str) -> None:
    print(f"[augment {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_mono(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


def core(x: np.ndarray) -> np.ndarray:
    b = speech_bounds(x.astype(np.float32))
    return x[b[0]:b[1]] if b else x


def clip_rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def lufs(x: np.ndarray, meter) -> float:
    y = x if len(x) >= int(0.4 * SR) else np.tile(x, int(0.4 * SR) // max(1, len(x)) + 1)
    try:
        val = meter.integrated_loudness(y)
    except Exception:
        val = float("-inf")
    if not np.isfinite(val):
        val = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) - 3.0
    return float(val)


def set_level(y: np.ndarray, meter, target: float) -> np.ndarray:
    """Loudness of the speech core (pads excluded) to `target` LUFS, then peak <= -1 dBFS."""
    y = y * 10 ** ((target - lufs(y[LEAD:len(y) - TAIL], meter)) / 20)
    return peak_limit(y)


def peak_limit(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    return y * (PEAK_MAX / peak) if peak > PEAK_MAX else y


def seal(y: np.ndarray) -> np.ndarray:
    """Exact digital silence in the build's lead/tail pads, 5 ms fades into them.

    A codec LM learns when to stop from what precedes EOS; filter ringing or a
    background running into the tail would teach "sound continues after speech".
    """
    y = y.copy()
    f = int(0.005 * SR)
    hi = len(y) - TAIL
    y[:LEAD] = 0.0
    y[hi:] = 0.0
    y[LEAD:LEAD + f] *= np.linspace(0, 1, f)
    y[hi - f:hi] *= np.linspace(1, 0, f)
    return y


def level_target(clean: np.ndarray, meter) -> float:
    # The clean input is already at the build's level (-23 LUFS unless the build's gain clamp or
    # peak limit held it lower); the augmented copy keeps that level, so a whisper stays quiet.
    return min(TARGET_LUFS, lufs(clean, meter))


def write_wav(path: Path, y: np.ndarray) -> None:
    if np.any(y[:LEAD] != 0) or np.any(y[len(y) - TAIL:] != 0):
        raise SystemExit(f"{path.name}: sound in the silent lead/tail pad")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), SR, subtype="PCM_16")


def frame_view(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop]


def frame_db(x: np.ndarray, n: int = 480, hop: int = 240) -> np.ndarray:
    return 10 * np.log10(np.mean(frame_view(x, n, hop) ** 2, axis=1) + 1e-12)


def active_db(x: np.ndarray, rel: float = 30.0) -> float:
    """Mean power (dB) of the 20 ms frames within `rel` dB of the clip's loud (p95) frames."""
    e = frame_db(x)
    sel = e > np.percentile(e, 95) - rel
    return float(10 * np.log10(np.mean(10 ** (e[sel] / 10)) + 1e-12))


# ============================================================== metadata and clip selection

def load_meta(path: Path) -> list[dict]:
    if path.is_dir():   # HF `data/` root, indexed like p01_fetch (Japanese rows only)
        rows = []
        for split in ("train", "validation", "test"):
            for line in (path / split / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["language"] == "ja":
                        p = path / split / r["file_name"]
                        r.update(id=p.stem, audio=str(p), split=split)
                        rows.append(r)
        return rows
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_phone(r: dict) -> bool:
    tags = set(r.get("tags") or []) | set(r.get("condition_tags") or [])
    return r["character_variant"] == "kurisu" and "[digital]" in tags


def punct_class(text: str) -> str:
    if re.search(r"[？?]", text):
        return "question"
    if re.search(r"[！!]", text):
        return "exclamation"
    if re.search(r"[…‥]|――", text):
        return "ellipsis"
    return "plain"


def dur_bin(d: float) -> str:
    return "short" if d < 3 else ("mid" if d < 6 else "long")


def clean_candidates(meta: list[dict], args, split: str) -> list[dict]:
    """Clean game-Kurisu tier-A clips usable as augmentation sources. With --kept (the v3 build's
    kept.jsonl) a source must be a kept `kurisu` row whose v3 text carries no tag, and that text
    becomes the augmented rows' text, so clean and augmented copies always agree."""
    kept, allowed = None, None
    if args.kept:
        kept = {k["id"]: k for k in (json.loads(l) for l in Path(args.kept).read_text(encoding="utf-8").splitlines()
                                     if l.strip())}
    elif args.wav_dir:
        allowed = {p.stem for p in Path(args.wav_dir).glob("*.wav")}
    out = []
    for r in meta:
        if (r["split"] != split or r["medium"] != "game" or r["character_variant"] != "kurisu"
                or r["quality_tier"] != "A" or r.get("style", "speech") != "speech" or r["id"] == REF_ID
                or BRACKET.search(r["transcription"]) or r.get("tags") or r.get("contamination")
                or not re.sub(r"[\W_]+", "", r["text"])):
            continue
        if allowed is not None and r["id"] not in allowed:
            continue
        text = train_text(r["transcription"])
        if kept is not None:
            k = kept.get(r["id"])
            if (k is None or k.get("speaker", "kurisu") != "kurisu" or BRACKET.search(k.get("text", ""))
                    or k.get("events") or k.get("bg")):
                continue
            text = k.get("text") or text
        if args.wav_dir:
            info = sf.info(str(Path(args.wav_dir) / f"{r['id']}.wav"))
            dur = info.frames / info.samplerate
        else:   # what p03_build.normalise_audio will leave: speech span + 0.12 s + 0.25 s pads
            dur = r["duration_seconds"] - r.get("lead_silence_s", 0) - r.get("tail_silence_s", 0) + 0.37
        if args.min_dur <= dur <= args.max_dur:
            out.append(dict(r, dur_est=round(dur, 3), text_train=text,
                            stratum=(r["series"], dur_bin(dur), punct_class(r["text"]))))
    return out


def stratified_order(cands: list[dict], rng: np.random.Generator) -> list[dict]:
    """Round-robin over (series x length bin x punctuation) strata: near-equal allocation."""
    groups: dict[tuple, list] = defaultdict(list)
    for c in sorted(cands, key=lambda c: c["id"]):
        groups[c["stratum"]].append(c)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    out = []
    while any(groups[k] for k in keys):
        rng.shuffle(keys)
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop())
    return out


def load_clean(r: dict, args, meter) -> np.ndarray | None:
    if args.wav_dir:
        return read_mono(Path(args.wav_dir) / f"{r['id']}.wav")
    y, _ = normalise_audio(read_mono(r["audio"]).astype(np.float32), meter)
    return None if y is None else y.astype(np.float64)


def base_row(r: dict, rid: str, path: Path, text: str, speaker: str, augment: str) -> dict:
    return {"id": rid, "audio": str(path), "text": text, "language": "japanese", "speaker": speaker,
            "source_id": r["id"], "augment": augment, "sfx_class": None, "sfx_source": None,
            "sfx_license": None, "snr_db": None, "split": "train"}


# ============================================================== (A) phone: measurement

def band_power(x: np.ndarray) -> np.ndarray | None:
    """Power per 1/6-octave band (Welch, 4096-sample window, 2.9 Hz bins) of the speech core."""
    x = core(x)
    if len(x) < 2048:
        return None
    n = min(4096, len(x))
    f, p = signal.welch(x, SR, nperseg=n, noverlap=3 * n // 4, nfft=8192)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    lo, hi = np.searchsorted(f, F_LO), np.searchsorted(f, F_HI)
    return cs[hi] - cs[lo] + 1e-30


def norm_db(bp: np.ndarray) -> np.ndarray:
    """Per-clip LTAS normalised to the clip's own 62 Hz - 10 kHz power, dB."""
    return 10 * np.log10(bp / bp.sum(axis=-1, keepdims=True))


def smooth(v: np.ndarray, passes: int = 1) -> np.ndarray:
    for _ in range(passes):
        p = np.pad(v, 1, mode="edge")
        v = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return v


def voice_stats(x: np.ndarray) -> dict:
    """Non-linear character on the speech core: HNR proxy (normalised autocorrelation peak,
    Boersma window correction, 40 ms frames, F0 75-600 Hz, median of active frames), crest
    factor, 20 ms frame dynamic range (p95-p10 of active frames), spectral flatness 150 Hz-8 kHz."""
    x = core(x)
    fr = frame_view(x, 960, 240)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    fr = fr[e > np.percentile(e, 95) - 25]
    w = np.hanning(960)
    spec = np.fft.rfft(fr * w, 2048)
    ac = np.fft.irfft(np.abs(spec) ** 2, 2048)[:, :400]
    wac = np.fft.irfft(np.abs(np.fft.rfft(w, 2048)) ** 2, 2048)[:400]
    r = (ac / (ac[:, :1] + 1e-20)) / (wac / wac[0])
    rmax = np.clip(r[:, SR // 600:SR // 75].max(axis=1), 1e-3, 0.999)
    f = np.fft.rfftfreq(2048, 1 / SR)
    pb = np.abs(spec[:, (f >= 150) & (f <= 8000)]) ** 2 + 1e-20
    flat = 10 * np.log10(np.exp(np.mean(np.log(pb), axis=1)) / np.mean(pb, axis=1))
    e20 = frame_db(x, 480, 480)
    e20 = e20[e20 > e20.max() - 40]
    return {"hnr_db": float(np.median(10 * np.log10(rmax / (1 - rmax)))),
            "crest_db": float(20 * np.log10(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))),
            "dyn_range_db": float(np.percentile(e20, 95) - np.percentile(e20, 10)),
            "flatness_db": float(np.median(flat))}


def group_stats(stats: list[dict]) -> dict:
    return {k: round(float(np.median([s[k] for s in stats])), 2) for k in stats[0]} if stats else {}


def measure(items, loader) -> tuple[np.ndarray, list[dict], list[str]]:
    bps, stats, ids = [], [], []
    for it in items:
        x = loader(it)
        if x is None:
            continue
        bp = band_power(x)
        if bp is None:
            continue
        bps.append(bp)
        stats.append(voice_stats(x))
        ids.append(it["id"])
    return np.array(bps), stats, ids


# ============================================================== (A) phone: the chain

def dense_db(curve: np.ndarray, warp: float = 1.0, tilt_db: float = 0.0) -> np.ndarray:
    """Band curve -> FFT grid. warp scales the frequency axis (features move by x warp); tilt is
    a ramp linear in log f from -tilt at 150 Hz to +tilt at 8 kHz. Flat beyond the band range."""
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    g = np.interp(np.log(np.clip(f / warp, FC[0], FC[-1])), np.log(FC), curve)
    if tilt_db:
        lo, hi = np.log(150.0), np.log(8000.0)
        pos = (np.clip(np.log(np.maximum(f, 1.0)), lo, hi) - (lo + hi) / 2) / ((hi - lo) / 2)
        g = g + tilt_db * pos
    return g


def minphase_fir(g_db: np.ndarray) -> np.ndarray:
    """Minimum-phase FIR (homomorphic/cepstral folding) with magnitude g_db on the NFFT grid."""
    logmag = (g_db - g_db.max()) * (np.log(10) / 20)
    cep = np.fft.irfft(logmag, NFFT)
    fold = np.zeros(NFFT)
    fold[0], fold[1:NFFT // 2], fold[NFFT // 2] = 1.0, 2.0, 1.0
    h = np.fft.irfft(np.exp(np.fft.rfft(cep * fold)), NFFT)[:N_TAPS]
    t = N_TAPS // 8
    h[-t:] *= np.hanning(2 * t)[t:]
    return h


def compress(y: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """Mild feed-forward RMS compressor (10 ms frames, 5 ms attack / 80 ms release)."""
    ratio, below = float(rng.uniform(2.0, 3.5)), float(rng.uniform(8.0, 14.0))
    e = frame_db(y, 240, 240)
    thr = np.percentile(e, 95) - below
    target_gain = -np.maximum(e - thr, 0.0) * (1 - 1 / ratio)
    g = np.empty_like(target_gain)
    att, rel = math.exp(-10 / 5), math.exp(-10 / 80)
    cur = 0.0
    for i, tg in enumerate(target_gain):
        a = att if tg < cur else rel
        cur = a * cur + (1 - a) * tg
        g[i] = cur
    gs = np.interp(np.arange(len(y)), np.arange(len(g)) * 240 + 120, g)
    return y * 10 ** (gs / 20), {"ratio": round(ratio, 2), "thr_below_p95_db": round(below, 1)}


def saturate(y: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    drive = float(rng.uniform(1.5, 3.0))
    pk = np.max(np.abs(y)) + 1e-12
    return np.tanh(drive * y / pk) / np.tanh(drive) * pk, {"drive": round(drive, 2)}


def add_noise_floor(y: np.ndarray, rng, snr_db: float, h: np.ndarray) -> tuple[np.ndarray, dict]:
    n = signal.fftconvolve(rng.standard_normal(len(y)), h)[:len(y)]
    n *= 10 ** ((active_db(y) - snr_db - active_db(n)) / 20)
    return y + n, {"noise_snr_db": round(snr_db, 1)}


NB_SOS = signal.butter(6, [300, 3400], btype="bandpass", fs=SR, output="sos")


def narrowband(y: np.ndarray) -> np.ndarray:
    """Telephone channel: 300-3400 Hz, 8 kHz, G.711-style 8-bit mu-law, back to 24 kHz."""
    z = signal.resample_poly(signal.sosfilt(NB_SOS, y), 1, 3)
    pk = np.max(np.abs(z)) + 1e-12
    z = z / pk * 0.9
    mu = 255.0
    c = np.sign(z) * np.log1p(mu * np.abs(z)) / np.log1p(mu)
    c = np.round(c * 127) / 127
    z = np.sign(c) * np.expm1(np.abs(c) * np.log1p(mu)) / mu * pk / 0.9
    out = signal.resample_poly(z, 3, 1)[:len(y)]
    return np.pad(out, (0, len(y) - len(out)))


def nonlinear_decision(st: dict, args, phone_snr: float, pool_snr: float) -> dict:
    """Rule: a non-linear stage is added only if the EQ-only copies miss the real phone clips by
    more than the source difference explains: (aug - phone) - (pool - anime) beyond 3 dB.
    The source difference (game studio vs source-separated anime) must not be imitated (§9.2).
    No noise floor unless the phone clips are noisier between words than the pool by > 6 dB."""
    def excess(k):
        return round((st["eq_only"][k] - st["phone"][k]) - (st["pool"][k] - st["anime"][k]), 2)
    ex = {k: excess(k) for k in ("hnr_db", "crest_db", "dyn_range_db", "flatness_db")}
    auto = {"compress": ex["crest_db"] > 3 or ex["dyn_range_db"] > 3,
            "saturate": ex["hnr_db"] > 3,
            "noise": (pool_snr - phone_snr) > 6}
    final = {k: (auto[k] if getattr(args, f"phone_{k}") == "auto" else getattr(args, f"phone_{k}") == "on")
             for k in auto}
    return {"excess_db": ex, "snr_est_db_median": {"phone": phone_snr, "pool": pool_snr},
            "auto": auto, "applied": final}


def run_phone(args, meta: list[dict], sel: list[dict], fresh: list[dict], meter,
              out: Path) -> tuple[list[dict], dict]:
    # train takes only: the eval compares the phone row with the real takes, so the
    # curve must not be fitted on val/test ones
    phone = [r for r in meta if is_phone(r) and r.get("split") == "train"]
    series = {r["series"] for r in phone}
    anime = [r for r in meta if r["medium"] == "anime" and r["character_variant"] == "kurisu"
             and r["series"] in series and r["quality_tier"] in ("A", "B")
             and r.get("style", "speech") == "speech" and not is_phone(r)]
    log(f"phone: {len(phone)} real phone clips, {len(anime)} matched anime clips, {len(sel)} pool clips")
    raw = lambda r: read_mono(r["audio"])          # noqa: E731  (real clips: raw source audio, speech core)
    B_ph, S_ph, _ = measure(phone, raw)
    B_an, S_an, _ = measure(anime, raw)
    clean = {}
    for r in sel:
        x = load_clean(r, args, meter)
        if x is not None:
            clean[r["id"]] = x.astype(np.float32)
    sel = [r for r in sel if r["id"] in clean]
    B_po, S_po, _ = measure(sel, lambda r: clean[r["id"]].astype(np.float64))
    L_ph, L_an, L_po = (np.median(norm_db(B), axis=0) for B in (B_ph, B_an, B_po))

    # Comparisons are of spectral *shape*: per-clip normalisation followed by a per-band median
    # leaves a group-dependent constant (it depends on how dispersed the clips' spectra are, and
    # no EQ can change it), so every error is taken after removing its mean over 150 Hz - 8 kHz.
    def shape_err(med):
        e = med - L_ph
        return e - e[VAL].mean(), float(e[VAL].mean())

    rng = np.random.default_rng(args.seed + 1)
    n_nb = int(round(args.phone_narrowband_share * len(sel)))
    nb_ids = {sel[i]["id"] for i in rng.choice(len(sel), size=n_nb, replace=False)} if n_nb else set()
    eq_sel = [r for r in sel if r["id"] not in nb_ids]

    def jitter(rid: str) -> tuple[float, float]:     # deterministic per clip: (warp, tilt_db)
        g = clip_rng(args.seed, "phone_jitter:" + rid)
        return (float(g.uniform(1 - args.phone_warp, 1 + args.phone_warp)),
                float(g.uniform(-args.phone_tilt_db, args.phone_tilt_db)))

    # Per-clip curve variation as wide as the curve's own uncertainty: each clip adds the
    # deviation of one bootstrap resample of the real phone takes (review: the fixed
    # tilt/warp jitter was narrower than that in 27 of 35 bands, mostly below 500 Hz).
    brng = np.random.default_rng(args.seed + 7)
    boot = [smooth(np.median(norm_db(B_ph[brng.integers(0, len(B_ph), len(B_ph))]), axis=0) - L_ph)
            for _ in range(64)]

    def eq(x: np.ndarray, c: np.ndarray, rid: str) -> np.ndarray:
        k = zlib.crc32(("phone_boot:" + rid).encode("utf-8")) % len(boot)
        h = minphase_fir(dense_db(c + boot[k], *jitter(rid)))
        return signal.fftconvolve(x, h)[:len(x)]

    def predict(c):     # band domain: each pool clip's band powers x gain, renormalised
        return np.median(norm_db(B_po * 10 ** (c / 10)), axis=0)

    def med_of(ys):
        return np.median(norm_db(np.array([band_power(y) for y in ys])), axis=0)

    # applied curve: smooth(phone - pool), refined (1) in the band domain and (2) on the jittered
    # EQ-only copies themselves (the jitter is deterministic per clip), so what is written is what
    # was calibrated. Each correction is smoothed so the curve does not chase the 1/6-octave
    # ripple of 28 short clips; an out-of-sample check on fresh clips is reported below.
    curve = smooth(L_ph - L_po)
    for _ in range(args.phone_calib_iters):
        curve = curve - smooth(shape_err(predict(curve))[0])
    for _ in range(args.phone_audio_iters):
        e = shape_err(med_of([eq(clean[r["id"]].astype(np.float64), curve, r["id"]) for r in eq_sel]))[0]
        curve = curve - smooth(e)
    curve -= curve[np.argmin(np.abs(FC - 1000))]       # 0 dB at 1 kHz (offset is irrelevant)
    probe_y = [eq(clean[r["id"]].astype(np.float64), curve, r["id"]) for r in eq_sel]
    calib_err = shape_err(med_of(probe_y))[0]
    log(f"phone: calibrated shape error (jittered EQ-only) max {np.max(np.abs(calib_err[VAL])):.2f} dB")
    fresh_ys = []
    for r in fresh:
        x = load_clean(r, args, meter)
        if x is not None:
            fresh_ys.append(eq(x, curve, r["id"]))
    fresh_err = shape_err(med_of(fresh_ys))[0] if fresh_ys else None
    if fresh_err is not None:
        log(f"phone: out-of-sample ({len(fresh_ys)} fresh clips) shape error max {np.max(np.abs(fresh_err[VAL])):.2f} dB")

    # FIR check: designed magnitude (no jitter) vs target on 100 Hz - 10 kHz
    h0 = minphase_fir(dense_db(curve))
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    resp = 20 * np.log10(np.abs(np.fft.rfft(h0, NFFT)) + 1e-12)
    tgt = dense_db(curve)
    m = (f >= 100) & (f <= 10000)
    off = np.median(resp[m] - tgt[m])
    fir_err = float(np.max(np.abs(resp[m] - tgt[m] - off)))
    log(f"phone: FIR max |designed - target| 100 Hz-10 kHz = {fir_err:.2f} dB")

    # non-linear probe: the EQ-only copies, before any non-linear stage
    st = {"phone": group_stats(S_ph), "anime": group_stats(S_an), "pool": group_stats(S_po),
          "eq_only": group_stats([voice_stats(y) for y in probe_y])}
    del probe_y, fresh_ys
    snr = lambda rows: round(float(np.median([r["snr_est_db"] for r in rows if r.get("snr_est_db") is not None])), 1)  # noqa: E731
    dec = nonlinear_decision(st, args, snr(phone), snr(sel))
    log(f"phone: non-linear stats {json.dumps(st)}; decision {json.dumps(dec)}")

    # process
    rows, out_bp, out_nb, out_stats = [], [], [], []
    for i, r in enumerate(sel):
        x = clean.pop(r["id"]).astype(np.float64)
        crng = clip_rng(args.seed, "phone:" + r["id"])
        warp, tilt = jitter(r["id"])
        h = minphase_fir(dense_db(curve, warp, tilt))
        y = signal.fftconvolve(x, h)[:len(x)]
        params = {"warp": round(warp, 3), "tilt_db": round(tilt, 2)}
        if dec["applied"]["compress"]:
            y, p = compress(y, crng)
            params.update(p)
        if dec["applied"]["saturate"]:
            y, p = saturate(y, crng)
            params.update(p)
        if dec["applied"]["noise"]:
            y, p = add_noise_floor(y, crng, float(crng.uniform(*args.phone_noise_snr)), h)
            params.update(p)
        is_nb = r["id"] in nb_ids
        if is_nb:
            y = narrowband(y)
        y = seal(set_level(y, meter, level_target(x, meter)))
        kind = "phone_narrowband" if is_nb else "phone_eq"
        rid = f"aug_{'phonenb' if is_nb else 'phone'}_{r['id']}"
        path = out / "wavs" / "phone" / f"{rid}.wav"
        write_wav(path, y)
        row = base_row(r, rid, path, r["text_train"], "phone", kind)
        row["params"] = params
        rows.append(row)
        bp = band_power(y)
        if bp is not None:
            out_bp.append(bp)
            out_nb.append(is_nb)
        out_stats.append(voice_stats(y))
        if (i + 1) % 100 == 0:
            log(f"phone: {i + 1}/{len(sel)}")

    out_bp, out_nb = np.array(out_bp), np.array(out_nb, bool)
    err_eq, off_eq = shape_err(np.median(norm_db(out_bp[~out_nb]), axis=0))
    err_all, off_all = shape_err(np.median(norm_db(out_bp), axis=0))
    err_dev, _ = shape_err(predict(smooth(L_ph - L_an)))   # what the device-only curve would give
    err_pool, _ = shape_err(L_po)                           # unprocessed pool vs real phone
    # how well 28 clips pin the target down: bootstrap SD of the (shape-normalised) phone median
    n_ph, brng = norm_db(B_ph), np.random.default_rng(args.seed + 5)
    boot_sd = np.array([shape_err(np.median(n_ph[brng.integers(0, len(n_ph), len(n_ph))], axis=0))[0]
                        for _ in range(300)]).std(axis=0)
    mx = lambda e: round(float(np.max(np.abs(e[VAL]))), 2)  # noqa: E731
    r2 = lambda v: [round(float(a), 2) for a in v]  # noqa: E731
    validation = {
        "tolerance_db": 1.5, "band_range_hz": [150, 8000],
        "metric": "median normalised LTAS of augmented minus real phone, mean over 150 Hz-8 kHz removed",
        "eq_only_max_abs_err_db": mx(err_eq),
        "eq_only_mean_abs_err_db": round(float(np.mean(np.abs(err_eq[VAL]))), 2),
        "eq_only_pass": bool(mx(err_eq) <= 1.5),
        "eq_only_removed_offset_db": round(off_eq, 2),
        "eq_only_raw_max_abs_err_db": mx(err_eq + off_eq),     # without removing the constant
        "all_incl_narrowband_max_abs_err_db": mx(err_all),
        "all_removed_offset_db": round(off_all, 2),
        "device_curve_only_predicted_max_abs_err_db": mx(err_dev),
        "unprocessed_pool_max_abs_err_db": mx(err_pool),
        "out_of_sample_fresh_clips_max_abs_err_db": mx(fresh_err) if fresh_err is not None else None,
        "out_of_sample_n": len(fresh),
        "phone_target_bootstrap_sd_db_range": [round(float(boot_sd[VAL].min()), 2), round(float(boot_sd[VAL].max()), 2)],
        "per_band": [{"fc_hz": round(float(FC[i]), 1), "real_phone_db": round(float(L_ph[i]), 2),
                      "phone_boot_sd_db": round(float(boot_sd[i]), 2),
                      "eq_only_err_db": round(float(err_eq[i]), 2), "all_err_db": round(float(err_all[i]), 2),
                      "fresh_err_db": round(float(fresh_err[i]), 2) if fresh_err is not None else None,
                      "device_only_err_db_pred": round(float(err_dev[i]), 2)}
                     for i in np.where(VAL)[0]],
    }
    log(f"phone: validation EQ-only max |err| {validation['eq_only_max_abs_err_db']} dB "
        f"(pass={validation['eq_only_pass']}), incl. narrowband {validation['all_incl_narrowband_max_abs_err_db']} dB")

    curve_json = {
        "band_centres_hz": r2(FC), "band_resolution": "1/6 octave",
        "ltas_method": "speech core (p03_build.speech_bounds), Welch 4096/8192, band power normalised "
                       "to the clip's 62 Hz-10 kHz total, median over clips (dB)",
        "n_clips": {"phone": int(len(B_ph)), "anime_matched": int(len(B_an)), "pool": int(len(B_po))},
        "anime_series": sorted(series),
        "median_ltas_db": {"phone": r2(L_ph), "anime_matched": r2(L_an), "pool": r2(L_po)},
        "device_response_db_raw": r2(L_ph - L_an),
        "device_response_db": r2(smooth(L_ph - L_an)),
        "medium_offset_db": r2(L_an - L_po),
        "applied_response_db": r2(curve),
        "phone_median_bootstrap_sd_db": r2(boot_sd),
        "applied_note": "phone - pool, smoothed (1/4,1/2,1/4), shape-calibrated in the band domain and "
                        "on filtered audio; 0 dB at 1 kHz; held flat outside 62.5 Hz-10.08 kHz",
        "fir": {"type": "minimum phase (cepstral)", "taps": N_TAPS, "max_abs_err_db_100_10k": round(fir_err, 2)},
        "jitter": {"tilt_db_uniform": [-args.phone_tilt_db, args.phone_tilt_db], "tilt_span_hz": [150, 8000],
                   "warp_uniform": [1 - args.phone_warp, 1 + args.phone_warp]},
    }
    (out / "phone_curve.json").write_text(json.dumps(curve_json, indent=1), encoding="utf-8")
    report = {
        "n_rows": len(rows), "by_augment": dict(Counter(r["augment"] for r in rows)),
        "selection_strata": {"series": dict(Counter(s["series"] for s in sel)),
                             "length": dict(Counter(s["stratum"][1] for s in sel)),
                             "punct": dict(Counter(s["stratum"][2] for s in sel))},
        "fir_max_abs_err_db": round(fir_err, 2),
        "nonlinear": {"stats_median": {**st, "augmented_final": group_stats(out_stats)}, **dec},
        "validation": validation, "source_ids": [r["source_id"] for r in rows],
    }
    return rows, report


# ============================================================== (B) background SFX: library

FSD_META = ["labels/dev.csv", "labels/eval.csv", "labels/vocabulary.csv",
            "metadata/dev_clips_info_FSD50K.json", "metadata/eval_clips_info_FSD50K.json",
            "metadata/pp_pnp_ratings_FSD50K.json"]
# token class -> FSD50K labels (any of), kind, SNR range in dB (speech active level vs SFX active
# level), co-labels to exclude, a Freesound title/tag rule (ringtone), and the AST floor a library
# source must reach to count as its class (None: AST cannot tell this class even on clean
# sources - crowd is heard as "Speech"/"Applause", wind as rumble - so no filter; use CLAP or ears).
# Curation came from AST profiles of the candidates: FSD50K "Raindrop" close-ups (drips) and
# musical "Ringtone" tunes are not recognisable as rain or a phone ringing. Tokens reuse the
# real-data names footsteps / ringtone / knock.
# v3 lead: continuous backgrounds capped at 12 dB SNR - the tagger barely heard rain
# at 13-16 dB, so quieter mixes would not teach (or let us measure) the token.
SFX = {
    "footsteps": {"labels": ("Walk_and_footsteps",), "kind": "continuous", "snr": (3.0, 12.0), "tagger_floor": 0.05},
    "knock":     {"labels": ("Knock",), "kind": "event", "snr": (0.0, 12.0), "tagger_floor": 0.2},
    "ringtone":  {"labels": ("Ringtone",), "kind": "event", "snr": (0.0, 12.0), "tagger_floor": 0.1,
                  "require_text": r"ring|phone|cell|mobile|call|nokia|handy|telefon",
                  "avoid_text": r"music|melod|song|marimba|piano|guitar|tune|synth|chiptune|jingle"},
    "rain":      {"labels": ("Rain",), "kind": "continuous", "snr": (3.0, 12.0), "exclude": ("Raindrop",),
                  "tagger_floor": 0.2},
    "crowd":     {"labels": ("Crowd", "Chatter"), "kind": "continuous", "snr": (5.0, 20.0),
                  "exclude": ("Applause", "Cheering", "Clapping"), "tagger_floor": None},
    "typing":    {"labels": ("Typing", "Computer_keyboard"), "kind": "continuous", "snr": (3.0, 12.0),
                  "tagger_floor": 0.2},
    "microwave": {"labels": ("Microwave_oven",), "kind": "event", "snr": (0.0, 12.0), "tagger_floor": 0.1},
    "traffic":   {"labels": ("Traffic_noise_and_roadway_noise",), "kind": "continuous", "snr": (3.0, 12.0),
                  "tagger_floor": 0.1},
    "wind":      {"labels": ("Wind",), "kind": "continuous", "snr": (5.0, 20.0), "tagger_floor": None},
}
# a background must not carry intelligible voice or music (crowd babble is labelled Crowd/Chatter)
VOICE_MUSIC = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
               "Child_speech_and_kid_speaking", "Conversation", "Speech_synthesizer", "Singing",
               "Male_singing", "Female_singing", "Whispering", "Laughter", "Giggle", "Shout", "Yell",
               "Screaming", "Music", "Musical_instrument"}
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-3.0", "CC-BY-4.0"}
EVENT_MAX_S, CONT_MIN_S, MAX_SRC_S = 5.0, 4.0, 30.0


def license_id(url: str) -> str:
    if "publicdomain/zero/1.0" in url:
        return "CC0-1.0"
    m = re.search(r"/licenses/([a-z+\-]+)/(\d\.\d)", url)
    return f"CC-{m.group(1).upper()}-{m.group(2)}" if m else url


def hf_base(args) -> str:
    return f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/datasets/{args.fsd_repo}"


def http(url: str, data: bytes | None = None, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={
                "User-Agent": "kurisu-p03c/1", **({"Content-Type": "application/json"} if data else {})})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            log(f"retry {url}: {e}")
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def fetch(args, rel: str, dest: Path, size: int | None = None, sha256: str | None = None) -> int:
    """Download one repo file (pinned revision); verify size and LFS sha256. Returns bytes fetched."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return 0
    data = http(f"{hf_base(args)}/resolve/{args.fsd_revision}/{rel}")
    if size is not None and len(data) != size:
        raise SystemExit(f"{rel}: size {len(data)} != {size}")
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{rel}: sha256 mismatch")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def paths_info(args, paths: list[str], cache: Path) -> dict:
    known = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [p for p in paths if p not in known]
    for i in range(0, len(todo), 500):
        body = json.dumps({"paths": todo[i:i + 500], "expand": False}).encode()
        for e in json.loads(http(f"{hf_base(args).replace('/datasets/', '/api/datasets/')}"
                                 f"/paths-info/{args.fsd_revision}", data=body)):
            known[e["path"]] = {"size": e["size"], "sha256": (e.get("lfs") or {}).get("oid")}
    cache.write_text(json.dumps(known), encoding="utf-8")
    return known


def fsd_index(args, cache: Path) -> tuple[dict, dict, dict, dict]:
    meta_dir = cache / "meta"
    for rel in FSD_META:
        fetch(args, rel, meta_dir / Path(rel).name)
    info = {}
    for name in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
        info.update(json.loads((meta_dir / name).read_text(encoding="utf-8")))
    labels = {}
    for name, split in (("dev.csv", "dev"), ("eval.csv", "eval")):
        with open(meta_dir / name, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                labels[r["fname"]] = (set(r["labels"].split(",")), split)
    with open(meta_dir / "vocabulary.csv", encoding="utf-8", newline="") as fh:
        vocab = {row[1]: row[2] for row in csv.reader(fh)}
    pp = json.loads((meta_dir / "pp_pnp_ratings_FSD50K.json").read_text(encoding="utf-8"))
    return info, labels, vocab, pp


def select_sfx_sources(args, classes: list[str], cache: Path) -> tuple[dict, dict]:
    """Per class: licence-checked, co-label- and duration-filtered candidates, ranked (predominant
    first, then random)."""
    info, labels, vocab, pp = fsd_index(args, cache)
    for c in classes:
        missing = [l for l in SFX[c]["labels"] + SFX[c].get("exclude", ()) if l not in vocab]
        if missing:
            raise SystemExit(f"FSD50K vocabulary lacks {missing}")
    all_target = {l for c in SFX for l in SFX[c]["labels"]}
    cand, lic_counts = {}, {}
    for c in classes:
        spec = SFX[c]
        labs = spec["labels"]
        req = re.compile(spec["require_text"], re.I) if "require_text" in spec else None
        avoid = re.compile(spec["avoid_text"], re.I) if "avoid_text" in spec else None
        mids = [vocab[l] for l in labs]
        others = all_target - set(labs)
        lc, rows = Counter(), []
        for fname, (ls, split) in labels.items():
            if not ls & set(labs):
                continue
            lic = license_id(info[fname]["license"])
            lc[f"all:{lic}"] += 1
            if lic not in ALLOWED_LICENSES:
                continue
            lc[f"licence_ok:{lic}"] += 1
            if ls & VOICE_MUSIC or ls & others or ls & set(spec.get("exclude", ())):
                continue
            words = info[fname].get("title", "") + " " + " ".join(info[fname].get("tags", []))
            if (req and not req.search(words)) or (avoid and avoid.search(words)):
                continue
            lc[f"clean_colabels:{lic}"] += 1
            ratings = [v for m in mids for v in pp.get(fname, {}).get(m, []) if v >= 0]
            pp_score = float(np.mean(ratings)) if ratings else 0.0
            rows.append({"fname": fname, "split": split, "license": lic, "pp": round(pp_score, 2),
                         "uploader": info[fname].get("uploader", ""), "title": info[fname].get("title", ""),
                         "license_url": info[fname]["license"], "path": f"clips/{split}/{fname}.wav"})
        cand[c], lic_counts[c] = rows, dict(sorted(lc.items()))
    pinfo = paths_info(args, sorted({r["path"] for rows in cand.values() for r in rows}), cache / "paths_info.json")

    ranked = {}
    for c in classes:
        kind = SFX[c]["kind"]
        rng = clip_rng(args.seed, "sfx_sources:" + c)
        rows = []
        for r in cand[c]:
            pi = pinfo.get(r["path"])
            if not pi:
                continue
            dur = (pi["size"] - 44) / (44100 * 2)
            if dur > MAX_SRC_S or (kind == "continuous" and dur < CONT_MIN_S) or dur < 0.3:
                continue
            rows.append(dict(r, size=pi["size"], sha256=pi["sha256"], dur_est=round(dur, 2)))
        rng.shuffle(rows)
        rows.sort(key=lambda r: 0 if r["pp"] >= 0.75 else (1 if r["pp"] >= 0.5 else 2))   # predominant first
        ranked[c] = rows
        lic_counts[c]["usable_after_duration"] = len(rows)
    return ranked, lic_counts


def pick_sources(rows: list[dict], k: int, h: int, cap: int, accept) -> list[dict]:
    """Train: the first k accepted rows, at most `cap` per uploader. Held-out: h accepted rows from
    uploaders that supply no train source for the class, one per uploader."""
    train, per_up, rest = [], Counter(), []
    for r in rows:
        if len(train) >= k or per_up[r["uploader"]] >= cap:
            rest.append(r)
        elif accept(r):
            train.append(dict(r, role="train"))
            per_up[r["uploader"]] += 1
    held, held_up = [], set()
    for r in rest:
        if len(held) >= h:
            break
        if r["uploader"] not in per_up and r["uploader"] not in held_up and accept(r):
            held.append(dict(r, role="heldout_eval"))
            held_up.add(r["uploader"])
    return train + held


def load_sfx(path: Path) -> np.ndarray:
    x = read_mono(path)            # mono, resampled 44.1 kHz -> 24 kHz (resample_poly 80/147)
    return x - np.mean(x)


# ============================================================== (B) background SFX: mixing

def loop_to(src: np.ndarray, n: int, rng, xfade: int = int(0.1 * SR)) -> np.ndarray:
    if len(src) >= n:
        s = int(rng.integers(0, len(src) - n + 1))
        return src[s:s + n].copy()
    xfade = min(xfade, len(src) // 4)
    fade_in = np.sqrt(np.linspace(0, 1, xfade))
    fade_out = np.sqrt(np.linspace(1, 0, xfade))
    out = src.copy()
    while len(out) < n + len(src):
        joint = out[-xfade:] * fade_out + src[:xfade] * fade_in
        out = np.concatenate([out[:-xfade], joint, src[xfade:]])
    s = int(rng.integers(0, len(src)))
    return out[s:s + n]


def fades(y: np.ndarray, fin: float, fout: float) -> np.ndarray:
    y = y.copy()
    a, b = min(len(y) // 2, int(fin * SR)), min(len(y) // 2, int(fout * SR))
    if a:
        y[:a] *= np.linspace(0, 1, a)
    if b:
        y[-b:] *= np.linspace(1, 0, b)
    return y


def pauses(x: np.ndarray, min_s: float = 0.15) -> list[tuple[int, int]]:
    """Low-energy runs (10 ms frames, 35 dB below the loud frames) of at least min_s, in samples."""
    e = frame_db(x, 240, 240)
    quiet = e < np.percentile(e, 95) - 35
    runs, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.01 >= min_s:
                runs.append((start * 240, i * 240))
            start = None
    return runs


def event_segment(src: np.ndarray, max_len: int, rng) -> np.ndarray:
    """Crop an event to max_len around its loudest frame (random position within the window)."""
    if len(src) <= max_len:
        return src.copy()
    e = frame_db(src, 240, 240)
    peak = int(np.argmax(e)) * 240
    lo = max(0, peak - max_len + int(0.1 * SR))
    hi = min(peak, len(src) - max_len)
    s = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(peak, len(src) - max_len))
    return fades(src[s:s + max_len], 0.01, 0.05)


def mix_one(x: np.ndarray, src: np.ndarray, cls: str, rng) -> tuple[np.ndarray, dict]:
    """Background under the speech core only: the build's silent pads stay silent."""
    kind, snr_rng = SFX[cls]["kind"], SFX[cls]["snr"]
    full, x = x, x[LEAD:len(x) - TAIL]
    n = len(x)
    fits: list = []
    if kind == "continuous":    # covers the whole clip from t=0 (the token is utterance-initial)
        bg = fades(loop_to(src, n, rng), 0.08, 0.15)
        onset, ev_len = 0, n
    else:
        # an event stays an event: at most ~60% of the clip (1-5 s), onset in a pause if one fits
        cap = min(EVENT_MAX_S, max(1.0, 0.6 * n / SR), n / SR - 0.1)
        ev = event_segment(src, int(cap * SR), rng)
        ev_len = len(ev)
        fits = [(a, b) for a, b in pauses(x) if a + ev_len <= n]
        if fits:
            a, b = fits[int(rng.integers(len(fits)))]
            onset = int(rng.integers(a, min(b, n - ev_len) + 1))
        else:
            onset = int(rng.integers(0, n - ev_len + 1))
        bg = np.zeros(n)
        bg[onset:onset + ev_len] = ev
    snr = float(rng.uniform(*snr_rng))
    seg = bg[onset:onset + ev_len]
    g = 10 ** ((active_db(x) - snr - active_db(seg)) / 20)
    y = full.copy()
    y[LEAD:LEAD + n] = x + g * bg
    return y, {"snr_db": round(snr, 1), "onset_s": round((LEAD + onset) / SR, 2),
                        "sfx_len_s": round(ev_len / SR, 2), "in_pause": bool(fits)}


def run_sfx(args, meta, sfx_pool, eval_pool, meter, out: Path) -> tuple[list[dict], dict]:
    classes = [c.strip() for c in args.sfx_classes.split(",") if c.strip()]
    cache = Path(args.sfx_cache) if args.sfx_cache else out / "sfx_src"
    ranked, lic_counts = select_sfx_sources(args, classes, cache)
    k, h, cap = args.sfx_sources_per_class, args.sfx_holdout, args.sfx_max_per_uploader
    n_pre = math.ceil(args.sfx_prefetch * (k + h))
    report = {"classes": classes, "licence_counts": lic_counts,
              "fsd50k": {"repo": args.fsd_repo, "revision": args.fsd_revision}}
    if args.sfx_dry_run:    # selection without the tagger filter: sizes are a lower bound
        chosen = {c: pick_sources(ranked[c], k, h, cap, lambda r: True) for c in classes}
        for c in classes:
            lic_counts[c]["selected"] = dict(Counter(f"{r['role']}:{r['license']}" for r in chosen[c]))
        report.update(dry_run=True, download_bytes_selected=sum(r["size"] for v in chosen.values() for r in v),
                      download_bytes_prefetch=sum(r["size"] for c in classes for r in ranked[c][:n_pre]))
        log(f"sfx: dry run, {report['download_bytes_selected'] / 1e6:.1f} MB selected, "
            f"{report['download_bytes_prefetch'] / 1e6:.1f} MB prefetch")
        return [], report

    tagger = not args.sfx_no_tagger
    if tagger:
        try:
            _ast()
            report["source_filter"] = ("AST P(own class) >= per-class floor on the loudest 10.24 s, best first: "
                                       + json.dumps({c: SFX[c]["tagger_floor"] for c in classes}))
        except Exception as e:  # noqa: BLE001  (no torch/transformers: keep going, and say so)
            tagger = False
            report["source_filter"] = f"skipped: {e!r}"
            log(f"sfx: tagger source filter skipped: {e!r}")
    else:
        report["source_filter"] = "off"
    score_file = cache / "tagger_scores.json"
    scores = json.loads(score_file.read_text(encoding="utf-8")) if score_file.exists() else {}
    fetched, needed = 0, 0
    chosen = {}
    for c in classes:
        floor = SFX[c]["tagger_floor"] if tagger else None
        n = n_pre
        while True:     # score the first n ranked candidates (fetched in parallel), extend if short
            cand = ranked[c][:n]
            with ThreadPoolExecutor(args.workers) as ex:
                fetched += sum(ex.map(lambda r: fetch(args, r["path"], cache / r["path"], r["size"], r["sha256"]),
                                      cand))
            if floor is None:
                good = cand
            else:
                for r in cand:
                    if r["fname"] not in scores:
                        x = event_segment(load_sfx(cache / r["path"]), int(10.24 * SR), clip_rng(args.seed, r["fname"]))
                        scores[r["fname"]] = {k: round(v, 4) for k, v in ast_class_probs(x).items()}
                good = [dict(r, tagger_p=scores[r["fname"]][c]) for r in cand if scores[r["fname"]][c] >= floor]
                good.sort(key=lambda r: -r["tagger_p"])       # most recognisable first
            chosen[c] = pick_sources(good, k, h, cap, lambda r: True)
            done = (sum(r["role"] == "train" for r in chosen[c]) >= k
                    and sum(r["role"] == "heldout_eval" for r in chosen[c]) >= h)
            if done or n >= len(ranked[c]):
                break
            n += n_pre
        lic_counts[c]["selected"] = dict(Counter(f"{r['role']}:{r['license']}" for r in chosen[c]))
        lic_counts[c]["tagger_floor"] = floor
        needed += sum(r["size"] for r in ranked[c][:n])
        lic_counts[c]["candidates_scored" if floor is not None else "candidates_fetched"] = min(n, len(ranked[c]))
        if floor is not None:
            lic_counts[c]["passed_floor"] = len(good)
        n_train = sum(r["role"] == "train" for r in chosen[c])
        log(f"sfx: {c}: {n_train} train + {len(chosen[c]) - n_train} held-out sources "
            f"from {min(n, len(ranked[c]))} candidates (floor {floor})")
        score_file.write_text(json.dumps(scores), encoding="utf-8")
    report.update(download_bytes_fetched=fetched, download_bytes_all_candidates_touched=needed,
                  download_bytes_selected=sum(r["size"] for v in chosen.values() for r in v))
    log(f"sfx: {sum(len(v) for v in chosen.values())} sources selected, {fetched / 1e6:.1f} MB downloaded")

    # interleave classes over a stratified speech order so every class gets a similar mix
    it = iter(sfx_pool)
    assign = defaultdict(list)
    for _ in range(args.sfx_per_class):
        for c in classes:
            nxt = next(it, None)
            if nxt is not None:
                assign[c].append(nxt)
    rows, eval_rows, uses = [], [], Counter()
    for c in classes:
        train_src = [r for r in chosen[c] if r["role"] == "train"]
        held_src = [r for r in chosen[c] if r["role"] == "heldout_eval"]
        crng = clip_rng(args.seed, "sfx_mix:" + c)
        audio = {r["fname"]: load_sfx(cache / r["path"]) for r in train_src + held_src}
        for j, sp in enumerate(assign[c]):
            avail = [r for r in train_src if uses[r["fname"]] < args.sfx_max_uses]
            if not avail:
                log(f"sfx: {c}: sources exhausted after {j} mixes")
                break
            least = min(uses[r["fname"]] for r in avail)
            pick = [r for r in avail if uses[r["fname"]] == least]
            src = pick[int(crng.integers(len(pick)))]
            x = load_clean(sp, args, meter)
            if x is None:
                continue
            y, p = mix_one(x, audio[src["fname"]], c, crng)
            y = seal(peak_limit(y))     # speech keeps its build level; the background adds
            uses[src["fname"]] += 1
            rid = f"aug_sfx_{c}_{sp['id']}"
            path = out / "wavs" / "sfx" / f"{rid}.wav"
            write_wav(path, y)
            row = base_row(sp, rid, path, f"[bg_{c}]" + sp["text_train"], "kurisu", "sfx")
            row.update(sfx_class=c, sfx_source=f"fsd50k:{src['fname']}", sfx_license=src["license"],
                       snr_db=p["snr_db"], params=p)
            rows.append(row)
        # evaluation-only mixes: held-out sounds x validation-split speech (calibrates the tagger)
        for j in range(min(args.sfx_eval_per_class, len(eval_pool)) if held_src else 0):
            sp = eval_pool.pop()
            src = held_src[j % len(held_src)]
            x = load_clean(sp, args, meter)
            if x is None:
                continue
            y, p = mix_one(x, audio[src["fname"]], c, crng)
            y = seal(peak_limit(y))     # speech keeps its build level; the background adds
            rid = f"eval_sfx_{c}_{sp['id']}"
            path = out / "wavs" / "sfx_eval" / f"{rid}.wav"
            write_wav(path, y)
            eval_rows.append(dict(base_row(sp, rid, path, f"[bg_{c}]" + sp["text_train"],
                                           "kurisu", "sfx"),
                                  sfx_class=c, sfx_source=f"fsd50k:{src['fname']}", sfx_license=src["license"],
                                  snr_db=p["snr_db"], split="eval_heldout", params=p,
                                  clean_audio=str(sp["audio"]) if not args.wav_dir
                                  else str(Path(args.wav_dir) / f"{sp['id']}.wav")))
        log(f"sfx: {c}: {sum(1 for r in rows if r['sfx_class'] == c)} mixes, "
            f"{sum(1 for r in eval_rows if r['sfx_class'] == c)} eval mixes")

    (out / "sfx_eval_rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_rows), encoding="utf-8")
    attribution = {
        "dataset": {"name": "FSD50K", "license": "CC-BY-4.0 (dataset); per-clip licences below",
                    "citation": "E. Fonseca, X. Favory, J. Pons, F. Font, X. Serra. FSD50K: an Open Dataset of "
                                "Human-Labeled Sound Events. IEEE/ACM TASLP 30, 2022. arXiv:2010.00475",
                    "source": f"https://huggingface.co/datasets/{args.fsd_repo} @ {args.fsd_revision} "
                              "(copy of zenodo.org/record/4060432)"},
        "clips": [{"fsd50k_id": r["fname"], "class": c, "role": r["role"], "license": r["license"],
                   "license_url": r["license_url"], "uploader": r["uploader"], "title": r["title"],
                   "freesound_url": f"https://freesound.org/s/{r['fname']}/", "uses": uses[r["fname"]],
                   "tagger_p": r.get("tagger_p"),
                   "attribution": (f"\"{r['title']}\" by {r['uploader']} (https://freesound.org/s/{r['fname']}/), "
                                   f"CC BY {r['license'].rsplit('-', 1)[-1]}") if r["license"].startswith("CC-BY") else None}
                  for c in classes for r in chosen[c]],
    }
    (out / "sfx_attribution.json").write_text(json.dumps(attribution, indent=1, ensure_ascii=False), encoding="utf-8")
    snrs = defaultdict(list)
    for r in rows:
        snrs[r["sfx_class"]].append(r["snr_db"])
    report.update({
        "n_rows": len(rows), "n_eval_rows": len(eval_rows),
        "mixes_per_class": dict(Counter(r["sfx_class"] for r in rows)),
        "snr_db_p10_p50_p90": {c: [round(float(v), 1) for v in np.percentile(s, [10, 50, 90])] for c, s in snrs.items()},
        "events_in_pause_share": {c: round(float(np.mean([r["params"]["in_pause"] for r in rows if r["sfx_class"] == c])), 2)
                                  for c in classes if SFX[c]["kind"] == "event" and snrs.get(c)},
        "source_uses_max": max(uses.values()) if uses else 0,
        "rows_licence": dict(Counter(r["sfx_license"] for r in rows)),
        "source_ids": [r["source_id"] for r in rows],
    })
    return rows, report


# ============================================================== optional evaluator (see augment_notes.md)

AUDIOSET_LABELS = {   # AudioSet display names in MIT/ast-finetuned-audioset-10-10-0.4593 (id2label)
    "footsteps": ["Walk, footsteps", "Shuffle", "Run"], "knock": ["Knock"],
    "ringtone": ["Ringtone", "Telephone bell ringing", "Telephone"],
    "rain": ["Rain", "Rain on surface", "Raindrop"], "crowd": ["Crowd", "Chatter", "Hubbub, speech noise, speech babble"],
    "typing": ["Typing", "Computer keyboard"], "microwave": ["Microwave oven", "Beep, bleep"],
    "traffic": ["Traffic noise, roadway noise", "Vehicle", "Car", "Motor vehicle (road)", "Car passing by"],
    "wind": ["Wind", "Wind noise (microphone)"],
}


AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"   # BSD-3-Clause; trained on AudioSet only


@functools.lru_cache(maxsize=2)
def _ast(model_id: str = AST_ID):
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = ASTFeatureExtractor.from_pretrained(model_id)
    model = ASTForAudioClassification.from_pretrained(model_id).to(device).eval()
    lab = {v: int(k) for k, v in model.config.id2label.items()}
    missing = [n for names in AUDIOSET_LABELS.values() for n in names if n not in lab]
    if missing:
        raise RuntimeError(f"AudioSet labels not in {model_id}: {missing}")
    return fe, model, {c: [lab[n] for n in names] for c, names in AUDIOSET_LABELS.items()}, device


def ast_class_probs(x: np.ndarray, model_id: str = AST_ID) -> dict:
    """24 kHz mono -> {sfx class: max sigmoid probability over its AudioSet labels}.
    AST sees the first 10.24 s."""
    import torch

    fe, model, idx, device = _ast(model_id)
    x16 = signal.resample_poly(x, 2, 3).astype(np.float32)
    with torch.no_grad():
        feats = fe(x16, sampling_rate=16000, return_tensors="pt").to(device)
        prob = torch.sigmoid(model(**feats).logits)[0].float().cpu().numpy()
    return {c: float(prob[i].max()) for c, i in idx.items()}


def sfx_class_scores(wav_paths: list[str], model_id: str = AST_ID) -> list[dict]:
    """Evaluator (plan in v3/augment_notes.md): per generated clip, P(class) for every SFX class.
    Compare a [bg_x] prompt's P(x) with its untagged twin's; calibrate on sfx_eval_rows.jsonl."""
    return [ast_class_probs(read_mono(p), model_id) for p in wav_paths]


# ============================================================== main

def merge_outputs(out: Path, part: str, rows: list[dict], report: dict) -> None:
    (out / f"rows_{part}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    merged = []
    for p in ("phone", "sfx"):
        f = out / f"rows_{p}.jsonl"
        if f.exists():
            merged += [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    (out / "augment_rows.jsonl").write_text("".join(l + "\n" for l in merged), encoding="utf-8")
    rp = out / "report.json"
    full = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    full[part] = report
    full["augment_rows"] = len(merged)
    rp.write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=str(WORK / "src" / "ja_meta.jsonl"),
                    help="ja_meta.jsonl, or the HF dataset data/ directory")
    ap.add_argument("--wav_dir", default=None, help="processed clean wavs <id>.wav (pod: $KURISU_WORK/build_v3/wavs)")
    ap.add_argument("--kept", default=None, help="optional kept.jsonl restricting usable ids")
    ap.add_argument("--out", default=str(WORK / "augment"))
    ap.add_argument("--parts", default="phone,sfx")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--min_dur", type=float, default=1.5)
    ap.add_argument("--max_dur", type=float, default=12.0)
    ap.add_argument("--phone_n", type=int, default=400)
    ap.add_argument("--phone_narrowband_share", type=float, default=0.15)
    ap.add_argument("--phone_tilt_db", type=float, default=2.5)
    ap.add_argument("--phone_warp", type=float, default=0.10)
    ap.add_argument("--phone_calib_iters", type=int, default=6, help="band-domain curve refinements")
    ap.add_argument("--phone_audio_iters", type=int, default=2, help="refinements on filtered audio")
    ap.add_argument("--phone_fresh_check", type=int, default=150, help="unwritten clips for the out-of-sample check")
    ap.add_argument("--phone_compress", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--phone_saturate", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--phone_noise", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--phone_noise_snr", type=float, nargs=2, default=(30.0, 45.0))
    ap.add_argument("--sfx_classes", default=",".join(SFX))
    ap.add_argument("--sfx_per_class", type=int, default=80)
    ap.add_argument("--sfx_sources_per_class", type=int, default=30)
    ap.add_argument("--sfx_max_per_uploader", type=int, default=2)
    ap.add_argument("--sfx_holdout", type=int, default=4)
    ap.add_argument("--sfx_max_uses", type=int, default=4)
    ap.add_argument("--sfx_eval_per_class", type=int, default=8)
    ap.add_argument("--sfx_no_tagger", action="store_true",
                    help="skip the AST recognisability filter on library sources (floors are per class in SFX)")
    ap.add_argument("--sfx_prefetch", type=float, default=1.5, help="parallel prefetch, x (sources + holdout)")
    ap.add_argument("--sfx_cache", default=None, help="FSD50K download dir (default <out>/sfx_src)")
    ap.add_argument("--sfx_dry_run", action="store_true", help="select sources and report sizes, no audio")
    ap.add_argument("--fsd_repo", default="Fhrozen/FSD50k")
    ap.add_argument("--fsd_revision", default="ccf1acaff12f3f4a4c10052dddafbb6d22152c9b")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import pyloudnorm
    meter = pyloudnorm.Meter(SR)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(Path(args.meta))
    parts = {p.strip() for p in args.parts.split(",")}

    # one stratified order over all clean train candidates: the first phone_n are the phone pool,
    # the rest feed the SFX mixes, so the two pools are disjoint and stable whichever part runs.
    pool = stratified_order(clean_candidates(meta, args, "train"), np.random.default_rng(args.seed))
    phone_sel, sfx_pool = pool[:args.phone_n], pool[args.phone_n:]
    eval_pool = stratified_order(clean_candidates(meta, args, "validation"), np.random.default_rng(args.seed + 7))
    log(f"{len(meta)} meta rows; clean train candidates {len(pool)}, phone pool {len(phone_sel)}, "
        f"sfx pool {len(sfx_pool)}, validation pool {len(eval_pool)}")
    common = {"seed": args.seed, "args": vars(args), "ref_id_excluded": REF_ID,
              "clean_train_candidates": len(pool)}
    if "phone" in parts:
        fresh = [sfx_pool[i] for i in np.random.default_rng(args.seed + 3).choice(
            len(sfx_pool), size=min(args.phone_fresh_check, len(sfx_pool)), replace=False)]
        rows, rep = run_phone(args, meta, phone_sel, fresh, meter, out)
        merge_outputs(out, "phone", rows, {**common, **rep})
    if "sfx" in parts:
        rows, rep = run_sfx(args, meta, sfx_pool, eval_pool, meter, out)
        if not args.sfx_dry_run:
            merge_outputs(out, "sfx", rows, {**common, **rep})
        else:
            (out / "sfx_dry_run.json").write_text(json.dumps({**common, **rep}, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
