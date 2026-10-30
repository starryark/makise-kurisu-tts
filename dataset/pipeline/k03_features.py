"""Stage 03 - deterministic acoustic features for every clip.

Everything here is signal processing on the published 24 kHz WAV: no model, no
threshold, no decision.  Stage 08 calibrates thresholds on these numbers against
the human gold set; stages 09 onwards apply them.  Keeping measurement and
judgement in separate files is what makes the thresholds auditable.

What is measured and why it earns its place
-------------------------------------------
``roll995_hz`` / ``hf_ratio_db`` / ``lf_ratio_db`` / ``spectral_tilt_db``
    Playback-path shape.  A line heard through the Amadeus phone speaker is
    band-limited and bass-cut; a line heard in the room is not.  Measured on the
    per-bin *maximum* over the loudest frames rather than the mean, so the answer
    is "what does this clip ever contain up there", not "what is the average
    phoneme", which otherwise makes a sentence ending in a fricative look wide-band.

``f0_*`` / ``voiced_ratio`` / ``hnr_db``
    Pitch register, pitch movement, and how periodic the voice is.  A whisper is
    speech with the periodicity removed, so it shows up as low ``voiced_ratio``
    with normal active level -- something no loudness measure can see.

``decay_db_per_s`` / ``late_energy_ratio_db``
    Room.  Fitted on the energy that follows each speech offset.  Studio game
    voice falls off at the frame limit; a reverberant take holds energy for
    a couple of hundred milliseconds.

``noise_floor_dbfs`` / ``snr_est_db`` / ``flatness``
    What is left between the words.  For separated anime audio this is the music
    bed and the separation artefacts, and it is the honest way to rank clips by
    how clean they actually are.

``units_per_active_s``
    Speaking rate over *speaking* time, so a long lead-in silence does not make a
    line look slow.  Characters for Japanese and Chinese, words elsewhere.
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf
import soxr

from kcommon import (MANIFESTS, PROJECT, REPORTS, TARGET_SR, lexical_units,
                     read_jsonl, utf8_stdout, write_json, write_jsonl)

utf8_stdout()

PITCH_SR = 16000
F0_MIN, F0_MAX = 70.0, 500.0
PITCH_FRAME_S, PITCH_HOP_S = 0.040, 0.010
VOICED_R = 0.38            # normalised autocorrelation peak that counts as voiced

SPEC_N, SPEC_HOP = 2048, 512
ENERGY_FRAME_S = 0.020


def _db(x) -> float:
    return float(10.0 * np.log10(max(float(x), 1e-20)))


# ------------------------------------------------------------------ spectrum

def spectral(x: np.ndarray) -> dict:
    if len(x) < SPEC_N * 2:
        pad = SPEC_N * 2 - len(x)
        x = np.pad(x, (0, pad))
    win = np.hanning(SPEC_N).astype(np.float32)
    starts = np.arange(0, len(x) - SPEC_N + 1, SPEC_HOP)
    fr = np.stack([x[s:s + SPEC_N] * win for s in starts])
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    freqs = np.fft.rfftfreq(SPEC_N, 1.0 / TARGET_SR)
    e = P.sum(axis=1)
    loud = P[e >= np.percentile(e, 70)]
    if len(loud) == 0:
        loud = P
    Pmax = loud.max(axis=0)
    Pmean = loud.mean(axis=0)
    tot = Pmax.sum() + 1e-20
    c = np.cumsum(Pmax) / tot

    def roll(q: float) -> float:
        return float(freqs[min(int(np.searchsorted(c, q)), len(freqs) - 1)])

    m_tel = (freqs >= 300) & (freqs < 3400)
    m_hi = freqs >= 4500
    m_lo = freqs < 250
    tel = Pmax[m_tel].sum()
    hf = _db(Pmax[m_hi].sum() / (tel + 1e-20))
    lf = _db(Pmax[m_lo].sum() / (tel + 1e-20))
    centroid = float((freqs * Pmean).sum() / (Pmean.sum() + 1e-20))
    gm = float(np.exp(np.log(Pmean + 1e-20).mean()))
    flat = gm / (float(Pmean.mean()) + 1e-20)
    return {
        "roll95_hz": round(roll(0.95), 1),
        "roll995_hz": round(roll(0.995), 1),
        "hf_ratio_db": round(hf, 2),
        "lf_ratio_db": round(lf, 2),
        "spectral_tilt_db": round(hf - lf, 2),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_flatness": round(float(flat), 5),
    }


# --------------------------------------------------------------------- pitch

def pitch(x24: np.ndarray) -> dict:
    x = soxr.resample(x24, TARGET_SR, PITCH_SR, quality="HQ").astype(np.float32)
    n = int(PITCH_FRAME_S * PITCH_SR)
    hop = int(PITCH_HOP_S * PITCH_SR)
    if len(x) < n + hop:
        return {"f0_median_hz": None, "f0_p10_hz": None, "f0_p90_hz": None,
                "f0_range_st": None, "f0_std_st": None, "voiced_ratio": 0.0,
                "hnr_db": None}
    starts = np.arange(0, len(x) - n + 1, hop)
    fr = np.stack([x[s:s + n] for s in starts])
    fr = fr - fr.mean(axis=1, keepdims=True)
    energy = (fr ** 2).mean(axis=1)
    live = energy >= max(energy.max() * 10 ** (-2.5), 1e-10)   # -25 dB of loudest
    if not live.any():
        live = np.ones(len(fr), dtype=bool)
    fr = fr[live]

    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :n]
    r0 = ac[:, :1] + 1e-20
    acn = ac / r0
    lag_lo = int(PITCH_SR / F0_MAX)
    lag_hi = min(int(PITCH_SR / F0_MIN), n - 1)
    seg = acn[:, lag_lo:lag_hi]
    # taper the bias of the unbiased-less estimator: normalise by overlap length
    taper = (n - np.arange(lag_lo, lag_hi)) / float(n)
    seg = seg / np.maximum(taper, 1e-6)
    best = seg.argmax(axis=1)
    rmax = seg[np.arange(len(seg)), best]
    f0 = PITCH_SR / (best + lag_lo)
    voiced = (rmax >= VOICED_R) & (f0 >= F0_MIN) & (f0 <= F0_MAX)
    ratio = float(voiced.mean())
    if voiced.sum() < 3:
        return {"f0_median_hz": None, "f0_p10_hz": None, "f0_p90_hz": None,
                "f0_range_st": None, "f0_std_st": None,
                "voiced_ratio": round(ratio, 4),
                "hnr_db": round(float(10 * np.log10(
                    np.clip(np.median(rmax), 1e-3, 0.999) /
                    (1 - np.clip(np.median(rmax), 1e-3, 0.999)))), 2)}
    v = f0[voiced]
    st = 12.0 * np.log2(v / np.median(v))
    rv = np.clip(rmax[voiced], 1e-3, 0.999)
    return {
        "f0_median_hz": round(float(np.median(v)), 1),
        "f0_p10_hz": round(float(np.percentile(v, 10)), 1),
        "f0_p90_hz": round(float(np.percentile(v, 90)), 1),
        "f0_range_st": round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2),
        "f0_std_st": round(float(st.std()), 2),
        "voiced_ratio": round(ratio, 4),
        "hnr_db": round(float(10 * np.log10(np.median(rv) / (1 - np.median(rv)))), 2),
    }


# ---------------------------------------------------------------- room / floor

def room_and_floor(x: np.ndarray, thr_dbfs: float) -> dict:
    n = int(ENERGY_FRAME_S * TARGET_SR)
    m = len(x) // n
    if m < 6:
        return {"decay_db_per_s": None, "late_energy_ratio_db": None,
                "noise_floor_dbfs": None, "snr_est_db": None, "offsets": 0}
    f = x[: m * n].reshape(m, n)
    rd = 10.0 * np.log10((f ** 2).mean(axis=1) + 1e-20)
    act = rd >= thr_dbfs
    floor = rd[~act]
    active = rd[act]
    noise_floor = float(np.median(floor)) if floor.size >= 3 else None
    snr = (round(float(np.median(active) - noise_floor), 2)
           if noise_floor is not None and active.size else None)

    slopes, lates = [], []
    tail = int(round(0.20 / ENERGY_FRAME_S))        # 200 ms of decay
    hold = int(round(0.10 / ENERGY_FRAME_S))        # silence must actually persist
    for i in range(1, m - tail):
        # A real speech offset is followed by silence. Every 20 ms dip inside a
        # word is also an active -> inactive transition, and measuring "decay"
        # across one of those measures the next syllable arriving, not the room.
        if not (act[i] and not act[i + 1] and not act[i + 1:i + 1 + hold].any()):
            continue
        seg = rd[i:i + tail + 1]
        if seg.size < 4:
            continue
        t = np.arange(seg.size) * ENERGY_FRAME_S
        stop = int(np.argmin(seg)) + 1              # fit only while it is falling
        if stop < 4:
            stop = min(4, seg.size)
        sl, _ = np.polyfit(t[:stop], seg[:stop], 1)
        slopes.append(float(sl))
        lates.append(float(seg[min(seg.size - 1, int(round(0.15 / ENERGY_FRAME_S)))])
                     - float(seg[0]))
    return {
        "decay_db_per_s": round(float(np.median(slopes)), 1) if slopes else None,
        "late_energy_ratio_db": round(float(np.median(lates)), 2) if lates else None,
        "noise_floor_dbfs": None if noise_floor is None else round(noise_floor, 2),
        "snr_est_db": snr,
        "offsets": len(slopes),
    }


def main() -> None:
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    out = []
    for i, r in enumerate(rows):
        x, sr = sf.read(PROJECT / r["audio"], dtype="float32", always_2d=False)
        assert sr == TARGET_SR
        feats = {}
        feats.update(spectral(x))
        feats.update(pitch(x))
        feats.update(room_and_floor(x, r["activity_threshold_dbfs"]))
        active_s = max(r["duration"] * r["speech_ratio"], 1e-3)
        units = lexical_units(r["text"], r["language"])
        feats["lexical_units"] = units
        feats["units_per_active_s"] = round(units / active_s, 3)
        feats["units_per_s"] = round(units / max(r["duration"], 1e-3), 3)
        out.append({"id": r["id"], **feats})
        if (i + 1) % 1500 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    write_jsonl(MANIFESTS / "03_features.jsonl", out)

    def summ(key):
        v = np.array([o[key] for o in out if o.get(key) is not None], dtype=float)
        if v.size == 0:
            return None
        return {"n": int(v.size), "p05": round(float(np.percentile(v, 5)), 2),
                "median": round(float(np.median(v)), 2),
                "p95": round(float(np.percentile(v, 95)), 2)}

    write_json(REPORTS / "03_features.json", {
        "clips": len(out),
        "features": {k: summ(k) for k in (
            "roll995_hz", "hf_ratio_db", "lf_ratio_db", "spectral_tilt_db",
            "spectral_centroid_hz", "spectral_flatness", "f0_median_hz",
            "f0_range_st", "f0_std_st", "voiced_ratio", "hnr_db",
            "decay_db_per_s", "late_energy_ratio_db", "noise_floor_dbfs",
            "snr_est_db", "units_per_active_s")},
    })
    print(json.dumps(json.loads((REPORTS / "03_features.json").read_text("utf-8")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
