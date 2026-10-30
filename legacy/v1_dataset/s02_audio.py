"""Stage 02 - audio standardisation.

Every clip becomes 24 kHz mono PCM-16 WAV. Loudness is aligned **per source group**
(game:SG, game:SG0, game:PHENO, game:SGRE, anime:SG, ...) with a single gain per
group, never per clip - so the loudness relationships inside a game are untouched
(guide s9.5) while the four games and the anime stop disagreeing by ~6 dB.

Loudness is ITU-R BS.1770-4 integrated, gated, pooled over every 400 ms block in
the group. The common target is lowered automatically if any group would otherwise
push a clip past the peak ceiling.

Also standardises ref_audio.wav (32 kHz -> 24 kHz) into audio/ref.wav.
"""
from __future__ import annotations

import collections
import json
import math
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

from common import (MANIFESTS, PEAK_CEILING_DBFS, REF_OUT, REF_SRC, REPORTS,
                    TARGET_LUFS, TARGET_SR, WAVS, read_jsonl, sha256_file,
                    utf8_stdout, write_jsonl)

utf8_stdout()

BLOCK_S = 0.400
OVERLAP = 0.75
ABS_GATE = -70.0
REL_GATE = -10.0


def _biquad(kind: str, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """BS.1770 K-weighting biquads, derived for an arbitrary sample rate."""
    if kind == "high_shelf":
        G, Q, fc = 4.0, 1 / math.sqrt(2), 1681.974450955533
        A = 10 ** (G / 40)
        w0 = 2 * math.pi * fc / rate
        alpha = math.sin(w0) / (2 * Q)
        cw = math.cos(w0)
        sa = math.sqrt(A)
        b = np.array([A * ((A + 1) + (A - 1) * cw + 2 * sa * alpha),
                      -2 * A * ((A - 1) + (A + 1) * cw),
                      A * ((A + 1) + (A - 1) * cw - 2 * sa * alpha)])
        a = np.array([(A + 1) - (A - 1) * cw + 2 * sa * alpha,
                      2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - 2 * sa * alpha])
    else:  # high_pass
        Q, fc = 0.5003270373238773, 38.13547087602444
        w0 = 2 * math.pi * fc / rate
        alpha = math.sin(w0) / (2 * Q)
        cw = math.cos(w0)
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


class KWeighting:
    def __init__(self, rate: int):
        from scipy.signal import lfilter
        self._lfilter = lfilter
        self.stages = [_biquad("high_shelf", rate), _biquad("high_pass", rate)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x
        for b, a in self.stages:
            y = self._lfilter(b, a, y)
        return y


def block_powers(x: np.ndarray, rate: int, kw: KWeighting) -> np.ndarray:
    """Mean-square of each gated-loudness block (BS.1770 z_i), single channel."""
    n = int(BLOCK_S * rate)
    step = int(n * (1 - OVERLAP))
    if len(x) < n:
        return np.empty(0, dtype=np.float64)
    y = kw(x.astype(np.float64))
    starts = range(0, len(y) - n + 1, step)
    return np.array([np.mean(y[s:s + n] ** 2) for s in starts], dtype=np.float64)


def integrated_from_powers(z: np.ndarray) -> float | None:
    """Gated integrated loudness (LUFS) from pooled block powers."""
    if z.size == 0:
        return None
    with np.errstate(divide="ignore"):
        l = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    keep = l > ABS_GATE
    if not keep.any():
        return None
    rel = -0.691 + 10 * math.log10(max(z[keep].mean(), 1e-20)) + REL_GATE
    keep &= l > rel
    if not keep.any():
        return None
    return -0.691 + 10 * math.log10(max(z[keep].mean(), 1e-20))


def load_mono_24k(path: str, sr_hint: int = 0) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        w = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        w = torchaudio.functional.resample(
            w, sr, TARGET_SR, resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64, rolloff=0.945)
        x = w.squeeze(0).numpy()
    return x.astype(np.float64)


def edge_silence(x: np.ndarray, rate: int, thresh_db: float = -50.0) -> tuple[float, float]:
    """Seconds of sub-threshold audio at the head and tail (10 ms frames)."""
    n = max(1, int(0.010 * rate))
    frames = len(x) // n
    if frames == 0:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(x[:frames * n].reshape(frames, n) ** 2, axis=1) + 1e-20)
    loud = 20 * np.log10(rms) > thresh_db
    if not loud.any():
        return frames * n / rate, 0.0
    first, last = int(np.argmax(loud)), frames - 1 - int(np.argmax(loud[::-1]))
    return first * n / rate, (frames - 1 - last) * n / rate


AUDIO_FIELDS = ("audio", "audio_abs", "sr", "dur", "gain_db", "peak_dbfs",
                "lufs_clip", "lead_sil_s", "tail_sil_s", "clipped_samples", "sha256")


def refresh_manifest_only() -> None:
    """Re-derive 02_audio.jsonl from 01 without touching the wavs.

    For transcript/status changes: the audio bytes do not depend on the text, so
    re-encoding 10k files to update one metadata field is wasted work. Fails loudly
    if the clip set changed, because then the audio really does need rebuilding.
    """
    old = {r["id"]: r for r in read_jsonl(MANIFESTS / "02_audio.jsonl")}
    new = list(read_jsonl(MANIFESTS / "01_inventory.jsonl"))
    missing = [r["id"] for r in new if r["id"] not in old]
    if missing:
        raise SystemExit(f"{len(missing)} clip(s) have no audio yet "
                         f"({missing[:5]}) - run a full stage 02")
    out = []
    for r in new:
        m = dict(r)
        m.update({k: old[r["id"]][k] for k in AUDIO_FIELDS})
        out.append(m)
    write_jsonl(MANIFESTS / "02_audio.jsonl", out)
    dropped = len(old) - len(out)
    print(f"refreshed 02_audio.jsonl for {len(out)} clips "
          f"({dropped} no longer selected); wavs untouched")


def main() -> None:
    if "--manifest-only" in sys.argv:
        refresh_manifest_only()
        return
    rows = list(read_jsonl(MANIFESTS / "01_inventory.jsonl"))
    kw = KWeighting(TARGET_SR)

    # ---- pass 1: decode once, measure, keep the waveform in memory group by group
    print(f"pass 1/2  measuring {len(rows)} clips ...")
    pooled: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    peaks: dict[str, float] = collections.defaultdict(float)
    meas: dict[str, dict] = {}

    for i, r in enumerate(rows):
        if i % 1000 == 0:
            print(f"   {i}/{len(rows)}")
        x = load_mono_24k(r["src_path"], r["sr_src"])
        pk = float(np.max(np.abs(x))) if x.size else 0.0
        g = r["source_group"]
        peaks[g] = max(peaks[g], pk)
        z = block_powers(x, TARGET_SR, kw)
        if z.size:
            pooled[g].append(z)
        lead, tail = edge_silence(x, TARGET_SR)
        meas[r["id"]] = {
            "peak": pk, "n": len(x),
            "lufs": integrated_from_powers(z),
            "lead_sil_s": round(lead, 3), "tail_sil_s": round(tail, 3),
        }

    group_lufs = {g: integrated_from_powers(np.concatenate(v)) for g, v in pooled.items()}
    group_peak_db = {g: 20 * math.log10(max(p, 1e-9)) for g, p in peaks.items()}

    # ---- choose one common target that no group can clip at
    feasible = [group_lufs[g] + (PEAK_CEILING_DBFS - group_peak_db[g])
                for g in group_lufs if group_lufs[g] is not None]
    target = min([TARGET_LUFS] + feasible)
    gains = {g: (target - group_lufs[g]) if group_lufs[g] is not None else 0.0
             for g in group_lufs}

    print("\nsource-group loudness alignment")
    print(f"  common target: {target:.2f} LUFS  (requested {TARGET_LUFS})")
    for g in sorted(gains):
        print(f"  {g:26s} {group_lufs[g]:7.2f} LUFS  peak {group_peak_db[g]:6.2f} dBFS"
              f"  ->  gain {gains[g]:+6.2f} dB")

    # ---- pass 2: apply gain, write
    print(f"\npass 2/2  writing {len(rows)} clips to {WAVS} ...")
    out_rows = []
    for i, r in enumerate(rows):
        if i % 1000 == 0:
            print(f"   {i}/{len(rows)}")
        x = load_mono_24k(r["src_path"], r["sr_src"])
        gdb = gains[r["source_group"]]
        x = x * (10 ** (gdb / 20))
        clipped = int(np.sum(np.abs(x) >= 1.0))
        x = np.clip(x, -1.0, 1.0)
        dst = WAVS / f"{r['id']}.wav"
        sf.write(str(dst), x.astype(np.float32), TARGET_SR, subtype="PCM_16")

        m = meas[r["id"]]
        out = dict(r)
        out.update({
            "audio": f"./audio/wavs/{r['id']}.wav",
            "audio_abs": str(dst).replace("\\", "/"),
            "sr": TARGET_SR,
            "dur": round(len(x) / TARGET_SR, 4),
            "gain_db": round(gdb, 3),
            "peak_dbfs": round(20 * math.log10(max(float(np.max(np.abs(x))), 1e-9)), 2),
            "lufs_clip": round(m["lufs"] + gdb, 2) if m["lufs"] is not None else None,
            "lead_sil_s": m["lead_sil_s"],
            "tail_sil_s": m["tail_sil_s"],
            "clipped_samples": clipped,
        })
        out_rows.append(out)

    for r in out_rows:
        r["sha256"] = sha256_file(WAVS / f"{r['id']}.wav")
    write_jsonl(MANIFESTS / "02_audio.jsonl", out_rows)

    # ---- reference audio
    ref = load_mono_24k(str(REF_SRC), 0)
    ref_z = block_powers(ref, TARGET_SR, kw)
    ref_lufs = integrated_from_powers(ref_z)
    ref_gain = target - ref_lufs if ref_lufs is not None else 0.0
    ref_peak_after = 20 * math.log10(max(float(np.max(np.abs(ref))), 1e-9)) + ref_gain
    if ref_peak_after > PEAK_CEILING_DBFS:
        ref_gain -= (ref_peak_after - PEAK_CEILING_DBFS)
    ref_out = np.clip(ref * (10 ** (ref_gain / 20)), -1.0, 1.0)
    sf.write(str(REF_OUT), ref_out.astype(np.float32), TARGET_SR, subtype="PCM_16")

    lead, tail = edge_silence(ref_out, TARGET_SR)
    report = {
        "clips": len(out_rows),
        "target_lufs": round(target, 2),
        "requested_target_lufs": TARGET_LUFS,
        "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
        "groups": {g: {"lufs_before": round(group_lufs[g], 2),
                       "peak_dbfs_before": round(group_peak_db[g], 2),
                       "gain_db": round(gains[g], 2),
                       "clips": sum(1 for r in rows if r["source_group"] == g)}
                   for g in sorted(gains)},
        "clips_with_clipped_samples": sum(1 for r in out_rows if r["clipped_samples"]),
        "edge_silence_s": {
            "lead_median": round(float(np.median([r["lead_sil_s"] for r in out_rows])), 3),
            "tail_median": round(float(np.median([r["tail_sil_s"] for r in out_rows])), 3),
            "lead_p90": round(float(np.percentile([r["lead_sil_s"] for r in out_rows], 90)), 3),
            "tail_p90": round(float(np.percentile([r["tail_sil_s"] for r in out_rows], 90)), 3),
        },
        "ref_audio": {
            "src": str(REF_SRC).replace("\\", "/"),
            "out": str(REF_OUT).replace("\\", "/"),
            "sr": TARGET_SR,
            "dur_s": round(len(ref_out) / TARGET_SR, 3),
            "lufs_before": round(ref_lufs, 2) if ref_lufs else None,
            "gain_db": round(ref_gain, 2),
            "peak_dbfs_after": round(20 * math.log10(max(float(np.max(np.abs(ref_out))), 1e-9)), 2),
            "lead_sil_s": round(lead, 3), "tail_sil_s": round(tail, 3),
        },
    }
    (REPORTS / "02_audio.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
