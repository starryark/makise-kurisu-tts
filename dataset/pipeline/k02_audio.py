"""Stage 02 - one standardised 24 kHz mono WAV per clip, with the source's own
dynamics put back.

Three things happen here, in this order.

1. **Resample.**  Game audio is 48 kHz; everything else already arrives at
   24 kHz.  ``soxr`` very-high quality, no trimming, no denoising, no gating.

2. **Undo the upstream per-clip loudness normalisation.**  The anime and song
   builds normalised every clip independently to a fixed active-speech RMS of
   -22 dBFS.  That is convenient and it is also exactly what the preparation
   guide warns against (s9.5): it flattens the difference between a whisper and
   a shout, which is the single most useful thing an expressive TTS corpus can
   teach.  Each clip records the gain it was given, so the operation is exactly
   invertible -- multiply by ``10 ** (-gain_db / 20)`` and the clip is back on
   the scale of its own programme, every one of which was normalised as a whole
   to -23 LUFS before any cuts were made.

3. **One gain per source group.**  Group integrated loudness is measured to
   ITU-R BS.1770-4 by pooling the K-weighted 400 ms blocks of every clip in the
   group and applying the absolute and relative gates across the pool, which is
   the same number a single concatenated render would give.  Each group is then
   moved to a common target with a single gain, so loudness relationships
   *inside* a group survive untouched.  If that gain would push any clip in the
   group past the peak ceiling, the whole group is backed off together.
"""
from __future__ import annotations

import collections
import json

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import soxr

from kcommon import (GROUP_LUFS_TARGET, MANIFESTS, PEAK_CEILING_DBFS, REPORTS,
                     TARGET_SR, WAVS, read_jsonl, utf8_stdout, write_json,
                     write_jsonl)

utf8_stdout()

FRAME_S = 0.020
BLOCK_S = 0.400
BLOCK_STEP_S = 0.100
ACTIVE_REL_DB = 25.0      # active = within this many dB of the clip's loudest frame
ACTIVE_FLOOR_DBFS = -65.0

_METER = pyln.Meter(TARGET_SR)


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def to_target_sr(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    return soxr.resample(x, sr, TARGET_SR, quality="VHQ").astype(np.float32)


def frame_rms(x: np.ndarray, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    n = max(1, int(round(FRAME_S * sr)))
    m = len(x) // n
    if m == 0:
        return np.array([float(np.sqrt(np.mean(x ** 2)) if len(x) else 0.0)]), n
    f = x[: m * n].reshape(m, n)
    return np.sqrt((f ** 2).mean(axis=1) + 1e-20), n


def activity(x: np.ndarray) -> dict:
    """Frame-level speech activity, measured never modified."""
    r, n = frame_rms(x)
    rd = 20.0 * np.log10(r + 1e-12)
    peak_frame = float(rd.max())
    thr = max(peak_frame - ACTIVE_REL_DB, ACTIVE_FLOOR_DBFS)
    act = rd >= thr
    idx = np.flatnonzero(act)
    dur = len(x) / TARGET_SR
    if len(idx) == 0:
        return {"active_rms_dbfs": _db(float(np.sqrt(np.mean(x ** 2)) if len(x) else 0)),
                "activity_threshold_dbfs": round(thr, 2), "speech_ratio": 0.0,
                "lead_silence_s": round(dur, 3), "tail_silence_s": round(dur, 3),
                "internal_pause_count": 0, "internal_pause_total_s": 0.0,
                "internal_pause_max_s": 0.0, "internal_pauses": []}
    active_rms = float(np.sqrt((r[act] ** 2).mean()))
    lead = idx[0] * n / TARGET_SR
    tail = dur - (idx[-1] + 1) * n / TARGET_SR
    gaps, run = [], 0
    for i in range(idx[0], idx[-1] + 1):
        if not act[i]:
            run += 1
        elif run:
            gaps.append((i - run, run))
            run = 0
    pauses = [{"start_s": round(s * n / TARGET_SR, 3),
               "duration_s": round(L * n / TARGET_SR, 3)}
              for s, L in gaps if L * n / TARGET_SR >= 0.20]
    return {
        "active_rms_dbfs": round(_db(active_rms), 2),
        "activity_threshold_dbfs": round(thr, 2),
        "speech_ratio": round(float(act.mean()), 4),
        "lead_silence_s": round(float(lead), 3),
        "tail_silence_s": round(float(max(tail, 0.0)), 3),
        "internal_pause_count": len(pauses),
        "internal_pause_total_s": round(sum(p["duration_s"] for p in pauses), 3),
        "internal_pause_max_s": round(max((p["duration_s"] for p in pauses), default=0.0), 3),
        "internal_pauses": pauses,
    }


def k_blocks(x: np.ndarray) -> np.ndarray:
    """Mean square of every gated-loudness block, K-weighted (BS.1770-4)."""
    y = x.astype(np.float64)
    for f in _METER._filters.values():
        y = f.apply_filter(y)
    n = int(BLOCK_S * TARGET_SR)
    step = int(BLOCK_STEP_S * TARGET_SR)
    if len(y) < n:
        return np.empty(0, dtype=np.float64)
    starts = range(0, len(y) - n + 1, step)
    return np.array([float((y[s:s + n] ** 2).mean()) for s in starts])


def integrated_from_blocks(z: np.ndarray) -> float | None:
    """Pooled BS.1770-4 integrated loudness for a set of block mean squares."""
    if z.size == 0:
        return None
    l = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-20))
    keep = l >= -70.0
    if not keep.any():
        return None
    rel = -0.691 + 10.0 * np.log10(z[keep].mean()) - 10.0
    keep2 = keep & (l >= rel)
    if not keep2.any():
        return None
    return float(-0.691 + 10.0 * np.log10(z[keep2].mean()))


def main() -> None:
    rows = list(read_jsonl(MANIFESTS / "01_inventory.jsonl"))
    print(f"clips: {len(rows)}")

    group_blocks: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    group_peak: dict[str, float] = collections.defaultdict(lambda: -120.0)
    staged: list[dict] = []

    # ---- pass 1: read, resample, restore dynamics, measure -------------------
    for i, r in enumerate(rows):
        x, sr = load_mono(r["src_path"])
        x = to_target_sr(x, sr)

        restore_db = 0.0
        if r["medium"] in ("anime", "song"):
            g = r["upstream_scores"].get("gain_db")
            if g is not None:
                restore_db = -float(g)
                x = x * (10.0 ** (restore_db / 20.0))

        g = r["source_group"]
        group_blocks[g].append(k_blocks(x))
        pk = _db(float(np.abs(x).max()) if len(x) else 0.0)
        group_peak[g] = max(group_peak[g], pk)
        staged.append({"id": r["id"], "group": g, "restore_db": round(restore_db, 3),
                       "pre_peak_dbfs": round(pk, 2), "src_sr": sr,
                       "samples": int(len(x))})
        if (i + 1) % 1500 == 0:
            print(f"  measured {i + 1}/{len(rows)}", flush=True)

    # ---- group gains ---------------------------------------------------------
    group_gain, group_loudness = {}, {}
    for g, blocks in group_blocks.items():
        z = np.concatenate([b for b in blocks if b.size]) if any(
            b.size for b in blocks) else np.empty(0)
        lufs = integrated_from_blocks(z)
        group_loudness[g] = None if lufs is None else round(lufs, 3)
        gain = 0.0 if lufs is None else GROUP_LUFS_TARGET - lufs
        headroom = PEAK_CEILING_DBFS - group_peak[g]
        if gain > headroom:                       # back the whole group off together
            gain = headroom
        group_gain[g] = round(float(gain), 3)

    # ---- pass 2: apply and write --------------------------------------------
    out_rows = []
    by_id = {s["id"]: s for s in staged}
    for i, r in enumerate(rows):
        s = by_id[r["id"]]
        x, sr = load_mono(r["src_path"])
        x = to_target_sr(x, sr)
        if s["restore_db"]:
            x = x * (10.0 ** (s["restore_db"] / 20.0))
        gain = group_gain[s["group"]]
        x = x * (10.0 ** (gain / 20.0))
        clipped = int(np.count_nonzero(np.abs(x) > 1.0))
        if clipped:
            x = np.clip(x, -1.0, 1.0)

        out = WAVS / f"{r['id']}.wav"
        sf.write(out, x.astype(np.float32), TARGET_SR, subtype="PCM_16")

        act = activity(x)
        z = k_blocks(x)
        lufs = integrated_from_blocks(z)
        m = dict(r)
        m.pop("upstream_scores_dummy", None)
        m.update({
            "audio": f"audio/wavs/{r['id']}.wav",
            "sr": TARGET_SR,
            "duration": round(len(x) / TARGET_SR, 4),
            "samples": int(len(x)),
            "source_sr": s["src_sr"],
            "dynamics_restore_db": s["restore_db"],
            "group_gain_db": gain,
            "total_gain_db": round(s["restore_db"] + gain, 3),
            "peak_dbfs": round(_db(float(np.abs(x).max()) if len(x) else 0.0), 2),
            "clipped_samples": clipped,
            "lufs_clip": None if lufs is None else round(lufs, 2),
        })
        m.update(act)
        out_rows.append(m)
        if (i + 1) % 1500 == 0:
            print(f"  wrote {i + 1}/{len(rows)}", flush=True)

    write_jsonl(MANIFESTS / "02_audio.jsonl", out_rows)

    dur = np.array([m["duration"] for m in out_rows])
    arms = np.array([m["active_rms_dbfs"] for m in out_rows])
    report = {
        "clips": len(out_rows),
        "hours": round(float(dur.sum()) / 3600.0, 4),
        "sample_rate": TARGET_SR,
        "format": "WAV PCM_16 mono",
        "duration_s": {"min": round(float(dur.min()), 3),
                       "p05": round(float(np.percentile(dur, 5)), 3),
                       "median": round(float(np.median(dur)), 3),
                       "p95": round(float(np.percentile(dur, 95)), 3),
                       "max": round(float(dur.max()), 3)},
        "group_loudness_before_lufs": group_loudness,
        "group_gain_db": group_gain,
        "loudness_target_lufs": GROUP_LUFS_TARGET,
        "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
        "clips_with_clipped_samples": int(sum(1 for m in out_rows if m["clipped_samples"])),
        "dynamics_restored_clips": int(sum(1 for m in out_rows if m["dynamics_restore_db"])),
        "active_rms_dbfs": {"p05": round(float(np.percentile(arms, 5)), 2),
                            "median": round(float(np.median(arms)), 2),
                            "p95": round(float(np.percentile(arms, 95)), 2),
                            "spread_p05_p95_db": round(
                                float(np.percentile(arms, 95) - np.percentile(arms, 5)), 2)},
        "active_rms_spread_by_medium": {
            k: round(float(np.percentile(v, 95) - np.percentile(v, 5)), 2)
            for k, v in {
                m: [x["active_rms_dbfs"] for x in out_rows if x["medium"] == m]
                for m in sorted({x["medium"] for x in out_rows})}.items()},
    }
    write_json(REPORTS / "02_audio.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
