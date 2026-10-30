"""Turn the rough Japanese corpus into a consistent single-speaker training set.

Every decision is recorded per clip in WORK/build/decisions.jsonl (kept or dropped,
and why) and summarised in WORK/build/report.json, so the build can be audited
without re-running anything.

1. Population.  Japanese speech only (Asami Imai). Songs are out: UVR-separated
   singing lands nowhere near the speech voice. Quality tier C is out.
2. Device-processed voice is out.  Amadeus is heard through a device EQ (79% of
   game SG0 Amadeus takes sit above +5 dB spectral tilt, against 21% of ordinary
   SG0 Kurisu), and the Japanese human `[digital]` labels mark phone calls (28/29
   on the ordinary variant). Neither is tagged consistently, so both would teach
   the *untagged* default voice a filter. ~950 clips.
3. Transcript agreement.  Script text and both ASR hypotheses (Qwen3-ASR,
   anime-whisper) are compared as katakana readings (pyopenjtalk + a cast-name
   table), so 岡部/おかべ or 2010年/にせんじゅうねん are not errors. A clip is kept
   when the better hypothesis is within the bar for its length. Sub-second clips
   must match almost exactly: they are the least stable rows for a codec LM.
4. Source-separated anime audio must be at least as clean as the studio game
   takes: estimated SNR between the words (`snr_est_db`, published per clip) at or
   above the game corpus's 5th percentile. (A learned MOS was tried; UTMOSv2 and
   DNSMOS both ran at <3 clips/s on this pod's CPU feature path.)
5. Edges.  Leading/trailing silence is measured against the clip's own active
   level, trimmed, faded (5 ms) and replaced by fixed padding. Upstream edges ranged
   0.0-2.2 s; a codec LM learns when to stop from what precedes EOS, so the tail
   is a constant 0.25 s (trailing silence > 0.5 s is the commonest cause of
   endless generation).
6. Level.  Per-clip loudness to -23 LUFS, gain clamped to [-6, +9] dB and peaks
   to -1 dBFS: extreme level differences between sources are removed while a
   whisper still lands quieter than a shout.
7. Text.  Bracket tags are not used in the main training text: Qwen3-TTS has no
   native tag support, the corpus holds 148 [sigh] / 58 [gasp] / 16 [breath]
   Japanese examples (too few to bind a control), and the previous evaluation's
   runaway generations clustered on tagged prompts. [pause] becomes "……", the
   script's own way of writing a held pause; event tags are dropped from the text
   while the audio keeps the sound, as in ordinary TTS corpora. Tag-only takes are
   dropped. The tagged text is kept in `text_tagged` for an ablation.
   ASCII spaces next to Japanese characters (a script display convention) are
   removed; spaces between Latin/digit tokens ("B14 7H") stay.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
SR = 24000
LEAD_PAD = 0.12
TAIL_PAD = 0.25
TARGET_LUFS, GAIN_MIN, GAIN_MAX, PEAK_MAX = -23.0, -6.0, 9.0, 10 ** (-1 / 20)
TAG_RE = re.compile(r"\[[a-z_]+\]")
MIN_DUR, MAX_DUR = 0.6, 18.0


# ---------------------------------------------------------------- reading-space CER

_KANA_KEEP = re.compile(r"[ァ-ヺ]")
# OpenJTalk's dictionary misreads the cast (紅莉栖 -> クレナイリース); both sides of
# the comparison go through this table so script kanji and ASR katakana meet.
READINGS = {
    "紅莉栖": "クリス", "牧瀬": "マキセ", "鳳凰院": "ホウオウイン", "凶真": "キョウマ",
    "倫太郎": "リンタロウ", "鈴羽": "スズハ", "萌郁": "モエカ", "比屋定": "ヒヤジョウ",
    "真帆": "マホ", "漆原": "ウルシバラ", "阿万音": "アマネ", "橋田": "ハシダ",
    "椎名": "シイナ", "桐生": "キリュウ", "天王寺": "テンノウジ",
    "綯": "ナエ", "由季": "ユキ", "HENTAI": "ヘンタイ", "SERN": "セルン",
    "Dメール": "ディーメール", "IBN5100": "アイビーエヌゴーセンイチゼロゼロ",
}


def reading(text: str) -> str:
    """Katakana reading with punctuation, long marks and small/large kana folded."""
    import pyopenjtalk

    text = unicodedata.normalize("NFKC", TAG_RE.sub("", text))
    for k, v in READINGS.items():
        text = text.replace(unicodedata.normalize("NFKC", k), v)
    kata = pyopenjtalk.g2p(text, kana=True) if text.strip() else ""
    kata = unicodedata.normalize("NFKC", kata)
    kata = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kata)
    small = str.maketrans("ァィゥェォッャュョヮヵヶ", "アイウエオツヤユヨワカケ")
    return "".join(_KANA_KEEP.findall(kata)).translate(small)


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(ref)


def cer_bar(n_kana: int, dur: float) -> float:
    if dur < 1.0:
        return 0.25
    if n_kana <= 4:
        return 0.5
    if n_kana <= 10:
        return 0.35
    return 0.25


# ---------------------------------------------------------------- text

_JA = r"[　-〿぀-ヿ一-鿿＀-￯…―〜]"
_SPACE_JA = re.compile(rf"(?<={_JA})[ 　]+|[ 　]+(?={_JA})")


def normalise_text(text: str) -> str:
    text = _SPACE_JA.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def train_text(transcription: str) -> str:
    text = re.sub(r"[、。]?\[pause\]", "……", transcription)   # "、[pause]" -> "……"
    text = TAG_RE.sub("", text)
    return normalise_text(text)


def tagged_text(transcription: str) -> str:
    text = transcription
    for tag in set(TAG_RE.findall(text)) - {"[sigh]", "[gasp]", "[pause]"}:
        text = text.replace(tag, "")
    return normalise_text(text)


# ---------------------------------------------------------------- audio

def speech_bounds(wav: np.ndarray, hop: int = 240) -> tuple[int, int] | None:
    """First/last sample of audible speech: frames within 40 dB of the clip's loud frames."""
    n = len(wav) // hop
    if n < 3:
        return None
    frames = wav[: n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    ref = np.percentile(db, 95)
    active = np.where(db > max(ref - 40.0, -75.0))[0]
    if active.size == 0:
        return None
    start = max(0, active[0] * hop - hop)
    end = min(len(wav), (active[-1] + 2) * hop)
    return start, end


def normalise_audio(wav: np.ndarray, meter) -> tuple[np.ndarray | None, float]:
    b = speech_bounds(wav)
    if b is None:
        return None, 0.0
    core = wav[b[0]:b[1]].astype(np.float32).copy()
    fade = int(0.005 * SR)
    if len(core) <= 2 * fade:
        return None, 0.0
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    core[:fade] *= ramp
    core[-fade:] *= ramp[::-1]
    try:
        loud = meter.integrated_loudness(core if len(core) >= int(0.4 * SR)
                                         else np.tile(core, int(0.4 * SR) // len(core) + 1))
    except Exception:
        loud = float("-inf")
    if not np.isfinite(loud):
        loud = 20 * np.log10(np.sqrt(np.mean(core ** 2)) + 1e-9) - 3.0
    gain_db = float(np.clip(TARGET_LUFS - loud, GAIN_MIN, GAIN_MAX))
    gain = 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(core))) * gain
    if peak > PEAK_MAX:
        gain *= PEAK_MAX / peak
        gain_db = 20 * np.log10(gain)
    core = core * gain
    out = np.concatenate([np.zeros(int(LEAD_PAD * SR), np.float32), core,
                          np.zeros(int(TAIL_PAD * SR), np.float32)])
    return out, gain_db


# ---------------------------------------------------------------- main

def main() -> None:
    import pyloudnorm

    meter = pyloudnorm.Meter(SR)
    meta = [json.loads(l) for l in (WORK / "src" / "ja_meta.jsonl").read_text(encoding="utf-8").splitlines() if l]
    asr = {r["id"]: r for r in (json.loads(l) for l in (WORK / "src" / "asr.jsonl").read_text(encoding="utf-8").splitlines() if l)}
    out_dir = WORK / "build"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    game_snr = sorted(r["snr_est_db"] for r in meta
                      if r["medium"] == "game" and r.get("snr_est_db") is not None)
    snr_floor = game_snr[int(0.05 * len(game_snr))]

    decisions, kept = [], []
    for r in meta:
        a = asr.get(r["id"], {})
        d = {"id": r["id"], "split": r["split"], "medium": r["medium"], "series": r["series"],
             "variant": r["character_variant"], "dur_src": r["duration_seconds"],
             "text": r["text"], "transcription": r["transcription"], "snr_est_db": r.get("snr_est_db")}
        reason = None
        if r["style"] != "speech":
            reason = "singing"
        elif r["quality_tier"] == "C":
            reason = "tier_c"
        elif r["character_variant"] == "amadeus":
            reason = "device_voice_amadeus"
        elif "[digital]" in r["tags"]:
            reason = "device_voice_phone"
        elif not r["text"].strip():
            reason = "no_lexical_text"
        elif (r["medium"] == "anime" and r.get("snr_est_db") is not None
              and r["snr_est_db"] < snr_floor):
            reason = "anime_below_studio_snr"

        if reason is None:
            ref = reading(r["text"])
            cq = cer(ref, reading(a.get("qwen") or ""))
            ca = cer(ref, reading(a.get("anime") or ""))
            d.update(ref_kana=len(ref), cer_qwen=round(cq, 3), cer_anime=round(ca, 3),
                     asr_qwen=a.get("qwen"), asr_anime=a.get("anime"))
            if min(cq, ca) > cer_bar(len(ref), r["duration_seconds"]):
                reason = "transcript_mismatch"

        if reason is None:
            wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
            if sr != SR:
                raise SystemExit(f"{r['id']}: {sr} Hz")
            wav = wav.mean(axis=-1) if wav.ndim > 1 else wav
            new, gain_db = normalise_audio(wav, meter)
            if new is None:
                reason = "no_audible_speech"
            else:
                dur = len(new) / SR
                d.update(dur=round(dur, 3), gain_db=round(gain_db, 2))
                if dur < MIN_DUR:
                    reason = "too_short"
                elif dur > MAX_DUR:
                    reason = "too_long"
                else:
                    path = wav_dir / f"{r['id']}.wav"
                    sf.write(str(path), new, SR, subtype="PCM_16")
                    d["text_train"] = train_text(r["transcription"])
                    kept.append({"id": r["id"], "audio": str(path), "text": d["text_train"],
                                 "text_tagged": tagged_text(r["transcription"]),
                                 "split": r["split"], "dur": d["dur"], "medium": r["medium"],
                                 "series": r["series"], "quality_tier": r["quality_tier"],
                                 "snr_est_db": r.get("snr_est_db"),
                                 "speaker_margin": r.get("speaker_margin")})
        d["kept"] = reason is None
        d["reason"] = reason
        decisions.append(d)

    (out_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf-8")
    (out_dir / "kept.jsonl").write_text(
        "".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")

    by_split = Counter(k["split"] for k in kept)
    report = {
        "input": len(meta),
        "kept": len(kept),
        "kept_hours": round(sum(k["dur"] for k in kept) / 3600, 3),
        "by_split": dict(by_split),
        "hours_by_split": {s: round(sum(k["dur"] for k in kept if k["split"] == s) / 3600, 3) for s in by_split},
        "kept_by_medium": dict(Counter(k["medium"] for k in kept)),
        "dropped": dict(Counter(d["reason"] for d in decisions if not d["kept"])),
        "mismatch_by_source": dict(Counter(f"{d['medium']}:{d['series']}" for d in decisions
                                           if d["reason"] == "transcript_mismatch")),
        "snr_floor_from_game_p05": snr_floor,
        "snr_median_kept": {m: float(np.median([k["snr_est_db"] for k in kept if k["medium"] == m
                                                and k["snr_est_db"] is not None]))
                            for m in ("game", "anime") if any(k["medium"] == m for k in kept)},
        "tags_in_tagged_variant": dict(Counter(t for k in kept for t in TAG_RE.findall(k["text_tagged"]))),
        "lead_pad_s": LEAD_PAD, "tail_pad_s": TAIL_PAD, "target_lufs": TARGET_LUFS,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
