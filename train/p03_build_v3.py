"""v3 build: v2's audio policy, driven by the per-clip v3 labels.

Differences from p03_build.py (v2):
1. Amadeus and phone takes are kept. They train their own speaker rows so the
   default `kurisu` row never sees a device filter. Which row a take goes to comes
   from labels_v3.jsonl (`speaker_row`): `amadeus` = every game Amadeus take (one
   consistent device EQ); `amadeus_anime` = anime Amadeus takes whose spectrum
   matches the anime's (different) device EQ, judged within Steins;Gate 0 so the
   production difference between the two anime series is not mistaken for it;
   anime Amadeus takes that sound clean go to `kurisu`; `phone` = [digital] takes.
2. Text keeps what is audible, as atomic tokens: `[sigh]` and `[nv]` (any other
   non-verbal sound; `[inhale]` and `[laugh]` merge into it, too few to stand
   alone) inline, `[bg_*]` (background sound) at the start. `[pause]` is still
   "……". Tags are only where the label pass trusted them (no listening was done).
3. Tag-only takes (an isolated sigh or other non-verbal sound) are kept when they
   hold at least 0.25 s of sound. They carry no lexical text, so the transcript check
   does not apply, and they get the median gain of the speech takes of their series
   instead of being pushed to speech loudness: a sigh stays quieter than a line.
   (Takes v2 already trained on keep v2's audio policy, so E1 replicates v2.)
   `[sigh]` that rests only on the dataset's auto-tagger must be mostly unvoiced or
   heard as sigh kana by an ASR, and panting is not a sigh: both become `[nv]`.
   Every anime Amadeus take goes to `amadeus_anime` (review: the "clean-sounding"
   ones still carried part of the device EQ).
4. Every row carries `language` ("japanese") and `speaker`.

Everything else - tier C out, songs out, the anime SNR floor, the reading-space
transcript check, edge trimming and padding, -23 LUFS with clamped gain - is v2's,
imported from p03_build.py so the two builds cannot drift apart.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import p03_build as V2

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
SR = V2.SR
MIN_SOUND_TAG_ONLY = 0.25        # seconds of actual sound, padding excluded
SPEAKER_ROWS = ("kurisu", "amadeus", "amadeus_anime", "phone")
# Events with fewer than 50 trusted takes merge into the catch-all (v3_synthesis.md
# s2): [inhale] 45 and [laugh] 23 after the label pass. The original type stays in
# `events` for a later split once there is more data.
MERGE = {"[inhale]": "[nv]", "[laugh]": "[nv]"}
ALLOWED = re.compile(r"\[(sigh|inhale|laugh|nv|bg_[a-z]+)\]")
ANY_TAG = re.compile(r"\[[^\]]*\]")
SIGH_KANA = re.compile(r"(は[ぁあ]|ふ[ぅう]|ハ[ァア]|フ[ゥウ]|ため息)")
PANT = re.compile(r"(は[ぁあ]|ハ[ァア])[、。…―\s]*(\[[a-z]+\])?(は[ぁあ]|ハ[ァア])")


def check_sigh(text_v3: str, lab: dict, r: dict, a: dict) -> tuple[str, str | None]:
    """`[sigh]` only where it is likely a sigh; otherwise the event stays explained as [nv].

    - Panting (はぁ、はぁ / はぁ……はぁ) is repeated breathing, not a sigh.
    - A [sigh] that rests only on the dataset's auto-tagger (isolated clips) is kept
      when the take is mostly unvoiced (voiced_ratio <= 0.6) or an ASR hears sigh kana;
      on lexical clips the same tagger was corroborated 1 time in 42.
    """
    if "[sigh]" not in text_v3:
        return text_v3, None
    if PANT.search(text_v3):
        return text_v3.replace("[sigh]", "[nv]"), "panting"
    srcs = {e.get("source") for e in lab.get("events") or [] if e.get("type") == "[sigh]"}
    if srcs and srcs <= {"isolated"}:
        heard = SIGH_KANA.search((a.get("qwen") or "") + (a.get("anime") or ""))
        vr = r.get("voiced_ratio")
        if not heard and (vr is None or vr > 0.6):
            return text_v3.replace("[sigh]", "[nv]"), "sigh_unverified"
    return text_v3, None
EVENT_ONLY = re.compile(r"^(\[bg_[a-z]+\])*(\[(sigh|inhale|laugh|nv)\]|[\s…―ー〜！？!?、。っッぁ-ゖァ-ヺ]){1,40}$")


def lexical(text_v3: str) -> str:
    """The words a transcript check can compare: tags and their kana removed."""
    return ALLOWED.sub("", text_v3)


def is_tag_only(text_v3: str) -> bool:
    body = ALLOWED.sub("", text_v3)
    has_event = any(t in text_v3 for t in ("[sigh]", "[inhale]", "[laugh]", "[nv]"))
    return has_event and not re.search(r"[一-鿿A-Za-z0-9]", body) and bool(EVENT_ONLY.match(text_v3))


def apply_gain(wav: np.ndarray, gain_db: float) -> np.ndarray | None:
    b = V2.speech_bounds(wav)
    if b is None:
        return None
    core = wav[b[0]:b[1]].astype(np.float32).copy()
    fade = int(0.005 * SR)
    if len(core) <= 2 * fade:
        return None
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    core[:fade] *= ramp
    core[-fade:] *= ramp[::-1]
    gain = 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(core))) * gain
    if peak > V2.PEAK_MAX:
        gain *= V2.PEAK_MAX / peak
    core = core * gain
    return np.concatenate([np.zeros(int(V2.LEAD_PAD * SR), np.float32), core,
                           np.zeros(int(V2.TAIL_PAD * SR), np.float32)])


def main() -> None:
    import pyloudnorm

    meter = pyloudnorm.Meter(SR)
    meta = [json.loads(l) for l in (WORK / "src" / "ja_meta.jsonl").read_text(encoding="utf-8").splitlines() if l]
    asr = {r["id"]: r for r in (json.loads(l) for l in (WORK / "src" / "asr.jsonl").read_text(encoding="utf-8").splitlines() if l)}
    labels = {r["id"]: r for r in (json.loads(l) for l in (WORK / "src" / "labels_v3.jsonl").read_text(encoding="utf-8").splitlines() if l)}
    missing = [r["id"] for r in meta if r["id"] not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} clips have no v3 label, e.g. {missing[:3]}")
    out_dir = WORK / "build_v3"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    game_snr = sorted(r["snr_est_db"] for r in meta
                      if r["medium"] == "game" and r.get("snr_est_db") is not None)
    snr_floor = game_snr[int(0.05 * len(game_snr))]

    decisions, kept, deferred = [], [], []
    for r in meta:
        lab = labels[r["id"]]
        a = asr.get(r["id"], {})
        text_v3 = V2.normalise_text(lab.get("text_v3") or "")
        for src, dst in MERGE.items():
            text_v3 = text_v3.replace(src, dst)
        text_v3, sigh_note = check_sigh(text_v3, lab, r, a)
        while "[nv][nv]" in text_v3:
            text_v3 = text_v3.replace("[nv][nv]", "[nv]")
        speaker_row = lab.get("speaker_row")
        if r["character_variant"] == "amadeus" and r["medium"] == "anime":
            # Review: anime Amadeus takes judged "clean" still sat 62% past the midpoint
            # of the Kurisu->Amadeus spectral axis. None go to the default row.
            speaker_row = "amadeus_anime"
        stray =[t for t in ANY_TAG.findall(text_v3) if not ALLOWED.fullmatch(t)]
        d = {"id": r["id"], "split": r["split"], "medium": r["medium"], "series": r["series"],
             "variant": r["character_variant"], "speaker": speaker_row, "sigh_note": sigh_note,
             "dur_src": r["duration_seconds"], "transcription": r["transcription"],
             "text_v3": text_v3, "snr_est_db": r.get("snr_est_db"),
             "label_reason": lab.get("reason")}
        # the label pass decided which takes are isolated events; the regex is a fallback
        tag_only = bool(lab["tag_only"]) if "tag_only" in lab else is_tag_only(text_v3)
        reason = None
        if stray:
            raise SystemExit(f"{r['id']}: unexpected tags {stray} in {text_v3!r}")
        # The label pass's transcript/duration verdicts are recomputed below; for an
        # isolated event there is no transcript to check.
        recomputed = {"transcript_mismatch", "too_short", "no_lexical_text"}
        if (not lab.get("keep") and not lab.get("needs_cer_check")
                and not (tag_only and lab.get("reason") in recomputed)):
            reason = lab.get("reason") or "label_pass"
        elif r["style"] != "speech":
            reason = "singing"
        elif r["quality_tier"] == "C":
            reason = "tier_c"
        elif speaker_row not in SPEAKER_ROWS:
            reason = "no_speaker_row"
        elif not text_v3:
            reason = "no_text"
        elif (r["medium"] == "anime" and r.get("snr_est_db") is not None
              and r["snr_est_db"] < snr_floor):
            reason = "anime_below_studio_snr"

        if reason is None and not tag_only:
            ref = V2.reading(r["text"])         # the script text, exactly as v2 and the label pass
            cq = V2.cer(ref, V2.reading(a.get("qwen") or ""))
            ca = V2.cer(ref, V2.reading(a.get("anime") or ""))
            d.update(ref_kana=len(ref), cer_qwen=round(cq, 3), cer_anime=round(ca, 3))
            if not ref:
                reason = "no_lexical_text"
            elif min(cq, ca) > V2.cer_bar(len(ref), r["duration_seconds"]):
                reason = "transcript_mismatch"

        if reason is None and tag_only and not lab.get("v2_kept"):
            # gain needs the speech takes first; takes v2 already trained on keep v2's
            # audio policy so the E1 replicate is byte-identical to v2's data
            deferred.append((r, d, text_v3, speaker_row))
            decisions.append(d)
            continue

        if reason is None:
            wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
            if sr != SR:
                raise SystemExit(f"{r['id']}: {sr} Hz")
            wav = wav.mean(axis=-1) if wav.ndim > 1 else wav
            new, gain_db = V2.normalise_audio(wav, meter)
            if new is None:
                reason = "no_audible_speech"
            else:
                dur = len(new) / SR
                d.update(dur=round(dur, 3), gain_db=round(gain_db, 2))
                if dur < V2.MIN_DUR:
                    reason = "too_short"
                elif dur > V2.MAX_DUR:
                    reason = "too_long"
                else:
                    path = wav_dir / f"{r['id']}.wav"
                    sf.write(str(path), new, SR, subtype="PCM_16")
                    kept.append(row_for(r, lab, path, text_v3, dur, tag_only, speaker_row))
        d["kept"] = reason is None
        d["reason"] = reason
        decisions.append(d)

    # Tag-only takes: the median gain of the kept speech takes of the same series.
    gains = defaultdict(list)
    for d in decisions:
        if d.get("kept") and "gain_db" in d:
            gains[d["series"]].append(d["gain_db"])
    all_gains = [g for v in gains.values() for g in v]
    for r, d, text_v3, speaker_row in deferred:
        g = float(np.median(gains.get(r["series"]) or all_gains))
        wav, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
        wav = wav.mean(axis=-1) if wav.ndim > 1 else wav
        new = apply_gain(wav, g)
        reason = None
        if new is None:
            reason = "no_audible_sound"
        else:
            dur = len(new) / SR
            d.update(dur=round(dur, 3), gain_db=round(g, 2), tag_only=True)
            if dur - V2.LEAD_PAD - V2.TAIL_PAD < MIN_SOUND_TAG_ONLY:
                reason = "too_short"
            elif dur > V2.MAX_DUR:
                reason = "too_long"
            else:
                path = wav_dir / f"{r['id']}.wav"
                sf.write(str(path), new, SR, subtype="PCM_16")
                kept.append(row_for(r, labels[r["id"]], path, text_v3, dur, True, speaker_row))
        d["kept"] = reason is None
        d["reason"] = reason

    write_jsonl(out_dir / "decisions.jsonl", decisions)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_refs(out_dir, kept)

    tag_re = re.compile(r"\[[a-z_]+\]")
    report = {
        "input": len(meta), "kept": len(kept),
        "kept_hours": round(sum(k["dur"] for k in kept) / 3600, 3),
        "by_speaker": dict(Counter(k["speaker"] for k in kept)),
        "hours_by_speaker": {s: round(sum(k["dur"] for k in kept if k["speaker"] == s) / 3600, 3)
                             for s in {k["speaker"] for k in kept}},
        "by_split": dict(Counter(k["split"] for k in kept)),
        "by_speaker_split": dict(Counter(f"{k['speaker']}:{k['split']}" for k in kept)),
        "tag_only": sum(k["tag_only"] for k in kept),
        "tokens": dict(Counter(t for k in kept for t in tag_re.findall(k["text"]))),
        "tokens_by_split": dict(Counter(f"{t}:{k['split']}" for k in kept
                                        for t in tag_re.findall(k["text"]))),
        "dropped": dict(Counter(d["reason"] for d in decisions if not d["kept"])),
        "snr_floor_from_game_p05": snr_floor,
        "lead_pad_s": V2.LEAD_PAD, "tail_pad_s": V2.TAIL_PAD, "target_lufs": V2.TARGET_LUFS,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def row_for(r, lab, path, text_v3, dur, tag_only, speaker_row):
    return {"id": r["id"], "audio": str(path), "text": text_v3,
            "text_plain": V2.train_text(r["transcription"]),
            "language": "japanese", "speaker": speaker_row,
            "split": r["split"], "dur": round(dur, 3), "medium": r["medium"],
            "series": r["series"], "variant": r["character_variant"],
            "quality_tier": r["quality_tier"], "snr_est_db": r.get("snr_est_db"),
            "speaker_margin": r.get("speaker_margin"), "tag_only": tag_only,
            "bg": lab.get("bg"), "events": lab.get("events") or []}


def write_refs(out_dir: Path, kept: list[dict]) -> None:
    """Clip lists averaged into each saved speaker's locked vector (train split only).

    kurisu: v2's rule (tier A game takes, 3-10 s, best speaker margin, spread over
    SG/SGRE/PHENO, no tags). amadeus: the same rule over game Amadeus takes. phone:
    every real phone take in train (augmented clips are not the target sound).
    """
    plain = [k for k in kept if k["split"] == "train" and "[" not in k["text"]]

    def pick(rows, n, series):
        pool = sorted([k for k in rows if 3.0 <= k["dur"] <= 10.0 and k["quality_tier"] == "A"],
                      key=lambda k: -(k.get("speaker_margin") or -9))
        out = []
        for s in series:
            out += [k for k in pool if k["series"] == s][: n // len(series)]
        return out

    refs = {
        "kurisu": pick([k for k in plain if k["speaker"] == "kurisu" and k["medium"] == "game"
                        and (k.get("speaker_margin") or -1) > 0], 64, ("SG", "SGRE", "PHENO")),
        "amadeus": pick([k for k in plain if k["speaker"] == "amadeus" and k["medium"] == "game"],
                        64, ("SG0",)),
        # anime takes are tier B more often than A; take the longest-margin 3-10 s ones
        "amadeus_anime": sorted([k for k in plain if k["speaker"] == "amadeus_anime"
                                 and 3.0 <= k["dur"] <= 10.0],
                                key=lambda k: -(k.get("speaker_margin") or -9))[:64],
        "phone": [k for k in kept if k["split"] == "train" and k["speaker"] == "phone"],
    }
    if len(refs["amadeus"]) < 32:     # tier A game Amadeus is scarce: relax to tier B
        refs["amadeus"] = sorted([k for k in plain if k["speaker"] == "amadeus" and k["medium"] == "game"
                                  and 3.0 <= k["dur"] <= 10.0],
                                 key=lambda k: -(k.get("speaker_margin") or -9))[:64]
    for name, rows in refs.items():
        (out_dir / f"ref_{name}.txt").write_text("".join(k["audio"] + "\n" for k in rows), encoding="utf-8")
        print(f"[ref] {name}: {len(rows)} clips, {sum(k['dur'] for k in rows):.0f}s")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
