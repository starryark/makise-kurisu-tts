"""Stage 11 - write the Hugging Face repository.

Hugging Face ``AudioFolder`` layout, one directory per split, audio hard-linked
from ``combined/audio/wavs`` so the staging copy costs no extra disk.

The published schema is deliberately wide.  Everything the pipeline measured is a
column, because the useful thing about an automatically labelled corpus is that a
consumer can re-cut it: keep only clips whose speaker margin clears their own
bar, drop everything a human marked as containing a doorbell, train on Japanese
alone, or take the six-way parallel lines and nothing else.  A dataset that only
publishes its own verdict forces everyone to accept its thresholds.

``transcription`` is the training label; ``text`` is the same line with every tag
removed, for consumers who want none of this.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from kcommon import (ALL_TAGS, CONDITION_TAGS, EVENT_TAGS, HF_OUT, MANIFESTS,
                     PROJECT, REPORTS, STRUCTURAL_TAGS, VOICE_ACTOR, read_jsonl,
                     utf8_stdout, write_json, write_jsonl)

utf8_stdout()

SPLITS = ("train", "validation", "test")
SHARD_SIZE = 4000         # Hub hard limit is 10,000 files per directory


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def speaker_thresholds(rows: list[dict], pct: float = 2.0) -> dict:
    """Per-(language, medium) outlier bar on the speaker-identity cosine.

    An absolute rule -- "flag every clip whose margin over the closest rival is
    negative" -- looks principled and is not.  ``max over rivals`` is an
    upward-biased order statistic, so the flag fires more often the more rival
    characters a cell has: it marked 30% of the game corpus, whose speaker labels
    come from the games' own voice-bank bytecode and are about as close to ground
    truth as this project has.  What that rule measures is rival count, not label
    quality.  Centred cosines also sit at very different scales per cell (0.34
    for dry game voice, 0.70 for the French dub), so one number cannot serve both.

    So the bar is relative: the bottom ``pct``% of each cell's own-centroid
    cosine.  ``speaker_similarity``, ``speaker_rival_similarity`` and
    ``speaker_margin`` are all published per clip, so anyone who wants a stricter
    or looser rule has the raw numbers.
    """
    cells = collections.defaultdict(list)
    for m in rows:
        sc = m.get("speaker_check") or {}
        if sc.get("spk_cos") is not None:
            cells[(m["language"], m["medium"])].append(sc["spk_cos"])
    return {k: round(float(np.percentile(v, pct)), 4)
            for k, v in cells.items() if len(v) >= 50}


def _uninformative(v) -> bool:
    """A value Arrow cannot infer a type from."""
    return v is None or v == [] or v == ""


def schema_anchored(rs: list[dict]) -> list[dict]:
    """Order a split's metadata so its schema is inferable from the first rows.

    ``datasets``' folder builder reads one 10 MB batch of ``metadata.jsonl`` at a
    320 kB Arrow block size and *breaks*, then demands that every split's
    inferred schema match.  So the schema really comes from each file's first
    ~200 rows.  An empty list types as ``list<null>`` and a null as ``null``, and
    this corpus is ordered by source: the first thousands of train rows are game
    clips with no human review, no emotion note and no parallel group, so train
    inferred nulls where validation inferred strings and ``load_dataset``
    refused the dataset outright.

    Hoisting one row per nullable column to the front costs nothing, is
    deterministic, and makes every split infer the same correct schema whatever
    order the corpus is in.  Row order in ``metadata.jsonl`` carries no meaning
    to a consumer -- ``file_name`` already says where each clip lives.
    """
    if not rs:
        return rs
    anchors, taken = [], set()
    for col in rs[0]:
        if not any(_uninformative(r[col]) for r in rs):
            continue                      # never empty: any row types it
        for i, r in enumerate(rs):
            if i not in taken and not _uninformative(r[col]):
                anchors.append(i)
                taken.add(i)
                break
    return ([rs[i] for i in anchors]
            + [r for i, r in enumerate(rs) if i not in taken])


def quality(m: dict, spk_bar: dict) -> tuple[str, list[str]]:
    flags = []
    sc = m.get("speaker_check") or {}
    bar = spk_bar.get((m["language"], m["medium"]))
    if bar is not None and sc.get("spk_cos") is not None and sc["spk_cos"] < bar:
        flags.append("speaker_outlier")
    if m["contamination"]:
        flags.append("nonspeech_contamination")
    snr = (m["features"] or {}).get("snr_est_db")
    if snr is not None and snr < 15.0:
        flags.append("low_snr")
    if m["clipped_samples"]:
        flags.append("clipped")
    if m["active_rms_dbfs"] is not None and m["active_rms_dbfs"] < -55.0:
        flags.append("near_silent")
    if m["duration"] < 0.5:
        flags.append("very_short")
    if m["duration"] > 20.0:
        flags.append("very_long")
    al = m.get("alignment")
    # The aligner emits zero-length spans for some short function words, so
    # coverage understates how much of the line it actually found.  This bar is
    # set low on purpose: it is here to catch a transcript that does not match its
    # audio, not to grade alignment quality.
    if al and al.get("ok") and al.get("coverage") is not None and al["coverage"] < 0.25:
        flags.append("low_alignment_coverage")
    if m["lead_silence_s"] < 0.02 or m["tail_silence_s"] < 0.02:
        flags.append("speech_at_edge")

    hard = {"speaker_outlier", "nonspeech_contamination", "low_snr",
            "clipped", "near_silent"}
    if set(flags) & hard:
        return "C", flags
    trusted = (m["human_reviewed"] or m["medium"] == "game"
               or m["upstream_tier"] in ("strict", "A"))
    if not flags and trusted:
        return "A", flags
    return "B", flags


def main() -> None:
    rows = [m for m in read_jsonl(MANIFESTS / "10_split.jsonl") if m["split"]]
    held = [m for m in read_jsonl(MANIFESTS / "10_split.jsonl") if not m["split"]]
    print(f"exporting {len(rows)} clips ({len(held)} held back)")
    spk_bar = speaker_thresholds(rows)
    print(f"speaker outlier bar per cell: "
          f"{ {f'{a}:{b}': v for (a, b), v in sorted(spk_bar.items())} }")

    if HF_OUT.exists():
        backup = HF_OUT.with_name(HF_OUT.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        HF_OUT.rename(backup)
        print(f"previous staging directory moved to {backup}")
    for s in SPLITS:
        (HF_OUT / "data" / s).mkdir(parents=True, exist_ok=True)

    (HF_OUT / ".gitattributes").write_text(
        "*.wav filter=lfs diff=lfs merge=lfs -text\n"
        "*.jsonl text eol=lf\n"
        "*.md text eol=lf\n", encoding="utf-8")

    writers = {s: open(HF_OUT / "data" / s / "metadata.jsonl", "w",
                       encoding="utf-8", newline="\n") for s in SPLITS}
    meta = {s: [] for s in SPLITS}
    published = []
    try:
        per_split = collections.Counter()
        for i, m in enumerate(rows):
            split = m["split"]
            # The Hub rejects any commit that puts more than 10,000 files in one
            # directory, and train alone is 12,832 clips.  Numeric shards keep
            # every directory well under that and leave room to grow; the
            # metadata file stays at the split root and names each clip by its
            # path relative to itself, which is what audiofolder resolves.
            shard = f"{per_split[split] // SHARD_SIZE:03d}"
            per_split[split] += 1
            name = f"{shard}/{m['id']}.wav"
            src = PROJECT / m["audio"]
            dst = HF_OUT / "data" / split / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

            f = m["features"] or {}
            sc = m["speaker_check"] or {}
            al = m.get("alignment") or {}
            tier, flags = quality(m, spk_bar)
            present = [t for t in ALL_TAGS if t in m["transcription"]]
            src_of = m.get("condition_tag_source") or {}
            human_tags = sorted({e["tag"] for e in m["events"]
                                 if e.get("source") == "human" and e.get("tag")}
                                | {t for t, s in src_of.items() if s == "human"})
            auto_tags = sorted({e["tag"] for e in m["events"]
                                if e.get("source") == "auto" and e.get("accepted")
                                and e.get("tag")}
                               | {t for t, s in src_of.items() if s in ("auto", "source")}
                               | ({"[pause]"} if m["measured_pauses"] else set()))
            row = {
                "file_name": name,
                "transcription": m["transcription"],
                "text": m["text"],
                "language": m["language"],
                "voice_id": m["voice_id"],
                "voice_actor": VOICE_ACTOR[m["language"]],
                "speaker_id": "kurisu",
                "speaker_name": "Makise Kurisu",
                "character_variant": m["character_variant"],
                "medium": m["medium"],
                "style": m["style"],
                "series": m["series"],
                "source_group": m["source_group"],
                "scene": m["scene"],
                "duration_seconds": round(m["duration"], 3),
                "sample_rate": m["sr"],
                "tags": [t for t in present],
                "event_tags": [t for t in present if t in EVENT_TAGS],
                "condition_tags": [t for t in present if t in CONDITION_TAGS],
                "human_tags": [t for t in human_tags if t in present],
                "auto_tags": [t for t in auto_tags if t in present],
                "human_reviewed": bool(m["human_reviewed"]),
                "from_quarantine": bool(m["from_quarantine"]),
                "parallel_group": m["parallel_group"],
                "emotion_notes": m["emotion_notes"],
                "contamination": m["contamination"],
                "contamination_score": m["contamination_score"],
                "measured_pauses": m["measured_pauses"],
                "transcript_source": m["transcript_authority"],
                "quality_tier": tier,
                "quality_flags": flags,
                # --- acoustics, all measured on the published WAV ---
                "active_rms_dbfs": m["active_rms_dbfs"],
                "peak_dbfs": m["peak_dbfs"],
                "lufs": m["lufs_clip"],
                "clipped_samples": m["clipped_samples"],
                "speech_ratio": m["speech_ratio"],
                "lead_silence_s": m["lead_silence_s"],
                "tail_silence_s": m["tail_silence_s"],
                "internal_pause_max_s": m["internal_pause_max_s"],
                "snr_est_db": f.get("snr_est_db"),
                "noise_floor_dbfs": f.get("noise_floor_dbfs"),
                "f0_median_hz": f.get("f0_median_hz"),
                "f0_range_st": f.get("f0_range_st"),
                "voiced_ratio": f.get("voiced_ratio"),
                "hnr_db": f.get("hnr_db"),
                "bandwidth_roll995_hz": f.get("roll995_hz"),
                "spectral_tilt_db": f.get("spectral_tilt_db"),
                "reverb_decay_db_per_s": f.get("decay_db_per_s"),
                "speaking_rate_units_per_s": f.get("units_per_active_s"),
                "lexical_units": f.get("lexical_units"),
                # --- verification ---
                "speaker_similarity": sc.get("spk_cos"),
                "speaker_rival_similarity": sc.get("spk_rival_cos"),
                "speaker_margin": sc.get("spk_margin"),
                "speaker_rival": sc.get("spk_rival"),
                "alignment_coverage": al.get("coverage"),
                "sha256": sha256_file(dst),
            }
            meta[split].append(row)
            published.append({**row, "split": split, "id": m["id"]})
            if (i + 1) % 2000 == 0:
                print(f"  staged {i + 1}/{len(rows)}", flush=True)
        for split, rs in meta.items():
            for row in schema_anchored(rs):
                writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for w in writers.values():
            w.close()

    write_jsonl(MANIFESTS / "11_published.jsonl", published)

    # ---- one reference clip per voice ---------------------------------------
    # Qwen3-TTS reads a reference on every training row and writes its embedding
    # into the speaker table, so a badly chosen reference quietly poisons a whole
    # fine-tune.  Six voices means six references, and the choice is made from
    # measurements rather than by ear: highest margin over the closest rival
    # character, among clean, untagged, well-sized clips of that voice.
    (HF_OUT / "reference").mkdir(parents=True, exist_ok=True)
    refs = {}
    by_voice = collections.defaultdict(list)
    for p in published:
        by_voice[p["voice_id"]].append(p)
    for voice, ps in sorted(by_voice.items()):
        cands = [p for p in ps
                 if p["quality_tier"] == "A" and not p["tags"]
                 and p["style"] == "speech"
                 and 4.0 <= p["duration_seconds"] <= 10.0
                 and p["speaker_margin"] is not None
                 and p["lead_silence_s"] >= 0.05 and p["tail_silence_s"] >= 0.05
                 and (p["snr_est_db"] or 0) >= 30]
        if not cands:
            cands = [p for p in ps if p["speaker_margin"] is not None
                     and 2.0 <= p["duration_seconds"] <= 12.0]
        if not cands:
            continue
        best = max(cands, key=lambda p: (p["speaker_margin"], p["speaker_similarity"]))
        src = HF_OUT / "data" / best["split"] / best["file_name"]
        dst = HF_OUT / "reference" / f"{voice}.wav"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        refs[voice] = {
            "file": f"reference/{voice}.wav",
            "clip_id": best["id"], "language": best["language"],
            "voice_actor": best["voice_actor"], "medium": best["medium"],
            "series": best["series"], "text": best["text"],
            "duration_seconds": best["duration_seconds"],
            "speaker_similarity": best["speaker_similarity"],
            "speaker_margin": best["speaker_margin"],
            "closest_other_character": best["speaker_rival"],
            "snr_est_db": best["snr_est_db"],
            "candidates_considered": len(cands),
            "sha256": best["sha256"],
        }
    (HF_OUT / "reference" / "reference.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(REPORTS / "11_reference.json", refs)
    print(f"reference clips: {sorted(refs)}")

    dur = collections.defaultdict(float)
    cnt = collections.Counter()
    for p in published:
        dur[p["split"]] += p["duration_seconds"]
        cnt[p["split"]] += 1
    stats = {
        "clips": len(published),
        "hours": round(sum(dur.values()) / 3600.0, 4),
        "splits": {s: {"clips": cnt[s], "hours": round(dur[s] / 3600.0, 4)}
                   for s in SPLITS},
        "by_language": dict(collections.Counter(p["language"] for p in published)),
        "hours_by_language": {
            l: round(sum(p["duration_seconds"] for p in published
                         if p["language"] == l) / 3600.0, 3)
            for l in sorted({p["language"] for p in published})},
        "by_medium": dict(collections.Counter(p["medium"] for p in published)),
        "by_variant": dict(collections.Counter(p["character_variant"] for p in published)),
        "by_series": dict(collections.Counter(p["series"] for p in published).most_common()),
        "by_quality_tier": dict(collections.Counter(p["quality_tier"] for p in published)),
        "quality_flags": dict(collections.Counter(
            f for p in published for f in p["quality_flags"]).most_common()),
        "tags": dict(collections.Counter(t for p in published for t in p["tags"]).most_common()),
        "tag_provenance": {
            "human": dict(collections.Counter(
                t for p in published for t in p["human_tags"]).most_common()),
            "auto": dict(collections.Counter(
                t for p in published for t in p["auto_tags"]).most_common())},
        "clips_with_any_tag": sum(1 for p in published if p["tags"]),
        "parallel_groups": len({p["parallel_group"] for p in published
                                if p["parallel_group"]}),
        "clips_in_parallel_group": sum(1 for p in published if p["parallel_group"]),
        "from_quarantine": sum(1 for p in published if p["from_quarantine"]),
        "human_reviewed": sum(1 for p in published if p["human_reviewed"]),
        "duration_seconds": {
            "min": round(min(p["duration_seconds"] for p in published), 3),
            "p05": round(float(np.percentile([p["duration_seconds"] for p in published], 5)), 3),
            "median": round(float(np.median([p["duration_seconds"] for p in published])), 3),
            "p95": round(float(np.percentile([p["duration_seconds"] for p in published], 95)), 3),
            "max": round(max(p["duration_seconds"] for p in published), 3)},
        "held_back": len(held),
        "reference_clips": {k: v["clip_id"] for k, v in refs.items()},
        "speaker_outlier_bar": {f"{a}:{b}": v for (a, b), v in sorted(spk_bar.items())},
    }
    write_json(REPORTS / "11_export.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
