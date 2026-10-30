"""Stage 01 - gather every Makise Kurisu / Amadeus Kurisu clip from every source.

Three upstream datasets feed this build:

``voice_dataset_game``   four MAGES./M2 games, Japanese, clip and transcript paired
                         by the games' own scenario bytecode.  Tier A/B are ordinary
                         speech; tier C is the quarantine pool -- lines whose script
                         text is only ``……``, i.e. the take is a non-verbal sound.
                         Both are taken; the quarantine is relabelled, not dropped.

``datasets/<language>``  the six-language anime build (TV, movie, OVA, specials),
                         dual-ASR transcripts, ECAPA speaker verification, and a
                         completed manual review whose bracket annotations are the
                         gold set this pipeline calibrates against.

``Cleaned_Songs/dataset``  three character songs, lyric-sheet transcripts.

``voice_dataset_clean`` (the older Japanese-only anime cut) is deliberately NOT
read: it covers the same episodes as ``datasets/japanese`` with the pre-correction
timing, including the episode 10/13 label files that were later found to contain
episode 1 dialogue.  Taking both would put two cuts of the same performance in the
corpus and leak them across the split boundary.
"""
from __future__ import annotations

import collections
import json

from kcommon import (ANIME_ROOT, ANIME_SPEAKERS, GAME_ROOT, GAME_SPEAKERS,
                     LANGUAGES, MANIFESTS, REPORTS, SONG_ROOT, VARIANT,
                     VOICE_ACTOR, is_nonlexical, lexical_units, normalize_text,
                     read_jsonl, split_human_tags, utf8_stdout, write_json,
                     write_jsonl)

utf8_stdout()

GAME_SERIES_TITLE = {
    "SG": "STEINS;GATE",
    "SG0": "STEINS;GATE 0",
    "PHENO": "STEINS;GATE: Linear Bounded Phenogram",
    "SGRE": "STEINS;GATE RE:BOOT",
}


def _base(row_id, medium, lang, speaker_id):
    return {
        "id": row_id,
        "medium": medium,
        "language": lang,
        "voice_id": lang,
        "voice_actor": VOICE_ACTOR[lang],
        "speaker_id": "kurisu",
        "character_variant": VARIANT[speaker_id],
        "upstream_speaker_id": speaker_id,
    }


def from_game():
    """Tier A/B speech plus the tier C quarantine, Japanese."""
    for r in read_jsonl(GAME_ROOT / "metadata.jsonl"):
        sid = r.get("speaker_id")
        if sid not in GAME_SPEAKERS:
            continue
        tier = r.get("tier")
        quarantined = tier == "C"
        rel = r["rel_path"]
        src = GAME_ROOT / rel
        if not src.exists():
            continue
        text_src = r.get("text") or ""
        text, tflags = normalize_text(text_src, "ja")
        row = _base(f"game_ja_{r['clip_id']}", "game", "ja", sid)
        row.update({
            "src_path": str(src),
            "upstream_id": r["clip_id"],
            "series": r.get("series"),
            "series_title": GAME_SERIES_TITLE.get(r.get("series"), r.get("series")),
            "source_group": f"game:{r.get('series')}",
            "scene": r.get("scene"),
            "scene_order": r.get("order"),
            "text_src": text_src,
            "text": text,
            "text_flags": tflags,
            "human_events": [],
            "human_notes": {"conditions": [], "pauses": [], "contamination": [],
                            "emotion": [], "unmapped": [], "raw": []},
            "human_reviewed": False,
            "from_quarantine": quarantined,
            "upstream_tier": tier,
            "upstream_flags": r.get("flags") or [],
            "transcript_authority": "game_script_bytecode",
            "src_dur": r.get("dur"),
            "src_sr": r.get("src_sr"),
            "upstream_scores": {
                "chars_per_sec": r.get("chars_per_sec"),
                "speech_ratio": r.get("speech_ratio"),
                "spk_score": r.get("spk_score"),
                "spk_margin": r.get("spk_margin"),
                "spk_agrees": r.get("spk_agrees"),
                "label_source": r.get("label_source"),
                "label_confidence": r.get("label_confidence"),
                "rolloff_hz": r.get("rolloff_hz"),
            },
            "voice_stem": r.get("voice_stem"),
            "source_file": r.get("source"),
            "style": "speech",
        })
        yield row


def from_anime():
    """Six languages of the anime build, plus the human bracket annotations."""
    for lang_dir, lang in LANGUAGES.items():
        root = ANIME_ROOT / lang_dir
        meta = root / "metadata.jsonl"
        if not meta.exists():
            continue
        for r in read_jsonl(meta):
            sid = r.get("speaker")
            if sid not in ANIME_SPEAKERS:
                continue
            src = root / r["audio"].replace("/", "\\")
            if not src.exists():
                continue
            raw_text = r.get("transcript") or ""
            # Normalise first, then pull the brackets out: the reviewers' offsets
            # are positions in the *normalised* line, so no rule that changes a
            # character count can slide a tag off the word it was written on.
            # normalize_text() does not touch square brackets.
            norm_tagged, tflags = normalize_text(raw_text, lang)
            text, events, notes = split_human_tags(norm_tagged)
            reviewed = r.get("curation_tier") == "manual" or bool(notes["raw"])
            row = _base(f"anime_{lang}_{r['clip_id']}", "anime", lang, sid)
            row.update({
                "src_path": str(src),
                "upstream_id": r["clip_id"],
                "series": r.get("series"),
                "series_title": r.get("series"),
                "source_group": f"anime:{lang}",
                "scene": r.get("source_id"),
                "scene_order": None,
                "reference_start": r.get("reference_start"),
                "reference_end": r.get("reference_end"),
                "src_start": r.get("start"),
                "src_end": r.get("end"),
                "text_src": raw_text,
                "text": text,
                "text_flags": tflags,
                "human_events": [{"offset": o, "tag": t} for o, t in events],
                "human_notes": notes,
                "human_reviewed": reviewed,
                "from_quarantine": False,
                "upstream_tier": r.get("curation_tier"),
                "upstream_flags": r.get("flags") or [],
                "transcript_authority": r.get("transcript_source"),
                "src_dur": r.get("duration"),
                "src_sr": 24000,
                "upstream_scores": {
                    "speaker_score": r.get("speaker_score"),
                    "speaker_margin": r.get("speaker_margin"),
                    "speaker_method": r.get("speaker_method"),
                    "asr_agreement": r.get("asr_agreement"),
                    "whisper_mean_word_probability":
                        r.get("whisper_mean_word_probability"),
                    "gain_db": r.get("gain_db"),
                    "output_active_rms_dbfs": r.get("output_active_rms_dbfs"),
                    "peak_dbfs": r.get("peak_dbfs"),
                    "edge_rms_dbfs": r.get("edge_rms_dbfs"),
                    "first_word_start": r.get("first_word_start"),
                    "last_word_end": r.get("last_word_end"),
                },
                "source_file": r.get("source_video"),
                "style": "speech",
            })
            yield row


def from_songs():
    meta = SONG_ROOT / "metadata.jsonl"
    if not meta.exists():
        return
    for r in read_jsonl(meta):
        if r.get("speaker") != "makise_kurisu":
            continue
        src = SONG_ROOT / r["audio"].replace("/", "\\")
        if not src.exists():
            continue
        norm_tagged, tflags = normalize_text(r.get("transcript") or "", "ja")
        text, events, notes = split_human_tags(norm_tagged)
        row = _base(f"song_ja_{r['clip_id']}", "song", "ja", "makise_kurisu")
        row.update({
            "src_path": str(src),
            "upstream_id": r["clip_id"],
            "series": r.get("source_title_en"),
            "series_title": r.get("source_title"),
            "source_group": f"song:ja:{r.get('source_id')}",
            "scene": r.get("source_id"),
            "scene_order": r.get("stanza"),
            "text_src": r.get("transcript") or "",
            "text": text,
            "text_flags": tflags,
            "human_events": [{"offset": o, "tag": t} for o, t in events],
            "human_notes": notes,
            "human_reviewed": True,
            "from_quarantine": False,
            "upstream_tier": r.get("curation_tier"),
            "upstream_flags": [],
            "transcript_authority": r.get("transcript_source"),
            "src_dur": r.get("duration"),
            "src_sr": 24000,
            "upstream_scores": {
                "asr_agreement": r.get("asr_agreement"),
                "gain_db": r.get("gain_db"),
                "output_active_rms_dbfs": r.get("output_active_rms_dbfs"),
                "peak_dbfs": r.get("peak_dbfs"),
                "whisper_mean_word_probability":
                    r.get("whisper_mean_word_probability"),
            },
            "source_file": r.get("source_audio"),
            "style": "singing",
        })
        yield row


def main() -> None:
    rows = list(from_game()) + list(from_anime()) + list(from_songs())

    seen = set()
    for r in rows:
        assert r["id"] not in seen, f"duplicate id {r['id']}"
        seen.add(r["id"])

        lang = r["language"]
        # Isolated non-verbal candidate: the script line carries no lexical content.
        # For the game quarantine this is the normal case and the reason the clip
        # was set aside in the first place.
        r["speech_status"] = ("nonlexical_candidate" if is_nonlexical(r["text"])
                              else "speech")
        r["lexical_units"] = lexical_units(r["text"], lang)

    write_jsonl(MANIFESTS / "01_inventory.jsonl", rows)

    by = collections.Counter((r["medium"], r["language"]) for r in rows)
    report = {
        "clips": len(rows),
        "by_medium_language": {f"{m}:{l}": n for (m, l), n in sorted(by.items())},
        "by_variant": dict(collections.Counter(r["character_variant"] for r in rows)),
        "by_source_group": dict(collections.Counter(r["source_group"] for r in rows)),
        "from_quarantine": sum(1 for r in rows if r["from_quarantine"]),
        "human_reviewed": sum(1 for r in rows if r["human_reviewed"]),
        "nonlexical_candidates": sum(1 for r in rows
                                     if r["speech_status"] == "nonlexical_candidate"),
        "human_event_tags": dict(collections.Counter(
            e["tag"] for r in rows for e in r["human_events"]).most_common()),
        "human_conditions": dict(collections.Counter(
            c for r in rows for c in r["human_notes"]["conditions"]).most_common()),
        "human_pauses": dict(collections.Counter(
            c for r in rows for c in r["human_notes"]["pauses"]).most_common()),
        "human_contamination": dict(collections.Counter(
            c for r in rows for c in r["human_notes"]["contamination"]).most_common()),
        "human_emotion": dict(collections.Counter(
            c for r in rows for c in r["human_notes"]["emotion"]).most_common()),
        "unmapped_human_tags": dict(collections.Counter(
            c for r in rows for c in r["human_notes"]["unmapped"]).most_common()),
        "text_flags": dict(collections.Counter(
            f for r in rows for f in r["text_flags"]).most_common()),
        "excluded_sources": {
            "voice_dataset_clean": "superseded by datasets/japanese (same episodes, "
                                   "pre-correction timing, ep10/13 label defect)",
        },
    }
    write_json(REPORTS / "01_inventory.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
