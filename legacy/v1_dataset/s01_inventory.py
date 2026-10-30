"""Stage 01 - inventory.

Selects every Kurisu / Amadeus-Kurisu clip the user asked for, normalises the
transcript, hashes the source audio and classifies each row's *speech status*
(lexical speech / isolated-NVV candidate / onomatopoeia-only candidate).

Writes manifests/01_inventory.jsonl and reports/01_inventory.json.
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

import soundfile as sf

from common import (CLEAN, GAME, MANIFESTS, REPORTS, SOURCE_RULES, SPEAKERS,
                    is_nonlexical, normalize_text, read_jsonl, sha256_file,
                    text_prior, utf8_stdout, write_jsonl)

utf8_stdout()

DATASETS = {"voice_dataset_game": GAME, "voice_dataset_clean": CLEAN}


def source_group(row: dict) -> str:
    """The recording chain a clip belongs to - the unit for loudness alignment."""
    return f"{row['dataset_short']}:{row['series']}"


def main() -> None:
    rows = []
    seen_ids = collections.Counter()

    for ds_name, ds_root in DATASETS.items():
        rules = SOURCE_RULES[ds_name]
        for m in read_jsonl(ds_root / "metadata.jsonl"):
            if m["speaker_id"] not in SPEAKERS:
                continue
            top = m["rel_path"].split("/")[0]
            if not rules.get(top, False):
                continue

            src = ds_root / m["rel_path"]
            if not src.exists():
                raise FileNotFoundError(src)

            text_src = m["text"]
            text, tflags = normalize_text(text_src)
            prior, prior_rule = text_prior(text)
            nonlex = is_nonlexical(text)

            info = sf.info(str(src))
            short = "game" if ds_name.endswith("game") else "anime"
            clip_id = m["clip_id"]
            seen_ids[clip_id] += 1

            if nonlex:
                status = "isolated_nvv_candidate"
            elif prior is not None:
                status = "onomatopoeia_candidate"
            else:
                status = "speech"

            rows.append({
                "id": clip_id,
                "src_path": str(src).replace("\\", "/"),
                "dataset": ds_name,
                "dataset_short": short,
                "from_quarantine": top == "wavs_quarantine",
                "speaker_id": m["speaker_id"],
                "variant": "amadeus" if m["speaker_id"] == "amadeus_kurisu" else "kurisu",
                "series": m.get("series") or "NA",
                "scene": m.get("scene") or m.get("episode") or "NA",
                "tier_src": m["tier"],
                "flags_src": m.get("flags") or [],
                "text_src": text_src,
                "text": text,
                "text_flags": tflags,
                "text_prior": sorted(prior) if prior else None,
                "text_prior_rule": prior_rule,
                "speech_status": status,
                "dur_src": round(info.frames / info.samplerate, 4),
                "sr_src": info.samplerate,
                "channels_src": info.channels,
                "subtype_src": info.subtype,
                "peak_dbfs_src": m.get("peak_dbfs"),
                "rms_dbfs_src": m.get("rms_dbfs"),
                "sha256_src": sha256_file(Path(src)),
            })

    dup = [k for k, v in seen_ids.items() if v > 1]
    if dup:
        raise SystemExit(f"clip_id collision across datasets: {dup[:10]}")

    for r in rows:
        r["source_group"] = source_group(r)

    # Exact-duplicate source audio: the engine can reference one voice file from two
    # script points. Keep the first, drop the rest - duplicate supervision and a
    # guaranteed train/val leak otherwise.
    first_seen: dict[str, str] = {}
    dropped = []
    keep = []
    for r in rows:
        h = r["sha256_src"]
        if h in first_seen:
            r["dropped_reason"] = f"duplicate_audio_of:{first_seen[h]}"
            dropped.append(r)
        else:
            first_seen[h] = r["id"]
            keep.append(r)
    if dropped:
        write_jsonl(MANIFESTS / "01_dropped_duplicates.jsonl", dropped)
    rows = keep

    n = write_jsonl(MANIFESTS / "01_inventory.jsonl", rows)

    # ---- report
    durs = sorted(r["dur_src"] for r in rows)
    by = lambda k: collections.Counter(r[k] for r in rows)
    audio_hash = collections.Counter(r["sha256_src"] for r in rows)
    report = {
        "clips": n,
        "hours": round(sum(durs) / 3600, 3),
        "duration_s": {
            "min": durs[0], "p05": durs[int(0.05 * len(durs))],
            "median": round(statistics.median(durs), 3),
            "p95": durs[int(0.95 * len(durs))], "max": durs[-1],
        },
        "by_dataset": dict(by("dataset_short")),
        "by_speaker": dict(by("speaker_id")),
        "by_series": dict(by("series")),
        "by_source_group": dict(by("source_group")),
        "by_speech_status": dict(by("speech_status")),
        "from_quarantine": sum(r["from_quarantine"] for r in rows),
        "sample_rates": dict(by("sr_src")),
        "text_flags": dict(collections.Counter(f for r in rows for f in r["text_flags"])),
        "dropped_duplicate_audio": len(dropped),
        "distinct_texts": len({r["text"] for r in rows}),
    }
    (REPORTS / "01_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
