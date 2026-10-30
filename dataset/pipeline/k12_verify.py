"""Stage 12 - verify the published repository before anything is uploaded.

Checks the files that will actually be pushed, not the manifests that produced
them: every declared clip exists, hashes to what the metadata says, decodes as
24 kHz mono PCM-16 for the declared duration, and carries a transcript whose tags
all come from the published inventory and whose ``text`` is exactly that
transcript with the tags removed.  Then the dataset-level invariants: no clip in
two splits, no substantive line on both sides of a split boundary, no parallel
group straddling one, every language present in every split, and no tag shipped
with too few examples to be steerable.

Exits non-zero on any failure so it can gate an upload.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from kcommon import (ALL_TAGS, HF_OUT, MIN_TAG_EXAMPLES, PEAK_CEILING_DBFS,
                     REPORTS, TAG_RE, TARGET_SR, leak_key, strip_tags,
                     utf8_stdout, write_json)

utf8_stdout()

SPLITS = ("train", "validation", "test")
AUDIO_SAMPLE = 400          # clips per split fully decoded and hashed
HUB_FILES_PER_DIR = 10000   # the Hub rejects a commit that exceeds this
ARROW_BLOCK_BYTES = 320 << 10   # datasets reads metadata at chunksize // 32


def _uninformative(v) -> bool:
    return v is None or v == [] or v == ""


def main() -> int:
    failures, warnings = [], []
    rows = {}
    for s in SPLITS:
        p = HF_OUT / "data" / s / "metadata.jsonl"
        if not p.exists():
            failures.append(f"missing {p}")
            continue
        rows[s] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    if failures:
        print("\n".join(failures))
        return 1

    seen = {}
    for s, rs in rows.items():
        for r in rs:
            if r["file_name"] in seen:
                failures.append(f"{r['file_name']} appears in {seen[r['file_name']]} and {s}")
            seen[r["file_name"]] = s

    # ---- per-row checks ------------------------------------------------------
    rng = np.random.default_rng(20260905)
    for s, rs in rows.items():
        idx = set(rng.choice(len(rs), size=min(AUDIO_SAMPLE, len(rs)),
                             replace=False).tolist())
        for i, r in enumerate(rs):
            f = HF_OUT / "data" / s / r["file_name"]
            if not f.exists():
                failures.append(f"missing audio {s}/{r['file_name']}")
                continue
            if not r["transcription"].strip():
                failures.append(f"empty transcription {r['file_name']}")
            bad = [t for t in TAG_RE.findall(r["transcription"]) if t not in ALL_TAGS]
            if bad:
                failures.append(f"unknown tag(s) {bad} in {r['file_name']}")
            if strip_tags(r["transcription"]) != r["text"]:
                failures.append(f"text != untagged transcription in {r['file_name']}")
            if r["sample_rate"] != TARGET_SR:
                failures.append(f"declared sample rate {r['sample_rate']} in {r['file_name']}")
            if r["peak_dbfs"] is not None and r["peak_dbfs"] > PEAK_CEILING_DBFS + 0.05:
                failures.append(f"peak {r['peak_dbfs']} dBFS over ceiling in {r['file_name']}")
            if r["clipped_samples"]:
                failures.append(f"{r['clipped_samples']} clipped samples in {r['file_name']}")
            if i in idx:
                info = sf.info(str(f))
                if (info.samplerate, info.channels, info.subtype) != (TARGET_SR, 1, "PCM_16"):
                    failures.append(f"{r['file_name']} is {info.samplerate}Hz "
                                    f"{info.channels}ch {info.subtype}")
                d = info.frames / info.samplerate
                if abs(d - r["duration_seconds"]) > 0.01:
                    failures.append(f"{r['file_name']} duration {d:.3f} != "
                                    f"{r['duration_seconds']}")
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        h.update(b)
                if h.hexdigest() != r["sha256"]:
                    failures.append(f"sha256 mismatch {r['file_name']}")

    # ---- reference clips -----------------------------------------------------
    refp = HF_OUT / "reference" / "reference.json"
    refs = {}
    if refp.exists():
        refs = json.loads(refp.read_text(encoding="utf-8"))
        for voice, r in refs.items():
            f = HF_OUT / r["file"]
            if not f.exists():
                failures.append(f"missing reference audio {r['file']}")
                continue
            info = sf.info(str(f))
            if (info.samplerate, info.channels, info.subtype) != (TARGET_SR, 1, "PCM_16"):
                failures.append(f"reference {voice} is {info.samplerate}Hz "
                                f"{info.channels}ch {info.subtype}")
            if r["speaker_margin"] is not None and r["speaker_margin"] <= 0:
                failures.append(f"reference {voice} does not out-score its "
                                f"closest rival ({r['closest_other_character']})")
        missing = {r["language"] for rs in rows.values() for r in rs} - set(refs)
        if missing:
            warnings.append(f"no reference clip for {sorted(missing)}")
    else:
        warnings.append("no reference/reference.json")

    # ---- orphan files --------------------------------------------------------
    for s in SPLITS:
        root = HF_OUT / "data" / s
        on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*.wav")}
        declared = {r["file_name"] for r in rows[s]}
        for extra in sorted(on_disk - declared):
            failures.append(f"undeclared file {s}/{extra}")

    # ---- metadata schema inferability ---------------------------------------
    # `datasets` infers each split's metadata schema from the first Arrow block
    # of metadata.jsonl and refuses the dataset if the splits disagree.  An
    # empty list or a null types as null, so a column that happens to be empty
    # throughout one split's opening rows breaks load_dataset for everyone.
    # Reproduce that here in plain Python: within the prefix Arrow actually
    # looks at, every column that carries data anywhere in the split must carry
    # data at least once, and the splits must agree on which columns those are.
    typed_by_split = {}
    for s in SPLITS:
        prefix, used = [], 0
        for r in rows[s]:
            if used >= ARROW_BLOCK_BYTES:
                break
            prefix.append(r)
            used += len(json.dumps(r, ensure_ascii=False).encode("utf-8")) + 1
        populated = {c for r in rows[s] for c in r if not _uninformative(r[c])}
        typed = {c for r in prefix for c in r if not _uninformative(r[c])}
        typed_by_split[s] = typed
        for c in sorted(populated - typed):
            failures.append(
                f"column {c!r} carries data in {s} but is empty throughout the "
                f"first {len(prefix)} rows, so Arrow will type it null")
    if len({frozenset(v) for v in typed_by_split.values()}) > 1:
        for a, b in itertools.combinations(SPLITS, 2):
            diff = typed_by_split[a] ^ typed_by_split[b]
            if diff:
                failures.append(f"{a} and {b} would infer different metadata "
                                f"schemas; columns differing: {sorted(diff)}")

    # ---- Hub directory limit -------------------------------------------------
    # A commit is rejected outright if any single directory holds more than
    # 10,000 files, and the failure only shows up at push time after the whole
    # upload.  Catching it here costs nothing.
    per_dir = collections.Counter()
    for p in HF_OUT.rglob("*"):
        if p.is_file():
            per_dir[p.parent] += 1
    for d, n in sorted(per_dir.items()):
        if n > HUB_FILES_PER_DIR:
            failures.append(f"{n} files in {d.relative_to(HF_OUT).as_posix()} "
                            f"(Hub limit is {HUB_FILES_PER_DIR} per directory)")

    # ---- dataset-level invariants -------------------------------------------
    def keys(s):
        return {k for k in (leak_key(r["text"], r["language"]) for r in rows[s]) if k}
    # All three pairs, not just train against each evaluation split.  A line in
    # both validation and test is not train/eval leakage, but it does mean the
    # two evaluation sets are not independent -- and checking only train hid a
    # repair loop in stage 10 that ping-ponged groups between them indefinitely.
    ks = {s: keys(s) for s in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        n = len(ks[a] & ks[b])
        if n:
            failures.append(f"{n} substantive lines shared between {a} and {b}")

    pg = collections.defaultdict(set)
    for s, rs in rows.items():
        for r in rs:
            if r["parallel_group"]:
                pg[r["parallel_group"]].add(s)
    broken = [g for g, v in pg.items() if len(v) > 1]
    if broken:
        failures.append(f"{len(broken)} parallel groups straddle a split boundary")

    langs = {r["language"] for rs in rows.values() for r in rs}
    for s in SPLITS:
        missing = langs - {r["language"] for r in rows[s]}
        if missing:
            warnings.append(f"{s} split is missing language(s) {sorted(missing)}")

    tagcount = collections.Counter(t for rs in rows.values() for r in rs for t in r["tags"])
    for t, n in tagcount.items():
        if n < MIN_TAG_EXAMPLES:
            failures.append(f"tag {t} shipped with only {n} examples")

    all_rows = [r for rs in rows.values() for r in rs]
    report = {
        "clips": len(all_rows),
        "splits": {s: len(rows[s]) for s in SPLITS},
        "hours": round(sum(r["duration_seconds"] for r in all_rows) / 3600.0, 4),
        "audio_fully_checked": min(AUDIO_SAMPLE, min(len(rows[s]) for s in SPLITS)) * 3,
        "tags": dict(tagcount.most_common()),
        "languages": sorted(langs),
        "reference_clips": {k: v["clip_id"] for k, v in refs.items()},
        "failures": failures,
        "warnings": warnings,
    }
    write_json(REPORTS / "12_verify.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        return 1
    print(f"\nOK: {len(all_rows)} clips, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
