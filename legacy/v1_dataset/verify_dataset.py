"""Audit what actually shipped, against the guide's acceptance checklist (§37).

Reads only the exported artefacts - not the intermediate manifests - so it fails if
the export is wrong even when the pipeline thought it was right.

    python verify_dataset.py            # exits non-zero if a MUST check fails
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from common import (FAMILY_TO_QWEN, MANIFESTS, PROJECT, QWEN, REPORTS, TARGET_SR,
                    leak_key, read_jsonl, utf8_stdout)

utf8_stdout()

TAGS = {t for t in FAMILY_TO_QWEN.values() if t}
TAG_LIKE = re.compile(r"\[[^\]]{1,24}\]")

fails: list[str] = []
warns: list[str] = []


def must(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        fails.append(msg)


def should(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  warn  ") + msg)
    if not ok:
        warns.append(msg)


def main() -> int:
    splits = {}
    for name in ("train", "val", "test"):
        p = QWEN / f"{name}_raw.jsonl"
        splits[name] = list(read_jsonl(p)) if p.exists() else []
    rows = [r for v in splits.values() for r in v]
    print(f"\n== export ==  train {len(splits['train'])}  val {len(splits['val'])}  "
          f"test {len(splits['test'])}  total {len(rows)}")

    must(bool(rows), "export is non-empty")
    if not rows:
        return 1

    print("\n== schema ==")
    must(all(set(r) == {"audio", "text", "ref_audio"} for r in rows),
         "every row has exactly audio/text/ref_audio")
    must(all(r["text"].strip() for r in rows), "no empty transcript")

    print("\n== reference audio ==")
    refs = {r["ref_audio"] for r in rows}
    must(len(refs) == 1, f"one ref_audio across every row (found {len(refs)})")
    ref_path = (PROJECT / next(iter(refs))).resolve()
    must(ref_path.exists(), f"ref_audio exists: {ref_path.name}")
    if ref_path.exists():
        i = sf.info(str(ref_path))
        must(i.samplerate == TARGET_SR, f"ref_audio is {TARGET_SR} Hz (got {i.samplerate})")
        must(i.channels == 1, f"ref_audio is mono (got {i.channels})")
        should(3.0 <= i.frames / i.samplerate <= 10.0,
               f"ref_audio is 3-10 s (got {i.frames / i.samplerate:.2f} s)")
        x, _ = sf.read(str(ref_path), dtype="float32")
        should(float(np.max(np.abs(x))) < 0.99, "ref_audio is not clipped")

    print("\n== training audio ==")
    bad_sr = bad_ch = missing = clipped = 0
    peaks, durs = [], []
    for r in rows:
        p = (PROJECT / r["audio"]).resolve()
        if not p.exists():
            missing += 1
            continue
        i = sf.info(str(p))
        bad_sr += i.samplerate != TARGET_SR
        bad_ch += i.channels != 1
        durs.append(i.frames / i.samplerate)
    must(missing == 0, f"every referenced wav exists ({missing} missing)")
    must(bad_sr == 0, f"every wav is {TARGET_SR} Hz ({bad_sr} wrong)")
    must(bad_ch == 0, f"every wav is mono ({bad_ch} wrong)")

    sample = rows[:: max(1, len(rows) // 600)]
    for r in sample:
        p = (PROJECT / r["audio"]).resolve()
        if p.exists():
            x, _ = sf.read(str(p), dtype="float32")
            pk = float(np.max(np.abs(x))) if x.size else 0.0
            peaks.append(pk)
            clipped += pk >= 0.999
    must(clipped == 0, f"no clipping in a {len(sample)}-clip sample ({clipped} clipped)")
    should(bool(peaks) and max(peaks) < 0.95,
           f"peak headroom kept (max {max(peaks):.3f})" if peaks else "no sample")

    print("\n== tags ==")
    used = collections.Counter()
    unknown = collections.Counter()
    for r in rows:
        for m in TAG_LIKE.findall(r["text"]):
            (used if m in TAGS else unknown)[m] += 1
    must(not unknown, f"no unknown tag strings ({dict(unknown)})")
    must(all(t == t.lower() for t in used), "tag spelling is canonical lowercase")
    for t, n in used.most_common():
        should(n >= 25, f"{t}: {n} examples (guide §12 wants >=25 for a dedicated control)")
    print(f"  info  tags in export: {dict(used.most_common())}")

    print("\n== splits ==")
    ids = {k: {r["audio"] for r in v} for k, v in splits.items()}
    must(not (ids["train"] & ids["val"]), "train/val disjoint")
    must(not (ids["train"] & ids["test"]), "train/test disjoint")
    must(not (ids["val"] & ids["test"]), "val/test disjoint")

    master = {m["id"]: m for m in read_jsonl(MANIFESTS / "09_master.jsonl")}
    scenes = collections.defaultdict(set)
    for m in master.values():
        if m.get("split"):
            scenes[(m["source_group"], m["scene"])].add(m["split"])
    bleed = {k: sorted(v) for k, v in scenes.items() if len(v) > 1}
    must(not bleed, f"no scene spans two splits ({len(bleed)} do)")

    # Control tokens are SUPPOSED to appear on both sides - '[sigh]' in train and in
    # val is what makes the tag testable. Only real lexical content can leak.
    train_text = {leak_key(r["text"]) for r in splits["train"]}
    train_text.discard(None)
    leak = [r for k in ("val", "test") for r in splits[k]
            if leak_key(r["text"]) in train_text]
    must(not leak, f"no substantive transcript shared train<->eval ({len(leak)})")

    print("\n== duplicates ==")
    dup = [a for a, n in collections.Counter(r["audio"] for r in rows).items() if n > 1]
    must(not dup, f"no wav appears twice in the export ({len(dup)})")

    print("\n== coverage ==")
    total = len(master)
    exported = len(rows)
    held = sum(1 for m in master.values() if m["status"] != "auto_accept")
    print(f"  info  {exported}/{total} clips exported, {held} held back for review")
    should(exported / total > 0.95, f"exported share {exported / total:.1%}")
    hours = sum(durs) / 3600
    print(f"  info  {hours:.2f} h of training audio")

    print(f"\n{'=' * 60}\n{len(fails)} failure(s), {len(warns)} warning(s)")
    for f in fails:
        print("  FAIL  " + f)
    for w in warns:
        print("  warn  " + w)
    (REPORTS / "verify.json").write_text(json.dumps(
        {"failures": fails, "warnings": warns, "exported": exported,
         "hours": round(hours, 3)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
