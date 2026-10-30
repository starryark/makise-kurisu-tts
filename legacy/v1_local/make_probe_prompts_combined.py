"""Build the ft03_eval probe prompt set from the new combined-build test split.

Mirrors the old s10_export.py probe recipe (guide s26): tagged prompts, the
same text untagged, negative controls, and synthetic tag-placement probes for
every shipped tag. Sources rows from qwen/test_raw.jsonl (the lane's test
split of the combined build, transcription targets), so every prompt string is
in-distribution for the r16 run.

Output: qwen/probe_prompts_combined.jsonl
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

QWEN = Path(__file__).resolve().parents[1] / "qwen"   # kurisu_qwen3tts/qwen in the original layout
TAG_RE = re.compile(r"\[[a-z_]+\]")


def strip_first_tag(text: str, tag: str) -> str:
    return " ".join(text.replace(tag, " ").split()) or "……"


def main() -> int:
    rows = [json.loads(line) for line in (QWEN / "test_raw.jsonl").open(encoding="utf-8")]
    tagged = [r for r in rows if TAG_RE.search(r["text"])]
    untagged = [r for r in rows if not TAG_RE.search(r["text"])]

    probe: list[dict] = []
    for r in tagged[:12]:
        tag = TAG_RE.search(r["text"]).group(0)
        probe.append({"kind": "tagged", "text": r["text"], "from": r["audio"]})
        probe.append({"kind": "same_untagged", "text": strip_first_tag(r["text"], tag),
                      "from": r["audio"]})
    negatives = [r for r in untagged if 8 <= len(r["text"]) <= 40]
    for r in negatives[:4]:
        probe.append({"kind": "negative_control", "text": r["text"], "from": r["audio"]})

    for tag in sorted({t for r in tagged for t in TAG_RE.findall(r["text"])}):
        probe.append({"kind": "isolated_tag", "text": tag, "from": None})
        probe.append({"kind": "tag_initial", "text": f"{tag} そう、そういうこと。", "from": None})
        probe.append({"kind": "tag_medial", "text": f"それは違う。{tag} もう一度説明するわ。", "from": None})

    # Deduplicate exact (kind, text) pairs while keeping order.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for p in probe:
        key = (p["kind"], p["text"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    out = QWEN / "probe_prompts_combined.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    kinds = OrderedDict()
    for p in unique:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"wrote {len(unique)} prompts -> {out.name}  kinds={dict(kinds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
