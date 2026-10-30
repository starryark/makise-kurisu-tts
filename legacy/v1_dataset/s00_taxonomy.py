"""Stage 00 - dump the taxonomy held in common.py to versionable YAML.

common.py is the single source of truth; these files exist so the mapping can be
diffed, reviewed and shipped with the dataset (supplement s6: "always version-control
this mapping").
"""
from __future__ import annotations

import collections

from common import (EASY_FAMILIES, FAMILY_TO_QWEN, FINE_TO_FAMILY, HARD_FAMILIES,
                    ONOMATOPOEIA_PRIOR, QWEN_TAGS, TAXONOMY, utf8_stdout)

utf8_stdout()


def main() -> None:
    fam_to_fine = collections.defaultdict(list)
    for fine, fam in FINE_TO_FAMILY.items():
        fam_to_fine[fam].append(fine)

    lines = ["# laion/vocalburst-classifier-single fine class -> canonical family.",
             "# Generated from pipeline/common.py - edit there, not here.",
             "version: 1", "family_map:"]
    for fam in sorted(fam_to_fine):
        lines.append(f"  {fam}:")
        for fine in sorted(fam_to_fine[fam]):
            lines.append(f"    - {fine!r}")
    (TAXONOMY / "vocalburst_to_family.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Canonical family -> the string that reaches the Qwen transcript.",
             "# 'null' means the family is recognised and stored in metadata but is",
             "# deliberately NOT exposed as a control token in v1.",
             "# Generated from pipeline/common.py - edit there, not here.",
             "version: 1",
             "qwen_tag_inventory:"]
    for t in QWEN_TAGS:
        lines.append(f"  - {t!r}")
    lines.append("family_to_qwen:")
    for fam in sorted(FAMILY_TO_QWEN):
        v = FAMILY_TO_QWEN[fam]
        lines.append(f"  {fam}: {repr(v) if v else 'null'}")
    lines.append("review_policy:")
    lines.append("  always_manual_review:   # semantically overlapping, supplement s16")
    for f in sorted(HARD_FAMILIES):
        lines.append(f"    - {f}")
    lines.append("  may_auto_accept_when_calibrated:")
    for f in sorted(EASY_FAMILIES):
        lines.append(f"    - {f}")
    (TAXONOMY / "family_to_qwen.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    lines = ["# Japanese onomatopoeia priors.",
             "# These are a PRIOR only. A written vocalisation never becomes a tag on",
             "# its own - the audio model has to agree first (supplement s3).",
             "# Generated from pipeline/common.py - edit there, not here.",
             "version: 1", "priors:"]
    for pat, fams, name in ONOMATOPOEIA_PRIOR:
        lines.append(f"  - name: {name}")
        lines.append(f"    expected_families: [{', '.join(sorted(fams))}]")
        lines.append(f"    pattern: {pat!r}")
    (TAXONOMY / "onomatopoeia_priors.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    for p in sorted(TAXONOMY.glob("*.yaml")):
        print("wrote", p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
