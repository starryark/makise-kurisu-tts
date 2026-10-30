"""Index the Japanese (Asami Imai) subset of starrydark/Kurisu_Voice on the pod.

The wavs arrive as a tar stream of the local, published `hf_dataset_upload/` tree
(`data/<split>/metadata.jsonl` + `data/<split>/<shard>/*_ja_*.wav`): Hub per-file
downloads crawl at ~1 file/s for 10k small files, and the Hub's Parquet conversion
of this dataset carries metadata only. Every wav is checked against the dataset's
published sha256, so what trains is byte-identical to what is on the Hub.

Writes WORK/src/ja_meta.jsonl: metadata + absolute `audio` path + the dataset split.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
EXPECTED_JA = 10039


def main() -> None:
    root = WORK / "src" / "hf"
    rows = []
    for split in ("train", "validation", "test"):
        for line in (root / "data" / split / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["language"] != "ja":
                continue
            path = root / "data" / split / r["file_name"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
                raise SystemExit(f"sha256 mismatch: {path}")
            r["id"] = path.stem
            r["audio"] = str(path)
            r["split"] = split
            rows.append(r)
    if len(rows) != EXPECTED_JA:
        raise SystemExit(f"expected {EXPECTED_JA} ja rows, got {len(rows)}")
    out = WORK / "src" / "ja_meta.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"[fetch] {len(rows)} ja rows, all sha256-verified -> {out}")


if __name__ == "__main__":
    main()
