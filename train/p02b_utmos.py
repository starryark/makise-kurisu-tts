"""Batched UTMOSv2 over the shard directories (one DataLoader per directory).

Per-clip `predict(input_path=...)` re-enters the whole pipeline for every file;
`predict(input_dir=...)` batches. Merges `utmos` into WORK/src/asr.jsonl.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))


def main() -> None:
    import utmosv2

    out_path = WORK / "src" / "asr.jsonl"
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l]
    by_id = {r["id"]: r for r in rows}
    model = utmosv2.create_model(pretrained=True)
    dirs = sorted({p.parent for p in (WORK / "src" / "hf" / "data").rglob("*.wav")})
    for d in dirs:
        res = model.predict(input_dir=str(d), batch_size=64, num_workers=16, verbose=False)
        for item in res:
            rid = Path(item["file_path"]).stem
            if rid in by_id:
                by_id[rid]["utmos"] = round(float(item["predicted_mos"]), 3)
        print(f"[utmos-batch] {d.relative_to(WORK)}: {len(res)}", flush=True)
    tmp = out_path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[utmos-batch] scored {sum('utmos' in r for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
