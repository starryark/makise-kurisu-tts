"""Codec-tokenise every v3 source once; mix_v3.py composes the per-experiment sets.

Sources (each optional except the first):
  build_v3/kept.jsonl          Kurisu JA takes (speakers kurisu / amadeus / phone)
  augment/augment_rows.jsonl   phone-EQ and background-SFX augmented takes
  replay/replay_rows.jsonl     native Mandarin / English takes from other speakers
  pseudo/pseudo_rows.jsonl     Kurisu-voiced native Mandarin from the Base model (E4)
Writes qwen_v3/<source>_<split>_with_codes.jsonl, keeping every field of the source
row and adding audio_codes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
SOURCES = {"ja": WORK / "build_v3" / "kept.jsonl",
           "aug": WORK / "augment" / "augment_rows.jsonl",
           "replay": WORK / "replay" / "replay_rows.jsonl",
           "pseudo": WORK / "pseudo" / "pseudo_rows.jsonl"}
SPLIT = {"train": "train", "validation": "val", "val": "val", "test": "test"}
BATCH = 48


def signature(items: list[dict]) -> str:
    """Rows + audio identity: any label, text or audio change re-tokenises the file."""
    h = hashlib.sha1()
    for r in sorted(items, key=lambda r: r["audio"]):
        st = os.stat(r["audio"])
        meta = {k: v for k, v in r.items() if k != "dur"}
        h.update(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode())
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def main() -> None:
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
    out_dir = WORK / "qwen_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"[codes] {name}: {path} missing, skipped")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        for r in rows:
            if "dur" not in r:
                info = sf.info(r["audio"])
                r["dur"] = info.frames / info.samplerate
        by_split: dict[str, list] = {}
        for r in rows:
            by_split.setdefault(SPLIT[r.get("split", "train")], []).append(r)
        for split, items in by_split.items():
            out = out_dir / f"{name}_{split}_with_codes.jsonl"
            sig_path = out.with_suffix(".sig")
            sig = signature(items)
            if out.exists() and sig_path.exists() and sig_path.read_text() == sig:
                print(f"[codes] {out.name}: up to date ({len(items)} rows)")
                continue
            items.sort(key=lambda r: r["dur"])
            lines = []
            for i in range(0, len(items), BATCH):
                chunk = items[i:i + BATCH]
                enc = tok.encode([r["audio"] for r in chunk])
                for r, codes in zip(chunk, enc.audio_codes):
                    codes = codes.cpu().tolist()
                    if abs(len(codes) - r["dur"] * 12.5) > 3:
                        raise SystemExit(f"{r.get('id')}: {len(codes)} frames for {r['dur']}s")
                    lines.append(json.dumps({**r, "audio_codes": codes}, ensure_ascii=False))
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            sig_path.write_text(sig)
            print(f"[codes] {out.name}: {len(lines)} rows")


if __name__ == "__main__":
    main()
