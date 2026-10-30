"""Stage FT-1 - add ``audio_codes`` to the Qwen JSONL exports.

Same job as the official ``finetuning/prepare_data.py``: run the 12 Hz speech
tokenizer over every target wav and write the discrete codes into each row. Four
differences, all of them things a 10 000-clip run on one desktop card needs:

* **Resumable.** Codes are appended to a ``.partial`` file as they are produced, so a
  crash at clip 8 000 costs the last batch, not the last hour. The final file is
  rewritten in the raw manifest's order once everything is present.
* **Length-bucketed batches.** The official script encodes in arbitrary batches of 32,
  which pads a 0.8 s clip out to the length of an 8.9 s one inside the same forward
  pass. Grouping by duration keeps padding - and any padding artefact at the tail of
  the codes - to a minimum.
* **OOM bisection.** A batch that does not fit is halved rather than killing the run.
* **It refuses bad input rather than encoding it.** Sample rate, channel count,
  duration and a single shared ``ref_audio`` are all checked here, because every one
  of them is silent at training time and obvious afterwards (guide s23, s37).

Writes ``qwen/<split>_with_codes.jsonl`` and ``reports/ft_01_prepare.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import ft_config as C

C.utf8_stdout()

MIN_DUR = 0.30      # below this the codec has almost nothing to quantise
MAX_DUR = 30.0      # above this a single sample dominates a batch's memory


def _resolve(rel: str) -> Path:
    """JSONL paths are written relative to the dataset root, not to the cwd."""
    p = Path(rel)
    return p if p.is_absolute() else (C.PROJECT / p).resolve()


def audit(rows: list[dict], split: str) -> tuple[list[dict], list[dict]]:
    """Split rows into (usable, rejected-with-reason) and validate the reference."""
    good: list[dict] = []
    bad: list[dict] = []
    refs = set()
    for row in rows:
        refs.add(row.get("ref_audio"))
        reason = None
        path = _resolve(row["audio"])
        if not row.get("text", "").strip():
            reason = "empty text"
        elif not path.exists():
            reason = "missing audio"
        else:
            info = sf.info(str(path))
            if info.samplerate != C.TARGET_SR:
                reason = f"sample rate {info.samplerate} != {C.TARGET_SR}"
            elif info.channels != 1:
                reason = f"{info.channels} channels, expected mono"
            elif not MIN_DUR <= info.duration <= MAX_DUR:
                reason = f"duration {info.duration:.2f}s outside [{MIN_DUR}, {MAX_DUR}]"
            else:
                good.append(dict(row, _dur=info.duration))
        if reason:
            bad.append(dict(row, reject=reason))

    if len(refs) > 1:
        raise SystemExit(
            f"{split}: {len(refs)} distinct ref_audio values. The official recipe and "
            f"the speaker-embedding write-back both assume one reference on every row "
            f"(guide s16): {sorted(refs)[:4]}")
    ref = _resolve(refs.pop())
    if not ref.exists():
        raise SystemExit(f"{split}: ref_audio {ref} does not exist")
    info = sf.info(str(ref))
    if info.samplerate != C.TARGET_SR or info.channels != 1:
        raise SystemExit(f"ref_audio must be {C.TARGET_SR} Hz mono, got "
                         f"{info.samplerate} Hz / {info.channels}ch: {ref}")
    return good, bad


def load_done(paths: list[Path]) -> dict[str, dict]:
    """Rows already carrying codes, keyed by audio path. Tolerates a torn last line."""
    done: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"   [resume] dropping torn line at the end of {path.name}")
                    continue
                if row.get("audio_codes"):
                    done[row["audio"]] = row
    return done


def encode_split(torch, tokenizer, split: str, batch: int, force: bool,
                 limit: int | None) -> dict:
    raw_path = C.SPLITS[split]
    if not raw_path.exists():
        raise SystemExit(f"no such split file: {raw_path}")
    rows = list(C.read_jsonl(raw_path))
    if limit:
        rows = rows[:limit]
    good, bad = audit(rows, split)
    print(f"[{split}] {len(rows)} rows, {len(bad)} rejected")
    for row in bad[:10]:
        print(f"   reject {row['audio']}: {row['reject']}")

    out_path = C.coded_path(split)
    partial = out_path.with_name(out_path.name + ".partial")
    if force:
        for p in (out_path, partial):
            p.unlink(missing_ok=True)
    done = load_done([out_path, partial])
    todo = [r for r in good if r["audio"] not in done]
    if done:
        print(f"[{split}] {len(done)} already encoded, {len(todo)} to go")

    # Bucket by duration so a batch pads as little as possible.
    todo.sort(key=lambda r: r["_dur"])
    t0 = time.time()
    n_done = 0
    with open(partial, "a", encoding="utf-8") as sink:
        i = 0
        while i < len(todo):
            size = min(batch, len(todo) - i)
            while True:
                chunk = todo[i:i + size]
                try:
                    enc = tokenizer.encode([str(_resolve(r["audio"])) for r in chunk])
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise
                    size = max(1, size // 2)
                    print(f"   [gpu] OOM -> batch {size}")
            for codes, row in zip(enc.audio_codes, chunk):
                out = {k: v for k, v in row.items() if not k.startswith("_")}
                out["audio_codes"] = codes.cpu().tolist()
                done[row["audio"]] = out
                sink.write(json.dumps(out, ensure_ascii=False) + "\n")
            sink.flush()
            i += size
            n_done += size
            if n_done % (batch * 20) < size:
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - n_done) / max(rate, 1e-6)
                print(f"   {n_done}/{len(todo)}  {rate:.1f} clip/s  eta {C.fmt_hms(eta)}"
                      f"  (vram {C.vram_used()[0]:.1f} GiB)", flush=True)

    # Rewrite in the raw manifest's order, then drop the crash-safety file.
    ordered = [done[r["audio"]] for r in good if r["audio"] in done]
    C.write_jsonl(out_path, ordered)
    partial.unlink(missing_ok=True)

    frames = [len(r["audio_codes"]) for r in ordered]
    durs = [r["_dur"] for r in good if r["audio"] in done]
    fps = [f / d for f, d in zip(frames, durs)]
    odd = [r["audio"] for r, f in zip(ordered, fps) if not 11.0 <= f <= 14.0]
    print(f"[{split}] wrote {len(ordered)} rows -> {out_path.name}  "
          f"({sum(durs) / 3600:.2f} h, {sum(frames)} code frames, "
          f"{statistics.median(fps):.2f} frames/s)")
    if odd:
        print(f"   [warn] {len(odd)} clips are not ~{C.CODEC_FPS} frames/s: {odd[:3]}")
    return {
        "rows_in": len(rows),
        "rows_out": len(ordered),
        "rejected": [{"audio": r["audio"], "reason": r["reject"]} for r in bad],
        "hours": round(sum(durs) / 3600, 4),
        "duration_s": {
            "median": round(statistics.median(durs), 3),
            "p05": round(float(np.percentile(durs, 5)), 3),
            "p95": round(float(np.percentile(durs, 95)), 3),
            "max": round(max(durs), 3),
        },
        "code_frames": {
            "total": sum(frames),
            "median": int(statistics.median(frames)),
            "p95": int(np.percentile(frames, 95)),
            "max": max(frames),
        },
        "frames_per_second_median": round(statistics.median(fps), 3),
        "frames_per_second_outliers": odd[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Add audio_codes to the Qwen JSONL splits.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=list(C.SPLITS))
    ap.add_argument("--batch", type=int, default=16,
                    help="clips per encoder forward pass; halved automatically on OOM")
    ap.add_argument("--tokenizer_model_path", default=C.TOKENIZER_12HZ)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram_fraction", type=float, default=0.92)
    ap.add_argument("--force", action="store_true", help="ignore existing output")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke test)")
    args = ap.parse_args()

    C.vram_guard(args.vram_fraction)
    import torch  # imported after the VRAM cap is installed
    from qwen_tts import Qwen3TTSTokenizer

    print(f"[model] {args.tokenizer_model_path} -> {args.device}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path,
                                                  device_map=args.device)
    report = {"tokenizer": args.tokenizer_model_path, "batch": args.batch, "splits": {}}
    for split in args.splits:
        report["splits"][split] = encode_split(torch, tokenizer, split, args.batch,
                                               args.force, args.limit)
    C.write_report("ft_01_prepare.json", report)


if __name__ == "__main__":
    main()
