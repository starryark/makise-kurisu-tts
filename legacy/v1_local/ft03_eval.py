"""Stage FT-3 - synthesize a fixed listening suite from one or more checkpoints.

Loss cannot reveal speaker drift, rate drift, or whether the custom ``[sigh]`` and
``[gasp]`` conventions work. This script makes checkpoint comparison repeatable and
writes simple signal/duration diagnostics beside the WAV files. Listening remains
the actual model-selection step.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

import ft_config as C

C.utf8_stdout()


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_checkpoints(run: Path, spec: str) -> list[Path]:
    candidates = sorted(
        (p for p in run.glob("checkpoint-epoch-*") if (p / "model.safetensors").exists()),
        key=checkpoint_number,
    )
    if spec == "all":
        chosen = candidates
    elif spec == "latest":
        chosen = candidates[-1:] if candidates else []
    else:
        path = Path(spec)
        if not path.is_absolute():
            direct = (Path.cwd() / path).resolve()
            path = direct if direct.exists() else (run / path).resolve()
        chosen = [path]
    if not chosen:
        raise SystemExit(f"no checkpoints matching {spec!r} under {run}")
    for path in chosen:
        if not (path / "model.safetensors").exists():
            raise SystemExit(f"not a loadable checkpoint directory: {path}")
    return chosen


def interleave_kinds(rows: list[dict], limit: int | None) -> list[dict]:
    """Keep a small eval balanced instead of taking only the first `[sigh]` pairs."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        groups.setdefault(row.get("kind", "prompt"), []).append(row)
    out: list[dict] = []
    while groups and (limit is None or len(out) < limit):
        for kind in list(groups):
            if limit is not None and len(out) >= limit:
                break
            out.append(groups[kind].pop(0))
            if not groups[kind]:
                del groups[kind]
    return out


def audio_stats(wav: np.ndarray, sr: int) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(wav).all())
    clean = np.nan_to_num(wav)
    peak = float(np.max(np.abs(clean))) if clean.size else 0.0
    rms = float(np.sqrt(np.mean(clean**2))) if clean.size else 0.0
    active = np.flatnonzero(np.abs(clean) >= 0.01)
    leading = float(active[0] / sr) if active.size else float(len(clean) / sr)
    trailing = float((len(clean) - 1 - active[-1]) / sr) if active.size else 0.0
    return {
        "duration_s": round(len(clean) / sr, 3),
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "clipped_fraction": round(float(np.mean(np.abs(clean) >= 0.999)), 7),
        "leading_silence_s": round(leading, 3),
        "trailing_silence_s": round(trailing, 3),
        "finite": finite,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a checkpoint listening suite.")
    ap.add_argument("--run", type=Path, default=C.RUNS / "kurisu-lora-r32")
    ap.add_argument("--ckpt", default="all", help="all | latest | checkpoint directory")
    ap.add_argument("--prompts", type=Path, default=C.QWEN / "probe_prompts.jsonl")
    ap.add_argument("--limit", type=int, default=12,
                    help="balanced prompt count; 0 means every prompt")
    ap.add_argument("--speaker", default="kurisu")
    ap.add_argument("--language", default="Japanese")
    ap.add_argument("--attn", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="codec-frame cap (~20.5 s at 12.5 Hz)")
    ap.add_argument("--deterministic", action="store_true",
                    help="greedy main/sub-talker decoding for exact repeatability")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--vram_fraction", type=float, default=0.92)
    args = ap.parse_args()

    run = args.run.resolve()
    checkpoints = resolve_checkpoints(run, args.ckpt)
    prompts = list(C.read_jsonl(args.prompts))
    prompts = interleave_kinds(prompts, None if args.limit == 0 else args.limit)
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")

    device = C.vram_guard(args.vram_fraction)
    if device == "cpu":
        raise SystemExit("evaluation requires CUDA")
    import torch
    from qwen_tts import Qwen3TTSModel

    root = (args.out or (C.EVAL / run.name)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generate_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.deterministic:
        generate_kwargs.update({
            "do_sample": False,
            "top_k": 1,
            "subtalker_dosample": False,
            "subtalker_top_k": 1,
        })

    report = {
        "run": str(run),
        "prompts": str(args.prompts.resolve()),
        "prompt_count": len(prompts),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "checkpoints": [],
    }
    for checkpoint in checkpoints:
        print(f"[model] {checkpoint}")
        model = Qwen3TTSModel.from_pretrained(
            str(checkpoint), device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation=args.attn)
        supported = [s.lower() for s in model.get_supported_speakers()]
        if args.speaker.lower() not in supported:
            raise SystemExit(
                f"speaker {args.speaker!r} is not in checkpoint speakers: {supported}")

        checkpoint_out = root / checkpoint.name
        checkpoint_out.mkdir(parents=True, exist_ok=True)
        manifest: list[dict] = []
        t0 = time.time()
        for index, prompt in enumerate(prompts):
            seed = args.seed + index
            torch.manual_seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            kind = prompt.get("kind", "prompt")
            wav_path = checkpoint_out / f"{index:03d}_{kind}.wav"
            print(f"  [{index + 1:>2}/{len(prompts)}] {kind}: {prompt['text']}", flush=True)
            try:
                wavs, sr = model.generate_custom_voice(
                    text=prompt["text"], speaker=args.speaker, language=args.language,
                    **generate_kwargs)
                wav = np.asarray(wavs[0], dtype=np.float32)
                sf.write(wav_path, wav, sr, subtype="PCM_16")
                row = {
                    "index": index,
                    **prompt,
                    "seed": seed,
                    "audio": str(wav_path.relative_to(root)),
                    "sample_rate": sr,
                    **audio_stats(wav, sr),
                }
                # The waveform decoder runs at roughly 12.5 codec frames/s. Reaching
                # this duration strongly suggests generation hit the safety cap rather
                # than producing EOS, which is an important checkpoint-quality signal.
                row["likely_hit_token_limit"] = (
                    row["duration_s"] >= 0.95 * args.max_new_tokens / C.CODEC_FPS)
            except Exception as exc:  # keep the rest of a long listening run usable
                row = {"index": index, **prompt, "seed": seed,
                       "error": f"{type(exc).__name__}: {exc}"}
                print(f"  [error] {row['error']}")
            manifest.append(row)

        C.write_jsonl(checkpoint_out / "manifest.jsonl", manifest)
        errors = sum("error" in row for row in manifest)
        durations = [row["duration_s"] for row in manifest if "duration_s" in row]
        entry = {
            "checkpoint": str(checkpoint),
            "output": str(checkpoint_out),
            "generated": len(manifest) - errors,
            "errors": errors,
            "wall_seconds": round(time.time() - t0, 1),
            "duration_s": {
                "median": round(float(np.median(durations)), 3) if durations else None,
                "min": round(min(durations), 3) if durations else None,
                "max": round(max(durations), 3) if durations else None,
            },
        }
        report["checkpoints"].append(entry)
        print(f"[done] {entry}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    C.write_report("ft_03_eval.json", report)
    (root / "README.txt").write_text(
        "Listen to the same numbered prompt across checkpoint folders. Prefer stable "
        "identity, natural rate, clean onset, and a real tagged-vs-untagged contrast; "
        "do not choose an epoch from training loss alone.\n",
        encoding="utf-8",
    )
    print(f"[eval] listening suite: {root}")


if __name__ == "__main__":
    main()
