"""Independent transcripts for every Japanese clip, from two unrelated ASR systems.

The game transcripts come from scenario bytecode (what the screen shows, not
necessarily what was recorded) and the anime ones from subtitles, so neither is
guaranteed to match the take. Two ASR hypotheses per clip let p03 keep a clip
when *either* system agrees with the script: a disagreement from both is a
transcript problem; a disagreement from one is usually that system's error.

The shared context only lists the series' recurring proper nouns. It is the same
string for every clip, so it cannot pull a hypothesis towards that clip's script.

Writes WORK/src/asr.jsonl: {id, qwen, anime, utmos}.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
CONTEXT = ("牧瀬紅莉栖、岡部倫太郎、鳳凰院凶真、椎名まゆり、橋田至、ダル、阿万音鈴羽、漆原るか、"
           "フェイリス、桐生萌郁、比屋定真帆、SERN、Dメール、タイムリープ、世界線、"
           "アマデウス、未来ガジェット研究所、ラボメン、HENTAI、@ちゃんねる")


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def save(done: dict, rows: list[dict], out_path: Path) -> None:
    """Checkpoint after every pass: each one is minutes of GPU time."""
    tmp = out_path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(done[r["id"]], ensure_ascii=False) + "\n"
                           for r in rows if r["id"] in done), encoding="utf-8")
    os.replace(tmp, out_path)


def load_audio(path: str) -> dict:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    return {"raw": wav.mean(axis=-1) if wav.ndim > 1 else wav, "sampling_rate": sr}


def main() -> None:
    rows = [json.loads(l) for l in (WORK / "src" / "ja_meta.jsonl").read_text(encoding="utf-8").splitlines() if l]
    # Longest first so a batch's padding is bounded by its neighbours.
    rows.sort(key=lambda r: -r["duration_seconds"])
    out_path = WORK / "src" / "asr.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r

    todo = [r for r in rows if r["id"] not in done or done[r["id"]].get("qwen") is None]
    if todo:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", max_inference_batch_size=64, max_new_tokens=256)
        for i, chunk in enumerate(batches(todo, 64)):
            res = model.transcribe([r["audio"] for r in chunk], language="Japanese",
                                   context=[CONTEXT] * len(chunk))
            for r, h in zip(chunk, res):
                done.setdefault(r["id"], {"id": r["id"]})["qwen"] = h.text.strip()
            if i % 20 == 0:
                print(f"[qwen-asr] {min((i + 1) * 64, len(todo))}/{len(todo)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        save(done, rows, out_path)

    # litagin/anime-whisper: a Whisper fine-tune on ~5.3k h of visual-novel speech,
    # i.e. exactly this register (CER 13.0 vs 16.5 for whisper-large-v3 on its eval).
    todo = [r for r in rows if done[r["id"]].get("anime") is None]
    if todo:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="litagin/anime-whisper",
                       torch_dtype=torch.float16, device="cuda:0", chunk_length_s=30.0)
        for i, chunk in enumerate(batches(todo, 32)):
            res = asr([load_audio(r["audio"]) for r in chunk], batch_size=32,
                      generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0,
                                       "repetition_penalty": 1.0})
            for r, h in zip(chunk, res):
                done[r["id"]]["anime"] = h["text"].strip()
            if i % 40 == 0:
                print(f"[anime-whisper] {min((i + 1) * 32, len(todo))}/{len(todo)}", flush=True)
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        save(done, rows, out_path)

    # Predicted MOS for every clip: the game takes are studio-clean by construction,
    # so their distribution is the yardstick p03 holds the source-separated anime to.
    todo = [r for r in rows if "utmos" not in done[r["id"]]]
    if todo:
        import utmosv2
        model = utmosv2.create_model(pretrained=True)
        for i, r in enumerate(todo):
            try:
                done[r["id"]]["utmos"] = round(float(model.predict(input_path=r["audio"], verbose=False)), 3)
            except TypeError:
                done[r["id"]]["utmos"] = round(float(model.predict(input_path=r["audio"])), 3)
            except Exception as exc:
                print(f"[utmos] {r['id']}: {exc}")
                done[r["id"]]["utmos"] = None
            if i % 1000 == 0:
                print(f"[utmos] {i}/{len(todo)}", flush=True)
            if i and i % 2000 == 0:
                save(done, rows, out_path)

    tmp = out_path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(done[r["id"]], ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[asr] {len(rows)} clips -> {out_path}")


if __name__ == "__main__":
    main()
