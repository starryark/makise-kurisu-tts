"""Stage FT-4 - local epoch playground: test any checkpoint with custom text.

A local Gradio page (127.0.0.1) for the listening work ft03 cannot do: type
your own text, pick any epoch, and hear it. Mirrors the deployed Space app
(``Kurisu_Qwen3_TTS/app.py``) - same load/verify/synthesize path - extended
for multi-checkpoint selection:

* discovers every ``runs/<run>/checkpoint-epoch-*`` that is a complete
  ``custom_voice`` checkpoint with the ``kurisu`` speaker (smoke runs are
  skipped), newest run first;
* keeps at most two checkpoints in VRAM (LRU; loading a third evicts the
  least recently used);
* exposes language, max_new_tokens, an optional seed, and a greedy-decode
  toggle so two epochs can be compared on equal terms;
* appends every generation to ``eval/playground_log.jsonl`` - the listening
  decision gets the same provenance discipline as the training run.

Run it:

    .venv-tts/Scripts/python.exe ft04_playground.py            # port 7860
    .venv-tts/Scripts/python.exe ft04_playground.py --port 7861

Then open http://127.0.0.1:7860. First use of a checkpoint loads it (~30 s);
later uses are instant until it is evicted.
"""
from __future__ import annotations

import argparse
import gc
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ft_config as C

C.utf8_stdout()

DEFAULT_PORT = 7860
DEFAULT_SPEAKER_NAME = "kurisu"
MAX_TEXT_CHARACTERS = 200
MAX_NEW_TOKENS_CAP = 512
LANGUAGES = ("Auto", "Japanese", "English", "Chinese")

LOG_PATH = C.EVAL / "playground_log.jsonl"

CSS = ".gradio-container { max-width: 860px !important; margin: 0 auto !important; }"


# ---------------------------------------------------------------- discovery

def verify_checkpoint(source: Path, speaker_name: str) -> None:
    """The Space app's checks: complete custom_voice checkpoint, speaker present."""
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("tts_model_type") != "custom_voice":
        raise ValueError(f"{source.name}: not a custom_voice checkpoint")
    speaker_ids = config.get("talker_config", {}).get("spk_id", {})
    if speaker_name not in {str(name).lower() for name in speaker_ids}:
        raise ValueError(f"{source.name}: no {speaker_name!r} speaker")


def discover_checkpoints() -> list[dict]:
    """Every loadable epoch checkpoint, ordered for epoch comparison.

    The current run comes first (its epochs are the open question), then the
    previous run with its deployed baseline marked. Smoke runs are skipped.
    """
    entries: list[dict] = []
    for run_dir in sorted(C.RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("smoke"):
            continue
        for ckpt in sorted(run_dir.glob("checkpoint-epoch-*"), key=_epoch_of):
            if not (ckpt / "model.safetensors").is_file() or not (ckpt / "config.json").is_file():
                continue
            try:
                verify_checkpoint(ckpt, DEFAULT_SPEAKER_NAME)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                print(f"[discover] skipping {ckpt.name}: {exc}")
                continue
            label = f"{run_dir.name} - epoch {_epoch_of(ckpt)}"
            if (run_dir.name, _epoch_of(ckpt)) == ("kurisu-lora-r32", 10):
                label += "  (deployed baseline A)"
            entries.append({"label": label, "path": str(ckpt.resolve()), "run": run_dir.name})
    run_order = {"kurisu-lora-r16": 0, "kurisu-lora-r32": 1}
    entries.sort(key=lambda e: (run_order.get(e["run"], 9), e["run"], _epoch_of(Path(e["path"]))))
    return entries


def _epoch_of(ckpt: Path) -> int:
    try:
        return int(ckpt.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


# ---------------------------------------------------------------- serving

class ModelPool:
    """LRU cache of loaded checkpoints; all GPU work serialized behind a lock."""

    def __init__(self, max_loaded: int = 2):
        self.max_loaded = max_loaded
        self.lock = threading.Lock()
        self._loaded: OrderedDict[str, object] = OrderedDict()

    def loaded_labels(self) -> list[str]:
        return list(self._loaded)

    def get(self, path: str):
        with self.lock:
            if path in self._loaded:
                self._loaded.move_to_end(path)
                return self._loaded[path]
            while len(self._loaded) >= self.max_loaded:
                old_path, old = self._loaded.popitem(last=False)
                print(f"[pool] evicting {Path(old_path).name}")
                del old
                gc.collect()
                import torch

                torch.cuda.empty_cache()
            import torch
            from qwen_tts import Qwen3TTSModel

            t0 = time.time()
            print(f"[pool] loading {Path(path).name} ...", flush=True)
            model = Qwen3TTSModel.from_pretrained(
                path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
            self._loaded[path] = model
            print(f"[pool] loaded in {time.time() - t0:.1f}s "
                  f"({len(self._loaded)}/{self.max_loaded} resident)", flush=True)
            return model

    def clear(self) -> int:
        with self.lock:
            count = len(self._loaded)
            self._loaded.clear()
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return count


def validate_text(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to synthesize.")
    normalized = text.strip()
    if len(normalized) > MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"Text is too long ({len(normalized)} characters). "
            f"Use at most {MAX_TEXT_CHARACTERS} characters."
        )
    return normalized


def prepare_audio(wavs: object, sample_rate: object) -> tuple[int, np.ndarray]:
    """The Space app's audio validation, unchanged."""
    if not isinstance(sample_rate, (int, np.integer)) or not 8_000 <= int(sample_rate) <= 192_000:
        raise RuntimeError("The model returned an invalid sample rate.")
    if not isinstance(wavs, (list, tuple)) or not wavs:
        raise RuntimeError("The model returned no audio.")
    waveform = np.asarray(wavs[0], dtype=np.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError("The model returned an empty or invalid waveform.")
    if not np.isfinite(waveform).all():
        raise RuntimeError("The model returned non-finite audio samples.")
    return int(sample_rate), waveform


def log_generation(row: dict) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_synthesize_fn(pool: ModelPool, checkpoints: list[dict]):
    by_label = {entry["label"]: entry for entry in checkpoints}

    def synthesize(label: str, text: str, language: str, max_new_tokens: int,
                   seed_text: str, greedy: bool):
        entry = by_label[label]
        try:
            normalized = validate_text(text)
        except ValueError as exc:
            yield None, f"Input error: {exc}"
            return
        if language not in LANGUAGES:
            language = "Auto"
        max_new_tokens = int(min(max(max_new_tokens or MAX_NEW_TOKENS_CAP, 32), MAX_NEW_TOKENS_CAP))
        seed_val: int | None = None
        if seed_text is not None and str(seed_text).strip():
            try:
                seed_val = int(str(seed_text).strip())
            except ValueError:
                yield None, f"Input error: seed {seed_text!r} is not an integer."
                return
        yield None, f"Preparing **{label}** (loads on first use, up to ~40 s) ..."
        try:
            model = pool.get(entry["path"])
            import torch

            if seed_val is not None:
                torch.manual_seed(seed_val)
                np.random.seed(seed_val & 0xFFFFFFFF)
            yield None, f"Generating with **{label}** ..."
            t0 = time.time()
            wavs, sample_rate = model.generate_custom_voice(
                text=normalized, speaker=DEFAULT_SPEAKER_NAME, language=language,
                non_streaming_mode=True, max_new_tokens=max_new_tokens,
                **({"do_sample": False, "top_k": 1, "subtalker_dosample": False,
                    "subtalker_top_k": 1} if greedy else {}))
            sr, waveform = prepare_audio(wavs, sample_rate)
            wall = time.time() - t0
            duration = waveform.size / sr
            status = (
                f"**{label}** - {duration:.1f} s audio in {wall:.1f} s "
                f"({language}, {max_new_tokens} tokens max"
                + (f", seed {seed_val}" if seed_val is not None else "")
                + (", greedy" if greedy else "") + ")"
            )
            log_generation({
                "model": label, "path": entry["path"], "text": normalized,
                "language": language, "seed": seed_val,
                "greedy": greedy, "max_new_tokens": max_new_tokens,
                "duration_s": round(duration, 3), "wall_s": round(wall, 2),
            })
            yield (sr, waveform), status
        except Exception as exc:  # keep the page usable; the traceback lands in the console
            import traceback

            traceback.print_exc()
            yield None, f"Generation failed: {type(exc).__name__}: {exc}"

    return synthesize


# ---------------------------------------------------------------- UI

def build_demo(pool: ModelPool, checkpoints: list[dict]):
    import gradio as gr

    default = next((e["label"] for e in checkpoints if "r16" in e["run"] and "epoch 1" in e["label"]),
                   checkpoints[0]["label"])

    def free_gpu():
        count = pool.clear()
        return f"Freed {count} loaded checkpoint(s). Next generation reloads on demand."

    with gr.Blocks(title="Kurisu Epoch Playground") as demo:
        gr.Markdown(
            "## Kurisu epoch playground\n"
            "Pick a checkpoint, type any text, and listen. Compare epochs on the "
            "same text: set a **seed** and/or **deterministic** for an equal-footing "
            "A/B, or leave both off to hear each epoch's real sampling behavior. "
            f"At most {pool.max_loaded} checkpoints stay in VRAM; others load on demand."
        )
        model = gr.Dropdown(choices=[e["label"] for e in checkpoints],
                            value=default, label="Checkpoint", interactive=True)
        text = gr.Textbox(
            label="Text", lines=4, max_lines=8,
            placeholder="Enter text in Japanese, English, or Chinese ...",
            value="未来のことは、誰にもわからない。だからこそ面白いんじゃない。",
            info=f"Up to {MAX_TEXT_CHARACTERS} characters.",
        )
        with gr.Row():
            language = gr.Dropdown(choices=list(LANGUAGES), value="Auto", label="Language")
            max_new_tokens = gr.Slider(minimum=64, maximum=MAX_NEW_TOKENS_CAP, step=32,
                                       value=MAX_NEW_TOKENS_CAP, label="Max codec frames "
                                       "(512 ≈ 41 s)")
            seed = gr.Textbox(label="Seed (blank = random)", value="",
                              placeholder="e.g. 20260905",
                              info="Same seed + same text = a repeatable take")
            greedy = gr.Checkbox(label="Deterministic (greedy)", value=False)
        generate = gr.Button("Generate speech", variant="primary", size="lg")
        audio = gr.Audio(label="Generated speech", type="numpy", autoplay=False,
                         interactive=False)
        status = gr.Markdown("Ready.", elem_id="status")
        gr.Examples(examples=[
            ["これが、運命石の扉の選択よ。"],
            ["すみません、[pause]急に、頼ってしまって……"],
            ["The Organization is watching us. We have to move, now."],
            ["ええっ、[gasp] それは、世界線の変動よ。"],
        ], inputs=[text], label="Examples (the last two use the corpus tag conventions)")
        with gr.Accordion("GPU", open=False):
            free = gr.Button("Free loaded checkpoints")
            free_status = gr.Markdown("")

        generate.click(
            make_synthesize_fn(pool, checkpoints),
            inputs=[model, text, language, max_new_tokens, seed, greedy],
            outputs=[audio, status], api_name="generate_speech",
        )
        free.click(free_gpu, inputs=None, outputs=free_status, api_name="free_gpu")
    return demo


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Kurisu epoch listening playground")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max_loaded", type=int, default=2,
                        help="checkpoints resident in VRAM at once (LRU)")
    parser.add_argument("--no_browser", action="store_true",
                        help="do not open a browser window on launch")
    args = parser.parse_args()

    C.vram_guard(0.92)
    checkpoints = discover_checkpoints()
    if not checkpoints:
        raise SystemExit("no loadable custom_voice checkpoints found under runs/")
    print(f"[playground] {len(checkpoints)} checkpoints:")
    for entry in checkpoints:
        print(f"  {entry['label']}")

    pool = ModelPool(max_loaded=args.max_loaded)
    import gradio as gr

    demo = build_demo(pool, checkpoints)
    for port in range(args.port, args.port + 5):
        try:
            demo.queue().launch(server_name=args.host, server_port=port,
                                inbrowser=not args.no_browser,
                                theme=gr.themes.Soft(), css=CSS)
            return
        except OSError as exc:
            print(f"[playground] port {port} unavailable ({exc.strerror}); "
                  "is another playground already running? trying the next port")
    raise SystemExit(f"no free port in {args.port}..{args.port + 4}")


if __name__ == "__main__":
    main()
