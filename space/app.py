from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path


LOGGER = logging.getLogger("kurisu_tts")

DEFAULT_SPEAKER_NAME = "kurisu"
DEFAULT_MODEL_ID = "starrydark/Makise_Kurisu_Qwen3_TTS"
MAX_TEXT_CHARACTERS = 200
MAX_NEW_TOKENS = 512
GPU_DURATION_SECONDS = 60

# The training transcripts carry no spaces next to Japanese characters (the game
# script's display spaces were removed), so the same normalisation is applied here.
_JA_CHAR = r"[　-〿぀-ヿ一-鿿＀-￯…―〜]"
_SPACE_NEAR_JA = re.compile(rf"(?<={_JA_CHAR})[ 　]+|[ 　]+(?={_JA_CHAR})")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")

# Display names for the speaker rows a checkpoint may carry (v3: one voice, several
# recording conditions; anything unknown is shown by its own name).
SPEAKER_LABELS = {
    "kurisu": "Kurisu (studio, default)",
    "amadeus": "Amadeus (device voice, game)",
    "amadeus_anime": "Amadeus (device voice, anime)",
    "phone": "Phone call",
}
TAG_HELP = {
    "[sigh]": "a sigh",
    "[nv]": "another non-verbal sound (breath, gasp, small laugh)",
    "[bg_footsteps]": "footsteps in the background",
    "[bg_ringtone]": "a phone ringing in the background",
    "[bg_rain]": "rain in the background",
    "[bg_typing]": "keyboard typing in the background",
    "[bg_traffic]": "street traffic in the background",
}
ABOUT = """
**Kurisu Qwen3-TTS v3**: a full fine-tune of Qwen3-TTS-12Hz-1.7B-Base on 13.7 h of
training rows.
- **Japanese:** 8,873 rows of Kurisu (game studio and anime), the Amadeus device voice
  and phone calls.
- **Native replay:** 1,566 rows of Mandarin (AISHELL-3) and English (LibriTTS-R),
  spoken by 55 other speakers.
- **Pseudo-Mandarin:** 531 rows generated in Kurisu's own voice by the Base model.

**How it was trained.**
- **Every row carries Qwen's language id.** This, plus the native data, is what
  removes the Japanese accent from her Mandarin.
- **Optimisation:** lr 5e-6, batch 32, one epoch (343 steps) on one H100, about 4
  minutes per epoch.
- **Checkpoint choice:** a selection rule written before any result, from 6 candidates
  (2 seeds × 3 checkpoints).

**Blind listening test.** One listener rated 16 Mandarin sentences on a 1–5 scale,
where 5 is native.

| System | Mean rating |
|---|---|
| v3 | 5.0 |
| v2, told the text is Chinese | 3.9 |
| v2, as previously served (Auto) | 3.6 |
| Native-speaker anchor | 5.0 |

v3 was rated above v2 on 9 of the 16 sentences and below it on none.

Full training details, code, logs and every evaluation are on the
[model card](https://huggingface.co/starrydark/Makise_Kurisu_Qwen3_TTS).
"""


def read_configuration() -> tuple[str, str, str | None]:
    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID).strip()
    if not model_id:
        raise RuntimeError("MODEL_ID cannot be empty.")

    speaker_name = os.getenv("SPEAKER_NAME", DEFAULT_SPEAKER_NAME).strip().lower()
    if not speaker_name:
        raise RuntimeError("SPEAKER_NAME cannot be empty.")

    token = os.getenv("HF_TOKEN", "").strip() or None
    return model_id, speaker_name, token


# Validate deployment configuration before importing the heavier ML stack.
MODEL_ID, SPEAKER_NAME, _MODEL_ACCESS_TOKEN = read_configuration()

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel


def resolve_model_source(model_id: str, token: str | None) -> Path:
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return local_path.resolve()

    try:
        return Path(snapshot_download(repo_id=model_id, token=token))
    except Exception as exc:
        access_hint = (
            " If the model repository is private, add an HF_TOKEN Space secret "
            "with read access to it."
        )
        raise RuntimeError(f"Could not download MODEL_ID={model_id!r}.{access_hint}") from exc


def verify_checkpoint(source: Path, speaker_name: str) -> dict:
    config_path = source / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"MODEL_ID does not contain a readable config.json: {source}") from exc

    if config.get("tts_model_type") != "custom_voice":
        raise RuntimeError(
            "MODEL_ID must point to a complete Qwen3-TTS custom_voice checkpoint."
        )

    speaker_ids = config.get("talker_config", {}).get("spk_id", {})
    configured_speakers = {str(name).lower() for name in speaker_ids}
    if speaker_name not in configured_speakers:
        raise RuntimeError(
            f"SPEAKER_NAME={speaker_name!r} is not present in the checkpoint. "
            f"Available speakers: {sorted(configured_speakers)}"
        )

    required_paths = (
        source / "model.safetensors",
        source / "speech_tokenizer" / "config.json",
        source / "speech_tokenizer" / "model.safetensors",
    )
    missing = [str(path.relative_to(source)) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"MODEL_ID is missing required checkpoint files: {missing}")

    # kurisu_v3.json: what the checkpoint was trained with (language ids, tag tokens).
    meta_path = source / "kurisu_v3.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}


def load_model(model_id: str, speaker_name: str, token: str | None) -> tuple[Qwen3TTSModel, dict]:
    source = resolve_model_source(model_id, token)
    meta = verify_checkpoint(source, speaker_name)

    # ZeroGPU emulates CUDA during startup, so current guidance is to place the
    # model on CUDA at module scope instead of transferring it per request.
    model = Qwen3TTSModel.from_pretrained(
        str(source),
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    supported = {name.lower() for name in model.get_supported_speakers()}
    if speaker_name not in supported:
        raise RuntimeError(
            f"Loaded model does not expose speaker {speaker_name!r}; available: {sorted(supported)}"
        )
    return model, meta


def validate_text(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some text to synthesize.")

    normalized = _SPACE_NEAR_JA.sub("", text.strip())
    # A tag is one token only when nothing is wedged inside it; spaces next to it are
    # not part of the training text either.
    normalized = re.sub(r"\s*(\[[a-z_]+\])\s*", r"\1", normalized)
    if len(normalized) > MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"Text is too long ({len(normalized)} characters). "
            f"Use at most {MAX_TEXT_CHARACTERS} characters."
        )
    return normalized


def pick_language(choice: str, text: str) -> str:
    """'Detect' -> the language the text is written in, if the model was trained with
    language ids; otherwise Auto (what a checkpoint without language training expects)."""
    if choice != "Detect":
        return choice
    if not META.get("lang_trained"):
        return "Auto"
    body = re.sub(r"\[[a-z_]+\]", "", text)
    if _KANA.search(body):
        return "Japanese"
    if _HAN.search(body):
        return "Chinese"
    if _LATIN.search(body):
        return "English"
    return "Auto"


def prepare_audio(wavs: object, sample_rate: object) -> tuple[int, np.ndarray]:
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


MODEL, META = load_model(MODEL_ID, SPEAKER_NAME, _MODEL_ACCESS_TOKEN)
del _MODEL_ACCESS_TOKEN
SPEAKERS = sorted({name.lower() for name in MODEL.get_supported_speakers()},
                  key=lambda n: (n != SPEAKER_NAME, n))
TAGS = [t for t in META.get("tokens", []) if t in TAG_HELP]


@spaces.GPU(duration=GPU_DURATION_SECONDS)
def synthesize(text: str, speaker: str, language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
    try:
        normalized = validate_text(text)
        if speaker not in SPEAKERS:
            raise ValueError(f"Unknown voice {speaker!r}.")
        lang = pick_language(language, normalized)
        # About 1.6 codec frames per Japanese character at 12.5 Hz; allowing ~2.5x that
        # bounds a rare non-stopping sample without clipping natural speech.
        max_new_tokens = min(MAX_NEW_TOKENS, 64 + 4 * len(normalized))
        wavs, sample_rate = MODEL.generate_custom_voice(
            text=normalized,
            speaker=speaker,
            language=lang,
            non_streaming_mode=True,
            max_new_tokens=max_new_tokens,
        )
        audio = prepare_audio(wavs, sample_rate)
        duration = audio[1].size / audio[0]
        return audio, f"Speech generated — {duration:.1f} seconds ({lang})."
    except ValueError as exc:
        return None, f"Input error: {exc}"
    except Exception:
        LOGGER.exception("Speech generation failed")
        return None, "Generation failed. Please try again with a shorter passage."


CSS = """
.gradio-container { max-width: 780px !important; margin: 0 auto !important; }
#generate-button { min-height: 48px; }
#status { min-height: 1.5rem; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Makise Kurisu AI Voice (STEINS;GATE) — Qwen3-TTS",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# Makise Kurisu AI Voice (STEINS;GATE)\n"
            "Type Japanese, Chinese or English and hear it in Makise Kurisu's voice "
            "(牧瀬紅莉栖 / 牧濑红莉栖): as Kurisu, as Amadeus, or on a phone call. "
            "Unofficial fan demo; no reference audio is needed."
        )
        text = gr.Textbox(
            label="Text",
            placeholder="Enter text in Japanese, English, Chinese or another supported language…",
            lines=5,
            max_lines=8,
            max_length=MAX_TEXT_CHARACTERS,
            info=f"Up to {MAX_TEXT_CHARACTERS} characters.",
            value="未来のことは、誰にもわからない。だからこそ面白いんじゃない。",
        )
        with gr.Row():
            speaker = gr.Dropdown(
                label="Voice",
                choices=[(SPEAKER_LABELS.get(s, s), s) for s in SPEAKERS],
                value=SPEAKER_NAME,
            )
            language = gr.Dropdown(
                label="Language",
                choices=["Detect", "Japanese", "Chinese", "English", "Auto"],
                value="Detect",
                info="Detect reads the script of the text.",
            )
        if TAGS:
            gr.Markdown(
                "**Experimental tags** (trained from few examples; they may be ignored): "
                + ", ".join(f"`{t}` {TAG_HELP[t]}" for t in TAGS)
                + ". Write them without spaces, e.g. `[sigh]はぁ……そうね。`"
            )
        generate = gr.Button(
            "Generate Speech",
            variant="primary",
            size="lg",
            elem_id="generate-button",
        )
        audio = gr.Audio(
            label="Generated speech",
            type="numpy",
            format="wav",
            interactive=False,
            autoplay=False,
        )
        status = gr.Markdown("Ready.", elem_id="status")
        examples = [
            ["これが、運命石の扉の選択よ。", SPEAKER_NAME, "Detect"],
            ["这个实验的结果，我们必须先验证再下结论。", SPEAKER_NAME, "Detect"],
            ["Science is built on questions, not certainty.", SPEAKER_NAME, "Detect"],
            ["私はアマデウス。牧瀬紅莉栖の記憶を持つ人工知能よ。", "amadeus", "Detect"],
            ["もしもし、岡部？今どこにいるの？", "phone", "Detect"],
        ]
        gr.Examples(
            examples=[e for e in examples if e[1] in SPEAKERS],
            inputs=[text, speaker, language],
            label="Examples",
        )
        with gr.Accordion("About this model", open=False):
            gr.Markdown(ABOUT)

        generate.click(
            fn=synthesize,
            inputs=[text, speaker, language],
            outputs=[audio, status],
            api_name="generate_speech",
            show_progress="full",
            concurrency_limit=1,
            concurrency_id="kurisu-tts",
        )

    return demo.queue(default_concurrency_limit=1, max_size=16)


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
