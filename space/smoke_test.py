"""Run one or more real synthesis checks through the Space application path."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_PROMPTS = [
    "Hello. This is Kurisu.",
    "こんにちは。牧瀬紅莉栖よ。",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Local checkpoint directory or model repo ID")
    parser.add_argument(
        "--text",
        action="append",
        dest="prompts",
        help="Prompt to test; repeat for multiple prompts (defaults to English and Japanese)",
    )
    args = parser.parse_args()

    os.environ["MODEL_ID"] = str(args.model)
    os.environ.setdefault("SPEAKER_NAME", "kurisu")

    # Import only after configuring MODEL_ID because app.py loads the model at module scope.
    from app import prepare_audio, synthesize

    prompts = args.prompts or DEFAULT_PROMPTS
    with tempfile.TemporaryDirectory(prefix="kurisu_space_smoke_") as temp_dir:
        for index, prompt in enumerate(prompts):
            audio, status = synthesize(prompt, os.environ["SPEAKER_NAME"], "Detect")
            if audio is None:
                raise RuntimeError(status)
            sample_rate, waveform = prepare_audio([audio[1]], audio[0])
            output = Path(temp_dir) / f"smoke-{index}.wav"
            sf.write(output, waveform, sample_rate, subtype="PCM_16")
            readback, readback_rate = sf.read(output, dtype="float32")
            if readback_rate != sample_rate or readback.size == 0 or not np.isfinite(readback).all():
                raise RuntimeError(f"WAV readback validation failed for prompt {index}")
            print(
                f"PASS prompt={index} sample_rate={sample_rate} samples={waveform.size} "
                f"duration={waveform.size / sample_rate:.2f}s"
            )


if __name__ == "__main__":
    main()
