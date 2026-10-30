# Makise Kurisu AI voice (STEINS;GATE): Qwen3-TTS fine-tuning code

**The code behind the [Makise Kurisu Qwen3-TTS model](https://huggingface.co/starrydark/Makise_Kurisu_Qwen3_TTS): an unofficial, fan-made text-to-speech voice of Makise Kurisu from STEINS;GATE, in Japanese, Mandarin Chinese and English, with no reference audio needed.**

The repository covers the full process:
- building the dataset from game and anime voice lines;
- the fixed Qwen3-TTS trainer;
- the cross-lingual accent work;
- the evaluation and pre-registered selection rule;
- the demo app;
- the research notes behind each decision.

| | |
|---|---|
| **Try it** | [Demo on Hugging Face Spaces](https://huggingface.co/spaces/starrydark/Kurisu_Qwen3_TTS): Japanese, Chinese or English; voices `kurisu`, `amadeus`, `phone` |
| **Model** | [starrydark/Makise_Kurisu_Qwen3_TTS](https://huggingface.co/starrydark/Makise_Kurisu_Qwen3_TTS) (v3, 2026-10-29) |
| **Dataset** | [starrydark/Makise_Kurisu_Voice](https://huggingface.co/datasets/starrydark/Makise_Kurisu_Voice) |
| **Everything on the Hub** | [Collection](https://huggingface.co/collections/starrydark/makise-kurisu-ai-voice-steins-gate) |
| **Fine-tuning guide** | [Four silent bugs in the official Qwen3-TTS fine-tuning script](https://github.com/starryark/tricolour-tts/blob/main/docs/qwen3-tts-finetuning-fixes.md). The fixes started in this project. |
| **Sibling project** | [tricolour-tts](https://github.com/starryark/tricolour-tts): the same trainer extended to 14 characters in one model, with emotion instructions |

**Also known as:**
- **Japanese:** 牧瀬紅莉栖, 紅莉栖.
- **Chinese:** 牧濑红莉栖, 红莉栖.
- **Korean:** 마키세 크리스.
- **Nicknames:** "Christina"; Amadeus Kurisu (アマデウス).
- **From:** STEINS;GATE (シュタインズ・ゲート / 命运石之门 / 슈타인즈 게이트).

This is an unofficial fan project, with no affiliation to the rights holders of STEINS;GATE or to the voice actors.

- **日本語：** 『シュタインズ・ゲート』牧瀬紅莉栖の非公式ファンメイド音声合成（TTS）モデルのコードです。
  - **対応言語：** 日本語・中国語・英語。参照音声は不要です。
  - **含まれるもの：** データセット構築、Qwen3-TTS のファインチューニング（公式スクリプトの不具合を修正済み）、評価をすべて含みます。
  - 公式および声優ご本人とは一切関係ありません。
- **中文：** 《命运石之门》牧濑红莉栖非官方粉丝向 AI 语音合成（TTS）模型的完整代码。
  - **语言：** 支持日语、中文、英语，无需参考音频。
  - **代码内容：** 包括数据集构建、Qwen3-TTS 微调（修复了官方 `sft_12hz.py` 的隐性问题）、减少中文日本口音的方法，以及预先登记的评测。
  - 与官方及声优本人无关。

## Use the model

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel   # pip install qwen-tts

tts = Qwen3TTSModel.from_pretrained("starrydark/Makise_Kurisu_Qwen3_TTS",
                                    device_map="cuda", dtype=torch.bfloat16)
wavs, sr = tts.generate_custom_voice(
    text="仮説を立てたなら、まずは検証しなさい。話はそれからよ。",
    speaker="kurisu",          # or "amadeus", "phone"
    language="Japanese",       # "Chinese" for Mandarin, "English" for English
)
sf.write("kurisu.wav", wavs[0], sr)
```

Usage notes:
- **Pass the language of the text.** That is what keeps the Mandarin free of a Japanese accent.
- **Tags are experimental.** `[sigh]` and `[nv]` work as a nudge, not a switch.
- **More options** are on the [model card](https://huggingface.co/starrydark/Makise_Kurisu_Qwen3_TTS).

## Results (v3 against v2, same texts and sampling seeds)

| | v2 | **v3** |
|---|---:|---:|
| Japanese CER ↓ (real recordings: 0.032) | 0.032 | **0.030** |
| Japanese speaker similarity ↑ | 0.668 | **0.671** |
| Mandarin shared-character syllable error ↓ (native reference 0.048) | 0.170 | **0.060** |
| Mandarin CER ↓ | 0.071 | **0.017** |
| English WER ↓ | 0.005 | 0.002 |
| Blind Mandarin nativeness rating, 1–5 (one listener, 16 sentences) | 3.94 | **5.00** |

**Compute:**
- **The shipped run:** 6 min 37 s on one H100 80GB.
- **The whole project:** about 10.5 GPU-hours, almost all of it ablations and evaluation.

Full tables, the selection rule and every decision: [`eval/`](eval/) and the [model card](https://huggingface.co/starrydark/Makise_Kurisu_Qwen3_TTS#evaluation).

## Layout

```
train/      v3: the exact code that built, trained, evaluated and selected the released model
            (identical to training_code/ on the model repo). Run order: train/README.md
  train_sft.py          full fine-tune with the four upstream fixes, fixed speaker vectors,
                        language ids, new tag tokens
  test_equivalence.py   trainer inputs == generate_custom_voice inputs (72 voice/language/tag cases)
  mix_v3.py, p01-p13    data mix, ASR, build, phone-EQ augmentation, native replay, codec codes,
                        accent diagnosis (E0), pseudo-Mandarin, evaluation, selection, adoption
  queue_v3.sh, bootstrap_v3.sh   the RunPod queue and the pinned environment
eval/       SELECTION_RULE.md (pre-registered, with disclosed amendments), decisions.txt,
            adoption verdicts, accent diagnosis and listening-test tables
dataset/    the build of starrydark/Makise_Kurisu_Voice (pipeline/k01-k15): merges the game
            voice set, the six-language anime build and three character songs; speaker checks,
            forced alignment, earned expressive tags, leakage-safe splits, verification, export.
            dataset/README.md and TAGGING.md explain every stage
space/      the Gradio demo (ZeroGPU)
legacy/v1_local/   v1: single-speaker LoRA fine-tuning on a 16 GB RTX 5060 Ti (Windows,
                   PowerShell), model revision v1-epoch10. Already has the four trainer fixes
legacy/v1_dataset/ v1: the earlier Japanese-only training-set build (s00-s11) and its tag taxonomy
research/   reports: base-model choice, fine-tuning methods, labelling, Qwen3-TTS internals,
            cross-lingual accent, results
```

**Not in this repository:**
- **Audio and transcripts:** use the [dataset](https://huggingface.co/datasets/starrydark/Makise_Kurisu_Voice).
- **Large evaluation tables:** `table_v3.json` and the per-clip `selected_scored*.jsonl` are in the model repo's `eval/`.
- **The upstream builds that `dataset/` reads:**
  - the game voice extraction;
  - the six-language anime build (source separation and subtitle alignment).

The scripts default to the author's local folders; set the paths at the top of each script, or in `train/ft_config.py`.

## Build your own character voice with Qwen3-TTS

The recipe works for any character with a few hours of clean lines:
1. **Collect and label lines.** One performer, one voice, with transcripts. The dataset card shows what to keep per clip: quality tier, ASR cross-checks and tags.
2. **Filter.** Drop:
   - clips whose transcript two ASR systems disagree with;
   - noisy clips and singing;
   - clips that are too short or too long.
3. **Freeze one speaker vector per voice.** Average the speaker encoder over clean reference takes, and fine-tune the whole model with that vector fixed. The result is a `custom_voice` checkpoint that needs no reference audio.
4. **Keep other languages native.**
   - Train with Qwen's language id.
   - Mix in about 15% native speech from other speakers.
   - Add about 5% target-voice speech in the other language, generated by the Base model.
5. **Use a fixed trainer.** The official script has four silent bugs ([guide](https://github.com/starryark/tricolour-tts/blob/main/docs/qwen3-tts-finetuning-fixes.md)); `train/train_sft.py` fixes them.
6. **Choose the checkpoint by a rule fixed in advance.**
   - Score held-out CER, speaker similarity and an accent metric. Validation loss barely moves, so don't use it.
   - Confirm the pick with a blind listening test.

For several characters in one model, or for emotion instructions, see [tricolour-tts](https://github.com/starryark/tricolour-tts).

## FAQ

**Can I hear it without installing anything?**
Yes, in the [Hugging Face Space](https://huggingface.co/spaces/starrydark/Kurisu_Qwen3_TTS).

**Does it need a reference clip? Is it voice cloning?**
No. The model has three fixed voices, called by name: `kurisu`, `amadeus` (the Amadeus device voice from STEINS;GATE 0) and `phone` (Kurisu on a phone call). No audio is needed at inference.

**Why did the earlier Mandarin have a Japanese accent, and how was it fixed?**
A fine-tune on Japanese lines alone drags other languages toward Japanese phonetics. v3 fixed it in three steps:
1. It trains with Qwen's language ids.
2. It mixes in about 15% native Mandarin and English replay from other speakers.
3. It adds about 5% Kurisu-voiced Mandarin generated by the Base model, which still reads Mandarin almost natively in her voice.

The shared-character syllable error fell from 0.170 to 0.060, and a blind listener rated v3's Mandarin 5.0/5, against 3.9 for v2. The diagnosis is in [`research/crosslingual_accent_2026-10.md`](research/crosslingual_accent_2026-10.md) and [`research/e0_results.md`](research/e0_results.md).

**How do I make her sigh or laugh?**
Put `[sigh]` or `[nv]` where the sound happens, for example `[sigh]はぁ……そうね。`. They are experimental: `[sigh]` produced an audible breath in about 1 of 7 prompts, against about 1 in 30 without the tag.

**Can I fine-tune Qwen3-TTS on a 16 GB GPU?**
Yes. [`legacy/v1_local/`](legacy/v1_local/) is a LoRA workflow (rank 32, bf16 frozen base) that trained v1 on a 16 GB RTX 5060 Ti with SDPA attention. The released v3 is a full fine-tune on one H100.

**What is wrong with the official Qwen3-TTS fine-tuning script?**
Four silent bugs: labels shifted twice, the sub-talker reading the frame it predicts, `text_projection` skipped, and the speaker encoder decaying under AdamW. Both trainers here fix all four. The details, with the upstream code and issue links, are in the [fine-tuning guide](https://github.com/starryark/tricolour-tts/blob/main/docs/qwen3-tts-finetuning-fixes.md).

**Can I use it in an Amadeus-style assistant?**
It runs locally as an ordinary Qwen3-TTS `custom_voice` model. Label its output as synthetic, and keep to the rights section below.

**Can I use the voice commercially?**
No. No rights to the character or the voice are granted. The code is Apache-2.0 and can be reused on your own data.

## License

The code is Apache-2.0 (see `LICENSE`). Parts of `train/train_sft.py` and `legacy/v1_local/ft02_sft.py` derive from the Apache-2.0 Qwen3-TTS fine-tuning scripts (© Alibaba Qwen team).

The model weights and the dataset are separate releases with no rights grant. STEINS;GATE, its characters and the voice recordings belong to their respective owners. Do not use the voice to impersonate the voice actors or anyone else, or to mislead. Disclose synthetic audio as synthetic. Rights holders may contact the repository owner for removal.

## Citation

```bibtex
@misc{makise_kurisu_qwen3_tts_2026,
  title  = {Makise Kurisu AI voice: a Qwen3-TTS fine-tune},
  author = {starrydark},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/starrydark/Makise_Kurisu_Qwen3_TTS}},
  note   = {Code: \url{https://github.com/starryark/makise-kurisu-tts}}
}
```
