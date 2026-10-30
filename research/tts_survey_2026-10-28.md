# Open-weight TTS models for a single-character Japanese fine-tune (survey as of 2026-10-28)

How to read the tags: a cell or fact is **primary** (taken from a HF model card or HF API timestamp, a GitHub README, tree or release, a LICENSE file or an arXiv paper) unless it is tagged **[S]** (secondary) or **[U]** (unverified). Dates are HF `createdAt`/`lastModified` or GitHub release/commit dates. "No official FT" means no training or fine-tune code in the official repo tree.

## Main table

| Model | Latest version + date | Japanese support (evidence) | Official fine-tuning | Single-speaker FT results | Cross-lingual | Non-verbal tags | Weights license | Params / VRAM | Maturity | Key URL |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fun-CosyVoice3-0.5B-2512** | HF 2025-12-11 (modified 2026-02-03); no 2026 successor (Fun-CineForge 2026-03 is zh/en dubbing) | Listed. CV3-Eval ja CER 5.15 (with RL), 10.4 (without) | **Full SFT only** (LLM, flow, HiFi-GAN), lr 1e-5 constant; Kaldi-style lists converted to parquet; no LoRA | Paper §5.7 (speaker SFT, figure only); §5.8: a monolingual speaker made polyglot reaches JP CER ~9% | Yes: ja→en WER 4.41, ja→zh CER 3.22 | Built-in special tokens: `[breath] [laughter] [cough] [sigh] [quick_breath] [lipsmack] [noise] [mn] <strong>` | Apache-2.0 | 0.5B (the 1.5B is paper-only); VRAM [U] | 24.2k★, 653 open issues | huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 |
| **IndexTTS-2.5** | HF 2026-08-10; repo push 2026-09-29 | Listed; 42k h of JP training data; CV3 ja WER 5.30 (RL) | **None** (issues #185, #518 open; #501 is unofficial code) | Community LoRA only (e.g. Turkish) [S] | **Best zh→ja** in its own paper (6.38 / SS 75.8) | No; 8-dim emotion vector, kana `<word\|reading>`, speed control | bilibili license: commercial use allowed below 100M MAU / RMB 1B revenue; you may not use it to improve other AI models | ~0.8B GPT; ~6 GB; 22.05 kHz | 24.8k★, 413 open issues | huggingface.co/IndexTeam/IndexTTS-2.5 |
| **Fish Audio S2 Pro** | HF 2026-03-09; GitHub v2.0.0-beta 2026-03-10 | Tier-1 language. CV3 ja 3.96; MiniMax ja 2.76 / SIM 0.796 | LoRA (r8/α16) plus full FT, **only at tag v2.0.0-beta**; removed from `main` on 2026-10-13 | None found | Claims strong cross-lingual timbre (paper) | Free-form `[tag]`, "15,000+", incl. `[sigh] [inhale] [exhale] [pause] [short pause] [panting]` | **Fish Audio Research License**: no commercial use; the HF gate says non-commercial only | 4B + 0.4B; ≥24 GB; RTF 0.195 on H200 | 33.5k★ (repo) | huggingface.co/fishaudio/s2-pro |
| **Fish Audio Drama 3** | HF 2026-10-01; v3.0.0-beta 2026-10-13; int4 variant 2026-10-14 | Listed, but MiniMax ja CER **5.02, worse than S2's 2.76** | **Removed** ("Fine-tuning code removed from this repo") | Aratako/Drama-3-Japanese LoRA, below | Multilingual, 80 languages | Point tags `[laughs]`, scoped `<whisper>…</whisper>`, director's notes, infill editing | **Fish Audio Community License v1.0**: free under US$500k annual revenue; **any Hosted Service needs a separate license**. HF metadata still says "research-license" | 35B-A3B MoE; ≥48 GB FP8; int4 build 22 GB (A100 or newer) | Same repo | huggingface.co/fishaudio/drama-3 |
| **OpenAudio S1-mini** | HF 2025-05-31 (modified 2026-02-06), gated | Listed. JSUT CER 9.03, CV3-ja SIM 61.4 (Sarashina paper) | LoRA doc (fish-speech v1.5.x / v2.0 tag) | — | Degrades 113% with non-JP prompts | Emotion markers [U] | CC-BY-NC-SA-4.0 | 0.5B | — | huggingface.co/fishaudio/s1-mini |
| **GPT-SoVITS** | Latest GitHub release **20250606v2pro**; **V5** merged to code 2026-10-06 (changelog 2026-10-05), no release tag | Native (another agent covers it) | Full GPT + SoVITS training | — | en/ja/ko/yue/zh | — | MIT (code and lj1995 weights) | — | 64k★, 906 open issues | github.com/RVC-Boss/GPT-SoVITS |
| **F5-TTS** | Release 1.1.22 (2026-07-23); weights from 2024-10 | Base is zh/en. Community JP: Jmica/F5TTS (Emilia-JA 1.7k h + Galgame 5.4k h) | Full FT via CLI/Gradio [S] | — | Limited | No | CC-BY-NC-4.0 (official and Jmica) | 0.3B | 15.5k★ | github.com/SWivid/F5-TTS |
| **VoxCPM2** | HF 2026-04-03 (modified 2026-08-18); v2.0.3 2026-05-11 | One of 30 languages. CV3 ja 5.96; MiniMax ja 4.63 / **SIM 82.8** | **LoRA and full SFT, documented**: LoRA r32/α32, lr 1e-4; full lr 1e-5; JSONL `{audio,text,ref_audio?}` | Docs say 1–3 epochs suffice for one speaker; community JP single-speaker FT (Zundamon) [S, no metrics] | Yes, no language tag needed; zh→ja 11.84 / SS 71.5 | No documented non-verbal list; style via `(description)` prefix | **Apache-2.0** | 2B; ~8 GB inference; FT ~20 GB (LoRA) / ~40 GB (full); 48 kHz out | 40.8k★, 170 open issues | github.com/OpenBMB/VoxCPM |
| **Higgs TTS 3** (higgs-tts-3-4b) | HF 2026-06-04 (modified 2026-09-04) | One of 100+ languages; only macro averages published | **None** (issue #183 open) | — | Yes | `<\|sfx:sigh\|> <\|sfx:laughter\|> <\|prosody:pause\|> <\|prosody:long_pause\|> <\|style:whispering\|>`, 21 emotions | Research/non-commercial plus a **Creator Use Grant** (monetized content allowed with credit; hosting an API or embedding in a product is not) | 4B; RTF 0.147 on H100 | 8.4k★ (higgs-audio repo) | huggingface.co/bosonai/higgs-tts-3-4b |
| Higgs Audio v2 | HF 2025-07-01 | Not listed (en/zh/de/ko) | Community scripts (#94) [S] | — | — | — | Llama-3-derived; commercial below 100k annual active users | 3.6B + 2.2B | — | huggingface.co/bosonai/higgs-tts-2-3b-base |
| **VibeVoice** | MS removed the TTS code 2025-09-05; 1.5B weights still on HF; Realtime-0.5B 2025-12-04 | Not official; Realtime has an "experimental JP voice" | MS: ASR only. **Community fork**: LoRA, lr 2.5e-5, 1 epoch | — | — | No | MIT | 1.5B / 7B (fork mirror) | Fork 1.6k★; in Transformers since 2026-08-27 | github.com/vibevoice-community/VibeVoice |
| **MOSS-TTS v1.5** (+ Local-v1.5 4B, Nano-100M, TTSD-v1.0) | v1.5 2026-05-25; Local-v1.5 2026-06-18; Nano 2026-04-13; TTSD v1.0 2026-02-08; "2.0 coming soon" (2026-04-29) | Listed (31 languages). CV3 ja 10.97 / SS 68.7 | **Full SFT** (`sft.py`, FSDP/ZeRO-3); JSONL `{audio,text,ref_audio,language}`; community LoRA in repo | — | zh→ja 11.52 | `[pause X.Ys]` only | Apache-2.0 | 8B (llama.cpp path fits 8 GB) | 4.3k★ | github.com/OpenMOSS/MOSS-TTS |
| **Step-Audio-EditX** | HF 2025-10-29; JP/KO model 2025-11-28; more tags and GRPO code 2026-01-23 | Via a `[Japanese]` tag; no JP metrics | SFT/DPO/GRPO scripts (`scripts/run_edit_sft.sh`) | — | — | `[Breathing] [Laughter] [Sigh] [Uhm] [Surprise-oh/ah/wa] [Question-ei] [Dissatisfaction-hnn]` and more | Code Apache-2.0; **weights: no license file [U]** | 3B; 12–16 GB (AWQ 6–8 GB) | 1k★ | huggingface.co/stepfun-ai/Step-Audio-EditX |
| **FireRedTTS-2** | HF 2025-09-08; FT code 2025-10-26 | Listed. JSUT CER 9.33, CV3-ja SIM 66.2 | Official FT tutorial (`bin/finetune_example`) | — | zh→ja 19.05; +168% CER with non-JP prompts | [U] | Apache-2.0 | 1.5B; 9 GB bf16 | 1.4k★ | github.com/FireRedTeam/FireRedTTS2 |
| **FireRedTTS3** | Base 2026-08-05; Instruct 2026-08-13 | Listed (24 languages). MiniMax ja 3.60 (Drama 3 paper) / 3.42, SIM 0.825 (Ming card); JP text normalization needs an LLM | **None in repo** | — | Recommends a same-language prompt | Instruct editing | Apache-2.0 | [U] | 1.7k★ | huggingface.co/FireRedTeam/FireRedTTS3 |
| **Chatterbox Multilingual V3** | Weights 2026-04-22; release 2026-06-10 | One of 23 languages; no JP single-language pack | **None** in repo tree | — | Card warns the reference must match the language tag | Tags only in Turbo (350M) / Nano (110M), English-only | MIT | 0.5B | 27.4k★ | github.com/resemble-ai/chatterbox |
| **Kani-TTS(-2)** | kani-tts-2-en 2026-02-12; successor gepard-1.x (2026-06/09) is en/es/pt/nl | JP: kani-tts-370m-expo2025-osaka-ja; **kani-tts-400m-kuroyukihime-ja** (an anime character) | Finetune pipeline repo [S] | Kuroyukihime single-character model (no metrics) | — | — | kani-2: LFM1.0; JP models Apache-2.0 | 370–400M; 3 GB | 457★ | huggingface.co/nineninesix/kani-tts-400m-kuroyukihime-ja |
| **OmniVoice** | HF 2026-03-30; v0.2.1 2026-07-16 | One of 600+ languages. MiniMax ja 4.03 / SIM 0.828; CV3 ja 5.38 | Training and FT examples (`examples/`) | — | zh→ja 9.09 / SS 69.1 | `[laughter] [sigh] [surprise-ah/oh/wa/yo] [question-*] [dissatisfaction-hnn]` | **CC-BY-NC** (code Apache) | ~0.8B (Qwen3-0.6B base); RTF 0.025 | 16k★ | github.com/k2-fsa/OmniVoice |
| **ZONOS2** | HF 2026-06-11 | **Tier-1** language; no metrics | None in repo | — | — | [U] | Apache-2.0 (HF) | MoE, 15.3 GB checkpoint, parameter count [U]; Linux only | 321★ | huggingface.co/Zyphra/ZONOS2 |
| **Irodori-TTS v4.1-Small / v4-Large** (new, JP-only) | 2026-08-11 / 2026-09-27; community Anime FT 2026-09-04 | **JP-only**. JKYB Kana-CER 7.29; JSUT CER 7.22 | **Full FT and PEFT LoRA configs** | phasefield-audio/Irodori-TTS-v4.1-Anime (anime data) | **None** (Japanese only) | Emoji-based style and non-verbal control | Small: MIT; Large: Gemma | 766M / 3.29B; 48 kHz | 1.5k★ | github.com/Aratako/Irodori-TTS |
| **Sarashina2.2-TTS** (new) | HF 2026-04-16 (modified 2026-07-01) | **JP-centric** (54% of 361k h). CV3-ja SIM 75.6 (best in its paper) | [U] | — | JA↔EN; no degradation with EN prompts | — | **Non-commercial** | 0.5B LLM + CosyVoice2 flow/vocoder | — | huggingface.co/sbintuitions/sarashina2.2-tts |
| **Ming-3.0-tts-0.5B** (new) | HF 2026-09-22 | Listed. MiniMax ja 3.77 / SIM 0.765 | [U] | — | 7 languages | 14 inline tags (breath, cough, sigh, laughter…), zh/en | MIT | 0.5B | 357★ | huggingface.co/inclusionAI/Ming-3.0-tts-0.5B |
| **Llasa** (1B/3B/8B; 1B-Multilingual) | HF 2025-01/02; training repo push 2026-01-21 | 1B-Multilingual includes ja (from Emilia-JA); no metrics | **Official finetune instructions** (LLaSA_training/finetune) | Official `Llasa-1B-multi-speakers-genshin-zh-en-ja-ko` (game-character FT, no metrics) | BPE tokenizer, so any language | — | CC-BY-NC-4.0 | 1–8B | 657★ (training repo) | huggingface.co/HKUSTAudio/Llasa-1B-Multilingual |
| **Orpheus** | HF 2025-03/04; repo push 2025-12-05 | Multilingual research release has no ja; community JP: Gapeleon/slim-orpheus-3b-JAPANESE-ft, kadirnar/Orpheus-StoryTTS-JA [S] | **Official** `finetune/train.py` and `lora.py` | Many English single-voice LoRAs [S] | — | `<laugh> <sigh> <gasp>` etc., English model [S] | Apache-2.0 (some pretrained checkpoints under Llama 3.2) | 3B | 6.4k★ | github.com/canopyai/Orpheus-TTS |
| **NVIDIA Magpie-TTS Multilingual** | v2607 (2026-07-21); v2602 (2026-03); v2512 (2026-01) | ja listed; v2607 ja CER 1.40 / SSIM 0.775 (no v2602 ja row). Training data includes japanese-anime-speech | Via NeMo | — | 5 fixed speakers across 12 languages | — | NVIDIA Open Model License (commercial use allowed) | 364M | NeMo | huggingface.co/nvidia/magpie_tts_multilingual_357m |

Dropped or excluded, one line each: **Spark-TTS** (zh/en, training code still unreleased, CC-BY-NC-SA). **Sesame CSM-1B** (English only: "likely won't do well" on other languages). **Breeze-TTS-2** (BreezeBlue, 2026-08-25, en/zh, non-commercial; MediaTek has no "Breeze TTS 2" on HF). **Voxtral-4B-TTS-2603** (no Japanese, CC-BY-NC-4.0). **GLM-TTS** (zh/en, MIT, LoRA and GRPO code). Kokoro, Supertonic-3, MeloTTS, XTTS, Piper: dated or non-cloning.

## Japanese benchmark numbers (all self-reported unless noted)

1. **CV3-Eval multilingual, ja CER (%)**, from the Fish S2 paper Table 2 and the VoxCPM README: Fish S2 **3.96**; S1 4.54; CosyVoice3-0.5B with RL 5.15; CosyVoice3-1.5B with RL 5.27; VoxCPM2 5.96; CosyVoice2 with RL 6.36; CosyVoice3-1.5B 7.57; CosyVoice2 9.13; CosyVoice3-0.5B without RL 10.4 (CosyVoice3 paper Table 5).
2. **IndexTTS-2.5 paper (arXiv 2601.03888), test-ja WER / SS**:
   - Zero-shot: IndexTTS2.5-RL **5.30/75.4**; IndexTTS2.5 5.66/74.6; Fish S2 Pro 5.15/66.2; OmniVoice 5.38/70.5; Qwen3-TTS 5.89/70.2; VoxCPM2 6.69/72.9; MOSS-TTS 1.5 10.97/68.7; CosyVoice3-0.5B –/76.4.
   - Cross-lingual zh→ja: IndexTTS2.5-RL **6.38/75.8**; OmniVoice 9.09/69.1; Fish S2 Pro 10.48/61.7; MOSS 1.5 11.52/65.5; VoxCPM2 11.84/71.5; CosyVoice3-1.5B 13.70; FireRedTTS-2 19.05/64.1; **Qwen3-TTS 36.09/65.7**.
3. **MiniMax Multilingual Test, ja**:
   - Fish S2 paper: S2 **2.76 / SIM 0.796**; S1 3.45/0.780; MiniMax 3.52/0.776; ElevenLabs 10.65/0.738.
   - VoxCPM README: Qwen3-TTS 3.82/0.788; VoxCPM2 4.63/**0.828**.
   - OmniVoice paper: 4.03/0.828.
   - Drama 3 paper Table 6 (error rate only; columns Drama3 | OmniVoice | FireRedTTS3-Base | S2 | ElevenLabs v4 | Inworld TTS-2 | Soniox): 5.02 | 4.03 | 3.60 | 2.76 | 3.66 | 4.23 | 5.36.
   - Ming card (one evaluation run applied to all models): FireRedTTS3 3.42/0.825; Fish S2 3.43/0.755; Ming-tiny 3.43/0.780; Ming-0.5B 3.77/0.765; VoxCPM2 4.36/0.825; dots.tts 5.13/0.838.
4. **Sarashina2.2-TTS paper (arXiv 2606.25369)**, the only independent JP-focused comparison found:

   | System | JSUT CER | Joyo Kana-CER (kanji) | CV3-ja SIM | CER change with a non-JP prompt |
   |---|---|---|---|---|
   | T5Gemma-TTS | 7.63 | 13.8 | 50.6 | +18% |
   | Qwen3-TTS | 14.87 | 185.9 | 69.9 | **+328%** |
   | S1-mini | 9.03 | 33.4 | 61.4 | +113% |
   | FireRedTTS-2 | 9.33 | 27.8 | 66.2 | +168% |
   | Sarashina2.2-TTS (stage 2) | 8.02 | 7.83 | 74.75 | −0.2% |

5. **Aratako's JP benchmarks.** These use different benchmark editions, so compare only within a row.
   - Drama-3-Japanese vs Drama 3:
     - JKYB-Parakeet Kana-CER: 6.55 vs 7.47.
     - JVS one-reference CAM++ similarity: 0.728 vs 0.733.
     - Whisper CER: 3.08 vs 2.89.
   - Irodori v4.1-Small:
     - JKYB (original edition) Kana-CER 7.29.
     - JVS CAM++ similarity rises from 0.661 with one clip to 0.775 with 120 s of reference.
6. **Community [S].** One Zenn author (2026-06-18) compared six models on their own voice with acoustic proxies only. S2 Pro sounded most natural, but ran at RTF 124 on a 16 GB RTX 5060 Ti. VoxCPM2 had the best ASR match. Irodori was the fastest (RTF 0.12).

**Qwen3-TTS comparisons found:** IndexTTS-2.5 paper (zh→ja WER 36.09); VoxCPM README (MiniMax ja 3.82/0.788); Sarashina paper (JSUT CER 14.87, +328% CER with non-JP prompts, CV3-ja SIM 69.9); Higgs 3 card (macro CV3 7.73 vs Higgs 3 4.41); Zenn article above.

## Key findings per model

- **CosyVoice3.** The official recipe is full SFT, not LoRA (`examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`: lr 1e-5, constantlr, warmup 2500, accum_grad 2, deepspeed stage 5). Japanese is converted to kana internally. In the paper's monolingual-speaker-to-polyglot experiment, Japanese CER is about 9%, worse than the other languages (<4%) (https://arxiv.org/abs/2505.17589 §5.7–5.8). The tag list is in `cosyvoice/tokenizer/tokenizer.py`. A community `tts-dataset/cosyvoice3-ja-delta` exists but has no metrics [S].
- **IndexTTS-2.5.** It has the strongest JP and zh→ja numbers among the open models in its own paper and is light (~6 GB). However, there is no official fine-tune code, and it has no non-verbal tags. https://github.com/index-tts/index-tts/issues/518
- **Fish S2 Pro / Drama 3.** The fine-tune doc at tag v2.0.0-beta (https://github.com/fishaudio/fish-speech/blob/v2.0.0-beta/docs/en/finetune.md) says:
  - It "highly do[es] not recommend" fine-tuning an RL-trained model.
  - LoRA "will only learn the speaker's speech patterns and not the timbre" unless you train more steps, which risks overfitting.
  - Defaults: lr 1e-4, batch size 4, max_steps 10000, `r_8_alpha_16`.

  Drama 3's release removed the fine-tune code (https://github.com/fishaudio/fish-speech/releases/tag/v3.0.0-beta). Aratako's Drama-3-Japanese used its own LoRA setup: r32/α32, lr 2e-4 for LoRA, 2e-5 for embeddings and 5e-5 for the audio decoder, one epoch on "approximately 70,000 hours" of JP speech (taken verbatim from the card; it is a large-scale continued fine-tune, not a single-speaker one). https://huggingface.co/Aratako/Drama-3-Japanese
  - **License concern for a public ZeroGPU Space.** The Community License requires a separate license for "any Hosted Service" (LICENSE §I). "Hosted Service" is defined in §V, which I did not read, so check that definition before deploying a public Space. S2 Pro stays under the non-commercial Research License.
- **VoxCPM2.** It has the most complete documented fine-tuning path: LoRA and full configs, VRAM figures, manifest format, overfitting guidance, and a hot-swappable LoRA at inference. It is Apache-2.0 and supports cross-lingual output. https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html. Its weakness is JP accuracy, which is mid-pack (CV3 ja 5.96; MiniMax 4.36–4.63). It has no non-verbal tag set, so tags would have to be learned during fine-tuning [U whether that works].
- **Higgs TTS 3.** It has the best tag vocabulary (`sfx`/`prosody`/`emotion`). However, there is no fine-tune code, and the license is non-commercial apart from the creator grant. https://huggingface.co/bosonai/higgs-tts-3-4b
- **MOSS-TTS.** Official full SFT code exists, but the model is 8B and its JP numbers are weak (CV3 ja 10.97). https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_delay/finetuning
- **Step-Audio-EditX.** It has official SFT/DPO/GRPO scripts and the richest built-in Mandarin-style paralinguistic tags. JP was added 2025-11-28, with no metrics. The weights have no license file. https://huggingface.co/stepfun-ai/Step-Audio-EditX
- **Irodori-TTS.** This is the strongest JP-only option with official LoRA and full fine-tuning, an MIT license, and an anime fine-tune precedent. It has **no cross-lingual capability**. https://github.com/Aratako/Irodori-TTS
- **Sarashina2.2-TTS.** It has the best JP speaker similarity and zero cross-lingual degradation in the independent comparison, but it is non-commercial and fine-tune code is unconfirmed. https://arxiv.org/abs/2606.25369
- **Chatterbox V3.** It is MIT and 0.5B, but has no official fine-tune code and no JP tags. https://github.com/resemble-ai/chatterbox
- **Magpie.** Zero-shot cloning has been removed, and the model has five fixed English speakers. A custom voice would require full NeMo fine-tuning. https://huggingface.co/nvidia/magpie_tts_multilingual_357m
- **VibeVoice.** It survives only through the community fork, which has LoRA fine-tuning. There is no official JP support. https://github.com/vibevoice-community/VibeVoice/blob/main/FINETUNING.md

**Single-character JP fine-tune precedents found** (none publish metrics):
- nineninesix/kani-tts-400m-kuroyukihime-ja
- WariHima/voxcpm-1.5-ja-single-spk-f-zundamon-ita
- HKUSTAudio/Llasa-1B-multi-speakers-genshin-zh-en-ja-ko (official)
- phasefield-audio/Irodori-TTS-v4.1-Anime
- Jmica/F5TTS JA (Galgame data)
- Gapeleon/slim-orpheus-3b-JAPANESE-ft

## Could not verify

- Step-Audio-EditX weights license (no LICENSE file in the HF repo).
- ZONOS2 parameter count (only a 15.3 GB `model.pth`) and fine-tune code.
- FireRedTTS3 parameter count and training code.
- Sarashina2.2-TTS and Ming-3.0-tts fine-tune code.
- MOSS-TTS 2.0 (announced 2026-04-29, not released).
- Commercial terms of the LFM1.0 license for Kani-TTS-2.
- Whether Voxtral's open weights allow cloning arbitrary voices locally.
- GPT-SoVITS V5 formal release tag (V5 exists only as a commit and README notes).
- Inference VRAM for CosyVoice3 and Higgs TTS 3.
- Which Magpie version first added ja (v2607 added ar/ko/pt; ja has no v2602 metric).
- Any official or credible single-speaker LoRA results with metrics for S2 Pro, IndexTTS, Higgs, VoxCPM2 or CosyVoice3.
- Drama 3 HF license metadata ("research") contradicts its README and LICENSE ("Community").
