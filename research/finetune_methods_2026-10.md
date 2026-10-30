# Fine-tuning methods for Kurisu v3 (research as of 2026-10-28)

This report covers how we should fine-tune, not which model to use. It extends `final_report.md` and `tts_survey_2026-10-28.md` and does not repeat their model survey, trainer-bug list or dataset-prep advice.

**Tags:**
- **[P]** primary source: a paper, repo code, issue/PR, model card or config.
- **[S]** secondary source: a blog, forum post or community model card's claims.
- **[I]** my inference.
- **[U]** unverified.
- **[L]** our local files.

Dates are submission or commit dates. Some arXiv full-text numbers were read by research sub-agents; I spot-checked the load-bearing ones (#25, PR #361, CSP-FT, CosyVoice3 §2.6, LASE, macminix, 2610.29351).

## 0. Bottom line

1. **The v2 recipe is still sound for the Japanese voice.**
   - Full FT, fp32 master weights, lr 5e-6, ~1 epoch, with the fixed trainer.
   - Nothing published in 2026 beats full FT on target-speaker quality at ~10 h.
   - Its known weakness is behaviour outside the training data. On Fun-CosyVoice3, full FT tripled held-out WER and Chinese CER, while a two-layer partial FT kept them at baseline with equal similarity (CSP-FT) [P].
   - **For v3, the gains come from what we train on and how we condition, not from the adapter type.**
2. **The Japanese accent on Chinese is most likely structural** [I].
   - v2 trained every row through the `nothink` path with no language id, so "Kurisu row ⇒ Japanese phonetics" was the only pattern on offer.
   - VALL-E X's ablation is the cleanest evidence that removing the language id is what produces a foreign accent: EN→ZH accent score 4.03 → 2.35 [P].
   - Our ZH metric could not see the problem, because Qwen3-ASR was *forced* to Chinese [L].
3. **Qwen3-TTS already has the slots we need** [P code]:
   - speaker rows, which change zero offsets in our collate;
   - a language id, which adds +1 token;
   - a dialect-via-speaker map.

   It lacks two things:
   - **Base does not support instruct**, per the maintainer (#25).
   - **There is no CFG.**
4. **Conditions** [I, with §3 evidence]:
   - **Amadeus:** a second speaker row, never dropped out, with a DSP-EQ control arm.
   - **Phone (26 clips):** DSP only.
   - **`[sigh]` and `[gasp]`:** dedicated tokens plus splice augmentation, as a best-effort feature.
   - **`[breath]` (11 clean):** drop.
   - **`[pause]`:** keep as "……".
5. **The Chinese dub cannot fix Chinese pronunciation by SFT.**
   - Of its 374 clips, 257 are Amadeus. The clean, non-device part is about **8.5 minutes** from a different actress [L].
   - Use it as a native reference for evaluation and as a voice-conversion source.
   - The English dub (93 min clean) is a much better source.
6. **The ranked accent levers** (§4.6):
   - language-id training;
   - native-speaker replay with language ids;
   - Kurisu-timbre native pseudo-data made by voice conversion;
   - freezing the text embedding;
   - language-contrastive guidance at inference;
   - finally, RL.
7. **RL is feasible on one H100 as offline DPO, but it belongs last.**
   - Japanese CER already equals the real recordings (0.030 vs 0.032).
   - No paper uses a language-ID or accent reward for RL.
   - GRPO collapse and slowed, monotone speech are documented failure modes [P].
8. **Fix the measurement before training v3** [I]:
   - Add an unforced language-ID accent metric.
   - Use at least 50 probes each for ZH and EN.
   - Stop using ECAPA similarity as a ZH/EN guard. It is language-biased (−0.105 cosine for the same voice across scripts, LASE [P]), so it rewards the accent.

## 1. What Qwen3-TTS gives us to condition on

Read from `QwenLM/Qwen3-TTS@main` (`qwen_tts/core/models/modeling_qwen3_tts.py`, `qwen_tts/inference/qwen3_tts_model.py`) and the three 1.7B `config.json` files on HF, fetched 2026-10-28 [P]. https://github.com/QwenLM/Qwen3-TTS

| Channel | How inference builds it | Who was trained on it | Cost in our collate |
|---|---|---|---|
| **Speaker row** | One codec-embedding vector after the think block: `[nothink, think_bos, think_eos, SPK, pad, bos]` | Base: the speaker-encoder x-vector in that slot. CustomVoice: 9 learned rows (`vivian` 3065, `serena` 3066, `ono_anna` 2873, …) | **Zero offset change.** Ids 2048–3071 are all suppressed at sampling (`suppress_tokens` covers the last 1024 ids except EOS), so any free row there is safe |
| **Language id** | `[think, think_bos, LANG, think_eos]` replaces `[nothink, think_bos, think_eos]` | All three 1.7B models ship `codec_language_id` (japanese 2058, chinese 2055, english 2050, …) | +1 token: speaker slot 6 → 7, every `8+n_text` offset +1 |
| **Dialect via speaker** | If `language` is `chinese` or `auto` and `spk_is_dialect[speaker]` is set, the speaker forces its own language id (`eric` → sichuan, `dylan` → beijing) | CustomVoice | Config only: a precedent for a speaker row that implies a language id |
| **Instruct text** | `<|im_start|>user\n{instruct}<|im_end|>\n` as text embedding + `text_projection` only, **prepended before** the `<|im_start|>assistant\n` role tokens | CustomVoice 1.7B and VoiceDesign; forced off for 0.6B CustomVoice. **Base: "our base model does not currently support instruct"** (#25, 2026-01-26) | Variable-length prefix: every offset moves |
| **Inline text tokens** | Ordinary Qwen BPE; `[sigh]` splits into several pieces | — | The tokenizer's ids stop at 151675 but `text_vocab_size` is 151936. **260 embedding rows (151676–151935) exist unused**, so new tag tokens need no resize, but those rows are untrained |
| **CFG / guidance** | **None.** `generate()` has no guidance argument (defaults: temperature 0.9, top-k 50, rep. penalty 1.05) | — | Needs a second forward per step (§6) |

Other facts:
- `generate_custom_voice` and `generate_voice_design` default to `non_streaming_mode=True`, the layout our collate trains. `generate_voice_clone` defaults to streaming [P].
- The CustomVoice and VoiceDesign checkpoints have **no speaker encoder**; it is built only when `tts_model_type == "base"` [P]. Starting v3 from CustomVoice would mean learning a new row, and #213 reports that training a row directly made timbre "randomly change" per text [P].
- The official finetuning README still says Base "currently supports single-speaker fine-tuning", yet its own JSONL example mixes a Chinese and an English line [P].

## 2. Q1: Full fine-tuning vs LoRA / partial / prompt tuning at ~10 h

### 2.1 Evidence

| Source | Model, data | Result |
|---|---|---|
| **CSP-FT** (arXiv [2501.14273](https://arxiv.org/abs/2501.14273) v2, 2026-03-06) [P] | GPT-SoVITS, VALL-E X, CosyVoice, Fun-CosyVoice3-0.5B; 244 h emotional English, 10 epochs | **Fun-CosyVoice3**, SS / WER:<br>• Origin 91.2 / 4.0<br>• **Full FT 94.5 / 12.1**<br>• LoRA 93.0 / 9.2<br>• **CSP-FT 94.8 / 3.8** (2 probe-selected layers, ~6% of params)<br>Cross-dataset Chinese CER: Origin 1.2, Full 3.9, LoRA 2.1, CSP-FT 1.2.<br>**The same ordering holds on all four models.** |
| VoxCPM docs ([link](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)) [P] | VoxCPM2 | • LoRA r32 reaches "approximately 98%" of full-FT similarity.<br>• Full FT: lr 1e-5, best within 1–2 epochs.<br>• For a new language, mix in "10–20%" zh/en to reduce forgetting. |
| PEFT-TTS (Interspeech 2025) [P via sub-search] | F5-TTS (non-AR), 12 h KSS | • Full FT: CER 5.67 / SIM 0.659.<br>• LoRA r16: CER 4.91 / SIM 0.640, with better NMOS.<br>• Full FT lost zero-shot ability (multi-speaker CER 13.45 vs 7.22). |
| NeuTTS LoRA study ([2603.10904](https://arxiv.org/abs/2603.10904), 2026-03-11) [P] | r8, 3.5–18 h per speaker | • SIM +0.05 to +0.11.<br>• Loss falls monotonically while DNSMOS degrades, so select by perceptual metrics. |
| "LoRA Learns Less and Forgets Less" ([2405.09673](https://arxiv.org/abs/2405.09673)) [P] | LLMs | The general pattern behind all of the above. |
| "A Deafening Silence" ([2610.09835](https://arxiv.org/abs/2610.09835), 2026-10-07) [P] | LLMs | • Forgetting concentrates in the output-embedding rows of tokens absent from the new data.<br>• Raising Adam ε on `lm_head` alone removes 39–68% of it.<br>• Our codec vocabulary is shared by all languages, so this probably explains little of the accent [I]. |
| **v2 (ours)** [L] | Qwen3-TTS 1.7B, 10 h | • Full FT ZH CER 0.067 vs LoRA r32 0.167, at equal JA quality.<br>• The LoRA used 6× the lr for 2 epochs, so this is not a clean "LoRA forgets less" test. |
| Qwen community [P/S] | Qwen3-TTS | • notmax123/QwenTTS-he-1.7B: "Do not run Chinese / English … with the adapter active"; adapter-off output is bit-identical to Base.<br>• #213: training the speaker row directly destabilised timbre. |

**Gaps:**
- No DoRA result for TTS was found [U].
- No embedding-only or speaker-prompt tuning study on an AR codec LM was found [U].

**The community's "tiny lr" advice does not transfer to our recipe.**
- The official `sft_12hz.py` trains bf16 weights under plain AdamW.
- The lr 1e-7 / 1e-8 reports come from such runs, e.g. #314: 7 h of Japanese, rated "better" while loss stayed at 14–15. In bf16, updates that small mostly round to zero.
- Meorav-TTS says so explicitly ([repo](https://github.com/Hebrish-ASR/Meorav-TTS)) [S].
- Runs that use the fixed trainer with fp32 master weights work at 5e-6 to 1e-5 [P/S]:
  - the PR #278 thread;
  - [sabari245](https://github.com/sabari245/qwen3-tts-audiobook-finetune): 8.5 h at lr 1e-5; best between epochs 1 and 3.

### 2.2 Verdict [I]

- **Keep full FT, lr 5e-6, fp32 masters, about 1 epoch over the Japanese portion.**
- It is the strongest option for the target voice. It is also the weakest for anything the data does not cover, which is why §4 adds regularising data and an optional frozen text path.
- Run two partial-FT ablation arms (E3f-text, E3f-layers in §9.4):
  - freeze only `text_embedding` and `text_projection`;
  - freeze those plus the bottom 14 of the 28 talker layers. This is the arm closer to CSP-FT's evidence.
- **LoRA is now only a mechanism.** It gives an adapter you can switch off or scale per language. The full-FT delta can be scaled the same way by weight interpolation (§4.4).

## 3. Q2: Multi-condition and instruction-conditioned SFT

### 3.1 How the big systems train control

Who checked what:
- **Read myself:** the Qwen3-TTS and CosyVoice 3 rows.
- **Read by a sub-search:** the CosyVoice 2, IndexTTS2, Fish S2, Step-Audio-EditX and Seed-TTS rows. Read their [P] as "[P via sub-search]".

| System | How conditions are injected | Data | Dropout / masking | Notes |
|---|---|---|---|---|
| **Qwen3-TTS** CustomVoice / VoiceDesign ([2601.15621](https://arxiv.org/abs/2601.15621), 2026-01-22) [P] | Instruct as a text-only prefix; speaker rows; dialect ids in the think block | Not disclosed | "probabilistically activated thinking pattern during training" | Order: pre-train S1–S3 → DPO → GSPO → "lightweight speaker fine-tuning on the base model". The CustomVoice card still recommends "each speaker's native language for the best quality" |
| **CosyVoice 2** ([2412.10117](https://arxiv.org/abs/2412.10117)) [P] | Instruction before `<|endofprompt|>`; inline `[laughter]`, `[breath]`, `<strong>` | 1,500 h instruct | **Multi-speaker SFT**: "Speaker A<|endofprompt|>"; unlabeled rows get an explicit `unknown<|endofprompt|>` | mSFT "mitigates potential catastrophic forgetting"; lr 1e-5 |
| **CosyVoice 3** ([2505.17589](https://arxiv.org/abs/2505.17589)) [P] | Natural-language instruction carrying speaker id **and language id** ("You are Speaker B. Please speak German.") | Instruct 5,000 h, "over 100" styles | Speaker SFT mixed with instruct data; they "randomly mask the speaker prompt or the style prompt" (rate not given) | §2.6.1 "auxiliary training dataset … studio-quality monolingual data from randomly-selected speakers" made a monolingual speaker polyglot: zh/en/de/es/fr/it/ru below 4%, **ja about 9%**, ko about 6%; mix ratio not given |
| **IndexTTS2** ([2506.21619](https://arxiv.org/abs/2506.21619)) [P] | Separate speaker and emotion conditioners; GRL keeps emotion timbre-free; Qwen3-1.7B maps text to an emotion vector | 135 h emotional speech, 361 speakers, 7 categories; 1,000 text→emotion pairs for the LoRA'd Qwen | Duration embedding zeroed 30% | Three stages; conditioners frozen in stage 3 |
| **Fish S2** ([2603.08823](https://arxiv.org/abs/2603.08823)) [P] | Free-form inline tags from a rich-transcription ASR (fine-tuned Qwen3-Omni) | Tagged hours not given | None reported | RL penalises missed tags (§5) |
| **Step-Audio-EditX** ([2511.03601](https://arxiv.org/abs/2511.03601)) [P] | Edit instructions | Emotion/style: one ~10 s actor clip per style, cloned, margin-filtered. Paralinguistics: NVSpeech real tagged audio as target | None reported | "A small set of quadruplet data is sufficient" (no number) |
| **Seed-TTS** ([2406.02430](https://arxiv.org/abs/2406.02430)) [P] | "additional speaker index token" | 5 speakers, 1–10 h each | — | Joint SFT: WER 3.15 → 2.83, SIM 0.779 → 0.789 |
| Qwen community [S] | SpragAI/qwen3-tts-emotion-tags: 9 inline tags on CustomVoice, `codec_embedding` frozen. TaoLiveAIGC TBLive: `<sil_L2…L6>`, `[speed_e2]` | SpragAI: 76,862 rows | — | "A tag conditions the whole utterance wherever it appears." No metrics |

### 3.2 Data needed per condition

- **No paper gives a minimum example count per tag for single-speaker codec-LM fine-tuning [U].** Do not treat any threshold, including the earlier report's "150–300", as established.
- **NonverbalTTS** ([2507.13155](https://arxiv.org/abs/2507.13155), 2025-07) [P]:
  - CosyVoice-300M, LM-only SFT, 25 epochs, lr 1e-5.
  - Tag counts: breath 3,612, laugh 1,018, cough 233, sigh 145.
  - Per-tag Jaccard: breath ≈ 0.91, laugh ≈ 0.23, cough ≈ 0.17–0.20.
  - **~200 multi-speaker cough examples were learned poorly.**
- **ISCSLP 2026 NVV challenge winner** ([2609.14231](https://arxiv.org/abs/2609.14231), 2026-09-13) [P]:
  - Each tag is one dedicated token, and tag frames are labelled "continue" for stop prediction.
  - Rare tags got +1,000 synthetic examples plus √-frequency resampling: EN cry 37 → +1,000 (ΔS +23.3); EN yawn 17 → +1,000 (ΔS +2.95).
  - Best-of-5 with a multi-metric selector cut EN WER 3.15 → 2.19.
- **Affectron** ([2603.14432](https://arxiv.org/abs/2603.14432), ACL 2026 Findings) [P]:
  - Starts from VoiceCraft (330M), a codec LM pretrained on verbal speech only.
  - Fine-tuned with NV clips **spliced** into 100 h of speech (~4 h of NVs over 15 types), with emotion-matched choice and placement.
  - NV accuracy 10.5% → 58.8% with splicing alone, 37.8% for the full method. Verbal WER improved 9.05 → 6.59 and SECS was nearly unchanged.
  - **Splicing is the best-evidenced way to teach NV tokens to a verbal-only codec LM.**
- **Our clean counts are far below all of these** (§8): 27 sigh, 20 gasp and 11 breath worded Kurisu examples, plus 68 sigh-only and 9 gasp-only takes.

### 3.3 Condition dropout

- **Published rates are mostly 10–30%, and mostly for CFG** [P]:
  - Koel-TTS: 10%.
  - VoXtream2: 10% each on text, prompt and speaker.
  - IndexTTS2: 30% on the duration embedding.
  - CosyVoice3 masks speaker and style prompts at an undisclosed rate.
- **Dropout exists to train an unconditional branch.** It is only safe when the unconditional output is acceptable.

### 3.4 Keeping the default voice clean when some data has device effects

| Source | Finding |
|---|---|
| Parler-TTS ([2402.01912](https://arxiv.org/abs/2402.01912), 2024-02) [P] | SNR and C50 binned into words in the description. Only ~1% of 45k h was clean, yet prompting "very good recording" gave PESQ 3.84 (GT 4.15) and MOS 3.92 (GT 3.67) |
| Audiobox Aesthetics ([2502.05139](https://arxiv.org/abs/2502.05139), 2025-02, §5) [P] | **Quality prompts beat filtering.** Filtering at p50 raised speech WER 2.95 → 5.06; prompting kept 2.80. "Improving audio quality generation using prompting is a better option compared to filtering" |
| Hsu et al., ICASSP 2019 ([link](https://research.google/pubs/pub47834/)) [P] | When speaker and noise are correlated, clean output for all speakers needed (a) a factorised condition, (b) augmentation adding the noise to other speakers' data, and (c) an adversarial term |
| Environment-aware TTS ([2110.03887](https://arxiv.org/abs/2110.03887)) [P] | Separate speaker and environment embeddings disentangle the two |
| x-vector probing ([1909.06351](https://arxiv.org/abs/1909.06351)) [P] | "x-vectors capture spoken content and channel-related information" |
| Parler-TTS Greek ([2609.10022](https://arxiv.org/abs/2609.10022), 2026-09-09) [P] | Varied LLM-written style prompts caused "speaker drift"; fixed prompts fixed it |

### 3.5 What this means for our conditions [I]

**Amadeus (920 JA clips ≈ 0.95 h, ~11% of rows)**
- **Mechanism:** a second speaker row (`amadeus`, e.g. row 3001), filled with the frozen speaker encoder's mean x-vector over clean Amadeus takes.
  - It needs zero offset changes.
  - It follows Qwen's own CustomVoice design.
  - The x-vector already carries channel information, so the row starts out "pointing at" the filter.
- **No dropout.** Never train an Amadeus clip under the `kurisu` row, because that is exactly what teaches the default voice to sometimes emit the filter.
  - This is the CosyVoice-2 "explicit `unknown` token" principle: every row says which condition it is.
- **Decorrelation (Hsu-style), optional:** add ~200–400 clean Kurisu clips passed through a fitted Amadeus EQ and labelled `amadeus`. The filter then becomes the row's job and is not tied to Amadeus-scene content.
- **DSP control arm:**
  - Fit a linear EQ, the long-term average spectrum ratio of Amadeus to Kurisu takes, and apply it to v3's default output.
  - An ~18 dB tilt difference is mostly a linear filter, so this may be indistinguishable from the learned variant at zero leakage risk.
  - Ship whichever wins a blind A/B.
- **Instruct is not the right channel:**
  - Base never learned it (#25).
  - Fine-tuned Bases ignore it (#121).
  - It shifts every offset.
  - It would need far more than 920 clips to bind from scratch.

**Phone / `[digital]` (26 JA phone takes)**
- Too few to learn. Implement as DSP: 300–3,400 Hz band-pass, light compression, optionally a narrow-band codec round-trip.
- If it must be a model condition later, it can be one more speaker row with EQ-augmented clean clips, as for Amadeus.

**Event tags**
- **Tokens:**
  - Add `<|sigh|>` and `<|gasp|>` as special tokens on unused text rows 151676+, so no resize is needed.
  - Initialise each from the text embedding of a matching onomatopoeia (はぁ, っ).
  - Train them inline (full FT updates `text_embedding`).
- **Data:**
  - The clean worded examples (27 / 20).
  - Tier-C rows whose transcripts verify.
  - Splice augmentation: put the 68 sigh-only and 9 gasp-only takes before or after ordinary utterances, with natural gaps (Affectron).
- **Gasp:** has too few distinct takes (≈ 30); treat it as experimental.
- **`[breath]` (11):** drop the tag and keep the audio untagged.
- **`[pause]`:** stays "……", which v2 already renders.
- **No tag dropout.** Untagged sigh audio teaches spontaneous sighs.
- **Watch stopping.** v1's runaways clustered on tag prompts. The NVV winner labels tag frames as "continue", so the model must not treat a sigh as the end of the utterance.
- **Ship rule:** a tag ships only if sigh realisation is ≥ 50% on tagged probes, with no rise in runaway rate and no rise in spontaneous sighs on untagged probes. Otherwise fall back to kana.

## 4. Q3: Forgetting and native pronunciation

### 4.1 Why v2 sounds Japanese in Chinese [I, ranked by my belief]

1. **No language id.**
   - All training rows took the `nothink` path, the same path `Auto` uses at inference.
   - In 10.7 h of training, the only pattern was "Kurisu row ⇒ Japanese phonetics and prosody", and nothing in the input told the model when to stop applying it.
   - VALL-E X shows what removing the language id does to accent (below) [P].
2. **Text-embedding drift on shared Han characters.**
   - Qwen's BPE gives most Han characters single tokens shared by Japanese and Chinese.
   - Japanese-only full FT pushes those rows toward Japanese readings and prosody.
   - Not measured for Qwen3-TTS [U].
3. **The speaker vector carries accent.**
   - x-vectors encode language and channel, so a vector averaged over Japanese takes partly encodes "Japanese".
   - The same entanglement is what makes zero-shot cross-lingual cloning leak accent (AAG: "the accent of the reference leaks into the target speech") [P].
4. **The metric was blind** [L]:
   - ZH CER came from Qwen3-ASR forced to Chinese.
   - The ZH/EN guard compared against a Base clone run with `language="Auto"`, which garbled Chinese (CER 0.88). That is a very low bar.

### 4.2 Evidence on the levers

| Lever | Evidence | Strength for us |
|---|---|---|
| **Language id** | **VALL-E X** ([2303.03926](https://arxiv.org/abs/2303.03926), Table 6) [P]: accent score (1–5, 5 = native)<br>• EN→ZH: **4.03** with correct LID, **2.35** without, 2.25 with a wrong one<br>• ZH→EN: 4.10 / 2.98 / 2.55<br>ASV similarity moves the other way: 0.48 vs 0.49. "Learning to Speak Fluently" ([1907.04448](https://arxiv.org/abs/1907.04448)) [P]: with speaker perfectly correlated with language, a language embedding plus a speaker-adversarial loss gave native or foreign accent on demand | Strong mechanism; architecture-matched (AR codec LM) |
| **Native-speaker replay with speaker and language ids** | • CosyVoice3 §2.6.1: monolingual speaker → polyglot via auxiliary monolingual data from other speakers, every row labelled with speaker and language [P].<br>• CosyVoice2 mSFT "mitigates potential catastrophic forgetting" [P].<br>• VoxCPM: 10–20% zh/en mix [P].<br>• Older lifelong-TTS work: replay beats plain FT ([2110.04482](https://arxiv.org/abs/2110.04482), [2103.14512](https://arxiv.org/abs/2103.14512)) [P].<br>• **No 2025–26 replay-ratio study for codec-LM TTS** [U] | Strong for preservation. By itself it does not break speaker↔language entanglement (the "standard polyglot" baseline still leaks accent; see the next row) |
| **Target-voice native pseudo-data via VC** | • Amazon VC-polyglot ([2309.08255](https://arxiv.org/abs/2309.08255), 2023) [P]: convert native target-language speakers into the target voice, then train. Against standard multi-speaker polyglot training, MUSHRA accent similarity 71.06 vs 66.26 (es-MX) and 75.84 vs 73.97 (fr-CA), speaker similarity on par or better.<br>• The 2026 closed-loop dual-reference paper ([2610.29351](https://arxiv.org/abs/2610.29351), 2026-10-21) also builds its native anchor by VC [P].<br>• Amazon's 2026-04-01 blog: accent leakage is "the" polyglot problem; fix with locale-weighted data plus LoRA [S] | The only lever that directly gives "Kurisu timbre + native Chinese" examples |
| **KL to base / self-distillation** | • macminix/qwen3_voice_design_t8 [S, verified card]: KL to the frozen base on codebook-0 logits over neutral minibatches, **β 0.02 with a 20% mix ("0.01 / 10% proved too weak")**; WER 0.0215 → 0.0187 while emotion accuracy rose.<br>• RL's Razor ([2509.04259](https://arxiv.org/abs/2509.04259)) [P]: forgetting tracks forward KL to the base.<br>• Retaining by Doing ([2510.18874](https://arxiv.org/abs/2510.18874)) [P]: approximately on-policy data, regenerated per epoch, reduced forgetting more than data from the initial policy. **No KL-regulariser ablation for TTS speaker SFT exists** [U] | Medium. Cheap: a bf16 base copy is 3.4 GB |
| **Partial freezing** | • CSP-FT (§2.1) [P]: two selected layers kept Chinese CER at baseline where full FT tripled it.<br>• UtterTune ([2508.09767](https://arxiv.org/abs/2508.09767)) [P]: frozen CosyVoice2 + LoRA + 2 new token embeddings fixed JA pitch accent (0.472 → 0.975).<br>• Freezing text embeddings specifically: mechanism only [I] | Medium, cheap to test |
| **Weight interpolation / task vector** | Accent Vector ([2603.07534](https://arxiv.org/abs/2603.07534), 2026-03-08) [P]: LoRA r16 on XTTS-v2 fine-tuned on native speech of language X. θft − θpre carries the accent; scaling it by α ∈ [0, 1] raises accent monotonically (English-accent probability in Spanish 1.2% → 44.7%) while WER rises. **v2's delta is a "Japanese accent vector" as much as a "Kurisu vector"** [I] | Inference-only; trades timbre for accent |
| **Inference guidance** | §6: language-contrastive guidance, dual-reference guidance (accent LID mass −73%, ECAPA −0.042), AAG | Medium, needs a custom decode loop |
| **RL with an LID reward** | §5: none published. KIT on Fish S2: language tags raised the LID probability but the accent persisted | Low priority |

### 4.3 Qwen3-TTS-specific evidence

- **Tech report Table 9** [P]:
  - Qwen's own "lightweight speaker fine-tuning" of Aiden was "exclusively on monolingual data". The 12Hz-1.7B-CustomVoice WER is still zh 0.903, en 0.899, ja 4.924.
  - Only WER is reported. That is the same blind spot as our forced-language CER.
- **Accent persists after FT into new languages even with lots of data** [P]:
  - #27: Turkish, 18 h: "Training doesn't help".
  - Discussion #348: Shona, 120 h: "foreign accent".
  - #323: Bengali, 7 h: "Hindi-influenced".
  - All of these used the `Auto` layout.
- **Language-id ports** [P/S]:
  - PR #361 (2026-08-28, open, untested) pads Auto rows to 6 slots and so **trains an Auto layout inference never produces**. Port with per-row offsets instead.
  - vadimbelsky/qwen3-TTS-KSA: a new language id on row 2072, initialised to the mean of the existing language rows; works; no metrics.
  - jmtl/hk-cantonese-cc0-qwen3-tts (2026-10-25): a two-phase LoRA with `language="chinese"`.
  - **Nobody has measured accent before and after language-id training on Qwen3-TTS** [U].
- GPT-SoVITS's wiki says large single-language data wipes cross-lingual ability (prior report §1) [P].

### 4.4 Weight interpolation: a zero-cost probe [I]

- θ = θ_base + α·(θ_v2 − θ_base) for ZH/EN inference.
- **Interpolate every parameter except the speaker rows.** Copy v2's `codec_embedding` row 3000 (and any later `amadeus` row) unchanged. Base has no trained vector in those rows, so a naive interpolation would also shrink Kurisu's speaker vector toward an unused row.
- Sweep α ∈ {0.25, 0.5, 0.75}.
- Measures how much accent and timbre live in the same weights.
- If α ≈ 0.5 keeps most of the Kurisu timbre with a clearly lower accent, it is a cheap serving fallback (two weight sets, or a LoRA scale).

### 4.5 Measuring accent

- **Whisper-large-v3 language-ID logit** on unforced audio [P]:
  - AAG: AUC 0.995 native vs L2; Spearman 0.59 with expert ratings.
  - For us: P(zh) on ZH probes and P(en) on EN probes. Watch for saturation.
- **VoxLingua107 ECAPA LID probability** [P]: the proxy used by KIT IWSLT 2026 and by the Accent Vector paper.
- **Qwen3-ASR with `language=None`** returns the detected language. Its LID accuracy is 97.9% averaged over four sets [P, [card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)]. Use it as a hard-label rate.
- **An LLM accent judge** (AAG used blind Gemini 2.5 Pro, 1–5: Spearman 0.74 with experts, κ 0.92) [P]. Optional, and it needs an API.
- **Native listeners**, for the final two candidates only.
- **Do not use ECAPA SIM against the Japanese centroid as a ZH/EN guard.**
  - ECAPA loses 0.105 cosine for the same voice across scripts (LASE, [2605.00777](https://arxiv.org/abs/2605.00777), 2026-05-01) [P].
  - Every guidance paper shows that making the accent more native costs measured similarity (VALL-E X, AAG, 2610.29351) [P].
  - Keep SIM as a floor on ZH/EN (≥ the Base x-vector clone) and maximise it only on JA.

### 4.6 Ranking for our case [I]

1. **Language-id training** on all Kurisu rows, with ~25% of JA rows left as `nothink`.
   - Cheapest; in-distribution for Qwen; strongest mechanistic evidence.
   - Probably insufficient alone, because the Kurisu row still only ever sees Japanese.
2. **Native-speaker replay with language ids**, ~15% of rows.
   - Per-clip x-vectors in the speaker slot: the Base's own voice-clone training regime, so no new rows are needed.
   - This teaches "language id ⇒ phonetics" independent of speaker, and anchors the non-Japanese behaviour.
   - **Sources:** Base self-generated ZH/EN from native reference speakers (closest to on-policy and KL-to-base), or real AISHELL-3 (Apache-2.0, 85 h, 218 speakers, [OpenSLR 93](https://www.openslr.org/93/)) and LibriTTS-R. Emilia-YODAS is CC-BY but has only 0.3k h of Chinese [P].
3. **Kurisu-timbre native pseudo-data** under the Kurisu row with zh/en ids, ~5–10% of rows.
   - **Route (a), if experiment E0 shows Base + Kurisu x-vector + `language="Chinese"` already sounds native:** Base self-generation.
   - **Route (b), otherwise:** VC.
     - Train RVC (MIT [U]) or Seed-VC (GPL-3.0 code [S]; it is used only as a tool) on Kurisu's 10 h.
     - Convert the EN dub (93 min clean, in-domain acting), the ZH dub (8.5 min clean) and expressive native Chinese, e.g. CustomVoice `vivian`/`serena` output.
   - **Filters:** unforced LID = target; CER ≤ 5%; ECAPA to the Kurisu centroid above the real-take p25; Whisper P(lang) ≥ 0.9.
4. **Partial freezing.** Two cheap ablation arms (text path only; text path plus the bottom half of the talker). Adopt only if JA is unchanged.
5. **KL to base on replay batches** (β 0.02 on codebook-0). Add it only if replay alone still drifts.
6. **Language-contrastive guidance** at inference (§6). Needs #1.
7. **RL with an LID term** (§5). Last.

## 5. Q4: RL and preference post-training (2025–2026)

### 5.1 What the papers did and measured

All numbers are as each paper reports them.

| Method | Algorithm, rewards | Data / hyper-parameters | Measured gain | Code |
|---|---|---|---|---|
| **CosyVoice3 DiffRO** ([2505.17589](https://arxiv.org/abs/2505.17589)) [P] | Back-prop through Gumbel-softmax token samples into a token-level "ASR on tokens" reward, with token-level KL. A multi-task variant adds SER, MOS and AED. **No language-ID reward** | not given | • CV3-Eval 0.5B: **ja CER 10.4 → 5.15**, ko 12.8 → 4.02, zh 3.89 → 2.89, en 5.24 → 3.68.<br>• SIM falls slightly ("hacking").<br>• DiffRO-EMO: happy 0.64 → 0.98, but it "can hurt pronunciation" | DiffRO not released. The repo has `examples/grpo/cosyvoice2` (veRL GRPO, zh CER 4.08 → 3.36) and `run_dpo.sh` |
| IndexTTS-2.5 ([2601.03888](https://arxiv.org/abs/2601.03888)) [P] | GRPO on the text-to-semantic LM; 4 samples per prompt; WER + SIM | not given | • ja WER 5.66 → 5.30, SS 74.6 → 75.4<br>• en WER 5.12 → 3.89, SS −0.27 | none |
| Seed-TTS ([2406.02430](https://arxiv.org/abs/2406.02430)) [P] | REINFORCE, WER + SIM; a separate SER run | not given | • zh WER 1.115 → 1.002, en 2.249 → 1.945; CMOS +0.14.<br>• **Reward hacking: slower, over-articulated, "standardized" speech** | no |
| **Koel-TTS** ([2502.05236](https://arxiv.org/abs/2502.05236)) [P] | DPO/RPO. Pairs come from the Pareto front of CER and SSIM; pairs whose "chosen" side is worse on either metric are dropped | 58k pairs × 6 samples; lr 2e-7; β 0.01; ≤ 4k iterations at batch 64 | • Unseen speakers: CER 2.56 → 0.62, SSIM 0.601 → 0.645.<br>• **Ground truth as "chosen" degenerated (CER > 90%) or did nothing** | NeMo |
| NeMo Magpie PO docs ([link](https://github.com/NVIDIA/NeMo/blob/main/docs/source/tts/magpietts-po.rst), 2026-08-11) [P] | **DPO:** lr 2e-7, β 0.01, group 6, chosen only if CER ≤ 0.01.<br>**Online GRPO** ("generally recommended"): 12 samples; β 0; reward 0.45 CER + 0.45 SSIM + 0.1 PESQ; lr 1e-7; half the rollouts with CFG 2.5 | Dropout off; validate every 50 steps | — | open |
| GLM-TTS ([2512.14291](https://arxiv.org/abs/2512.14291), 2025-12-16) [P] | GRPO with DAPO tricks. Rewards: CER, SIM, emotion, laughter, pitch/energy variance | `GRPO_cer.yaml`: lr 1e-6; 8 samples; cer 1.0 / sim 0.1; KL 0; 1,000 steps | • zh CER 1.03 → 0.89.<br>• **RL on an SFT'd model:** CER 2.13 → 2.09, EMO 0.695 → 0.705.<br>• Aggressive settings slurred pronunciation | open |
| Multi-reward GRPO ([2511.21270](https://arxiv.org/abs/2511.21270)) [P] | Llasa 1–8B. Rewards: intelligibility, SIM, length-ratio window, entropy, LLM-annotated prosody | G 12, lr 1e-6, 8×H20 | • 8B: zh CER 1.59 → 1.10, SIM 0.684 → 0.758, MOS 3.67 → 4.12.<br>• The prosody reward gave the largest MOS gain | no |
| GRPO CER+NLL ([2509.18798](https://arxiv.org/abs/2509.18798)) [P] | Whisper CER and NLL | 4k sentences, G 8, lr 1e-5, β 0.1 | CosyVoice2 **ja CER 12.45 → 9.09**, SIM 0.635 → 0.656 | claimed |
| GRPO vs DiffRO ([2509.18569](https://arxiv.org/abs/2509.18569)) [P] | ASR accuracy, duration vs group median, token/F0 diversity | G 8, KL 0.1 | • zh WER 4.28 → 3.33.<br>• **GRPO collapsed after ~1.5k steps; the model slowed down to game the ASR** | no |
| Fish S2 ([2603.08823](https://arxiv.org/abs/2603.08823)) [P] | Dr.GRPO. Rewards: rich-transcription ASR, quality model, voiceprint SIM. **The policy is rsLoRA r16 on MLPs; the reference is simply LoRA-off** | not given | Reward curve only | no |
| Step-Audio-EditX ([2511.03601](https://arxiv.org/abs/2511.03601)) [P] | PPO with a 3B token-level reward model | The repo also has DPO (lr 1e-6, β 0.1) and GRPO with CER/SIM/emotion/LLM-judge rewards | No RL-vs-SFT numbers | open |
| "No Verifiable Reward for Prosody" ([2509.18531](https://arxiv.org/abs/2509.18531), ICASSP 2026) [P abstract] | GRPO on CER/NLL "collapses prosody into monotone"; adding SIM "further destabilizes" | Fix: iterative DPO with a few hundred human-labelled pairs per round | Best ELO | — |
| CosyVoice 2 speaker E ([2412.10117](https://arxiv.org/abs/2412.10117)) [P] | Single-speaker DPO plus an ASR loss | 10k pairs | • WER: SFT 7.15 → 6.64 with both added; SIM flat.<br>• **DPO alone hurt the hard set** (8.31 vs 7.90) | — |
| Qwen3-TTS ([2601.15621](https://arxiv.org/abs/2601.15621)) [P] | Human-feedback DPO, then GSPO with rule-based rewards | not given | — | No RL code. No DPO/GRPO/CFG issue or PR in the repo |

### 5.2 Language and accent rewards

- **No TTS paper uses a language-ID or accent classifier as an RL reward** [U after targeted searches].
- **Closest work:**
  - [2606.19381](https://arxiv.org/abs/2606.19381) (2026-06-14) [P]: DPO on CosyVoice 2, with a critic that subtracts a Whisper-LID code-mixing index. MER 13.2 → 10.3.
  - KIT, IWSLT 2026 ([2606.07240](https://arxiv.org/abs/2606.07240), Fish S2 Pro) [P, full text via sub-search]:
    - Native-script language tags raised the LID confidence slightly (zh 90.99 → 92.13) but worsened zh CER (11.37 → 12.05).
    - GRPO on (1 − CER + SSIM) then took zh CER to 10.99.
    - **The source-language accent persisted.**
- **Every guidance paper trades accent against similarity** (§6), so an LID reward needs a similarity term or a floor, measured on JA [I].

### 5.3 Practicality on one H100 for our 1.7B model [I unless tagged]

**Memory:**
- The full-parameter policy needs about 27 GB (fp32 weights plus Adam). v2 peaked at 37 GiB including activations.
- A bf16 reference copy is about 3.4 GB. Alternatively, follow Fish: train a LoRA policy and use adapter-off as the reference.
- Qwen3-ASR-1.7B, ECAPA and an LID model add about 5 GB.
- **It fits.**

**Rollouts are the bottleneck:**
- vLLM-Omni serves 1.7B-CustomVoice at 33.5 audio-s/s at concurrency 10 (47 with CUDA graphs) on an H200, and 4.0 audio-s/s unbatched [P vLLM-Omni docs, 2026-10-16, via sub-search].
- The talker plus the 15-step code predictor is not a plain CausalLM, so TRL/veRL rollouts need a hand-written loop. Expect 5–20 audio-s/s [U].

**Offline DPO is the practical first step:**
- 3k prompts × 8 samples × ~4 s ≈ 27 h of audio: roughly 1.5–5 h to generate and about 1 h to score.
- DPO on 2–3k pairs takes under 1 h.
- The log-prob is the talker + sub-talker CE from our fixed forward pass.

**Online GRPO:**
- About 20–60 s per step at 8 × 8 rollouts, so 300–500 steps take 2–8 h.
- Needs custom code and carries collapse risk.

**Headroom:**
- JA CER 0.030 already equals the real takes (0.032).
- The published 10–50% relative gains came from undertrained bases.
- RL would mainly buy ZH/EN robustness and fewer bad samples.

**Guardrails from the literature:**
- Absolute thresholds for "chosen".
- Never use ground truth as "chosen".
- Include a duration-ratio term and an F0-variance term.
- lr 1e-7–1e-6; β 0.01–0.1.
- Checkpoint every 50–100 steps and select with the pre-registered suite.

## 6. Q5: Classifier-free guidance for AR codec LMs

### 6.1 Evidence

- **Koel-TTS** [P]:
  - Text and context dropped with p = 0.1 each during training; logits γ·ℓc + (1 − γ)·ℓu with γ = 2.5 (swept 1–3).
  - Unseen speakers: CER 2.56 → 0.69, SSIM 0.601 → 0.653.
  - CFG on top of DPO was best: CER 0.58, SSIM 0.678.
- **VoXtream2** ([2603.13518](https://arxiv.org/abs/2603.13518), 2026-03-13) [P]:
  - **The closest architecture to Qwen3-TTS**: 12.5 Hz, 16-codebook Mimi, temporal + depth transformers.
  - 10% dropout each on text, audio prompt and speaker embedding; γ 1.5 on the temporal transformer, 3.0 on the depth transformer.
  - Effect by CFG type:

    | CFG on | Effect |
    |---|---|
    | Text | WER 2.29 → 1.37 |
    | Audio prompt | SIM +0.083, UTMOS −0.26 |
    | Speaker | SIM +0.013 |

  - **CFG speeds up speech**, so the duration state is left unguided. High depth-γ copies prompt noise.
- **SemBridge** ([2608.07462](https://arxiv.org/abs/2608.07462)) [S, snippet]: γ = 2 is best; at 3–4, SIM falls.
- **Production AR TTS mostly does not guide the LM** [P code]:
  - CosyVoice guides only its flow decoder.
  - fish-speech text2semantic, Spark-TTS and IndexTTS's GPT have no CFG.
  - NeMo added CFG distillation into a single pass (2026-06-10).
  - siriusfreak's Qwen3-TTS Russian-stress fine-tune "tried and not kept" CFG on stress marks [S].
- **Accent and language guidance** (training-free):
  - **Phrase-localised language-contrastive guidance** ([2609.01016](https://arxiv.org/abs/2609.01016), EMNLP 2026, OmniVoice) [P full text via sub-search]: adds λ(c_phrase − c_src), where c_phrase swaps the language tag, at λ = 7.
    - Phrase LID 0.247 → 0.588.
    - Listeners preferred it for nativeness 59.8% vs 19.5%.
    - SIM 0.973 → 0.969.
  - **Closed-loop dual-reference guidance** ([2610.29351](https://arxiv.org/abs/2610.29351), v1 2026-10-21; a frozen TTS, VoxCPM2 per the full text) [P abstract]:
    - The second reference is **a native-accented anchor of the same speaker, made by VC**; the mixing weight is λ = 0.33.
    - Source-language LID mass −73% (−56% on Spanish without re-tuning); ECAPA −0.042; WER 0.009.
  - **Accent Analogy Guidance** ([2609.29123](https://arxiv.org/abs/2609.29123), 2026-09-24) [P]:
    - Subtracts an accent direction estimated from a proxy voice rendered in both languages, and works on **CosyVoice 2's AR token-LM log-probs**.
    - Along plain CFG reweighting, SIM falls as the accent becomes native (OmniVoice: accent 2.54 / SIM 0.651 → 3.51 / 0.429). AAG gains +0.11–0.27 SIM at equal accent.

### 6.2 Qwen3-TTS

- **No CFG.** `generate()` and `qwen3_tts_model.py` contain no `cfg`, `guidance` or `uncond`, and the repo has no CFG issue or PR [P].
- **A custom decode loop is needed** [I]:
  - Each decode step's `forward` first runs `code_predictor.generate()` for codebooks 1–15 of the previous frame, then adds the trailing text embedding (`modeling_qwen3_tts.py` L1636–1695) [P].
  - So a [cond, uncond] batch pair under a `LogitsProcessor` would sample different frames per row.
  - The loop (~150 lines) keeps both KV caches, mixes the codebook-0 logits, samples once, runs the code predictor on the conditional hidden state, and feeds the same frame to both branches.
- **Choice of the second branch** [I]:
  1. **Language-contrastive (best fit).** ℓ(zh) + λ(ℓ(zh) − ℓ(auto)).
     - Both branches are in-distribution if v3 trains with language ids and keeps ~25% `nothink` rows, so no extra dropout is needed.
     - It has not been tested on AR models.
  2. **Speaker-dropped.** Needs ~10% speaker-slot dropout in SFT. Base has probably never seen an empty slot [U].
  3. **Dual-reference.** Mix the logits of the Kurisu row with a VC-made native-ZH Kurisu anchor at λ ≈ 0.3.
  4. **Condition strength.** ℓ_kurisu + γ(ℓ_amadeus − ℓ_kurisu) to dial the device effect.
- **Cost:** 2× batch, or about 1.1–1.4× latency at batch 1, since decoding is bandwidth-bound [I]. Watch for faster speech and lower SIM above γ ≈ 2.5.

## 7. Q6: Qwen3-TTS community findings to Oct 2026 (new beyond `final_report.md`)

### 7.1 Language-id training

- **[#323](https://github.com/QwenLM/Qwen3-TTS/issues/323)** (2026-05-24, auto-closed 2026-08-26) [P]:
  - Bengali, ~7 h, lr 2e-6, 10 epochs, `Auto`. The result sounded "Hindi-influenced", and IPA input sounded "more robotic".
  - A 2026-06-11 reply proposes `[think, think_bos, lang_id, think_eos]`, but it gives a wrong id (4206 > vocab 3072) and says the speaker slot "is not needed", which `generate()` contradicts.
  - **No results were ever posted.**
- **[PR #361](https://github.com/QwenLM/Qwen3-TTS/pull/361)** "actually use the language field during fine-tuning" (2026-08-28, open, "Not a full training run") [P, diff read]:
  - It uses a uniform 6-slot prefix, so Auto rows become `[nothink, think_bos, pad, think_eos, SPK, pad]`.
  - Inference builds `[nothink, think_bos, think_eos, SPK, pad, bos]`.
  - **Do not copy its Auto layout.**
- **vadimbelsky/qwen3-TTS-KSA** (HF, 2026-03-17) [S]:
  - A new language id on unused row 2072, initialised to the mean of the existing language rows.
  - Full FT at lr 2e-6, 5 epochs; inference with `language="arabic"`.
  - Works; no metrics.
- **jmtl/hk-cantonese-cc0-qwen3-tts** v0.0.5 (2026-10-25) [S]:
  - Two LoRA phases (r32/α64, lr 1e-4, 1 epoch each, sub-talker weight 0): a language phase on 71 h of multi-speaker Common Voice with a speaker id per corpus, then a speaker phase on 0.54 h.
  - CER 0.150.
- Accent persists after new-language FT even with 18–120 h (#27, Discussion #348, #72) [P]. All of these used the default `Auto` layout.

### 7.2 Instruct and condition training

- **Base and instruct:**
  - The maintainer says Base does not support instruct ([#25](https://github.com/QwenLM/Qwen3-TTS/issues/25), quote verified) [P].
  - PR #281 wires `instruct` into `generate_voice_clone` as "experimental" [P].
  - Discussion #121: a fine-tuned Base "completely fails to follow instructions" [P].
- **macminix/qwen3_voice_design_t8** [S]:
  - LoRA r16 on the VoiceDesign talker, lr 1e-5, 46k clips, with the KL anchor in §4.2.
  - Emotion accuracy 0.559 → 0.712; WER 0.0215 → 0.0187; gender accuracy 0.847 → 0.746.
  - "Strong-emotion training data trains caricatured delivery."
- **SpragAI/qwen3-tts-emotion-tags** (2026-09-09) [S]: 9 inline tags on CustomVoice with LoRA r32 and a frozen `codec_embedding`. No metrics.
- **TaoLiveAIGC/Qwen3-TTS-TBLive-Base** (2026-07-30) [S]:
  - Continued pre-training plus GRPO, 6 speaker rows, pause-length and speed tokens.
  - SIM en 0.708 → 0.732, zh 0.766 → 0.782; WER en 1.54 → 1.63, zh 1.15 → 1.28.
- **siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf** (2026-10-04) [S]:
  - Plain SFT ignored stress marks the text already implied.
  - A counterfactual hinge loss raised mark-following from 77% to 96%.

### 7.3 Multi-speaker plumbing and settings

- **[PR #384](https://github.com/QwenLM/Qwen3-TTS/pull/384)** (2026-10-17, open) [P]: fixes the label shift, sub-talker alignment and `text_projection`, and adds a per-row `"speaker"` field registering rows from 3000 upward. It is the first upstream multi-speaker trainer.
- **Commit 022e286** (2026-03-17, "fix finetuning bug") changed the sub-talker hidden-state mask from `[:, 1:]` to `[:, :-1]` [P diff]. I read it as the change that introduced the #337 leak [I].
- **#362** (2026-08-28): streaming input causes speaking-rate drift [P]. Our non-streaming training matches `generate_custom_voice`.
- **#39, #72:** fine-tuned voices sound flatter than ICL clones [P].
- **Community RL:** only TBLive, and FelysNeko's work-in-progress SFT+GRPO (2026-09-06; open issues are speaker drift and long-audio CER) [S].

## 8. Data reality check and local facts [L]

### 8.1 Tags and variants

Counted from `kurisu_v2/artifacts/build/decisions.jsonl` (all 10,039 Japanese rows) and `hf_dataset_upload/data/*/metadata.jsonl`.

| Tag / variant | All JA rows | Kept clean in v2, with words | Tag-only takes | Tier C | Under Amadeus |
|---|---:|---:|---:|---:|---:|
| `[pause]` | 558 rows (648 tags) | 518 | — | 1 | 24 |
| `[sigh]` | 148 | **27** | 68 | 28 | 23 |
| `[gasp]` | 58 | **20** | 9 | 20 | 1 |
| `[breath]` | 16 | **11** | 1 | 1 | 1 |
| `[digital]` | 29 | 0 (26 are the phone takes) | — | 2 | 1 |
| Amadeus variant | 947 rows, 0.97 h | 920 dropped as device voice | | | |

- **Your counts are occurrences over all rows.** The figures sigh 148, gasp 62, breath 17, pause 648 and digital 29 include tag-only, tier-C and Amadeus rows. **The clean worded Kurisu examples are 27 sigh, 20 gasp and 11 breath.**
- **The other dubs add little:** EN has sigh 10, gasp 13 and breath 12; ZH, DE, FR and ES have at most 5 of any event tag.

### 8.2 The Chinese dub is mostly Amadeus

| Dub | Variant | Tier A+B clips | Minutes |
|---|---|---:|---:|
| ZH | kurisu | 105 | **8.5** |
| ZH | amadeus | 244 | 17.9 |
| EN | kurisu | 1,154 | 93.3 |
| EN | amadeus | 264 | 19.9 |

### 8.3 Pipeline facts

- **Training is cheap.** v2's 240 steps (1 epoch, 7,674 clips) took **216 s** on one H100, with peak 37 GiB under fp32 masters and gradient checkpointing (`artifacts/reports/ft_02_train_kurisu-full-r32.json`). Ablations cost minutes; evaluation is the bottleneck.
- **The Base clone used as the ZH/EN guard ran with `language="Auto"`** (`pod/p06_eval.py` L103). I found no run in `kurisu_v2/artifacts` that uses `language="Chinese"` for Base or v2.
- **ZH CER uses Qwen3-ASR forced to Chinese** (`pod/metrics.py` L120), and there are only 8 ZH and 8 EN probes.

## 9. v3 recipe

### 9.1 What stays from v2

- The fixed trainer (`train_sft.py`, with label alignment, `text_projection` and the frozen speaker encoder).
- Full FT with fp32 masters, AdamW (0.9/0.95), lr 5e-6, 3% warm-up, cosine to 10%, wd 0.01, batch 32.
- Kurisu's speaker vector stays a **fixed** mean x-vector in row 3000. Do not train the row (#213).
- **Budget:** about 1 epoch of Japanese exposure. With ~20–30% extra rows that is ~300–350 steps; checkpoint at 0.75, 1.0 and 1.5 epochs.
- Pre-registered selection on the JA confirm suite, plus the new ZH/EN accent metrics and guards below.

**Trainer changes the recipe implies.** `train_sft.py` currently hard-codes `spk_id = {name: 3000}` and writes one row (L397, L409), and `collate` assumes a fixed 8-slot prefix. Ranks 1–3 need:
- a per-row prefix length in `collate`, so `japanese`, `chinese`/`english` and `nothink` rows coexist with the exact `generate()` layouts. Every `8+n_text` offset becomes per-row.
- a speaker source per row: Kurisu's fixed row, Amadeus's fixed row, or a per-clip x-vector from the frozen encoder for replay rows.
- saving several rows, each with `spk_id` and `spk_is_dialect` config entries.

PR #384 (2026-10-17) is the upstream reference for per-row speakers [P]. Take its speaker plumbing, but keep our own label alignment and our language-id layout.

**Serving after rank 1.** Japanese can be served with either `language="Japanese"` (matching 75% of the training rows) or `Auto` (the other 25%). Use `"Chinese"`/`"English"` for those languages. Update the model card, whose current "use Auto" advice will be stale.

### 9.2 Ranked changes

Ranked by expected value for your three goals, discounted by risk.

| Rank | Change | Why (evidence) | Expected effect | Risk |
|---|---|---|---|---|
| **0** | **Measurement first**, plus inference-only diagnostics (E0) | Forced-language CER hides accent (§4.1); ECAPA is language-biased (LASE) | Lets every later change be judged; may show a zero-training fix | None |
| **1** | **Language-id training**: Kurisu JA rows as `[think, think_bos, japanese, think_eos, SPK, pad, bos]`; **~25% left as `nothink`** (exact inference layout, per-row offsets; *not* PR #361's padded Auto). Serve ZH/EN with `language="Chinese"/"English"` | VALL-E X 4.03 vs 2.35 [P]; it is Qwen's own conditioning path; it keeps `Auto` working and enables language-contrastive CFG | Some accent reduction on ZH/EN; JA unchanged | Unmeasured on Qwen. Alone it may do little, because the Kurisu row still only sees JA |
| **2** | **Native-speaker replay** ~15% of rows: ZH + EN from other speakers, per-clip x-vector, language id set. Sources: Base self-generation from native refs, or AISHELL-3 / LibriTTS-R, plus the EN dub (93 min) | CosyVoice2 mSFT, CosyVoice3 aux data, VoxCPM 10–20% [P]; Retaining by Doing favours self-generated data [P] | Preserves Base ZH/EN; teaches "language id ⇒ phonetics" independent of speaker | Domain shift (read speech); timbre bleed if speaker vectors are too similar to Kurisu |
| **3** | **Amadeus as a second row** (`amadeus`, row 3001, fixed mean x-vector of Amadeus takes), **no dropout**, optional ~300 EQ-augmented clean clips under that row. **Control arm: DSP EQ on default output.** Phone = DSP only | CustomVoice rows; CosyVoice2 explicit-condition principle; Audiobox prompt > filter; Hsu 2019 decorrelation [P]; x-vectors carry channel [P] | Selectable Amadeus; default unchanged | Filter leaking into the default (measure it); 920 clips may give a weaker filter than DSP |
| **4** | **Kurisu-timbre native ZH/EN pseudo-data** (~5–10% of rows, Kurisu row + zh/en id): Base self-generation if E0a passes, otherwise VC of the EN dub / ZH dub / expressive native ZH into Kurisu | Amazon VC-polyglot: accent similarity up, SIM on par [P]; 2610.29351 builds native anchors by VC [P] | The largest accent reduction that keeps Kurisu's timbre [I] | VC artifacts learned by the Kurisu row; source-speaker leakage; most engineering effort |
| **5** | **Partial freezing**, two arms: (a) freeze `text_embedding` + `text_projection`; (b) freeze those plus the bottom 14 of 28 talker layers. Ablation; adopt only if JA is unchanged | CSP-FT evidence is about *layers* [P], so arm (b) is the better-evidenced one; arm (a) rests on the shared-Han-row mechanism [I] | Less ZH drift | Slightly worse JA readings or timbre; tag tokens then need their rows unfrozen |
| **6** | **`<|sigh|>` / `<|gasp|>` tokens** + splice augmentation; drop `[breath]`; `[pause]` stays "……"; no tag dropout | NVV challenge: synthetic augmentation helps rare tags; Affectron splicing [P]; NonverbalTTS shows ~200 examples can still fail [P] | Usable sigh, maybe gasp | Runaway on tags (v1); too few distinct takes; spontaneous sighs |
| **7** | **Language-contrastive CFG** at inference for ZH/EN (λ 0.5–2), custom decode loop | LCG, AAG, dual-reference [P]; VoXtream2 CFG on a same-shape model [P] | Further accent reduction at inference | Faster speech, lower SIM, 2× compute; untested on AR codec LMs |
| **8** | **Offline DPO**: best-of-8 on JA/ZH/EN prompts; chosen = Pareto on CER + duration ratio + LID (ZH/EN) with a JA SIM floor; lr ≤ 1e-6, β 0.01–0.1 | Koel, NeMo, CosyVoice2 speaker DPO [P] | Fewer bad samples; small CER gains | Reward hacking (slow, monotone speech); CosyVoice2's DPO hurt the hard set |
| **—** | *Not recommended:* instruct-text conditioning on Base, starting from CustomVoice, training speaker rows, online GRPO | §1, §7.2, #213, §5.3 | — | — |

### 9.3 Evaluation for v3 (pre-register before training)

- **JA:**
  - The existing confirm suite (148 clips): CER, ECAPA SIM, runaway.
  - Acceptance: CER ≤ v2 + 0.005 and SIM ≥ v2 − 0.01, or within the seed-to-seed spread of E1.
- **ZH/EN:**
  - ≥ 50 probes each.
  - Primary: Whisper-large-v3 unforced P(target language), plus the Qwen3-ASR `language=None` hit rate.
  - Guards: forced CER, and SIM as a floor only (≥ the Base x-vector clone with the same language setting).
  - Final two candidates: native-listener A/B, with the ZH dub's own lines as native anchors.
- **Amadeus:**
  - Leakage check: long-term average spectrum and spectral tilt of *default* outputs vs v2; the difference must be within ±1 dB.
  - Fidelity check: the Amadeus row's tilt vs real Amadeus takes.
- **Tags:**
  - Realisation rate on tagged probes (listening, or a sigh/gasp detector).
  - Runaway rate on tagged prompts.
  - Spontaneous NV rate on untagged probes.

### 9.4 Experiment matrix

Each row changes one thing relative to its parent. Train ~5 min; evaluate on the suites above; 2 seeds where differences are near noise.

| ID | Parent | Change | Question it answers | Decision rule |
|---|---|---|---|---|
| **E0a** | Base, v2 | Inference only: Base (x-vector-only and ICL, Kurisu reference) with `language="Chinese"/"English"` vs `Auto`. Also report v2 with the language id, but **only as information**: that prefix is off-distribution for v2, so a bad result there cannot separate "weights" from "unseen prompt" | Does Base's own language-id path give native ZH/EN with a Kurisu x-vector? This decides rank 4 route (a) Base self-generation vs (b) VC | Base + id native (LID ≥ native-reference level) → route (a). Otherwise → route (b). Whether *training* the id path fixes v3 is answered by E2 vs E1, not here |
| **E0b** | v2 | Weight interpolation α ∈ {0.25, 0.5, 0.75} toward Base, ZH/EN only, **all parameters except the speaker rows** (keep v2's row 3000) | How entangled are timbre and accent? A free serving fallback? | Keep as fallback if an α gives LID ↑ with SIM ≥ floor |
| E1 | — | v2 recipe re-run, 2 seeds | Noise floor | — |
| **E2** | E1 | + language-id rows (25% `nothink`) | Does the id alone reduce accent? Is JA neutral? | Adopt if JA neutral |
| **E3** | E2 | + 15% native-speaker replay (zh/en ids, x-vectors) | Does replay add accent reduction and preservation? | Adopt if ZH/EN LID ↑ and JA neutral |
| E3k | E3 | + KL to base on replay batches (β 0.02) | Is extra anchoring needed? | Only if E3 still drifts |
| E3f-text | E3 | Freeze `text_embedding` + `text_projection` | Does the text path carry the accent? | Adopt if ZH/EN improve and JA is neutral |
| E3f-layers | E3 | Freeze the text path **and** the bottom 14 of 28 talker layers (the CSP-FT-like arm) | Does limiting which layers move preserve ZH/EN better? | Adopt if ZH/EN improve and JA is within noise |
| **E4** | best of E3 | + Kurisu-timbre ZH/EN pseudo-data (route from E0a) | The timbre-preserving accent fix | Adopt if LID ↑ beyond E3 at equal SIM floor |
| **E5** | E1 | + `amadeus` row (no dropout) ± EQ-augmented clean clips | Learned variant without leaking into the default | Adopt if default tilt within ±1 dB of v2 and it wins blind A/B vs E5d |
| E5d | E1 | DSP EQ applied to E1 output | Is DSP enough? | — |
| **E6** | E1 | + sigh/gasp tokens + splice augmentation | Can tags bind with ~100 takes? | Ship a tag only if ≥ 50% realisation, no runaway increase, no spontaneous-NV increase |
| E6k | E1 | Kana mapping (はぁ…, っ) instead of tokens | Cheap alternative | — |
| E7 | best SFT | + language-contrastive CFG, λ ∈ {0.5, 1, 2} | Inference-time accent lever | Adopt per language if LID ↑ and duration ratio within ±10% |
| E8 | best SFT | + offline DPO | Robustness | Adopt only if the confirm suite improves |
| **Final** | — | Combine the adopted changes, 2 seeds; pre-registered selection | — | — |

**Order of work** (E5 and E6 are independent of the accent branch and can run in parallel):
1. E0a and E0b: no training; about an hour of generation.
2. E1, E2, E3.
3. E4, if E3 is not enough.
4. E5 and E6.
5. E7.
6. E8 last.

## 10. Could not verify

- **Per-tag minimum examples** for single-speaker codec-LM fine-tuning. None published.
- **Condition-dropout rates** in CosyVoice3's speaker/style masking.
- **Qwen3-TTS internals:** instruct-data size, and whether Base ever saw an empty speaker slot or instructions.
- **Accent before and after language-id training on Qwen3-TTS.** Nobody has measured it.
- **Regularisers and adapters:**
  - A KL-to-base ablation in TTS speaker SFT, beyond the macminix card.
  - Any DoRA, or speaker-prompt-only tuning, result for AR codec LMs.
  - A replay-ratio study for codec-LM TTS (2025–26).
- **Methods untested on our model class:**
  - A pseudo-speaker row isolating a channel effect in a codec LM.
  - Language-contrastive guidance on an AR codec LM.
  - RL with an LID or accent reward.
- **Rollout throughput:** Qwen3-TTS rollouts with a hand-written loop (5–20 audio-s/s is an estimate).
- **Licences:** RVC's licence (believed MIT) and its suitability for anime timbre at 24 kHz.
- **Unread primary pages:** the vLLM-Omni throughput page, and the full texts of 2609.01016 and 2606.07240 (their numbers came via a sub-search, not my own reading).

## Sources (selected)

- Qwen3-TTS
  - Repo, issues and PRs: https://github.com/QwenLM/Qwen3-TTS (#25, #27, #39, #72, #121, #213, #314, #323, #348, #361, #362, #384)
  - Tech report: https://arxiv.org/abs/2601.15621
  - CustomVoice card: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- Large systems
  - CosyVoice 2: https://arxiv.org/abs/2412.10117
  - CosyVoice 3: https://arxiv.org/abs/2505.17589
  - IndexTTS2: https://arxiv.org/abs/2506.21619
  - IndexTTS-2.5: https://arxiv.org/abs/2601.03888
  - Fish S2: https://arxiv.org/abs/2603.08823
  - Step-Audio-EditX: https://arxiv.org/abs/2511.03601
  - Seed-TTS: https://arxiv.org/abs/2406.02430
- RL and CFG
  - Koel-TTS: https://arxiv.org/abs/2502.05236
  - GLM-TTS: https://arxiv.org/abs/2512.14291
  - Multi-reward GRPO: https://arxiv.org/abs/2511.21270
  - GRPO CER+NLL: https://arxiv.org/abs/2509.18798
  - GRPO vs DiffRO: https://arxiv.org/abs/2509.18569
  - No verifiable reward for prosody: https://arxiv.org/abs/2509.18531
  - Code-mixing DPO: https://arxiv.org/abs/2606.19381
  - KIT IWSLT 2026: https://arxiv.org/abs/2606.07240
  - VoXtream2: https://arxiv.org/abs/2603.13518
  - NeMo PO docs: https://github.com/NVIDIA/NeMo/blob/main/docs/source/tts/magpietts-po.rst
- Accent
  - VALL-E X: https://arxiv.org/abs/2303.03926
  - Learning to speak fluently: https://arxiv.org/abs/1907.04448
  - Accent Vector: https://arxiv.org/abs/2603.07534
  - AAG: https://arxiv.org/abs/2609.29123
  - LCG: https://arxiv.org/abs/2609.01016
  - Dual-reference: https://arxiv.org/abs/2610.29351
  - Amazon VC-polyglot: https://arxiv.org/abs/2309.08255
  - LASE: https://arxiv.org/abs/2605.00777
- Fine-tuning and forgetting
  - CSP-FT: https://arxiv.org/abs/2501.14273
  - NeuTTS LoRA: https://arxiv.org/abs/2603.10904
  - LoRA learns less: https://arxiv.org/abs/2405.09673
  - Deafening Silence: https://arxiv.org/abs/2610.09835
  - RL's Razor: https://arxiv.org/abs/2509.04259
  - Retaining by Doing: https://arxiv.org/abs/2510.18874
  - UtterTune: https://arxiv.org/abs/2508.09767
- Tags and conditions
  - NonverbalTTS: https://arxiv.org/abs/2507.13155
  - NVV challenge: https://arxiv.org/abs/2609.14231
  - Affectron: https://arxiv.org/abs/2603.14432
  - Parler-TTS: https://arxiv.org/abs/2402.01912
  - Audiobox Aesthetics: https://arxiv.org/abs/2502.05139
  - Hsu 2019: https://research.google/pubs/pub47834/
  - Environment-aware TTS: https://arxiv.org/abs/2110.03887
  - x-vector probing: https://arxiv.org/abs/1909.06351
  - Parler Greek: https://arxiv.org/abs/2609.10022
- Community
  - https://huggingface.co/macminix/qwen3_voice_design_t8
  - https://huggingface.co/SpragAI/qwen3-tts-emotion-tags
  - https://huggingface.co/TaoLiveAIGC/Qwen3-TTS-TBLive-Base
  - https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf
  - https://huggingface.co/vadimbelsky/qwen3-TTS-KSA
  - https://huggingface.co/jmtl/hk-cantonese-cc0-qwen3-tts
  - https://huggingface.co/notmax123/QwenTTS-he-1.7B
  - https://github.com/Hebrish-ASR/Meorav-TTS
  - https://github.com/sabari245/qwen3-tts-audiobook-finetune
- Data and tools
  - VoxCPM fine-tune docs: https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
  - AISHELL-3: https://www.openslr.org/93/
  - Emilia: https://huggingface.co/datasets/amphion/Emilia-Dataset
  - Qwen3-ASR: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
