# Labeling expressive and channel-conditioned TTS data: state of the art, late October 2026

Scope: the Japanese subset of `starrydark/Kurisu_Voice` (Asami Imai, 10,039 clips, 11.9 h), fine-tuned into Qwen3-TTS-12Hz-1.7B-Base. Goal for v3: keep Amadeus, learn phone-call and other channel effects plus non-verbal vocalizations (NVVs), and make them controllable without dirtying the default voice.

**Evidence tags**

- **[P]** primary: I or a research subagent read it in the paper, repo code, issue, or model/dataset card. I re-checked the load-bearing ones myself: NVSpeech Table 4, the LAION locator card, Fish S2, Higgs TTS 3, NV-Bench vs NVV-SuperBench, the Track 1 and 2 challenge papers, and the Qwen3-TTS code.
- **[S]** secondary: blog, mirror, search summary, or community post.
- **[I]** my inference.
- **[U]** unverified.
- **[L]** measured by me on your local files (`hf_dataset_upload/data/*/metadata.jsonl`, the audio, and the cached Qwen3-TTS tokenizer).

Dates are arXiv submission dates, model-card dates, or issue dates.

---

## 0. Key conclusions

1. **Use a different representation for each kind of control** [P survey, §2]:
   - Point events (sigh, laugh, breath): inline tokens at the event position.
   - Whole-utterance delivery: a global label or instruction placed before the text.
   - Recording channel: the least modelled category. Only Irodori (📞 / 📢 emoji) and Higgs v2 (scene caption) put it in the model input. MiniMax does it with DSP. Big systems mostly filter it out.
   - No controlled ablation of tags vs captions vs global labels exists [P negative result].
2. **For a 12 h single-speaker fine-tune, use atomic special tokens** for both events and conditions [I, from the convergence in §2 and the tokenization evidence in §5].
   - Free-text captions earn their keep at millions of clips.
   - Your v2 bracket tags are fragile: the same tag became 2–4 different BPE sequences depending on the preceding character [L].
   - Qwen3-TTS's text embedding has **260 unused rows** (151,676–151,935), so new tokens need no resize [L].
3. **For Japanese, keep the kana and add the category token: `<laugh>ふふっ`, `<sigh>はぁ……`** [I].
   - Kana is what VN scripts and anime-whisper write, and it is Qwen3-TTS's only native NVV pathway (issue #9).
   - Precedent for pairing the two: Irodori (emoji beside kana), Higgs TTS 3 (tag first, then onomatopoeia) and Fish S1 (`(laughing)` with "Ha,ha,ha") [P].
4. **No public labeler is accurate enough to trust, and none handles Japanese NVVs.**
   - Best in-dialogue F1: laugh ~0.8, sigh ~0.45–0.75, breath ~0.3–0.7, gasp ~0.2–0.3 [P].
   - Your locator's own references are ASR-made, F1 0.61 [P].
   - Audio LLMs hallucinate NVVs when asked open questions [P].
   - **At 10k clips, the strongest option is human verification of every candidate:**
     - Generate candidates broadly from script kana, an anime-whisper diff, a low-threshold locator, and a spectral-shape channel detector.
     - Use Gemini or Qwen3-Omni only as a yes/no verifier to rank the queue.
     - About 2k candidates is 4–7 h of listening [I].
5. **Precision matters more than recall, but don't leave events unexplained.**
   - A labeler at about 0.84 precision / 0.67 recall trained TTS as well as human labels at 573 h (NVSpeech) [P]. At your scale, aim for ≥0.9 precision on exposed tokens [I].
   - Route everything else to a catch-all `<nv>`, so stray events don't leak into the default voice [I; precedents: CosyVoice `[noise]`/`[mn]`, Meta's generic-token curriculum].
6. **Expose a tag as a control only past about 150 verified real examples.**
   - Around 250 consistent examples of a new tag can still be ignored (VoxCPM #292), and about 300 per tag rendered only 50–60% of the time (IaraTTS) [P].
   - Merge by family, use square-root resampling capped at 3×, and use only *diverse* synthetic top-up [P, NVVSpeech 2026 challenge papers].
7. **Amadeus is two filters.** Game Amadeus is a strong bright "presence" EQ (+11 dB at 4.8–7 kHz); anime Amadeus is a mild bass cut with a 2–4 kHz lift [L].
   - **Factorise the condition tokens:** `<src_anime>` for source-separation character, plus `<amadeus>` / `<phone>` for the channel.
   - Plain `<amadeus>` then yields the game sound, and the default stays the dry studio voice [I, design; analogues in Tan 2022, Hsu 2019].
8. **The show's "phone" is not a telephone band.** It cuts below 600 Hz by 10–13 dB, lifts 1.2–4.8 kHz by ~9.5 dB, and keeps the highs [L].
   - Either use DSP after generation (most reliable) and drop the 28 real clips, or keep them tagged `<phone>` and add 300–500 clean clips augmented with the *measured* curve for in-model control.
   - A phone clip must never be in training untagged.
   - Never use generic G.711 as the only phone augmentation [I].
9. **Do not add natural-language instructions or emotion captions in v3** [I].
   - The Base instruct path is untrained by the official SFT [P].
   - Japanese caption accuracy is unmeasured [U].
   - Captioners follow the transcript's implied emotion [S].
10. **Exclude from v3:** singing (53), non-vocal sound-effect contamination (50), whisper and echo clips below threshold. **Never drop a condition token** from a clip that has the effect, and keep one clean `ref_audio` [P/I].
    - **This disagrees with your request to learn sound effects.** Fifty incidental noises cannot teach a controllable effect.
    - If you want SFX control later, the route is CapSpeech's: library events mixed into clean speech with event markers (§9.2).

**The scheme is in §9.** The critique of your two planning docs and the v2 report is in §8.

---

## 1. What your data actually contains (local measurements)

All numbers in this section are [L]. They come from the local copy of the dataset (built 2026-09-05), which has kurisu 9,092 and amadeus 947 Japanese rows. Your brief gives kurisu 9,119 and amadeus 920, which suggests a slightly newer build; the differences are too small to matter here.

### 1.1 Where the current tags came from

| Tag | Clips | Human | Auto | Notes |
|---|---:|---:|---:|---|
| `[pause]` | 558 (648 occurrences) | 0 | 558 | Measured: an aligner gap of at least 1.28 s where the text has no "……" |
| `[sigh]` | 148 | 5 | 143 | **113 are isolated clips whose text is otherwise empty**. 114 game, 11 anime, 23 Amadeus |
| `[gasp]` | 58 (62) | 27 | 31 | 29 anime and 28 game |
| `[singing]` | 53 | 53 | 0 | All from the `song` medium |
| `[digital]` | 29 | 29 | 0 | 28 Kurisu phone/device clips plus 1 Amadeus. **All anime** |
| `[breath]` | 16 (17) | 16 | 0 | All anime, from human review |

- Only 427 anime clips (plus 52 songs) were ever human-reviewed. **No game clip has been reviewed.**
- The anime reviewers' tags are inconsistent. For example, `あっ おか[gasp]… 岡部… [gasp]もしもし` uses `[gasp]` for what is probably a stammer inhale, and `[breath]` and `[gasp]` overlap. Treat these labels as a noisy gold set, not ground truth.
- `emotion_notes` holds 36 labels across 7 values. That is too sparse to be anything but metadata.
- `contamination`: noise 44, footstep 4, ringtone 1, knock 1.

### 1.2 The game scripts already label many NVVs, as kana

VN script text is what the actress was reading, so the kana interjections in it are performance directions. A rough regex count, which includes false hits such as イクス and なきゃ:

| Family | Clips with kana | Examples | Currently tagged |
|---|---:|---|---|
| Laugh | ~80 | ふふっ, ふふ……, クスッ, あはは | **0** |
| Sigh / breath | ~56 | はあ……, はぁ……はぁ……, ふぅっ | 4 (as `[gasp]`) |
| Gasp | ~93 | ハッ…, ひっ, キャッ | 1 |
| Strain | ~122 | うっ……, うぅ, あうっ | 4 |
| Cough | 8 | ゴホン, ごほっ、げほげほ | 0 |

In addition:
- 189 clips have text that is only punctuation or empty (……っ, ――っ). Of these, 113 carry `[sigh]`, 20 `[gasp]`, and **55 carry nothing**.
- So the true number of NVV events is clearly higher than the tag counts. Laughter, in particular, exists in the audio but has no tag at all.

### 1.3 The bracket tags tokenize inconsistently

With Qwen3-TTS's own tokenizer (Qwen2 BPE, 151,676 entries), the same tag becomes different token sequences depending on the character before it. Counted across the Japanese transcripts that reconstruct cleanly:

| Tag | Distinct token sequences | Most common sequences |
|---|---:|---|
| `[pause]` | 3 | `[`·`pause`·`]` (297), ` [`·`pause`·`]` (42), `[p`·`ause`·`]` (14) |
| `[gasp]` | 4 | `[g`·`asp`·`]` (25), `[`·`g`·`asp`·`]` (9), ` [`…, `][`·`g`… |
| `[sigh]` | 2 | `[s`·`igh`·`]` (126), `[`·`s`·`igh`·`]` (12) |
| `[breath]` | 3 | including a merged `]['` token that crosses two tags |

Two further problems:
- `[sigh]` and `[singing]` share their first token, `[s`.
- `<|sfx:sigh|>` (Higgs style) costs 8 tokens.

What is stable:
- Added special tokens are atomic in every context I tested.
- Single-codepoint emoji (😮 🤭 📞 🤖 😤 ⏸) are one token each, regardless of the preceding Japanese punctuation.
- ZWJ emoji such as 😮‍💨 split into 4 byte-level pieces.

The model has room for new tokens without resizing:
- The talker's `text_vocab_size` is **151,936**, but the tokenizer has only 151,676 entries.
- That leaves **260 unused embedding rows**, so new special tokens can take ids 151,676 and up without resizing anything.

### 1.4 Amadeus and phone are different filters, and neither is a textbook telephone

I compared median long-term average spectra (LTAS): 150 tier-A clips per group (54 Amadeus anime clips, 12 phone clips), each spectrum normalised to its own total power. The table gives each group's difference from plain Kurisu from the same medium, in dB per band:

| Band (Hz) | 50–150 | 150–300 | 300–600 | 600–1.2k | 1.2–2.4k | 2.4–3.4k | 3.4–4.8k | 4.8–7k | 7–9.5k | 9.5–12k |
|---|---|---|---|---|---|---|---|---|---|---|
| Amadeus game − Kurisu game | −4.1 | −4.9 | −3.3 | +3.8 | +7.6 | +4.6 | +2.6 | **+11.3** | +1.6 | −3.3 |
| Amadeus anime − Kurisu anime | **−14.2** | −2.8 | −0.5 | +2.6 | +5.1 | +6.3 | +4.3 | +1.7 | +2.1 | +1.0 |
| Anime `[digital]` (phone) − Kurisu anime | −12.8 | **−13.4** | −10.2 | +3.2 | **+9.8** | +9.5 | +9.4 | +6.0 | +6.0 | +0.3 |

What this shows:
- **Amadeus (game)** is a thin, bright, "presence" EQ: lows down about 4 dB, 1–2.4 kHz up about 8 dB, and a strong peak around 5–7 kHz.
- **Amadeus (anime)** is milder: mostly a sub-bass cut plus a 2–4 kHz lift. That fits per-clip spectral tilt: median −2.6 dB in anime against +11.2 dB in game. Only 90 of 295 anime Amadeus clips are brighter than the 75th percentile of ordinary anime Kurisu.
- **The anime "phone"** cuts everything below 600 Hz by 10–13 dB and lifts 1.2–4.8 kHz by about 9.5 dB. It **keeps content above 3.4 kHz**, so it is a stylised broadcast phone EQ, not a 300–3,400 Hz telephone band. Generic G.711 or AMR telephony augmentation would therefore teach a different sound from the show's.
- **One feature cannot separate the groups.** Per-clip tilt above +2.4 dB catches 569 of 652 Amadeus game clips, but also 1,914 of 7,707 ordinary game clips. A detector needs the whole spectral shape (or an audio embedding), not one feature.

---

## 2. How leading systems represent control (research question 1)

### 2.1 System by system

| System (date) | Representation | Vocabulary / syntax | Token type | How the tags were labeled | Source |
|---|---|---|---|---|---|
| **CosyVoice 2** (2024-12) / **CosyVoice 3** (2025-05) | Inline point tags, span tags, and an NL instruction ending in `<\|endofprompt\|>` | `[breath] [noise] [laughter] [cough] [clucking] [accent] [quick_breath] [hissing] [sigh] [vocalized-noise] [lipsmack] [mn]`, spans `<strong>…</strong>`, `<laughter>…</laughter>` | **Added special tokens** | Instruct data 1,500 h (v2) / 5,000 h (v3); labeling method not stated | [P] `cosyvoice/tokenizer/tokenizer.py`; arXiv 2412.10117, 2505.17589 |
| **Fish S2 / S2 Pro** (2026-03-09) | Free-form inline `[tag]` and NL descriptions | Open vocabulary | **Plain text**, "without requiring dedicated control tokens" | **Rich-transcription ASR fine-tuned from Qwen3-Omni-30B-A3B**, seeded by video-based pseudo-labels plus human checks, >10M h. The same ASR is the RL reward. **Tag activation: en 0.881 (S1 0.626), zh 0.984** | [P] arXiv 2603.08823 |
| **Fish Drama 3** (2026-10-13) | Point `[gasps]`, scoped `<whisper>…</whisper>`, line-start direction, director's notes | Free NL | Plain text, except `<\|name\|>` | Not described | [P] arXiv 2610.18010; [S] docs |
| **OpenAudio S1** (2025) | `(laughing)` paired with "Ha,ha,ha" | 51 emotion, 5 tone, 10 special | — | — | [P] HF card |
| **ElevenLabs v3/v4** | Free-text `[tag]`, placed right before or after the line it modifies | `[laughs] [sighs] [whispers] [sarcastic]…`; "match tags to the voice"; no channel tags in the docs | Closed | — | [P] docs (now "Eleven v4", 2026-09-28 [S]) |
| **IndexTTS2** (2025-06) | Global emotion vector or emotion-reference audio, plus text → emotion distribution through a Qwen3-1.7B LoRA | 7 (paper) / 8 (code) emotions | Continuous | 135 h of emotional speech; GRL disentangles speaker and emotion. Removing the training strategy drops emotion similarity from 0.887 to 0.689 | [P] arXiv 2506.21619 |
| **Dia 1** (2025-04) / **Dia2** | Inline `(laughs)` etc. | 21 listed. "Might result in unexpected output" outside the list; overuse causes artifacts. **Dia2 lists none** | — | — | [P] GitHub |
| **MOSS-TTSD v1.0** (2026-02) / MOSS-TTS 1.5 | Speaker tags only; `[pause X.Ys]` | 0 / 1 | — | — | [P]/[S] |
| **Higgs Audio v2** (2025) | **Environment caption in the system prompt** (`<\|scene_desc_start\|>Audio is recorded from a quiet room.<\|scene_desc_end\|>`), plus `<SE>[Laughter]</SE>` events | ~7 events | — | AudioVerse: multiple ASR models, sound-event classifiers and an understanding model, 10M h | [P] HF card, examples |
| **Higgs TTS 3** (2026-06) | Sentence-start `<\|emotion:x\|>`, `<\|style:x\|>`, prosody; inline `<\|sfx:x\|>`, `<\|prosody:pause\|>` | 43 (21 emotion, 3 style including singing/whispering, 9 sfx, 10 prosody). Rule: **"the tag comes first, immediately followed by the onomatopoeia with no space"**; "anything else degrades output or gets read literally" | Special tokens in `extra_special_tokens`. Undocumented `<\|env:music\|>` and `<\|env:noise\|>` also exist | Not stated | [P] PROMPTING.md, tokenizer_config.json |
| **Step-Audio-EditX** (2025-11) | Edit instructions on existing audio, iterated | 22 paralinguistic tags (`[sigh] [inhale] [laugh] [Surprise-oh]…`); 14 emotions; 32 styles including "radio"; `denoise` edit | — | Paralinguistic: NVSpeech audio as the target, a tag-stripped clone as the input. Emotion and style: margin-filtered pairs. Emotion accuracy 55 → 71 over 3 edit rounds | [P] arXiv 2511.03601 |
| **Orpheus** (2025-03) | Inline `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` | 8 | Probably plain BPE [I] | — | [P] GitHub |
| **Chatterbox-Turbo** (2025-12) | One `[tag]` set mixing events and styles | 19 (`[sigh] [gasp] [laugh] [whispering] [narration]…`) | **Added tokens, ids 50257–50275** | — | [P] added_tokens.json |
| **Sesame CSM-1B** | Conversational context only | 0 | — | — | [P] |
| **Qwen3-TTS** (2026-01-22) | NL instruction in ChatML for CustomVoice and VoiceDesign | No tags. NVVs only "implicitly through expressions like 'hahaha'" | — | Not described | [P] arXiv 2601.15621; issue #9 |
| **MiniMax speech-2.6/2.8** | Global `emotion` enum (9); inline interjections `(laughs) (sighs) (breath) (inhale) (gasps)…` on 2.8; pause `<#x#>`; **channel as a post-effect parameter `voice_modify`: `lofi_telephone`, `robotic`, `spacious_echo`, `auditorium_echo`** | 19 interjections | API | — | [P] platform docs |
| **Seed-TTS 2.0 / Doubao 2.0** | NL `context_texts` | Free | — | — | [S] |
| **Gemini TTS** (doc 2026-10-28) | Per-turn NL `style`; inline `<sigh> <laugh> <breath> <short pause>`; English tags even for non-English text; advises against SFX tags and long "director's notes" | ~35 | — | — | [P] ai.google.dev |
| **Inworld TTS-2 / Voxtral TTS 3** | `[tag]`. Instruction tags persist until changed; non-verbal tags are points | Open / 34 emotions, 5 paces | — | — | [P] docs |
| **Qwen-Audio-3.0-TTS** (2026-07, API) | Inline `[gasp] [giggles] [angry]` plus NL instructions | "86 tags" [S] | — | — | [P] blog |
| **InstructTTS / PromptTTS 2 / Parler (Data-Speech) / ParaSpeechCaps / CapSpeech** | NL captions | Parler bins: SNR "very noisy…very clean"; reverberation "very distant-sounding…very close-sounding" (v02), or "very roomy…very confined sounding" (default script). **CapSpeech's `<telephone>` is a ringing-phone sound event, not a channel** | Text encoder | Signal estimators → bins → LLM rewrite (Parler); propagation plus Gemini filter (ParaSpeechCaps). **Parler's authors are "unsure as to why the model performs poorly at generating audio with the appropriate C50"** | [P] arXiv 2402.01912, 2503.04713, 2506.02863; dataspeech |
| **NVSpeech** (2025-08) | Inline, 18 categories | `[Laughter] [Breathing] [Uhm] [Question-ah]…` | Added to the vocabulary | 76 h hand-labeled **from Genshin/StarRail game voice**, then auto-labeled 573 h | [P] arXiv 2508.04195 |
| **NVVSpeech Track 2 winner** (2026-09) | Inline, 16 categories | `laugh`, `cry`, … | **"Dedicated special tokens… exactly one token id and a learnable embedding"** | Public NVV corpora plus synthetic top-up | [P] arXiv 2609.14231 |

### 2.2 Where the field has converged

1. **Point events (sigh, laugh, breath): inline tokens at the event position.** Nearly every 2025–26 system does this.
   - Square brackets dominate in plain-text systems.
   - Small fixed vocabularies are usually **added special tokens**: CosyVoice, Higgs 3, Chatterbox-Turbo, NVSpeech, the Track 2 winner.
   - Open vocabularies trained on 10M+ hours use plain text (Fish S2/Drama 3) [P; pattern I].
   - Two systems tell you to **attach onomatopoeia to the tag**: Higgs TTS 3 (tag first, no space) and Fish S1 ("Ha,ha,ha") [P]. That is the hybrid recommended here for Japanese.
2. **Whole-utterance style and emotion: a natural-language instruction placed before the text** (CosyVoice, Qwen3-TTS ChatML, Gemini, OpenAI, Seed, Drama 3).
   - Fixed global labels survive in small or open models and in API enums: MiniMax emotion, Higgs sentence-start tokens, Voxtral, IndexTTS2's vector [P].
   - Gemini and Inworld both formally separate a *momentary inline event* from a *sustained turn-level delivery* [P].
   - That is the same split this report uses between event tokens and condition tokens.
3. **Recording channel and environment: the least modelled category.**
   - As an in-model caption: Higgs Audio v2 scene description, Parler SNR/C50 bins (reverb control reported poor).
   - As a token: Irodori 📞/📢 (§7) and the undocumented Higgs `<|env:*|>`.
   - As DSP: MiniMax `lofi_telephone`/`robotic`.
   - As an edit: Step-Audio-EditX "radio" and `denoise`.
   - Actively removed from training data: Fish S2, Emilia.
   - **No system I found uses an inline phone or radio tag** [P, survey].

### 2.3 Evidence on representation choices

**No controlled ablation compares inline tags, captions and global labels**, and none compares new special tokens with plain-text brackets [P, negative result of the survey]. The closest evidence:
- **OV-InstructTTS** (arXiv 2601.01459, 2026-01-04), Table 3: adding emotion labels and NVSpeech tags to the transcript alone did not beat the plain transcript (Gemini score 66.98 vs 67.70). It helped only combined with reasoning (71.57). The tags were model-generated, not user-supplied [P].
- **Control vs no control** (not a comparison of representations):
  - Drama 3: a written direction raises whisper realisation from 10.6% to 62.2% and laughing from 27.1% to 64.4% [P].
  - CosyVoice 2: instruct MOS-I 4.06 vs 2.28, with CER rising from 0.97 to 1.52 [P].
- **Tokenization:**
  - IaraTTS shows BPE-split tags produced no audible events, while atomic tokens did (§4) [P].
  - NonverbalTTS attributes CosyVoice2's better laughter to "single-token laughter," presented as a hypothesis [P].

**Inference [I]:**
- For a 12 h, one-speaker fine-tune of a model with no tag prior, use **atomic inline event tokens, plus fixed global condition tokens.**
- Free-text captions earn their keep at millions of clips, not ten thousand.

---

## 3. Automatic labelers and their measured accuracy (research question 2)

### 3.1 NVV-aware ASR (rich transcription)

**There is no Japanese NVV-aware ASR.** Every tagged-ASR system below covers only Mandarin and/or English.

| System | Languages, tags | Measured accuracy | Source |
|---|---|---|---|
| **NVSpeech** (SenseVoice-Small with tags added as CTC tokens), 2025-08-06 | zh; 18 inline tags | In-domain: CER 4.61 with tags, tag F1 0.83. Open-domain: F1 0.85. Tag-only detection: P/R/F1 **0.84 / 0.67 / 0.73**. Fine-tuned Whisper reached F1 0.71 / 0.50 | [P] arXiv 2508.04195 |
| **Emilia-NV** | zh; 573 h auto-labeled with the NVSpeech ASR | None given | [P] HF amphion/Emilia-NV |
| **CharlesNi/Multilingual-NVASR** (SenseVoice fine-tune) | zh 13 tags, en 7; **no ja** | CER only, from NV-Bench | [P] card; NV-Bench arXiv 2603.15352 |
| **NonverbalTTS** pipeline, 2025-07-17 | en; 10 NVV types | BEATs at threshold 0.1 (recall-first), placement with MFA, 3 annotators in Argilla with 2-of-3 majority. **No accuracy of the automatic tags is reported** | [P] arXiv 2507.13155 |
| **NonVerbalSpeech-38K**, 2025-08 (rev. 2026-01-13) | zh/en; 10 tags; fully automatic | Detector frame F1 ~91%, **on synthetic mixes only**. Placement method TSA (§5) | [P] arXiv 2508.05385 |
| **MNV-17**, 2025-09 | zh, 17 classes | Best exact NVV-sequence accuracy only **57.3%** (SenseVoice, Qwen2.5-Omni) | [P] arXiv 2509.18196 |
| **Meta "Beyond Words"**, 2026-07-02 / v2 2026-10-07 | en; RNN-T trained on 800 h of curated NVV labels | Per-tag F1: breath 72.9, laugh 83.4, **sigh 74.5**, cough 89.0, cry 78.3. Whisper-D (22 h of podcasts) got breath 11.0, sigh 43.6. **Cry F1 rose from 32.1 to 69.0 when breath and laugh data were added**, i.e. rare tags borrow from frequent ones. A two-stage curriculum (generic NV token first, then categories) is also proposed | [P] arXiv 2607.01563 |
| **ISCSLP 2026 NVVSpeech Challenge, Track 1** | zh/en; 16 tags | 4th-place system: Qwen3-ASR-1.7B full SFT on 70 h, **sqrt class-balanced sampling**, pooled F1 0.59–0.69. Per-tag F1 (two-stage): laugh 0.82, breath 0.32, **gasp 0.19**, cough 0.26, cry 0.37, yawn 0.39. Another team's biggest gain came from 454 h of movies labeled by Gemini 2.5 Pro on top of ASR (micro-F1 ~0.51–0.57) | [P] arXiv 2609.23462 (2026-09-23); arXiv 2609.09929 |
| **Qwen3-ASR / Qwen3-ForcedAligner**, 2026-01 | Text plus a language tag. **No event, emotion or NVV tags** | Aligner, Japanese: mean timestamp shift 42 ms (MFA-labeled data) and 81 ms (300 s concatenations) | [P] arXiv 2601.21337 |
| **litagin/anime-whisper** (kotoba-whisper-v2.0 tuned on ~5.3k h of galgame audio with script text) | ja | Writes NVVs as kana and punctuation (はぁっ、はぁっ…, ふふ, っ). CER 13.0 against 16.5 for whisper-large-v3 on held-out games. **No NVV-level metric** | [P] HF card |

### 3.2 Vocal-burst classifiers and detectors (what your v2 pipeline uses)

**`laion/vocalburst-classifier-single`** [P card]:
- 83 classes. Macro mAP 0.68 at fine level and **0.87 at the 16-family level**. Top-1 accuracy 60%, top-3 88%.
- The test set is isolated bursts with Gemini-confirmed positives.
- v2 added 9k FLEURS negatives, including Japanese, because v1 "hallucinated bursts on non-European speech."
- Gemini and a second automatic annotator agree **64.8% on family but 19.0% on exact label**. Fine subtypes (moan, groan, grunt; relief vs exasperated sigh) are not reliably labelable, so aggregating at family level, as your pipeline does, is correct.

**Sibling LAION models** [P cards]:
- Real-audio recall for sighs is low: Relief Sigh 6.2% and Exasperated Sigh 22.6% (vocal-burst-detector-x2).
- Sharp Inhale is high: 89.7%.

**`laion/vocalburst-locator` v2** [P card, verified]:
- Event F1 0.607 at IoU 0.5 on 992 real clips.
- The card says plainly that the real-audio labels "come from an ASR model, not from human annotation" (CrisperWhisper, which handles English and German), and that the ceiling is unknown.
- **It is a weak primary detector for Japanese anime speech.** That fits the low recall your calibration found.

**Clean isolated-clip benchmarks are much easier** than in-dialogue detection [P/S]:
- VocalSound, 6 classes: baseline 90.5%; Kimi-Audio is reported at 94.85% [S].
- A-VB 2022 burst type, 8 classes: UAR 0.42–0.52 [S].

### 3.3 Audio-LLM captioners

| Model | MMAU | MMSU | Notes |
|---|---:|---:|---|
| Gemini 2.5 Pro | 77.4 | **77.7** | Best paralinguistic understanding in the table [P, Qwen3-Omni report, arXiv 2509.17765, Table 7] |
| Qwen3-Omni-30B-A3B-Instruct | 77.5 | 69.0 | Open weights. Needs about 70 GB in bf16, so quantize for a 16 GB card [I] |
| Gemini 2.5 Flash | 71.8 | 70.2 | |
| Qwen3-Omni-30B-A3B-Captioner | – | – | Covers emotion, sound effects and ambience. Clips of 30 s or less. "Low hallucination" is shown only by case studies [P] |

**The reproducible lesson about captioners is hallucination:**
- NVV-SuperBench (arXiv 2604.16211, 2026-04-17) found open-ended Gemini NVV detection "unreliable, mainly due to hallucinated events and confusion among acoustically similar categories." It uses Gemini 2.5 Pro **only as a verifier that is told the target tag** [P].
- When LAION pushed a captioner to find bursts, burst claims rose from 9 to 63 of 95 clips and precision collapsed [P, LAION project page; LLM-judged].
- Audio LLMs tend to report the emotion that the text implies over what the acoustics show (VoxParadox, 2026) [S]. That is a real risk for VN lines whose wording is already emotional [I].

**Japanese-specific numbers for any captioner: none found** [U]. Neither Gemini 3.x audio benchmarks nor MiDashengLM, Audio Flamingo (Next/3), Step-Audio-R1 or Kimi-Audio results on paralinguistic captioning were verified [U].

### 3.4 Caption and annotation datasets: how they label, and how accurate it is

| Dataset | What is labeled, and by what | Verified accuracy |
|---|---|---|
| **ParaSpeechCaps** (2025-03) | 59 style tags. Speaker tags are propagated by voice similarity ≥0.80; utterance tags filtered by Gemini 1.5 Flash (kept only at 5/5); basic tags from PENN pitch, Brouhaha SNR and speaking rate | Human-judged tag recall of auto labels 50.3% / 71.3%, vs 48.7% / 68.1% for human labels. Base plus auto-scaled data beat Base alone on consistency MOS and naturalness MOS [P] |
| **CapSpeech** (2025-06) | ~10M machine pairs plus 359k human pairs. Sound-event and NVV portions are synthetic or voice-acted | Audit of 500 random samples: average 95.5%, **expressiveness 86.3%** [P] |
| **SpeechCraft** (2024-08) | zh/en style descriptions | Emotion accuracy 84% (en, emotion2vec), 70.5% (zh); word-level emphasis 66.9% [P] |
| **EmoVoice-DB** (2025-04) | 40 h, generated entirely by GPT-4o; WER filter only | **Emotion labels never checked** [P] |
| **Data-Speech** (Parler-TTS) | Brouhaha SNR and C50 (reverberation), torchaudio SQUIM (PESQ, SI-SDR), PENN pitch, phoneme rate. Values binned into words, then an LLM writes the caption | Bins are histogram-based. No accuracy figure [P, github huggingface/dataspeech] |

### 3.5 What precision and recall TTS training needs

**Primary evidence:**
- **NVSpeech, Table 4** [P, verified]: CosyVoice2 trained on auto-labeled data (labeler P/R ≈ 0.84 / 0.67) matched or beat the same model trained on human-labeled data. In-domain CER including tags: 7.83 (auto, equal size) vs 8.61 (human) vs 7.51 (auto, large scale). Listeners heard requested tags realized at recall ≈ 0.60–0.62, and the model won 75–79% of preference comparisons against the original.
- **NV-Bench** (arXiv 2603.15352) [P]: human instruction-following MOS correlates with tag error rate (ρ = −0.65). So tag-level error metrics are a usable proxy.
- **Filtering wrong tags pays off** [P, arXiv 2609.09929, ASR side]: removing about 27% of public samples whose tags Gemini judged wrong raised the challenge score from 37.6 to 39.3.

**Inference from that evidence [I]:**
- About **0.85 precision with about 0.65 recall** is enough for tag control to work at scale.
- Your data is about 1/50 the size of NVSpeech's 573 h. With few examples per tag, each wrong example is a larger share of the signal, so aim for **≥0.9 precision on exposed tags**.
- No study isolates the effect of false negatives against false positives [U]. The logic:
  - A false positive (tag present, no sound) teaches that the tag sometimes does nothing, which directly erodes control.
  - A false negative (sound present, no tag) teaches spontaneous bursts from plain text. That is tolerable in small doses, and it is exactly what a catch-all token (§9.6) is for.

### 3.6 Human-in-the-loop practice in the literature

- **NVSpeech and NV-Bench** [P]: 10 trained annotators with positive and negative examples; 5% double-annotated; Cohen's κ > 0.85 "on major categories."
- **NonverbalTTS** [P]: 3 annotators, majority vote.
- **ParaSpeechCaps** [P]: qualification test, then 3–5 raters per item.
- **CapSpeech** [P]: 2 engineers audited 500 random samples.
- **NVV-SuperBench** [P]: Gemini as verifier, then a human filter.
- No per-type agreement figures (e.g. sigh vs breath κ) were found [U].

**The standard pattern** [P]: hand-label a seed set, train or calibrate a model, auto-label the rest, and audit a random sample (NVSpeech went from 76 h to 573 h this way).

**At your scale the arithmetic is different [I]:**
- The candidate pool is roughly 1.5–2.5k short clips.
- At about 10 s of listening and deciding per clip, that is **4–7 hours of human work**.
- Complete human verification of candidates is therefore affordable, and it beats every automatic labeler listed above (in-dialogue F1 0.2–0.85).

---

## 4. How many examples a tag needs, and what to do with rare ones (research question 3)

**No paper gives a minimum number of examples per tag for fine-tuning.** The data points that do exist:

| Evidence | Setting | Result | Tag |
|---|---|---|---|
| VoxCPM issue #292 (2026-05-01) | Full fine-tune, 5 epochs. **~250 synthetic tagged sentences per tag** (4 new tags, 12 speakers), mixed with 8k untagged | New tags were "ignored, no affect at all." Existing tags still worked. Maintainer: needs "consistent and sufficient" paired data, "careful balancing" | [P] |
| IaraTTS-SFT-v3 (MOSS-TTS-Nano, 2026) | ~2,480 tag-conditioned pairs across 8 tags (~300 per tag); full SFT, 10 epochs | v1 and v2 had BPE-split tags, and **the sounds never appeared**. With single special tokens: ~50–60% rendering. The author plans upsampling or 15–20 epochs | [P card] |
| Reso-1 (Orpheus-3B LoRA, r64) | 7,342 clips, 8.9 h, single voice, with tag-event verification | Counts per tag not given. Orpheus already knew these tags from pretraining | [P card] |
| Elise / Unsloth Orpheus recipe | ~1,200 clips; 336 `<laughs>`, 156 `<sighs>` | Works, but these are tags the base model already knew | [P docs] |
| Irodori-TTS LoRA, community | ~120 clips | Removing the whisper clips stopped 👂 from working, i.e. emoji control survives in LoRA only if the examples are present | [S, note.com] |
| NVVSpeech Track 2 winner (arXiv 2609.14231, 2026-09-13) | DiTAR; 16 NVV tags as dedicated special tokens; synthetic top-up for rare tags | **English cry: 37 real + 1,000 synthetic gave the largest gain (+23.3 on the challenge's score Δ)**; yawn (17 real + 1,000) and burp also improved. "Low-diversity synthetic data gave inconsistent gains" (sneeze fell) | [P] |
| Same paper: rebalancing | Per-utterance repeat factor r = clip(√(f_max/f_rarest)·b, 1, r_max) | Used for long-tail SFT. No isolated ablation | [P] |
| NVVSpeech Track 1 (ASR side, arXiv 2609.23462) | Class sampling p ∝ n^α | **α = 0.5 (sqrt) was best.** Natural (α = 1) and uniform (α = 0) were worse. A uniform second stage helped the rarest tag but hurt most others | [P] |
| Meta "Beyond Words" (ASR side) | 400 cry examples | F1 32 → 69 once breath and laugh data were added. Generic-NV-token-first curriculum | [P] |

**What I take from this [I]:**

1. **Under ~50 real, consistent examples, a new tag almost never becomes a reliable control.** It still has a job, though: it *explains* the audio, which keeps the event out of the default voice.
2. **About 150–300 consistent real examples, as atomic tokens, is where control starts to work.** Even then, expect partial rendering (50–60%) unless you upsample or train longer. This agrees with the 150–300 rule in your v2 report.
3. **Synthetic top-up helps most for the rarest classes, but only if it is diverse.** Ten copies of the same twenty sighs is overfitting, not augmentation.
4. **Rebalance with square-root sampling and cap the repeat factor at about 3–5.** Uniform rebalancing hurts frequent classes.
5. **Rare tags gain from sharing acoustic structure with frequent ones.** That argues for **family-level merges** (gasp + audible inhale) over dropping tags.

**On condition dropout [I]:** Qwen3-TTS has no classifier-free guidance, so there is no CFG reason to drop conditions randomly.
- Dropping a **channel** label from a filtered clip teaches the *default* voice to sometimes come out filtered. That is the exact contamination you want to avoid, so **do not drop condition labels**.
- For NVV tags, the residual false negatives already supply all the "untagged event" exposure you want, so **do not add deliberate tag dropout** either.

---

## 5. Placement and tokenization (research question 4)

### 5.1 Where the tag goes

- **Inline position is standard** for point events: NVSpeech, Emilia-NV, NonverbalTTS, CosyVoice 2/3, Fish, ElevenLabs v3, Higgs, Step-Audio-EditX, and the NVVSpeech challenge all use it (see §2).
- NVSpeech says tags allow "context-aware insertion at arbitrary token positions" [P].
- **Placement method matters, but sub-word precision doesn't** [P, NonVerbalSpeech-38K, arXiv 2508.05385]:
  - Ordering tags by ASR word timestamps misplaced them (English tag position distance 8.49 with Whisper).
  - MFA forced alignment "show[ed] greater temporal drift."
  - **Their TSA method** cuts the audio at NVV boundaries, runs ASR on each piece, and aligns the pieces to the full transcript by edit distance. That brought position distance down to 0.74.
- **You already have exact transcripts.** Qwen3-ForcedAligner reports a 42 ms mean shift on Japanese [P], so locator time plus aligner gives position to within one or two characters [I].
- **Placement errors to avoid** [I]:
  - A tag on the wrong side of a clause boundary (before vs after 「、」).
  - A tag placed inside a word.
  - Snap tags to the nearest punctuation or bunsetsu boundary, and **never insert a space** next to a tag (your v2 report found ASCII spaces after Japanese punctuation linked to runaway generations [L]).
- **Isolated clips** are cleanest as the tag alone (e.g. `<sigh>`). Clips you have trimmed to a single event teach the event's acoustics; contextual clips teach how it enters and leaves speech. Keep both kinds [P, as your planning doc says].

### 5.2 How to encode the tag in a BPE LM-TTS

| Option | Tokenization in Qwen3-TTS | Has a pretrained meaning? | Evidence |
|---|---|---|---|
| Plain-text brackets `[sigh]` (v2) | **2–4 sequences per tag depending on the previous character** (§1.3) | Weak (English subwords) | IaraTTS: BPE-split tags gave no audible NVVs; atomic tokens did [P card] |
| **New special tokens** `<sigh>` mapped to unused ids 151676+ | 1 atomic token | None. Initialize it yourself | NVVSpeech Track 2 winner: "dedicated special tokens… exactly one token id and a learnable embedding" [P]; NVSpeech "extend[s] its vocabulary" [P]; CosyVoice2/3 tokenizer specials [P, §2] |
| Single-codepoint emoji (😮, 🤭, 📞) | 1 token, stable in context [L] | Some, from pretraining | Irodori-TTS uses emoji as its control interface (§7) [P] |
| Kana onomatopoeia (はぁ……, ふふっ, っ) | 2–3 ordinary tokens | **Yes.** The Qwen3-TTS maintainer says NVVs come out only "implicitly through expressions like 'hahaha' or 'emmm'… with a certain degree of randomness" [P, issue #9, 2026-01-23] | anime-whisper and VN scripts use it (§7) |

**Recommendation [I, grounded in the rows above]:**
- **Use atomic special tokens for tags and conditions.** They are context-invariant, they cost one position, and they cannot collide with ordinary text. The two papers above that state their encoding (the Track 2 winner and NVSpeech) both extend the vocabulary.
- **Initialize each new embedding row as the mean of the embeddings of a few related existing tokens.** For example, `<sigh>` from ため, 息, `sigh` and はぁ. Alternatively, sample from a Gaussian fitted to the existing rows (Hewitt 2021, as IaraTTS did). Do not leave the unused rows at whatever values they hold now.
- **Train only those rows** with PEFT `trainable_token_indices` (PEFT ≥ 0.15) alongside LoRA [P, PEFT docs].
  - The text embedding is not tied to an output head in the talker, so a full `modules_to_save` copy would also work.
  - But that copy is 151,936 × 2,048 ≈ 311M parameters and pointless [I].

**Kana is a complementary channel, not a competitor.** VN scripts already write articulated interjections as kana, and that is what the actress performed. Keep the kana as lexical text, and use tags only for non-lexical sound that the text does not already spell out (§9.1).

---

## 6. Channel, device and environment conditions (research question 5)

### 6.1 Facts about Qwen3-TTS that constrain the design

- **The codec preserves the channel.** The model card says the 12 Hz tokenizer "fully preserves paralinguistic information and acoustic environmental features" [P, HF card]. So the Amadeus EQ and the phone EQ will be reproduced faithfully, and whichever one the text does not explain will leak into the default.
- **After SFT, `ref_audio` enters only as one speaker-encoder embedding.** It is detached, written at position 6, and saved into `codec_embedding[3000]` at export [P, `sft_12hz.py`]. That makes it much weaker than in-context codec prompting.
  - Zero-shot codec LMs copy the prompt's acoustic environment: VALL-E with reverberant prompts [P, arXiv 2301.02111]; noisy prompts [P, arXiv 2406.05699].
  - So a filtered `ref_audio` would still pull the default voice toward that channel [I]. **Use one clean game clip as `ref_audio` for every row.**
- **The instruct slot exists, but the official SFT never trains it** (§8.1 point 4).

### 6.2 How the field labels and models channel and environment

| Approach | Examples | Evidence for disentanglement |
|---|---|---|
| **Learned environment/condition embedding or ID, with "clean" as its own value** | Tan et al., *Environment Aware TTS* (arXiv 2110.03887, Interspeech 2022): RIR-convolved LibriTTS/VCTK, separate speaker and environment encoders. Hsu et al. (ICASSP 2019): augmentation "with noise uncorrelated with speaker identity" plus adversarial factorisation. Lakshminarayana et al. (arXiv 2306.10152, 2023): learned "Aug. ID" embeddings, with the original data as the "clean" ID used at inference | Tan: disentangles even when speaker and environment were **fully entangled** one-to-one in training [P]. Hsu: "consistently synthesize[s] clean speech for all speakers," and every ablated component matters [P]. Lakshminarayana [S] |
| **Natural-language environment caption** | VoiceLDM (arXiv 2309.13664): description prompt plus content prompt, dual CFG. VoiceDiT (arXiv 2412.19259). Audiobox. Parler/Data-Speech SNR bins ("very noisy … very clean") and C50 bins ("very distant-sounding … very close-sounding"). Higgs Audio v2 `<\|scene_desc_start\|>…` system prompt | Works at scale (millions of clips). **No phone or device bins** in Data-Speech [P] |
| **Inline or global tag in a commercial or community TTS** | **Irodori-TTS: 📞 "電話越し、スピーカー越しのような音", 📢 echo/reverb, 👂 whisper, 🤐 muffled.** Untagged means ordinary clean speech. The tags were auto-annotated by a fine-tuned Qwen3-Omni-30B-A3B [P, EMOJI_ANNOTATIONS.md and cards]. ElevenLabs v3 lists **no** phone, radio or echo tags in its official docs [P] | Irodori is the closest precedent to your case (Japanese, emoji tokens, phone and echo as tags) |
| **Render the channel with DSP** | Qwen-Audio-3.0-Gen-Preview (arXiv 2607.27011, 2026-07-29, a scene-audio DiT, not TTS): "device and channel conditions are implemented as ordered processing chains" (band limitation, EQ, codec, packet loss, RIR) and "we retain both the clean input and each processed source track"; it flags the sim-to-real "mismatch" itself [P] | Augmentation at scale |
| **Filter it out** | Emilia keeps only DNSMOS ≥ 3.0 (arXiv 2501.15907) [P] | Why base models have no channel vocabulary [I] |

**What the field has converged on [I, from the table]:**
- The **default (untagged) state is clean**, and every degraded clip carries its label.
- Augmentation that pairs clean content with a degraded copy, uncorrelated with speaker and content, is what makes the label separable.
- Free-text captions are used when there are millions of clips. Tokens or IDs are used when data is small.

### 6.3 Applying this to your three effects

- **Amadeus**:
  - The game filter is large and consistent (652 clips). It also changes HNR (6.8 vs 11.8 dB) and bandwidth, not only the EQ, which suggests nonlinear processing that a post-hoc linear EQ cannot reproduce [I]. **So learn it as a token** rather than doing it with DSP.
  - The anime filter is a different, milder curve (§1.4). A single token for both media would learn an average that fits neither [I].
  - Fix this by crossing `<amadeus>` with the source token `<src_anime>` (§9.2). Plain `<amadeus>` then means the game sound.
  - Optional: apply the measured LTAS ratio to ~200 normal lines, tagged `<amadeus>`, to loosen the token's link to Amadeus-topic wording. Keep that share small, or the token drifts toward "pure EQ."
- **Phone** (28 real clips, all anime, and *not* narrowband):
  - **Never leave the 28 real clips in training untagged**, or their EQ leaks into the default. Tag them, or drop them if you choose DSP.
  - **DSP after generation is the most reliable way to get a phone sound.** A model token trained on your own augmentation mostly learns your DSP [I, as agent analysis of Hsu/Tan suggests].
  - If you want it in-model, augment 300–500 *diverse* clean clips with the **measured show curve** (§1.4) plus jitter, optionally with a minority of true telephony chains (pedalboard `GSMFullRateCompressor`/`Resample`/`Bitcrush`, SoX AMR-NB) [S tools]. Then check sim-to-real on the 28 real clips.
- **Echo/reverb, whisper, yell**:
  - Echo is a channel: tag it or exclude it, and use RIR convolution (BUT ReverbDB, OpenSLR) if you want it [P Tan; I].
  - Whisper and yell are vocal modes that DSP cannot fake. Irodori has 👂 and 😱 for them [P], but your counts (17 and 13 human labels) are far below threshold. Exclude whisper clips in v3; leave yelling to text.

### 6.4 Environment and channel labeling tools

| Tool | What it gives | Notes |
|---|---|---|
| Normalised LTAS (32–64 bands) + logistic regression | Device-EQ similarity | Train on game Amadeus vs game Kurisu, then apply to anime [I] |
| Brouhaha (pyannote) | Frame VAD, SNR, **C50** (reverberation) | Used by Data-Speech and ParaSpeechCaps [P] |
| torchaudio SQUIM | PESQ / STOI / SI-SDR estimates | Data-Speech [P] |
| DNSMOS / NISQA | Overall quality, plus NISQA's coloration and noisiness dimensions | Emilia uses DNSMOS for filtering [P] |
| Irodori-style fine-tuned Qwen3-Omni-30B-A3B | Emoji annotations, including 📞 and 📢 | No accuracy published [P/U] |

---

## 7. Japanese specifics (research question 6)

### 7.1 Corpora and research

- **JNV** (arXiv 2305.12445, 2023-05) [P]:
  - 420 clips, 4 speakers, 6 emotions, 87 NV phrases.
  - Phrases were crowd-sourced and **stored as hiragana**.
- **JVNV** (arXiv 2310.06072, 2023-10) [P]:
  - Emotional scripts with NV phrases written as kana (e.g. はは, extendable to ははは), and NV durations annotated.
  - In its FastSpeech2 experiment, **the phoneme representation of NVs "could not converge."** MOS-Emo was 1.9 for phonemes vs 2.4 for HuBERT-code NVs. The authors conclude "phoneme is not a proper representation for NVs."
- **Laughter synthesis, Xin et al.** (arXiv 2305.12442) [P]: phoneme baseline MOS 1.25 vs ~3.0 for pseudo-phonetic tokens.
  - These findings concern phoneme-input FastSpeech2. They do not transfer directly to a BPE codec LM whose pretraining text already contains kana NVVs [I].
  - Still, they are a warning that **kana alone under-specifies a laugh or a sigh**. That supports adding an event token next to the kana.
- **CSJ** [P, NINJAL docs]:
  - Corpus annotation, not TTS control.
  - Fillers (F), fragments (D), whispery voice (L).
  - Separate marks for laughing, crying or coughing *while speaking* vs breath, laughter or cough *without* speech. This is the same split as `speech_laugh` vs discrete `<laugh>`.
- **Galgame_Speech_ASR** (5,355 h, the basis of anime-whisper) [P card]:
  - No tags; NVVs are kana.
  - Normalisation: "……" → "…", 3+ repeated kana cut to 2, and a character whitelist of `。、!?…♪♡○`.
- **joujiboi/japanese-anime-speech** (110 h): no NVV policy stated [P].

### 7.2 What Japanese practitioners do

- **anime-whisper** [P card]:
  - "他モデルでスキップされがちな言い淀みや、笑い声や叫びや吐息などの非言語発話も忠実に書き起こす" (it faithfully transcribes hesitations, laughter, shouts and breaths that other models skip).
  - Output is kana inside the text, nearly always a single `…`, and sentence-final 。 mostly omitted.
  - **No brackets and no emoji.**
- **Style-Bert-VITS2** [P, `normalizer.py`]:
  - The normaliser deletes every character OpenJTalk cannot read, so brackets and emoji are stripped. `…` becomes `...`; 〜 becomes ー.
  - NVVs are kana or nothing. v2.7.0 adopted anime-whisper for transcription.
- **GPT-SoVITS**: convention not verified [U]. In practice it is ASR text, i.e. kana if transcribed by anime-whisper [I].
- **Irodori-TTS** (Aratako) [P]:
  - 39 emoji controls for NVVs, styles and channels, auto-annotated with a fine-tuned Qwen3-Omni-30B-A3B.
  - Repeating an emoji strengthens it, and control is "not perfect."
  - **The card's examples put the emoji next to the kana:** 「うぅ…😭そんなに酷いこと、言わないで…😭」, 「…👂😮‍💨👂😮‍💨こういうのが好きなんだ？」.
  - That is a hybrid: kana carries the sound, the emoji carries the category.
  - Mapping: 🤭 giggle, 😮‍💨 breath/sigh, 😮 gasp, 🌬️ panting, ⏸️ pause, 😭 sob, 😱 scream, 👂 whisper, 📞 phone, 📢 echo [P/S].
  - The v4 models also take a caption (VoiceDesign) input [P].
- **Japanese Parler-TTS** (2121-8): English captions for gender, pitch, speed and recording quality; no NVV handling [P].
- **Sarashina2.2-TTS**: style from the reference audio; no tags [P].

### 7.3 Tokenization specifics for Japanese

All measured locally [L] (agent cross-check [P]):
- Kana NVVs take one token per kana and share ids with ordinary words. The は in はぁ is the topic-particle token.
- `…` and `……` are different single tokens.
- Single-codepoint emoji 📞 📢 🤖 👂 😭 😱 🤭 😮 are one stable token each.
- 😮‍💨 (ZWJ) splits into 4 pieces, including byte fragments. ⏸️ is 2 tokens.

**For this tokenizer:**
- If you adopt Irodori-style emoji, use **single-codepoint** emoji: 💨 or 😤 for sigh, never 😮‍💨; ⏸ without the variation selector.
- Dedicated special tokens avoid the question entirely (§5.2).

### 7.4 Kana vs tags vs emoji for Japanese

| | Kana only (anime-whisper, SBV2) | Tag or emoji only | **Hybrid: kana + category token** (Irodori) |
|---|---|---|---|
| Uses Qwen's pretrained implicit NVVs | Yes (issue #9) | No | Yes |
| Unambiguous category | No (はあ？ "huh" vs はぁ sigh) | Yes | Yes |
| Handles NVVs the script never spelled | Only if you write kana | Yes | Yes |
| Evidence | Practitioner default [P] | Every 2025–26 inline-tag TTS (§2) [P] | Irodori card [P] |

**Recommendation: the hybrid** [I]:
- Keep the script and anime-whisper kana.
- Add one atomic category token where the event is non-lexical.
- Use the token alone where no kana exists.

---

## 8. Critique of your planning documents

### 8.1 `qwen3_tts_dataset_preparation_humanlike_emotional.md` (2026-08-21)

**What holds up:**
- Keep a master manifest and export a minimal JSONL.
- Annotate what is audible, not what the text suggests.
- Use a compact vocabulary and keep nuance in metadata.
- Keep ordinary speech as the majority.
- Keep timestamps.
- Never cut through an NVV.
- Avoid destructive denoising and per-clip peak normalisation.
- Use a probe set with negative controls and tag/no-tag pairs, and evaluate controllability separately from quality.

All of these agree with the literature in §3–§5.

**What needs correcting:**

1. **It is written for English and new recordings, not for 10k existing Japanese VN/anime clips.**
   - §4.3 says to "avoid onomatopoeia as the primary label." That conflicts with how Japanese data is written: VN scripts and anime-whisper spell interjections and laughs as kana (§7).
   - It also ignores Qwen3-TTS's only native NVV pathway, which is implicit, from text like はぁ (issue #9).
   - The right rule for Japanese is "kana for articulated sounds, tag for non-lexical ones."
2. **It assumes plain-text brackets are fine.** In this tokenizer they are not (§1.3, §5.2). Use atomic tokens.
3. **Its starter vocabulary is too large for these counts.**
   - It proposes 13 tags, with a "first experiment" of 8: sigh, laugh, giggle, speech_laugh, moan, groan, breath_in, gasp.
   - You have zero tagged laughs, essentially no moan/groan/grunt, and 17 breaths.
   - Its count tiers (<25 merge, 75–200 reasonable, 200+ strong) are in the right range but optimistic for *new* tokens (VoxCPM #292 and IaraTTS, §4).
   - The fine splits are not reliably labelable either: LAION's annotators agree only 19% on exact subtype.
4. **It has no notion of global conditions.** Amadeus, phone, singing, whisper, reverb and the "keep the default voice clean" problem are all missing.
   - §15 says the official SFT "does not expose a separate instruction field." That is true of `finetuning/dataset.py`, but the model itself has a ChatML instruct slot. `generate()` puts `<|im_start|>user\n{instruct}<|im_end|>\n` before the role tokens, embeds it with the text embedding, and passes it through `text_projection` [P, `modeling_qwen3_tts.py`]. A custom trainer can fill it (§9.7).
5. **It cites one benchmark under two names.** "NVBench https://arxiv.org/abs/2604.16211" with "github.com/AmphionTeam/NV-Bench" mixes two different benchmarks:
   - arXiv 2604.16211 is **NVV-SuperBench** (NJU/HKUST, 45 types).
   - **NV-Bench** is arXiv 2603.15352 (CUHK-SZ, 14 categories) [P].
6. **The `speech_laugh` and `breath_in`/`breath_out` splits are good linguistics but not labelable here.** Leave them in metadata.
7. **The 20–35% NVV share is unreachable without synthetic data** (you have about 2–4%). That is fine: the literature does not require it. What matters is consistency and enough examples per exposed tag.

### 8.2 `qwen3_tts_local_nvv_labeling_pipeline_supplement.md` (2026-08-21)

Your `kurisu_qwen3tts/pipeline` (s00–s11) implements this almost exactly [L].

**What holds up:**
- Aggregate probabilities by family (strongly supported: family mAP 0.87 vs fine 0.68; family agreement 65% vs exact-label 19%).
- Optimise for precision.
- Calibrate per class on character gold data.
- Send moan/groan/grunt to review.
- Keep raw scores, pin revisions, cache.
- Never infer NVVs from punctuation alone.

**What needs correcting:**

1. **The locator is not a good primary detector here.** Its real-audio references are ASR-derived (CrisperWhisper, en/de), F1 0.607 at IoU 0.5, "ceiling unknown" [P]. Expect much lower recall on Japanese anime/VN audio, and your calibration showed exactly that.
2. **It throws away the best label source in the corpus: the script.**
   - VN script kana (はぁ……, ふふっ, クスッ, ゴホン) are the writer's direction to the actress.
   - Rule §3 ("don't infer NVVs from punctuation") is right about `……` and `！`, but **kana interjections are not punctuation**.
   - Requiring an audio classifier with 6–23% sigh recall to "agree first" (`onomatopoeia_priors.yaml`) removes most true events.
   - Use the script as a high-precision *candidate* source and verify with a human, not with the classifier.
3. **"Missing an NVV label is usually less damaging" is only half right.**
   - Unlabeled events (55 nonlexical-only clips with no tag, ~80 laugh clips untagged) teach spontaneous NVVs into the default voice.
   - They also starve the tag of positives.
   - The fix is a catch-all token and a recall pass by humans over candidates, not lower thresholds.
4. **Multilingual-NVASR has no Japanese** [P]. CLAP zero-shot was never calibrated. CED is AudioSet-generic. None of the secondary validators is strong on Japanese.
5. **No channel or condition pipeline.** Delivery conditions were later handled with logistic regression on acoustic features. That cannot work for Japanese `[digital]` on scalar features alone (LOO AUC 0.886, never reached the precision floor), and §1.4 shows why: you need spectral *shape*, not tilt.
6. **The gold set itself needs an audit.** The reviewed anime tags are inconsistent (§1.1). Recalibrating thresholds against that set inherits its noise.

### 8.3 `kurisu_v2/research/final_report.md`: "drop tags until ~150–300, write NVVs as kana"

- **The 150–300 threshold is consistent with the evidence** in §4. So is the warning that tags under ~30 examples won't be learned.
- **The kana suggestion is right for articulated interjections** and matches the anime-whisper and VN-script convention.
- **But "drop tags" leaves the events in the audio unexplained**, which is what produces spontaneous sighs in the default voice. Kana also gives no handle on non-lexical sounds that the text never spelled out.
- **The v3 refinement:**
  - Keep the kana that is already in the script.
  - Add a *small* set of **atomic** family tokens, each pushed past ~150 verified examples through a recall pass by humans over candidates.
  - Add a catch-all `<nv>` for everything else.
- The suggestion to render `[pause]` as "……" or "、" is reasonable, but `[pause]` is the one tag you have in volume (558 clips) and it is measured, not guessed. As an atomic token it is worth keeping (§9.1).

---

## 9. Recommended labeling scheme for v3

This section is the deliverable. §10 and the source list after it are appendices.

The design rests on four principles:
- **Text describes what is audible.** Articulated sound goes in kana; non-lexical sound gets an inline event token.
- **The acoustic channel gets a global condition token.** It is factorised into *source* and *channel*, so no effect is ever left unexplained.
- **Everything is an atomic special token.**
- **Every exposed tag is human-verified.** The corpus is small enough to listen to every candidate.

Tags on each bullet in this section show what it rests on: [I] for my design, [P] or [L] where a fact is cited.

### 9.1 Text-side vocabulary (inline, point events)

| Token | Meaning (perceptual) | Merges | Current | Candidates to verify [L] | Status |
|---|---|---|---:|---|---|
| `<sigh>` | Audible non-lexical exhalation or sigh | Relief, exasperated and wistful sighs; breath-out; unvoiced ふぅ | 148 (113 isolated) | ~56 script はぁ/ふぅ clips, locator hits, empty-text clips | **Expose** once ≥150 verified |
| `<inhale>` | Audible inhalation, from soft intake to a sharp gasp | `[gasp]` 62 + `[breath]` 17 + Sharp Inhale; ハッ/ひっ when unvoiced | 79 | ~90 script ハッ/ひっ/キャッ, anime stammer inhales | **Expose if ≥150** after the recall pass; otherwise train but don't advertise |
| `<laugh>` | Non-lexical laughter or giggle, discrete | Giggle, chuckle, snicker, breathy laugh | **0** | ~80 script ふふ/クスッ/あはは, anime | Experimental until ≥150 |
| `<pause>` | Measured internal silence of at least 1.28 s where the text has no "……" | — | 558 | (measured) | **Keep** (atomic token; measured, not guessed) |
| `<nv>` | Any other clearly audible non-lexical vocalisation: strain (うっ when non-lexical), cough, sob, scream, hum, unknown | Everything below threshold | — | 55 untagged nonlexical-only clips; strain/cough kana | **Train, never advertise.** It absorbs events so the default voice stays clean |

**The kana–tag rule** (decided per event by the reviewer):

1. If the event is **articulated** (a vowel or consonant is clearly pronounced: はあ？ as "huh?", うん, えっ, ふーん), keep the script kana. **No tag.**
2. If the event is **non-lexical** (air, laughter pulses, a gasp), insert the event token at the event, and **keep any script kana that spells it**: `<laugh>ふふっ`, `<sigh>はぁ……`. The kana gives realisation and length; the token says "this is a laugh" [I]. Keeping the script text means the script stays exactly what was performed. At inference both `<laugh>` alone and `<laugh>ふふっ` are valid.
3. If an isolated clip has no lexical content, its text is the token alone, e.g. `<sigh>`, optionally followed by the script's punctuation (`<sigh>……`).
4. **Never write the same event twice** (`<sigh>はぁ……<sigh>`), and never put a space next to a token.

**Inline placement:**
- Combine the event's start time (from a human, or a verified locator hit) with Qwen3-ForcedAligner character timings.
- Snap to the nearest punctuation or bunsetsu boundary; never place a token inside a word.
- If the event overlaps speech (laughing while talking), don't create a `speech_laugh` token in v3; record it in metadata as `speech_laugh`.
  - **Do put `<nv>` at the start of the laughing span**, so the laughter is explained and does not leak into the default voice.
  - Promote it to its own token only if it reaches ≥150.

### 9.2 Condition labels (global, utterance-initial)

Factorise conditions so that each token means one audible property [I]:

| Token | Group | Assign to | Count |
|---|---|---|---:|
| *(none)* | source = game studio (default) | Dry VN game audio | ~8.4k |
| `<src_anime>` | source | Every anime clip (source-separated, lower HNR 10.3 vs 11.8 dB [L]) | ~1.6k |
| `<amadeus>` | channel | All game Amadeus (652). Anime Amadeus only where the device EQ is audible: verify with the LTAS-shape detector (§9.4), then by ear | ~652 + verified anime |
| `<phone>` | channel | **Route (b), in-model:** the 28 real (anime) clips plus 300–500 EQ-matched synthetic (§9.5). **Route (a), DSP after generation:** drop the 28 real clips (0.3% of the data, nothing lost). Either way, a phone clip never enters training untagged | 28 + ~300–500, or 0 |
| `<echo>` | channel | Only if ≥50 verified real clips, or RIR-augmented; otherwise **exclude** echo clips | — |

**Order:** `<src_anime>` first, then the channel token. Example: `<src_anime><phone>岡部、聞こえる？`.

**Why separate source from channel:**
- All 28 real phone clips and every subtle Amadeus clip are anime.
- Without a source token, `<phone>` would also learn "source-separation artefacts."
- With one, the synthetic phone clips made from game audio teach `<phone>` alone, and `<src_anime>` soaks up separation artefacts. Inference without `<src_anime>` then gives the clean studio voice.

**Excluded from v3** (too rare to control, and harmful if left in untagged):
- `[singing]` (53).
- `contamination` clips (noise 44, footstep 4, ringtone 1, knock 1).
  - **Your brief asks for sound effects to be learned and controllable, and this report disagrees for v3.** Fifty incidental, overlapping, non-vocal sounds cannot teach a controllable effect, and untagged they only dirty the voice. Exclude them.
  - If you want SFX control later, the evidence-backed route is CapSpeech's: mix clean library sound events into clean speech at −3 to 6 dB SNR, marked `<B>…</B>` (background) or `<I>…</I>` (inserted), with an event token [P, arXiv 2506.02863].
  - That belongs in a later version with a sound library, not in this corpus. Until then, mix effects in after generation.
- Verified whisper (17) and echo (29) clips, unless you promote them to tokens later.
- Yelling is prosody that the text (！) already conveys, so leave it untagged.

**Never drop a condition token** from a row that has the effect (§4). The single `ref_audio` must be a clean game clip with no condition.

### 9.3 Natural-language instructions: not in v3

- The Base model's instruct pathway exists, but only CustomVoice and VoiceDesign are documented as instruct-tuned [P, `qwen3_tts_model.py`; tech report]. Instruction data construction is not described.
- Captioner accuracy for Japanese anime emotion is unmeasured [U]. Captioners hallucinate events and follow the transcript's implied emotion (§3.3).
- With about 10k clips from one speaker, fixed tokens are more sample-efficient than free text [I].

**If you later want emotion control:**
1. Label a closed set (e.g. 6 emotions) with emotion2vec+ and Gemini 2.5 Pro used as a verifier of each proposed label.
2. Human-audit 300 clips, and keep a label only where the two models agree.
3. Put it in the instruct slot as one token, e.g. `<|im_start|>user\n<emo_angry><|im_end|>\n`, and A/B it against no emotion conditioning.

Emotion is largely predictable from the text, which is a reason to expect a small gain [I].

### 9.4 Labeling and verification tooling, sized to your counts

1. **Candidate generation**, for high recall (union of sources, all automatic):
   - **(a) Script kana.** Family regexes over game text (§1.2), with the false-hit patterns fixed (なきゃ, イクス).
   - **(b) anime-whisper diff.** Transcribe all anime clips and any game clip whose script ends in `……`/`っ`, or has none. Kana, っ or … that the reference text lacks marks an *unscripted* vocalisation. Subtitles never write NVVs, so this is the main source for anime. No initial prompt; `no_repeat_ngram_size` 5–10 [P card].
     - **Normalise both sides before diffing:** `……`→`…`, drop sentence-final 。, collapse 3+ repeated kana to 2 (Galgame convention). Otherwise the diff is mostly punctuation noise.
     - anime-whisper's training set (Galgame_Dataset) may well contain Steins;Gate [U]. On *game* clips it may just reproduce the script, so treat the diff as an independent signal mainly on anime clips.
   - **(c) vocalburst locator plus family classifier at a low threshold** (e.g. 0.3), as a *proposer*, not a decider.
   - **(d)** All 189 nonlexical-only clips.
   - **(e) Channel candidates.** Compute a 32-band normalised LTAS per clip. Train a logistic regression (or a small MLP on CLAP/BEATs embeddings) on game Amadeus (652) vs game Kurisu, and score every anime clip for an Amadeus-like EQ. Score the phone-curve distance (§1.4) for phone. Use Brouhaha C50 or `reverb_decay_db_per_s` for echo.
   - **(f) Optional ranking.** Gemini 2.5 Pro, or local Qwen3-Omni-30B-A3B (quantized), as a **closed-question verifier** ("Is there an audible sigh between 1.9 s and 2.6 s? yes / no / unsure"). Use it to sort the queue, never to write labels.
2. **Human verification of 100% of candidates.**
   - Use a keyboard-driven review UI (Label Studio's audio template or a small Gradio app). Show the waveform, spectrogram, transcript with the proposed token position, and the family scores.
   - Actions: accept, move, relabel family, `<nv>`, articulated (kana only), reject clip.
   - Re-audit the 427 previously reviewed anime clips with the same guideline.
   - Double-annotate a random 10% and report κ per family. Merge any family with κ < 0.6 into its parent or `<nv>` [I].
   - Expected load: about 1.5–2.5k candidates at ~10 s each, i.e. **4–7 h**.
3. **Residual-miss audit.** Listen to 300 random non-candidate clips. If more than ~3% contain an untagged salient event of a family, mine that family again [I].
4. **Export checks.**
   - Every token encodes to exactly one id.
   - No space is adjacent to a token.
   - Condition-token order is canonical.
   - Per-token counts by real/synthetic and by game/anime.
   - The tag set used in the test split appears in train.

### 9.5 Augmentation

- **`<phone>`**:
  - Two routes:
    - **(a) DSP after generation.** Most reliable and zero training cost. A model token trained on augmentation mostly learns your own DSP.
    - **(b) In-model control, which you asked for.** The steps below.
  - Apply the measured anime-phone EQ curve (§1.4) to 300–500 *diverse* clean game Kurisu clips (different scenes, lengths and emotions). Add mild compression and per-clip curve jitter of ±2–3 dB, so the token learns a family of phone sounds rather than one filter.
  - Optionally add a small share (≤20%) of true narrowband telephony (300–3,400 Hz, G.711/AMR) if you also want that sound.
  - Tag them `<phone>` with no `<src_anime>`, and keep `source: synthetic` in the manifest.
  - Keep all 28 real clips. Diversity matters: the NVVSpeech Track 2 winner saw large gains from 1,000 synthetic examples per rare tag, but "low-diversity" synthesis gave inconsistent results [P].
- **`<amadeus>`**: there are enough real game clips (652). Augmentation is optional. Its main use is to break content confounds (Amadeus lines are about AI and memory): apply the measured Amadeus-game EQ curve to ~200 ordinary Kurisu lines. Keep synthetic ≤30% of `<amadeus>` rows [I].
- **`<echo>`**, only if wanted: convolve clean clips with room impulse responses (OpenSLR 26/28) [I].
- **NVV events**:
  - Square-root repeat factors, capped at 3×, for rows containing `<sigh>`, `<inhale>` or `<laugh>` [P method, §4].
  - Optional experiment: splice verified isolated real sighs or inhales into phrase boundaries of other clips, SynParaSpeech-style. Cap these at 1:1 with real contextual examples, and A/B against no splicing; transitions will be less natural [I].
- **Never augment `ref_audio`.**

### 9.6 Minimum counts and merges

| Verified real occurrences (after the recall pass) | Treatment |
|---|---|
| **≥150** | Exposed control token. Ship it only if probes show a hit rate of at least 60% and false triggers of at most 5% on matched untagged prompts |
| **50–149** | Train it (it explains the audio), with sqrt upsampling (≤3×). Optional diverse synthetic top-up for *conditions* only. Document as experimental |
| **<50** | Merge into its family; if there is none, an event becomes `<nv>`. A **condition** with fewer than 50 real clips is either augmented to ≥300 (and the real clips kept tagged) or its clips are dropped. `<phone>`: route (b) or route (a) in §9.2 |

Merges fixed in advance: giggle/chuckle/laugh → `<laugh>`; gasp/breath-in/sharp-inhale → `<inhale>`; sigh/breath-out → `<sigh>`; cough, throat-clear, sob, cry, scream, strain, moan, groan, hum → `<nv>`; whisper and yell → no token.

### 9.7 How a row maps into Qwen3-TTS training

**Master manifest** (rich; never fed to the model):

```json
{"id":"game_ja_SG0_amadeus_00412","audio":"...wav","script_text":"……はぁ……そう。","text_v3":"<amadeus>……<sigh>はぁ……そう。",
 "conditions":{"source":"game","channel":"amadeus"},"events":[{"type":"sigh","start_s":0.41,"end_s":1.02,"char_idx":2,
 "articulated_kana":"はぁ","origin":"script_kana","verified_by":"human","reviewer":"A","confidence":"clear"}],
 "synthetic":false,"augment":null,"split":"train"}
```

**Export row**, option A (zero trainer change; recommended for v3):

```json
{"audio":"...wav","text":"<amadeus>……<sigh>はぁ……そう。","ref_audio":"ref_kurisu_game_clean.wav","language":"ja"}
```

**Zero-code fallback.** If you cannot change the tokenizer in the serving app, use single-codepoint emoji instead of new tokens. They are one stable token each [L] and Irodori-like:

| Token | Emoji |
|---|---|
| `<sigh>` | 💨 |
| `<inhale>` | 😮 |
| `<laugh>` | 🤭 |
| `<pause>` | ⏸ |
| `<nv>` | 😶 |
| `<phone>` | 📞 |
| `<amadeus>` | 🤖 |
| `<src_anime>` | 📺 |

All eight are verified as one token each [L]. Never use ZWJ emoji (😮‍💨), and keep everything else in the scheme identical.

**Option B** (once you want instructions): condition tokens move into the ChatML instruct slot. Your trainer must:
- Prepend `<|im_start|>user\n<amadeus><|im_end|>\n` exactly as `generate()` does: before the role tokens, embedded through the text embedding and `text_projection` [P].
- Shift the speaker slot and the `8+n_text` offsets, as for the language-id change in your v2 report.

The text channel then holds only spoken content and events.

**Tokenizer and model:**

```python
NEW = ["<sigh>","<inhale>","<laugh>","<pause>","<nv>","<src_anime>","<amadeus>","<phone>"]  # ids 151676.. (≤151935 free)
tok.add_special_tokens({"additional_special_tokens": NEW})
emb = talker.get_text_embeddings().weight          # talker.model.text_embedding [151936, 2048]; rows 151676+ are untrained (norm ~0.31) -> re-init
seed = {"<sigh>":["ため","息","sigh","はぁ"], "<inhale>":["息","吸","gasp"], "<laugh>":["笑","ふふ","laugh"],
        "<pause>":["……","pause"], "<nv>":["声","sound"], "<src_anime>":["アニメ"], "<amadeus>":["Am","ade","us","AI"],
        "<phone>":["電話","phone"]}
for t, words in seed.items():                       # mean-of-related-token init (Hewitt-style alternative is fine)
    ids = [i for w in words for i in tok.encode(w, add_special_tokens=False)]
    emb.data[tok.convert_tokens_to_ids(t)] = emb.data[ids].mean(0)
# LoRA as in v2 + PEFT trainable_token_indices={"talker.model.text_embedding": new_ids} (PEFT >= 0.15; check the
# module path your ft02_sft.py wraps). The inference app must load the same extended tokenizer.
```

- Keep `label_alignment=fixed`, `text_projection` on, the frozen speaker encoder, and one clean `ref_audio` for every row, all from your v2 trainer [L].
- Assert that every exported `text` round-trips with each token as exactly one id.

### 9.8 Evaluation probes that decide whether a token ships

- **Per event token:**
  - 30 unseen texts × 3 positions (initial, medial, final), plus the identical untagged text, plus 20 neutral negatives.
  - Score: hit rate (vocalburst family classifier on the generated audio, then a human spot check of 20%), false-trigger rate on untagged twins, anime-whisper kana-CER, speaker similarity to `ref_audio`, cap-hit rate.
- **Per condition token:**
  - The same 30 texts with and without the token.
  - Score: LTAS distance to the target curve (§1.4), **and LTAS distance of the *untagged* outputs to clean game Kurisu.** That second number is your "default stays clean" check.
- **Gate:** ship a token only if hit rate ≥60%, false triggers ≤5%, and the untagged default does not drift on LTAS or speaker similarity compared with a no-token v3 control run [I].

---

## 10. Could not verify

- **Per-tag minimum counts for fine-tuning a new token.** No paper measures this. §4 is triangulated from issues, model cards and challenge papers.
- **Whether false negatives or false positives hurt TTS control more.** No study isolates them.
- **Accuracy of any audio LLM on Japanese NVV or emotion labeling.** Gemini 3.x audio benchmarks; MiDashengLM, Audio Flamingo Next, Step-Audio-R1 and Kimi-Audio paralinguistic tables; and the Qwen3-Omni-Captioner hallucination rate (only case studies exist) were all not verified.
- **Whether Qwen3-TTS-Base saw instruct data in pretraining.** The report does not say.
- **Where Irodori's 📞 / 📢 training audio came from** (real or augmented), and the accuracy of its Qwen3-Omni emoji annotator.
- **Official NVVSpeech challenge leaderboard and test-set construction.** The site returned 404.
- **SenseVoice per-event F1**, CLAP/VoiceCLAP zero-shot accuracy on vocal bursts, and AudioSet per-class AP for sigh, breath and gasp.
- **The GPT-SoVITS NVV convention.** Also the formats of Kotoba-Speech, ELYZA-TTS, Anime-Llasa, MioTTS and T5Gemma-TTS.
- **Any public anime/VN NVV ground truth for Japanese.** None found.
- **Some survey rows rest on secondary sources only:** Fish "15,000 tags", Qwen-Audio-3.0-TTS "86 tags", MiniMax 2.6 vs 2.8 tag support, and Seed-TTS 2.0 `context_texts`.
- **The LTAS curves in §1.4 come from samples:** 150, 54 and 12 clips. The phone curve rests on 12 tier-A clips longer than 1.5 s, so re-measure it on all 28 before you build augmentation from it.
- **The dataset build.** The local copy (2026-09-05) differs slightly from the counts in your brief. Recount on the current build.

---

## Sources (accessed 2026-10-28 unless dated)

**Qwen3-TTS:**
- Tech report, arXiv 2601.15621 (2026-01-22).
- github.com/QwenLM/Qwen3-TTS: `finetuning/dataset.py`, `sft_12hz.py`, `qwen_tts/inference/qwen3_tts_model.py`, `qwen_tts/core/models/modeling_qwen3_tts.py`; issue #9 (2026-01-23).
- HF Qwen/Qwen3-TTS-12Hz-1.7B-Base: config `text_vocab_size` 151936; safetensors `talker.model.text_embedding.weight` [151936, 2048] [L].

**NVV labeling and ASR:**
- NVSpeech arXiv 2508.04195 (2025-08-06).
- Emilia-NV: huggingface.co/datasets/amphion/Emilia-NV.
- NonverbalTTS arXiv 2507.13155 (2025-07-17).
- NonVerbalSpeech-38K arXiv 2508.05385.
- MNV-17 arXiv 2509.18196.
- Meta "Beyond Words" arXiv 2607.01563 (2026-07-02, v2 2026-10-07).
- NVVSpeech Challenge Track 1: arXiv 2609.23462 (2026-09-23), arXiv 2609.09929.
- NVVSpeech Challenge Track 2: arXiv 2609.14231 (2026-09-13).
- Preference optimisation for NVV: arXiv 2608.24163 (2026-08-25).
- NV-Bench arXiv 2603.15352 (2026-03-16).
- NVV-SuperBench arXiv 2604.16211 (2026-04-17, v3 2026-06-14).
- Qwen3-ASR / ForcedAligner arXiv 2601.21337.
- huggingface.co/litagin/anime-whisper.
- huggingface.co/CharlesNi/Multilingual-NVASR.
- huggingface.co/laion/vocalburst-classifier-single, /laion/vocalburst-locator, /laion/vocal-burst-detector-x2.
- Qwen3-Omni report arXiv 2509.17765.

**Caption datasets:**
- ParaSpeechCaps arXiv 2503.04713.
- CapSpeech arXiv 2506.02863.
- SpeechCraft arXiv 2408.13608.
- EmoVoice arXiv 2504.12867.
- Parler-TTS arXiv 2402.01912.
- github.com/huggingface/dataspeech (`v02_text_bins.json`, `metadata_to_text.py`).

**Control representations:**
- CosyVoice 2 arXiv 2412.10117; CosyVoice 3 arXiv 2505.17589; FunAudioLLM/CosyVoice `tokenizer.py`.
- Fish S2 arXiv 2603.08823 (2026-03-09); Drama 3 arXiv 2610.18010.
- IndexTTS2 arXiv 2506.21619.
- Step-Audio-EditX arXiv 2511.03601.
- huggingface.co/bosonai/higgs-tts-3-4b (PROMPTING.md); Higgs Audio v2 card.
- huggingface.co/ResembleAI/chatterbox-turbo.
- github.com/nari-labs/dia and /dia2.
- github.com/canopyai/Orpheus-TTS.
- github.com/OpenMOSS/MOSS-TTSD.
- elevenlabs.io/docs/best-practices/prompting.
- ai.google.dev/gemini-api/docs/speech-generation.
- platform.minimax.io T2A docs.
- docs.inworld.ai audio markups.
- docs.mistral.ai Voxtral emotion tags.
- OV-InstructTTS arXiv 2601.01459.

**Fine-tuning evidence:**
- github.com/OpenBMB/VoxCPM/issues/292 (2026-05-01).
- huggingface.co/marcosremar2/iaratts-sft-v3.
- huggingface.co/decibel-hq/reso1-3b-en.
- unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning.
- huggingface.co/docs/peft/package_reference/trainable_tokens (v0.21.0).

**Channel and environment:**
- Environment-aware TTS arXiv 2110.03887.
- Hsu et al. ICASSP 2019 (research.google pub47834).
- Lakshminarayana et al. arXiv 2306.10152.
- VoiceLDM arXiv 2309.13664.
- VoiceDiT arXiv 2412.19259.
- VALL-E arXiv 2301.02111.
- Noisy-prompt TTS arXiv 2406.05699.
- Qwen-Audio-3.0-Gen arXiv 2607.27011 (2026-07-29).
- Emilia arXiv 2501.15907.
- IDEA-TTS arXiv 2412.16977.
- SpeechX arXiv 2308.06873.

**Japanese:**
- JNV arXiv 2305.12445; JVNV arXiv 2310.06072; laughter synthesis arXiv 2305.12442.
- CSJ docs (clrd.ninjal.ac.jp).
- Galgame_Speech_ASR_16kHz card.
- joujiboi/japanese-anime-speech.
- github.com/Aratako/Irodori-TTS and HF `EMOJI_ANNOTATIONS.md`.
- Style-Bert-VITS2 `normalizer.py`.
- 2121-8/japanese-parler-tts.
- Sarashina2.2-TTS arXiv 2606.25369.
