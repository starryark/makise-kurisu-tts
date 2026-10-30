# Is Qwen3-TTS-12Hz-1.7B still the right base for Kurisu? (state as of 2026-10-28)

This report extends `tts_survey_2026-10-28.md` and `final_report.md`; it does not repeat them. It re-checks their conclusions against sources published or updated through October 2026. It adds models those reports missed, and it corrects some of their claims.

**Tags:**
- **[P]** I read it in a primary source: a repo, commit, issue, HF card, HF API, arXiv paper, LICENSE file or PyPI.
- **[S]** Secondary: a blog, news article, aggregator or note/zenn/qiita.
- **[I]** My own inference.
- **[U]** Unverified.

All sources were accessed on 2026-10-28. Publication dates are given inline.

---

## 0. Answer in brief

1. **Yes, conditionally.** Qwen3-TTS-12Hz-1.7B-Base is still the best *single* open base that meets all four requirements, and nothing released between May and October 2026 replaces it [I].
   - It does not lead on any single axis. Two things favour it:
     - Its Japanese→Chinese intelligibility (CER) is near the top, and it is the best of the bases that have a working fine-tuning path. CosyVoice3 is slightly better in Qwen's own table (3.05 against 3.43) [P].
     - It is the only base with primary evidence that a single-speaker fine-tune trained on one language still speaks other languages. The recipe behind that result is not published (§3.2).
   - Each challenger fails at least one hard requirement:
     - **Irodori:** Japanese only.
     - **Fish S2 / Drama 3:** fine-tuning code removed from `main`; licence restrictions.
     - **IndexTTS-2.5, FireRedTTS3, Ming-3.0, Higgs 3:** no fine-tuning code.
     - **VoxCPM2, dots.tts, Confucius4-TTS:** no built-in tags; Japanese accuracy is no better than Qwen's.
2. **There is no new open Qwen TTS.**
   - The Qwen3-TTS weights have not changed since 2026-01-23.
   - `qwen-tts` 0.1.1 (2026-02-06) is still the latest release.
   - The 25 Hz models were never released.
   - Qwen3.5-Omni and Qwen-Audio-3.0-TTS are papers or APIs with no weights under the Qwen org [P] (§1.1).
3. **The real weakness is requirement 2 (accent), and the existing reports understate it.**
   - On character error rate (CER), Qwen is strong for Japanese voice → Chinese text: 3.43 [P].
   - However, in the only human cross-lingual evaluation found (Confucius4-TTS paper, Aug 2026), Qwen3-TTS ranked **last or second-to-last on "pronunciation authenticity"** in all four directions tested. That includes en→zh, where it ranked 4.00 of 6 against VoxCPM2's 1.88 [P] (§3.2).
   - No source measures accent for a Japanese voice speaking Chinese [U].
   - Fix this inside Qwen before switching:
     - Train with language-ID tokens. Your run used `language="Auto"`, the code path that ignores the language field [P #323].
     - Mix in native Mandarin and English data from other speakers as replay data [I, §5].
4. **Requirement 3 does not separate the candidates.**
   - No open model has a telephone or "device" channel tag [P for Higgs 3; I for the others].
   - Non-verbal tags exist in CosyVoice3, Higgs 3, Step-Audio-EditX, OmniVoice, Fish S2 and Drama 3, and as emoji in Irodori. None of these has published evidence that its tags survive a 10-hour single-speaker fine-tune [U].
   - On any base, `[digital]` (the Amadeus device voice) has to be learned from your own labels or applied as DSP.
5. **Switch only on measured failure** (§6). The main trigger: after the language-ID and replay retrain, native Mandarin listeners still hear a Japanese accent, while a VoxCPM2 LoRA or a voice-conversion (VC) pipeline trained on the same data does not.

---

## 1. What is new since the prior reports, and what changes

### 1.1 Qwen: no new open checkpoints

| Item | Finding | Tag / source |
|---|---|---|
| HF `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | createdAt 2026-01-21, **lastModified 2026-01-23**, sha `fd4b2543…`, Apache-2.0, 3.26M downloads | [P] https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| Qwen HF org, newest 60 models | Audio releases since January are only `Qwen3-ASR-{0.6B,1.7B}-hf` and `Qwen3-ForcedAligner-0.6B-hf` (2026-06-26). No TTS or Omni weights. Newest models are Qwen-Image-2.1 (Sep–Oct) and Qwen3.8 (Aug) | [P] https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1 |
| PyPI `qwen-tts` | 0.1.1 (2026-02-06) is still the latest | [P] https://pypi.org/project/qwen-tts/#history |
| `QwenLM/Qwen3-TTS` main | Last commit **2026-03-17**, `022e286` "fix finetuning bug": `talker_hidden_states = hidden_states[codec_mask[:, :-1]]` (was `[:, 1:]`) in `sft_12hz.py` | [P] https://github.com/QwenLM/Qwen3-TTS/commit/022e286 |
| PR #278 (label alignment) | Still open. A reviewer approved it on 2026-09-27. #337 (2026-07-06) and #371 (2026-09-16) report misalignment after the March commit | [P] https://github.com/QwenLM/Qwen3-TTS/pull/278 |
| 25 Hz models | Listed in the tech report's Table 1. Only a tokenizer bug (#369, 2026-09-13) mentions 25 Hz; there are no weights | [P] arXiv 2601.15621; issues page |
| Qwen3.5-Omni | Tech report arXiv 2604.15804 (2026-04-17, v2 04-21). Speech generation in **10 languages**. No weights under the Qwen org; an HF search shows only third-party merges | [P] https://arxiv.org/abs/2604.15804 ; [S] The Decoder says API-only https://the-decoder.com/?p=33841 |
| **Qwen-Audio-3.0-TTS** (new; prior reports missed it) | arXiv 2607.23938, 2026-07-27. 16 languages incl. ja/zh. 12.5 Hz tokenizer plus LM and flow matching. **86 inline tags incl. laughter/breathing/coughing/sighing**. Two-stage speaker adaptation (LM+FM SFT with replay, then FM only). **The paper says nothing about releasing weights or code** | [P] https://arxiv.org/abs/2607.23938 |
| CosyVoice 3.5 | Exists only as the Alibaba Cloud API models `cosyvoice-v3.5-plus` / `-flash`. The FunAudioLLM HF org has no new TTS since 2025-12-11 | [P] https://help.aliyun.com/zh/model-studio/cosyvoice-v3-5-plus ; [P] HF API `author=FunAudioLLM` |

**What this means for you** [I]:
- Your custom trainer is still necessary.
- The March 17 commit changes only the code predictor's hidden-state slice, and later issues still report misalignment on `main`.
- I did not check whether `022e286` matches your #337 fix; diff it against `ft02_sft.py` [U].

### 1.2 New or updated models since June 2026 that bear on the decision

| Model | Date | Why it matters | Why it does not replace Qwen | Tag |
|---|---|---|---|---|
| **dots.tts** (rednote hilab / dots studio) 2B; base, SOAR and MeanFlow checkpoints; `dots.tts.edit` | Initial release 2026-06; edit model 2026-08; doteBench 2026-09 | **Apache-2.0** with **official training script** (`scripts/train_dots_tts.py`). Highest Japanese speaker similarity measured: MiniMax-ja SIM **83.7** | Japanese CER is worse than Qwen's: MiniMax-ja 5.28; CV3-ja **8.23** against Qwen 6.48; zh→ja 16.1. No non-verbal tags in TTS mode | [P] https://github.com/rednote-hilab/dots.tts ; [P] Qwen-Audio-3.0 paper Tables 4/6 |
| **Confucius4-TTS** (NetEase Youdao) | HF created 2026-06-02, modified 2026-10-23; paper 2026-08-12 | **Apache-2.0** (HF tag); **T2S and S2A training scripts**; 14 languages. CV3 ja→zh 4.87, ja→en 3.44. Best overall human rank on en→zh, zh→ko and zh→ja | MiniMax-ja 4.14 / SIM 0.788 (no better than Qwen). No tags. 849★; young project | [P] https://arxiv.org/abs/2608.11650 ; https://github.com/netease-youdao/Confucius4-TTS ; HF API |
| **IndexTTS-2.5** | Weights 2026-08-10; paper v5 2026-08-11 | Best open zh→ja (6.38 / SS 75.8); kana `<word\|reading>` control | **No fine-tuning code.** Issue #518 has been open since 2025-10-24 with no maintainer reply | [P] https://github.com/index-tts/index-tts ; issue #518 |
| **FireRedTTS3** (Base / Instruct) | 2026-08-05 / 2026-08-13 | Apache-2.0. MiniMax-ja 3.60 / SIM 82.8 | **No training code.** README advises a reference clip in the target language and says the Base model "relies on explicit language tags" | [P] https://github.com/FireRedTeam/FireRedTTS3 |
| **Irodori-TTS v4.1-Small / v4-Large** | 2026-08-11 / 2026-09-27 | Best measured Japanese readings: Joyo Kana-CER 6.88 (v4.1-Small). Official LoRA, full fine-tuning and speaker inversion. Emoji non-verbal control | **Japanese only.** The Large model is under the Gemma ToU and is *worse* than Small on readings (7.96) | [P] https://huggingface.co/Aratako/Irodori-TTS-v4-Large |
| **Ming-3.0-tts-0.5B / tiny** | 2026-09-22 | MIT. MiniMax-ja 3.77 / 0.765. 14 inline tags. Open-weights Artificial Analysis Elo 1187 (#3) | Tags work **only in zh/en**. No fine-tuning code on the card | [P] https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B ; [P] AA leaderboard |
| **Fish Audio Drama 3** | 2026-10-01; v3.0.0-beta 2026-10-13 | 35B-A3B; inline tags; tops the open-weights Artificial Analysis leaderboard at Elo 1240 | MiniMax-ja **5.02**, worse than S2's 2.76. **Fine-tuning code removed** in v3.0.0-beta. Needs ≥48 GB for FP8 | [P] https://arxiv.org/abs/2610.18010 ; https://github.com/fishaudio/fish-speech/releases |
| **Aratako/Drama-3-Japanese** | 2026-10-22 | Shows that a LoRA (r32/α32) on Drama 3 is feasible without official code. JKYB Kana-CER 6.55 against 7.47 for the base. Japanese inline tags such as `[笑いながら]` | Trained on ~70k h of Japanese, not one speaker. zh/en "not evaluated". Licence forbids using outputs to train other models (§IV(b)) | [P] https://huggingface.co/Aratako/Drama-3-Japanese |
| **GPT-SoVITS V5** | Changelog 2026-10-05; **no release tag** (latest is 20250606v2pro) | "Significantly improved voice similarity without SoVITS fine-tuning"; new vocoder | No details on languages or data in the V5 notes | [P] https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/docs/en/Changelog_EN.md |
| Higgs TTS 3 | 2026-06-04 | Richest tag set | Non-commercial plus a Creator grant; **no fine-tuning code** | [P] https://huggingface.co/bosonai/higgs-tts-3-4b |
| Not relevant to us | maya2-native (Indic+en, 2026-10-26); Breeze-TTS-2 (en/zh); gradium phonon (en/eu, 2026-10-19); kitten-tts-2; LongCat-AudioDiT 1B/3.5B (**zh/en only**, MIT); Ming-3.0-audio-flash (126B MoE, zh/en) | — | [P] HF cards / HF trending API |

**Checked and found nothing new** [P unless noted]:
- **MOSS-TTS 2.0:** no HF entry under `OpenMOSS-Team`.
- **StepFun:** no audio model after Step-Audio-R1.1 (2026-01-14).
- **ByteDance-Seed:** no TTS on HF.
- **MiniMax:** speech is API-only; Speech 2.8 Turbo shipped 2026-07-16 [S].
- **Xiaomi MiMo-V2.5-TTS:** proprietary [S].
- **MegaTTS 3:** en/zh only, no 2026 update [P card / S search].
- **MaskGCT / Vevo2 / Kimi-Audio:** no 2026 Japanese-capable release found [U].
- **Kotoba-tech:** no TTS after kotoba-speech v0.1 [S].
- **Style-Bert-VITS2:** no successor; v2.7.0 (2025-08-24) is the latest [S].

### 1.3 Corrections to the prior reports

1. **"Qwen's cross-lingual results are among the best published."**
   - The CER numbers are correct [P].
   - But CosyVoice3 beats Qwen on ja→zh in Qwen's own table: 3.05 against 3.43 [P].
   - Human raters judge Qwen's cross-lingual pronunciation poorly [P, Confucius4 §3.2].
   - **Revised:** good intelligibility, unproven accent.
2. **"Sarashina measured Qwen3-TTS worst of five on Japanese."** This result is an outlier, and the prior reports should present it that way.
   - Sarashina's Qwen3-TTS CER is **63.11 even with Japanese prompts** (cross-style test, std 125), and raw Joyo Kana-CER is 185.9 ± 105.6 [P].
   - A CER above 100% means generation ran away, not that readings were wrong.
   - Every other source puts Qwen's zero-shot Japanese CER at 3.8–6.5 [P]:
     - Tech report: 3.82.
     - Confucius4 run: 3.82.
     - IndexTTS paper: 5.89.
     - Qwen-Audio-3.0 paper, CV3-ja: 6.48.
   - The paper does not name the Qwen checkpoint or `language` setting [P].
   - **Revised:** an unknown harness exposed instability (endless generation) [I]. The capped Kana-CER of 21.7 is still a real warning about reading stability, and it matches your own cap-hit findings [I].
3. **Direction matters.** Qwen's bad cross-lingual numbers are **zh→ja**:
   - 36.09 in the IndexTTS-2.5 paper.
   - 8.40 in its own report.
   - Ranked last in the Confucius human evaluation.

   **You need ja→zh**, where every source with numbers puts Qwen near the top [P]. The prior reports quoted the zh→ja number as if it applied to you.
4. **Drama 3 licence: a free HF Space is allowed.**
   - §III(c) excludes from "Hosted Service" "a free, non-commercial research or demonstration service (for example, a Hugging Face Space…)" that is not monetized or promoted as a production API [P LICENSE].
   - The "could not verify" item in the prior report is now resolved.
   - Also new: §IV(b) forbids using Drama 3 *outputs* to train other speech models, so you cannot use it to generate synthetic training data for Qwen [P].
5. **OmniVoice is worse than reported for your use case.**
   - CV3 ja→zh CER is **52.64** [P Confucius4 table].
   - Its README warns that cross-lingual output "will carry an accent from the reference audio's language" [P].
   - Weights are CC-BY-NC [P card].

---

## 2. Comparison table (requirement-focused)

**Column notes:**
- "JA" uses the best independent-ish number available, with its source.
- "ja→zh" is CV3-Eval CER (%) for a Japanese prompt speaking Chinese text.
- "FT" means an official or proven fine-tuning path.

| Model (latest, date) | R1: JA fidelity evidence | R2: ja→zh / accent evidence | R3: non-verbal / channel | R4: FT path | Licence | Verdict |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base** (2026-01-21; unchanged) | MiniMax-ja 3.82 / 0.788 [P report]; CV3-ja 6.48 [P QA3]; Sarashina outlier (§1.3) | **3.43** (own) [P]; human pronunciation rank **4.00/6 en→zh, 4.85/6 zh→en** [P Confucius4]; Bengali FT drifted to a Hindi-like accent [P #323] | No tags; non-verbals only implicit [P #9]. EmergentTTS paralinguistics 50.7–62.7% (en) [P dots README] | Official script is buggy; fixed via #278/#337 (yours). **Only base with published monolingual-FT → 10-language result** [P Table 9] | Apache-2.0 | **Keep (#1)** |
| VoxCPM2 (2026-04-03; report 2026-06-05) | MiniMax-ja 4.63 / **0.828** [P]; CV3-ja 5.88–5.96 [P] | **4.20** [P Confucius4 table]; human pronunciation rank **1.88 en→zh, 1.65 zh→en** (best or near-best) [P] | No tag list; `(style)` prefix, InstructTTSEval [P] | **Official LoRA + full**, documented; replay advice (10–20% zh/en) [P docs] | Apache-2.0 | **Fallback (#3) and the R2 challenger** |
| dots.tts SOAR (2026-06) | MiniMax-ja 5.28 / **0.837** [P]; CV3-ja 8.23 [P QA3] | zh→ja 16.1, ja→en 8.02 [P QA3]; no ja→zh | No TTS tags; edit model has pause/emotion/background tags [P] | Official training script (smoke config) [P] | Apache-2.0 | Watch |
| Confucius4-TTS (2026-06/08) | MiniMax-ja 4.14 / 0.788 [P] | **4.87**; best human overall rank on en→zh [P own paper] | None | Official T2S/S2A training scripts [P] | Apache-2.0 (HF tag) [P] | Watch (R2 candidate) |
| Fun-CosyVoice3-0.5B-2512 (2025-12) | CV3-ja 10.4 without RL, 5.15 with RL [P]; polyglot-SFT Japanese ≈9% CER [P §5.8] | 3.05 (in Qwen's table) or 6.86 (0.5B, Confucius4 run) [P] — conflicting | **Built-in** `[breath] [laughter] [sigh] [cough]…`, `<strong>` [P] | Full SFT; **published polyglot-speaker recipe** [P §2.6.1] | Apache-2.0 | Borrow the recipe, not the model |
| IndexTTS-2.5 (2026-08-10) | test-ja 5.30 / SS 75.4 [P own] | zh→ja 6.38 (best); ja→zh not reported | Emotion vector only | **None** [P #518] | bilibili licence | Zero-shot helper only |
| Fish S2 Pro (2026-03-09) | MiniMax-ja **2.76** / 0.796 [P] | KIT IWSLT: en→zh with language tags, CER 11.4→12.1 [P] | Free-form `[tag]`; docs conflict on whether brackets are control tokens [S] | Only at tag v2.0.0-beta; doc says LoRA learns "patterns, not timbre" [P] | Research, non-commercial | No |
| Fish Drama 3 / Drama-3-Japanese (2026-10) | MiniMax-ja 5.02 (base) [P]; JKYB Kana-CER 6.55 (JA LoRA) [P] | Not reported | Point and scoped tags; JA tags in the Aratako model [P] | Official code removed; one large community LoRA proven [P] | Community v1.0; free Space OK; no training other models on its outputs [P] | Only if JA dominates and ≥48 GB is available |
| Irodori v4.1-Small (2026-08-11) | **Joyo Kana-CER 6.88; JSUT CER 7.22** [P] | **None (JA only)** | Emoji non-verbals, "not always perfectly consistent" [P] | **Official LoRA / full / speaker inversion** [P] | MIT | **JA-split option (#2)** |
| FireRedTTS3 (2026-08) | MiniMax-ja 3.60 / 82.8 [P] | Not reported; recommends a same-language prompt [P] | Instruct editing only | None [P] | Apache-2.0 | No |
| Ming-3.0-tts-0.5B (2026-09-22) | MiniMax-ja 3.77 / 0.765 [P] | Not reported | 14 tags, zh/en only [P] | None on card [P] | MIT | No |
| Higgs TTS 3 4B (2026-06-04) | "WER/CER < 5 tier" only [P] | Not reported | Best vocabulary (`sfx`/`prosody`/`style`); **no channel tags** [P] | None [P] | Non-commercial + Creator grant | No |
| OmniVoice (2026-03-30) | MiniMax-ja 4.03 / 0.828 [P] | **52.64**; README says accent carries over [P] | 13 tags [P] | `examples/` [P] | CC-BY-NC weights [P] | No |
| Step-Audio-EditX (2026-01-29 update) | `[Japanese]` tag; no metrics [P] | — | 22 tags [P] | SFT/DPO/GRPO scripts [P] | Code Apache; **weights licence unstated** [P] | No |
| Sarashina2.2-TTS (2026-06-24) | CV3-ja SIM 74.75; robust to non-JP prompts [P] | **JA+EN only** [P card] | — | Not documented [P card] | Non-commercial | No |
| T5Gemma-TTS-2b-2b (2025-12) | JSUT 7.63 (Sarashina) [P] | en/zh/ja; no metrics [P] | — | Training code on GitHub [P] | Gemma + CC-BY-NC | No |
| GPT-SoVITS V5 (2026-10-05, untagged) | No metrics [P] | Wiki: large single-language data erodes cross-lingual ability [P prior] | — | Full GPT+SoVITS training [P] | MIT | Baseline only |

---

## 3. Evidence by requirement

### 3.1 R1: Japanese fidelity

**Zero-shot Japanese numbers** (lower CER is better). Every number is from a different harness, so compare only within a column [I].

| System | Tech report / MiniMax-ja CER, SIM | CV3-ja CER (Qwen-Audio-3.0 Tbl 4) | IndexTTS-2.5 test-ja WER, SS | Kana-CER (Joyo, own card) |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B | 3.82, 0.788 [P] | 6.48 [P] | 5.89, 70.2 [P] | 21.7 capped (Sarashina) [P] |
| VoxCPM2 | 4.63, 0.828 [P] | 5.88 [P] | 6.69, 72.9 [P] | — |
| Fish S2 Pro | **2.76**, 0.796 [P] | — | 5.15, 66.2 [P] | — |
| IndexTTS-2.5-RL | — | — | **5.30, 75.4** [P] | — |
| dots.tts SOAR | 5.28, **0.837** [P] | 8.23 [P] | — | — |
| FireRedTTS3 | 3.60, 0.828 [P] | — | — | — |
| CosyVoice3-1.5B | — | 7.57 [P] | — | — |
| Irodori v4.1-Small | — | — | — | **6.88** (JSUT CER 7.22) [P] |
| Drama-3-Japanese | — | — | — | 6.55 (JKYB) [P] |
| Qwen-Audio-3.0-TTS (closed) | — | **4.78** [P] | — | — |

**What actually decides R1** [I]:
- Zero-shot numbers matter less than **post-fine-tune reading accuracy on unseen kanji-heavy text**, and no candidate publishes that for a 10-hour single speaker [U].
- On readings, the Japanese-specialised models (Irodori, Drama-3-Japanese) currently have the best published numbers.
- Qwen has the most stability warnings: the Sarashina runaways, your own cap hits, and issues #55 and #386.
- So the R1 test is your own anime-whisper Kana-CER plus cap-hit rate on held-out text, compared directly against an Irodori v4.1-Small LoRA trained on the same data.

### 3.2 R2: Chinese and English in the same voice, natively

**Cross-lingual CER (CV3-Eval)**, with a Japanese prompt and target text in another language:

| System | ja→zh | ja→en | Source |
|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base | **3.43** | **3.04** | [P] Qwen3-TTS report Tbl 7 (reproduced in Qwen-Audio-3.0 Tbl 6) |
| Qwen3-TTS-25Hz-1.7B-Base (unreleased) | 3.92 | 3.95 | [P] same |
| CosyVoice3 | 3.05 | 4.20 | [P] Qwen3-TTS report Tbl 7 |
| CosyVoice3-1.5B | 6.78 | 5.39–5.94 | [P] Qwen-Audio-3.0 Tbl 6; Confucius4 table |
| VoxCPM2 | 4.20 | 4.10 (6.22 in QA3) | [P] Confucius4 table (QA3 Tbl 6) |
| Confucius4-TTS | 4.87 | 3.44 | [P] own paper |
| MiniMax-Speech-2.8-HD (closed) | 6.07 | 5.79 | [P] QA3 Tbl 6 |
| ElevenLabs v3 (closed) | 6.17 | 5.30 | [P] QA3 Tbl 6 |
| OmniVoice | 52.64 | 5.25 | [P] Confucius4 table |
| Qwen-Audio-3.0-TTS (closed) | 3.29 | 3.15 | [P] QA3 Tbl 6 |

**Human ratings of cross-lingual pronunciation** come from one source only: Confucius4-TTS, arXiv 2608.11650, §human evaluation [P]. Each cell is a mean rank from 1 to 6, where 1 is best. Raters were native or fluent speakers of the target language. Rater count and significance are not reported [U].

| Direction | ElevenLabs | MiniMax | OmniVoice | **Qwen3-TTS** | **VoxCPM2** | Confucius4 |
|---|---|---|---|---|---|---|
| en→zh pronunciation | 2.10 | 4.12 | 2.73 | **4.00** | **1.88** | 1.80 |
| zh→en pronunciation | 2.48 | 4.48 | 3.05 | **4.85** | **1.65** | 1.85 |
| zh→ja pronunciation | 1.65 | 3.90 | 3.20 | **5.80** | 4.25 | 2.05 |
| zh→ko overall | 3.65 | 2.95 | 3.55 | **5.85** | 3.80 | 1.20 |

**Caveats on the human ratings:**
- The authors are a competitor, and they used an internal test set [I].
- There is **no ja→zh direction**.
- These are zero-shot clones, not fine-tuned voices.

Even so, it is the only human pronunciation judgement covering Qwen. It matches your symptom of Chinese with a Japanese accent [I].

**Evidence that matters after fine-tuning:**
- **Qwen tech report §4.2.5 / Table 9** [P]:
  - One speaker was fine-tuned "exclusively on monolingual data". The resulting CustomVoice model keeps 10 languages, with WER zh 0.903 and ja 4.924 for 12Hz-1.7B.
  - This is the only primary evidence for any candidate that a single-speaker fine-tune stays multilingual.
  - It measures WER only, not accent.
  - The in-house CustomVoice recipe is not described. Whether it used replay data or language-ID tokens is unknown [U].
  - So the table does **not** show that the public `sft_12hz.py` path, or yours, keeps other languages. The language-ID and replay plan in §4 is a hypothesis to test [I].
- **Issue #323 (2026-05-24, closed as "not planned" 2026-09-25)** [P]:
  - A 7-hour single-speaker Bengali fine-tune of 1.7B-Base came out with a Hindi-like accent.
  - A commenter traced this to the official `dataset.py` hard-coding the `nothink` prefill, so **the language field is ignored during training**.
  - The proposed fix (register a language ID, build the prefill dynamically, shift the speaker-embedding index) was never tested in the thread.
  - Your run (`language="Auto"`) has the same property [I].
- **CosyVoice3 polyglot-speaker recipe (arXiv 2505.17589 §2.6.1, §5.8)** [P]:
  - Mix studio-quality monolingual data from random speakers covering every language with the target speaker.
  - Put the speaker ID and language ID in an instruction, e.g. "You are Speaker B. Please speak German."
  - Result: CER/WER below 4% for every language except Japanese (about 9%) and Korean (about 6%).
- **Qwen-Audio-3.0 speaker adaptation** [P]:
  - Stage 1: joint SFT with a **replay subset**.
  - Stage 2: refine the flow model only, with the LM frozen.
  - Not reusable directly, but it confirms that replay is the in-house practice.
- **Accent-leakage literature, 2026** [P]:
  - **Region-Aware Masking** (Netflix, arXiv 2610.07524, 2026-10-05; F5-TTS, es→en):
    - Accent score (0–5) was 4.2–4.4 without adaptation.
    - **Bilingual adaptation alone brought it to 0.45–1.05.**
    - Masking cut it a further 18–55%.
  - **KIT, IWSLT 2026** (arXiv 2606.07240; S2-Pro): language tags raised language-ID confidence from 89.85% to 91.64%, but Chinese CER went slightly *up* (11.37 → 12.05).
  - Neither tests Japanese or Chinese voices; the transfer to your case is [I].

### 3.3 R3: non-verbal events and channel effects

- **Built-in tag sets** [P]:
  - CosyVoice3: `[breath] [laughter] [sigh] [cough] [quick_breath] [lipsmack]…`
  - Higgs 3: `sfx` (sigh, laughter, cough…), `prosody` (pause, long_pause…), `style` (whispering, singing, shouting).
  - Step-Audio-EditX: 22 tags.
  - OmniVoice: 13 tags.
  - Ming-3.0: 14 tags, zh/en only.
  - Drama 3: point and scoped tags.
  - Irodori: emoji.
  - Qwen-Audio-3.0: 86 tags (closed).
  - Qwen3-TTS: **none**.
- **Channel effects:**
  - No open model documents a telephone, radio or device tag. Higgs 3's full list has none [P].
  - S2 and Drama 3 tags are free-form and untested for this [S/I].
  - `dots.tts.edit` accepts a "background" edit instruction, but that edits existing audio; it is not a TTS tag [P].
  - Whether the closed Qwen-Audio-3.0's 86 tags include channel effects is not stated [U].
- **Do tags survive fine-tuning?** No primary single-speaker evidence exists for any model [U].
  - Irodori: a note.com LoRA user found that removing whisper clips from the data disabled 👂 [S, prior report].
  - Fish's own doc advises against fine-tuning the RL-trained S2 [P].
  - CosyVoice3's SFT keeps instruction following by mixing in the pre-training instruction data [P §2.6.2]. This is the same replay principle as for R2.
- **English benchmark (EmergentTTS-Eval, paralinguistics win-rate against gpt-4o-mini-tts)** [P dots.tts README]:
  - Qwen3-TTS: 50.7% (reference clone), 62.7% (Ryan preset).
  - dots.tts SOAR: 52.7%.
  - VoxCPM2: 44.1%.
  - Qwen's "Foreign words" category is weak at 25.4% / 17.1% [P].
- **Practical conclusion** [I]:
  - R3 is a data and recipe problem on every base.
  - Use dedicated tokens and at least about 150–300 examples per tag.
  - Use replay so the base's prosody is not overwritten.
  - Render `[digital]` and the telephone effect with deterministic DSP, so control does not depend on 17 labelled clips. Options:
    - A band-pass filter of about 300–3,400 Hz for telephone.
    - An EQ or impulse response for Amadeus, matched to your `[digital]` clips.

### 3.4 R4: fine-tuning paths and licences (personal fan project)

- **Workable today with documented code:**
  - Qwen3-TTS: needs your fixed trainer.
  - VoxCPM2: LoRA and full.
  - dots.tts.
  - Confucius4-TTS.
  - Irodori: LoRA, full and speaker inversion.
  - CosyVoice3: full SFT.
  - GPT-SoVITS.
- **Workable with your own code:** Drama 3, via Aratako's LoRA [P]; Fish S2 at v2.0.0-beta [P].
- **No path:** IndexTTS-2.5, FireRedTTS3, Ming-3.0, Higgs 3.
- **Licences:** all are acceptable for a non-monetized fan project. A free HF Space is explicitly allowed for Drama 3 [P]. The non-commercial ones are S2 Pro, Higgs 3, OmniVoice, Sarashina and T5Gemma; they would matter only if the project were monetized.
- **Ethics:** Irodori and VoxCPM cards forbid impersonation without consent [P]. The prior report's Nippairen note still applies [S].

---

## 4. Ranked recommendation

1. **Keep Qwen3-TTS-12Hz-1.7B-Base as the production base**, and retrain it as follows [I unless noted]:
   1. **Language-ID training.** Port the #323 change so each row carries its language ID [P design]. Japanese rows get `japanese`. At inference, pass `chinese` or `english`; stop using `Auto` for non-Japanese text.
   2. **Replay data.** Add 10–20% native Mandarin and English speech from many other speakers, each with its own speaker embedding and language ID. This is adapted from the VoxCPM docs' 10–20% advice, which they give for adding a new language, not for single-speaker replay. It also draws on the CosyVoice3 polyglot recipe [P] and the Netflix result that bilingual adaptation alone removes most accent [P]. Keep Kurisu's embedding fixed at inference.
      - Optionally include the 374 ZH-dub clips **as a separate speaker**. Never label them as Kurisu, or the two timbres will blend [I].
   3. **Tags** as described in the prior report: dedicated tokens, at least ~150–300 examples per tag, and kana for the rare ones. Do `[digital]` and the telephone voice as DSP.
   4. **Evaluate** with native-listener accent ratings (0–5, using the Netflix protocol) on fixed ZH and EN probes, plus anime-whisper Kana-CER and the cap-hit rate.
2. **Irodori-TTS v4.1-Small (MIT) as the Japanese engine, if the R1 test favours it.**
   - It has the best published Japanese reading accuracy and an official LoRA, full fine-tuning and speaker inversion path [P].
   - The cost is two engines: Irodori for Japanese, Qwen or option 3 for Chinese and English.
3. **VoxCPM2 LoRA (Apache-2.0) as the all-in-one fallback, and the R2 challenger.**
   - It has the best human-rated cross-lingual pronunciation among open models that can be fine-tuned [P Confucius4].
   - Its ja→zh CER is 4.20 [P], its fine-tuning is the best documented [P], and its Japanese speaker similarity is 0.828 [P].
   - It has no tags, and its Japanese CER is slightly worse than Qwen's [P].
4. **Watch list:**
   - **Confucius4-TTS:** Apache-2.0, training code, strong cross-lingual human ranks; the project is young.
   - **dots.tts:** Apache-2.0, training code, highest Japanese SIM, weaker Japanese CER.
   - Both need your own fine-tuning to prove anything [I].
5. **Fallback for R2 without changing TTS:**
   - Generate Chinese and English with a native-speaker voice: the ZH dub actress through VoxCPM2 or CosyVoice3 zero-shot, or a native preset.
   - Then convert it to Kurisu's timbre with a VC model fine-tuned on your Japanese data.
   - Pronunciation and prosody come from a native speaker, and timbre comes from Kurisu [I].
   - Seed-VC is the usual choice, but its original repo was archived on 2025-11-21 [S]. X-VC (2026) reports strong cross-lingual similarity [S, arXiv 2604.12456].
   - Untested on this data [U].

**Not recommended now:**
- **Fish S2 Pro:** fine-tuning lives only on an old tag, and the doc says LoRA does not learn timbre.
- **Drama 3:** heavy, Japanese CER of 5.02, and fine-tuning code removed. Reconsider Drama-3-Japanese only if R1 dominates and you have ≥48 GB.
- **IndexTTS-2.5, FireRedTTS3, Ming-3.0, Higgs 3:** no fine-tuning code.
- **CosyVoice3:** weaker Japanese, but borrow its recipe.
- **OmniVoice:** ja→zh is broken.
- **Sarashina:** no Chinese.

---

## 5. Minimal experiment plan to settle the decision

The first two experiments can run in parallel [I].

1. **A. Accent baseline (no training).**
   - Render 30 Chinese and 30 English probe sentences from the current production model with `language="Chinese"`/`"English"` and with `"Auto"`.
   - Render the same probes from VoxCPM2 and Confucius4-TTS zero-shot, with a 10 s Kurisu Japanese prompt.
   - Have three or more native Mandarin listeners rate accent from 0 to 5.
   - This tells you whether the problem is the fine-tune (language ID) or the base.
2. **B. Qwen retrain** with language ID and replay (§4, step 1). Repeat the same listener test.
3. **C. VoxCPM2 LoRA** (r32, lr 1e-4, 1–3 epochs, 30–50% of rows with `ref_audio`) on the same data and replay mix [P docs]. Run the listener test, plus Japanese Kana-CER.
4. **D. Irodori v4.1-Small LoRA** on the Japanese data only. Compare Japanese Kana-CER, cap hits and blind listening against B.

---

## 6. What would make us switch (explicit triggers)

| # | Trigger (measured on your data) | Switch to |
|---|---|---|
| 1 | After B, the mean native-listener accent rating for ZH stays **≥ 2/5**, while C or the VC pipeline reaches **≤ 1/5** with timbre preserved | VoxCPM2 LoRA (or VC) for zh/en; keep Qwen for Japanese if R1 holds |
| 2 | Qwen's held-out Japanese Kana-CER is **> ~2 points worse** than an Irodori LoRA, or its cap-hit rate stays **> 2%** on untagged held-out text after the fixes | Irodori v4.1-Small for Japanese (two-engine setup) |
| 3 | VoxCPM2 LoRA matches Qwen on Japanese Kana-CER **and** speaker similarity, and wins trigger 1 | VoxCPM2 as the single engine (simpler than two) |
| 4 | Qwen publishes open weights for Qwen-Audio-3.0-TTS, the 25 Hz Qwen3-TTS, or a successor with tags (check HF `author=Qwen` monthly) | Evaluate immediately; most likely a straight upgrade, since your pipeline carries over [I] |
| 5 | IndexTTS-2.5 or Fish S2 / Drama 3 ships official single-speaker fine-tuning code (#518; fish-speech `main`) | Trial: IndexTTS-2.5 has the best zh→ja result (6.38), Japanese SS of 75.4 (second only to CosyVoice3-0.5B's 76.4 in its own table) and kana control; S2 has the best Japanese CER |
| 6 | Tags (R3) become the top priority and fail on Qwen even with ≥150 examples per tag | CosyVoice3 (built-in non-verbal tokens and a polyglot SFT recipe), at the cost of Japanese accuracy [I] |
| 7 | The project becomes monetized or a hosted API | Drop S2, Drama 3 (above the revenue threshold or as a hosted service), Higgs, OmniVoice, Sarashina and T5Gemma; the Apache/MIT options remain |

**Not triggers:**
- Leaderboard Elo. Artificial Analysis is English and preset-voice; Qwen3 TTS sits at 929 with no open-weights badge [P]. I treat it as irrelevant to Japanese [I].
- Zero-shot Japanese CER differences under about 1 point.

---

## 7. Could not verify

1. **Any accent measurement for a Japanese voice speaking Mandarin (ja→zh), for any model.** CER tables exist; listener or classifier accent scores do not. This is the most decision-relevant gap.
2. Whether language-ID training plus replay removes the Japanese accent in a fine-tuned Qwen3-TTS. #323's fix was never tested. The recipe behind Qwen's Table 9 (monolingual CustomVoice fine-tune that stays multilingual) is undisclosed, so it does not show that replay or language IDs were used.
3. Any candidate's metrics for a 10 h+ single-speaker fine-tune, or whether built-in tags survive one.
4. Rater count and significance in the Confucius4-TTS human evaluation, and whether its Qwen3-TTS runs used `language` tags.
5. Which Qwen3-TTS checkpoint and settings the Sarashina and IndexTTS-2.5 papers used.
6. Why CosyVoice3's ja→zh CER differs between sources (3.05 in Qwen's table, 6.78/6.86 in the Confucius4 and Qwen-Audio-3.0 runs). The cause is probably different checkpoints or harnesses [I].
7. Whether commit `022e286` (2026-03-17) fully fixes the code-predictor alignment problem; #337 and #371 suggest it does not.
8. Whether any Qwen3.5-Omni variant has open weights. Third-party "Light on HF" claims [S] conflict with the HF org listing [P].
9. Whether Qwen-Audio-3.0-TTS weights will ever be released.
10. Whether ms-swift PRs merged in 2026-07 (mentioned in the #278 thread) give a working Qwen3-TTS fine-tuning path [U].
11. Confucius4-TTS parameter count and the licence text inside the repo (HF tag says Apache-2.0).
12. Whether dots.tts fine-tuning is full-parameter or supports LoRA (the config is a "smoke" config).
13. Step-Audio-EditX weights licence (still unstated).
14. Whether Fish S2 treats `[brackets]` as control tokens or plain text (sources conflict).
15. Seed-VC archive status, and X-VC licence and singing support.
16. 2026 updates to MaskGCT, Vevo2, Kimi-Audio and Spark-TTS (none found).

---

## 8. Sources (primary unless marked)

**Qwen**
- Qwen3-TTS repo commits (last commit 2026-03-17): https://github.com/QwenLM/Qwen3-TTS/commits/main ; commit https://github.com/QwenLM/Qwen3-TTS/commit/022e286
- Issues #323, #337, #371, #378, #386, PR #278: https://github.com/QwenLM/Qwen3-TTS/issues/323 , https://github.com/QwenLM/Qwen3-TTS/pull/278
- Official fine-tuning README (single-speaker only): https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
- HF API for the 1.7B-Base model (modified 2026-01-23): https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base ; Qwen org listing: https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1
- PyPI (0.1.1, 2026-02-06): https://pypi.org/project/qwen-tts/
- Qwen3-TTS Technical Report (2026-01): https://arxiv.org/abs/2601.15621
- Qwen3.5-Omni Technical Report (2026-04-17): https://arxiv.org/abs/2604.15804 ; [S] https://the-decoder.com/?p=33841
- Qwen-Audio-3.0-TTS (2026-07-27): https://arxiv.org/abs/2607.23938

**Competing models and their papers**
- IndexTTS-2.5 paper v5 (2026-08-11): https://arxiv.org/abs/2601.03888 ; repo https://github.com/index-tts/index-tts ; https://github.com/index-tts/index-tts/issues/518
- Sarashina2.2-TTS (2026-06-24): https://arxiv.org/abs/2606.25369 ; https://huggingface.co/sbintuitions/sarashina2.2-tts
- Confucius4-TTS (2026-08-12): https://arxiv.org/abs/2608.11650 ; https://github.com/netease-youdao/Confucius4-TTS ; https://huggingface.co/api/models/netease-youdao/Confucius4-TTS
- VoxCPM2 report (2026-06-05): https://arxiv.org/abs/2606.06928 ; repo https://github.com/OpenBMB/VoxCPM ; FT docs https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html
- dots.tts: https://github.com/rednote-hilab/dots.tts ; https://huggingface.co/rednote-hilab/dots.tts-soar
- CosyVoice3 paper: https://arxiv.org/abs/2505.17589 ; repo https://github.com/FunAudioLLM/CosyVoice ; CosyVoice 3.5 API: https://help.aliyun.com/zh/model-studio/cosyvoice-v3-5-plus
- Fish: Drama 3 report (2026-10-14): https://arxiv.org/abs/2610.18010 ; LICENSE https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE ; releases https://github.com/fishaudio/fish-speech/releases
- Aratako: https://huggingface.co/Aratako/Drama-3-Japanese (2026-10-22) ; https://huggingface.co/Aratako/Irodori-TTS-v4-Large (2026-09-27) ; https://github.com/Aratako/Irodori-TTS ; https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b ; https://huggingface.co/Aratako/MioTTS-2.6B
- FireRedTTS3 (2026-08): https://github.com/FireRedTeam/FireRedTTS3
- Ming-3.0-tts (2026-09-22): https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B
- Higgs TTS 3 (2026-06-04): https://huggingface.co/bosonai/higgs-tts-3-4b ; https://github.com/boson-ai/higgs-audio
- OmniVoice: https://huggingface.co/k2-fsa/OmniVoice ; https://github.com/k2-fsa/OmniVoice
- Step-Audio-EditX: https://github.com/stepfun-ai/Step-Audio-EditX
- GPT-SoVITS changelog (V5, 2026-10-05): https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/docs/en/Changelog_EN.md ; releases https://github.com/RVC-Boss/GPT-SoVITS/releases
- Excluded models: https://huggingface.co/maya-research/maya2-native (2026-10-26) ; https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B ; https://huggingface.co/inclusionAI/Ming-3.0-audio-flash

**Accent research and leaderboards**
- Region-Aware Masking (2026-10-05): https://arxiv.org/abs/2610.07524 ; KIT IWSLT 2026: https://arxiv.org/abs/2606.07240
- Artificial Analysis TTS leaderboard (open weights): https://artificialanalysis.ai/text-to-speech/leaderboard?open_weights=true
- HF trending TTS list: https://huggingface.co/api/models?pipeline_tag=text-to-speech&sort=trendingScore

**Org listings checked for new releases**
- https://huggingface.co/api/models?author=FunAudioLLM , ?author=stepfun-ai , ?author=ByteDance-Seed , ?author=OpenMOSS-Team , ?author=Aratako

**Secondary sources [S]**
- Zenn comparison: https://zenn.dev/fujinumagic/articles/local-japanese-tts-voice-clone
- Qiita comparison (2026-05-12): https://qiita.com/0h-n0/items/8f78f7acd31000612d13
- MiniMax Speech: https://www.datalearner.com/en/ai-models/pretrained-models/minimax-speech-2-6
- Seed-VC archive status and X-VC: https://github.com/Plachtaa/seed-vc ; https://arxiv.org/abs/2604.12456
