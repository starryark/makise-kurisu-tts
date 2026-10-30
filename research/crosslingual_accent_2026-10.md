# Japanese accent in Mandarin output from a Japanese-only Qwen3-TTS fine-tune (research, 2026-10-28)

**Scope.** Kurisu v2 is Qwen3-TTS-12Hz-1.7B-Base, fully fine-tuned on about 10.7 h of Japanese from one speaker, trained in `language="Auto"` (nothink) mode. When it speaks Mandarin, characters shared with Japanese drift toward Japanese readings, and the whole utterance has an audible Japanese accent. ASR CER on Chinese stays low (0.067 on 8 probes). This report collects what published work and public repos say about the problem, and turns it into options and an evaluation protocol for this setup.

**Tags.**
- **[P]** primary: a paper, model card, official repo code or config, or a GitHub issue or PR read directly. Dates are submission or creation dates.
- **[S]** secondary: a blog, a forum post, a review site, or a paraphrase I could not check against the original.
- **[I]** my inference.
- **[U]** unverified.

---

## 0. Bottom line

1. **SOTA has not solved this in a measured way.**
   - The *segmental* half has known fixes, and every one of them uses explicit language information in the input. The shared character is either
     - rewritten (CosyVoice 3 converts all Japanese to kana), or
     - tagged per token (IndexTTS-2.5 adds a language embedding to every text token), or
     - framed by a language-ID token (VALL-E X, and Qwen3-TTS's own `codec_language_id`).
   - The *suprasegmental* half (tones, vowels, rhythm) is handled by three groups of methods:
     - language-ID conditioning plus native data from other speakers (VALL-E X, Amazon Polyglot, CosyVoice 3's "polyglot" recipe);
     - adversarial or embedding disentanglement (Zhang 2019, DSE-TTS, CrossAccent-TTS 2026);
     - inference-time guidance (MegaTTS 3 text CFG, closed-loop dual-reference guidance 2026-10, activation steering 2026).
   - **Gaps in the literature:**
     - No paper evaluates our case: a Japanese single-speaker fine-tune speaking Mandarin, scored with an accent metric.
     - Nothing in the literature targets Mandarin tones for this problem.
     - CosyVoice 3's authors say outright that "there is currently no reliable objective method to evaluate the authenticity of accents" [P].
2. **The "it happens across models" claim holds** (§7).
   - The same Japanese→Chinese shared-kanji symptom is reported for Fish S2-Pro (#1263), CosyVoice (#411, #139), OpenVoice (#258) and GPT-SoVITS (#1054).
   - The F5-TTS and VoxCPM maintainers say accent transfer from the reference is by design [P].
3. **In our model the most likely cause is a speaker–language confound, not tokenization alone [I].**
   - In training, the Kurisu speaker slot always co-occurred with Japanese audio, and no language token was present.
   - So the speaker embedding (and the weight changes) became the model's language cue.
   - Zhang et al. 2019 describe this exact situation: speaker identity "perfectly correlated with language in the training data" [P].
   - The Qwen3-TTS fine-tune script hard-codes the language-blind `nothink` prefix [P code].
4. **The highest-value, lowest-cost steps, in order:**
   - (a) Run a diagnostic matrix that separates the effect of the base model, the speaker embedding, the language token and the fine-tune.
   - (b) Try the existing `language="Chinese"` codec token at inference (free).
   - (c) Retrain once (240 steps) with language IDs (PR #361) plus *native* Mandarin and English replay data under other speaker IDs (PR #384), using the Chinese and English dubs as those speakers.
   - The full ranking is in §8.
5. **Do not use CER as the accent metric.**
   - Use the target-language posterior from a language-ID model, P(ja | Mandarin audio). Two recent papers use this as their accent-leak proxy (VoxLingua107 ECAPA, Whisper LID).
   - Add a tone-aware pinyin error rate from an acoustic-only recognizer.
   - Add a small blind listening test as ground truth (§9).

---

## 1. How Qwen3-TTS handles language, and what that means for us (Q6)

### 1.1 Facts from the code and configs

- **Text channel.** Text goes through the standard Qwen BPE tokenizer. There is no G2P, pinyin or kana front-end. "Text is processed using the standard Qwen tokenizer" [P tech report arXiv:2601.15621, 2026-01-22].
  - A character such as 日, 月 or 大 is therefore *the same token* in Japanese and Chinese input.
  - Only context and the codec-channel prefix can tell the two apart [I].
- **The language ID lives in the codec channel, not the text channel** [P `modeling_qwen3_tts.py` L2110–2160, main branch, fetched 2026-10-28].
  - `language="Auto"` produces the prefix `[codec_nothink_id 2155, think_bos 2156, think_eos 2157]`.
  - An explicit language produces `[think 2154, think_bos, <lang id>, think_eos]`.
  - The speaker embedding (if any), `codec_pad` and `codec_bos` follow.
- **Language-ID table** in the Base config [P `Qwen3-TTS-12Hz-1.7B-Base/config.json`]: chinese 2055, english 2050, japanese 2058, korean 2064, german 2053, french 2061, spanish 2054, italian 2070, portuguese 2071, russian 2069.
- **Dialects are implemented as language IDs bound to a speaker** [P `-CustomVoice/config.json` and modeling L2118–2122].
  - The CustomVoice config adds `beijing_dialect 2074` and `sichuan_dialect 2062`.
  - It also sets `spk_is_dialect = {eric: sichuan_dialect, dylan: beijing_dialect}`.
  - If `language` is `chinese` or `auto` and the speaker is a dialect speaker, `generate()` *replaces* the language ID with the dialect ID.
  - Dialect IDs are hidden from `supported_languages` (L1831–1834). Issue #240 (2026-02-28) asks for that restriction to be lifted so other voices can use dialects [P].
  - **The implication [I]:** Qwen already uses the codec-channel language token to carry accent-like variation (Beijing erhua, Sichuan tones). That is the strongest evidence that this token is the intended handle for "which phonology". Instruction text is not that handle: in #248 (2026-03-03), "speak in Sichuan dialect" instructions failed [P].
- **The official fine-tune ignores language** [P `finetuning/dataset.py`].
  - `__getitem__` reads `item.get('language','Auto')`, but `collate_fn` always writes `[nothink, think_bos, think_eos, 0(speaker), pad]` into slots 3–7 of the codec channel.
  - The speaker embedding goes to slot 6.
  - `sft_12hz.py` stores the first batch's speaker embedding at `spk_id 3000` and sets `tts_model_type="custom_voice"` [P].
- **PR #361** (2026-08-28, open): "fix: actually use the language field during fine-tuning" [P].
  - For recognised languages it builds the think-mode prefix with the real language ID.
  - It moves every item to a uniform 6-slot layout (nothink rows get an extra pad).
  - It was tested only at the unit level: "Not a full training run."
- **PR #384** (2026-10-17, open): multi-speaker SFT [P].
  - Adds an optional `speaker` field per row.
  - Registers speakers from `spk_id 3000` upward.
  - Handles variable-length reference mels.
  - Also includes the label-shift, text_projection and sub-talker alignment fixes.
- **Issue #323** (Bengali, 2026-05-24) [P]:
  - Problem: a 7 h single-speaker fine-tune in Auto mode had a Hindi-like accent.
  - The reporter's observation: "IPA / romanized phoneme input reduced the Hindi accent but … became more robotic."
  - A commenter (mozi1924) diagnosed the cause as the hard-coded `nothink` prefix and proposed per-row language IDs.
  - **Nobody reported a result after training with language IDs** (thread inactive 2026-08-26).
  - **I found no public evidence that anyone has fine-tuned Qwen3-TTS with language IDs and kept native pronunciation in other languages [U, negative search].**

### 1.2 Facts from the tech report and model cards

- **Language handling is barely documented** [P arXiv:2601.15621].
  - The report never mentions language-ID tokens or the think/nothink switch. It mentions only a "probabilistically activated thinking pattern" for instruction following.
  - The cross-lingual section says Qwen3-TTS shows "superior content consistency and reduced accent drift" for zh↔en. **That accent claim rests only on WER/CER; no accent metric is reported** [P §4.2.3].
- **The ja→zh number does not measure accent.**
  - Table 7 is CV3-Eval cross-lingual cloning: a reference prompt in language A, text in language B.
  - The metric is CER/WER only [P].
  - ja→zh 3.43 is for the 12Hz-1.7B-Base model, against CosyVoice 3 at 3.05.
  - It was measured in Base cloning mode, not on a fine-tuned voice, and it says nothing about accent [I].
- **Speaker SFT on monolingual data is claimed to transfer to 10 languages** [P §4.2.5].
  - The "Aiden" voice was "fine-tuned exclusively on monolingual data" and "successfully transfers the target speaker's timbre and prosody to all 10 evaluated languages."
  - The only evidence is WER. The report does not say whether language IDs were used during that SFT [U].
- The CustomVoice card says: "We recommend using each speaker's native language for the best quality" [P HF card].
  - The same card says each preset (Ono_Anna is Japanese) "can speak any language."
  - The card's own evaluations used `language="auto"` only for Seed-Test and InstructTTS-Eval, and "explicitly passed the corresponding `language`" elsewhere [P].

### 1.3 Issues and community reports for Qwen3-TTS

| Issue / source | Date | What it shows | Tag |
|---|---|---|---|
| #134 "When I clone a voice in another language keep original accent" | 2026-01-29 | English→Italian cloning gives an American accent. "Same issue with german." One user says a 3 h African-English fine-tune still sounded American. No maintainer answer | [P] |
| #27 (Turkish FT, 18 h) | 2026-01-23 | A user says the output sounded "like a greek or American guy talking in turkish". Another user: "Without expanding the vocab, the model can speak another language, but with a strong accent. Training doesn't help" | [P] (anecdote) |
| #323 (Bengali FT, 7 h) | 2026-05-24 | Hindi accent in Auto mode. IPA input reduced the accent but sounded robotic. A language-ID fix was proposed but never tested | [P] |
| #151, #68 | 2026-02-01 / 01-25 | Chinese reference + English text: digits are read in Chinese ("nine 点 nine 九 yuan"). The reference language leaks into *segmental* choices for ambiguous tokens, the same mechanism as our shared kanji | [P] |
| #99, #273 (polyphones) | 2026-01-27 / 03-19 | Users say inline tone-marked pinyin (`běi`) works in Base but `bei3` does not. Other users could not reproduce it ("会把字母读出来", the letters get read out). Not documented by Qwen | [S] (mixed) |
| Sarashina2.2-TTS paper, Tables 5–6 | 2026-06-24 | On JSUT, Qwen3-TTS CER is 63.11 averaged over 9 diverse Japanese prompts (narration, news, rakugo and others; STD 125) and 270.12 with 3 English-accent prompts (+328%), the largest degradation of the systems tested. With one standard reading-style prompt (Table 2) the figure is 14.87, the number in `final_report.md`; the setups differ, so the two numbers do not conflict. The authors: Japanese pronunciation "partially relies on acoustic cues from Japanese prompts". This is the reverse direction, but it shows that Qwen3-TTS takes much of its pronunciation from the prompt or speaker side | [P arXiv:2606.25369] |
| Zhihu review of Qwen3-TTS-Flash | — | With auto language detection, Japanese 夢 was read as Chinese | [U] (snippet only; Zhihu returned 403) |

### 1.4 What this implies for Kurisu v2 [I]

- In every training row the fine-tune saw the same three things together:
  - the same speaker vector (slot 6 / `spk_id 3000`),
  - the `nothink` prefix,
  - Japanese audio.
- Nothing in training separates "Kurisu" from "Japanese". With Mandarin text in Auto mode, the model gets two conflicting cues:
  - the context of simplified-only characters (说, 这, 们) points to Chinese;
  - the speaker vector and the fine-tuned weights point to Japanese.
- Shared tokens with no strong local Chinese context (日, 月, 年, 大学, 先生, digits) fall to the Japanese reading. Prosody (tone) follows the speaker and fine-tune prior.
- **Why CER misses it.** A Mandarin ASR with a strong language model can map flattened tones and Japanese-coloured vowels back to the right characters. CER mostly catches whole-syllable substitutions [I].
- **What we do not know** is how much of the accent is already in the Base model with Kurisu's x-vector, and how much the fine-tune added. The diagnostic matrix in §9.1 answers that first.

---

## 2. Han-character homographs in multilingual LM-TTS (Q1)

| Approach | System / paper | What it does | Evidence on CJK homographs | Tag |
|---|---|---|---|---|
| **Kana conversion of all Japanese text** | CosyVoice 3, arXiv:2505.17589v2 (2025-05-27) §5.2.2 | Japanese text is converted to kana before tokenization, so Han characters only ever appear as Chinese in training | "CosyVoice 2 struggles with transferring voice from Japanese to Chinese due to the character overlap of two languages, a problem resolved in CosyVoice 3 by converting all Japanese characters into kana." CV3-Eval ja→zh CER: CosyVoice 2 48.1 → CosyVoice 3 about 6.8, 3.05 with DiffRO. **Cost:** "the conversion of kanji into kana before speech synthesis introduces additional errors", and Japanese CER is about 9% in their polyglot experiment (§5.8) | [P] (the Table 7 mapping is my reading of a flattened table [I]) |
| **Token-level language embedding** | IndexTTS-2.5, arXiv:2601.03888v3 (2026-01-09) §3.1.2 | "each input text token embedding is fused with a language-specific categorical embedding … particularly critical for characters shared across languages." Needs "fine-grained, token-level language identification" at inference | "cross-lingual homograph-induced mispronunciations are effectively suppressed." Table 1 compares this with boundary tags (`<ZH>…</ZH>`, §3.1.1) and natural-language instructions (§3.1.3). Token-level gives the best SS on all four test sets and the lowest WER on zh/en/es; on ja it is 10.50 against 9.23 for instructions. **No cross-lingual zh↔ja table** | [P] |
| **Boundary / language tags** | IndexTTS-2.5 §3.1.1; CosyVoice 1/2 `<\|zh\|>` `<\|jp\|>` text tags | Tags around the utterance | IndexTTS says boundary-tag control "degrades on longer utterances, occasionally leading to mispronunciations." CosyVoice #411 (2024-09-20): a Japanese prompt plus `<\|zh\|>` text gave "所有的汉字都是日语的读法" (every Han character read the Japanese way). #385: a maintainer blames "the drawback of bpe tokenize" for zh/yue confusion | [P] |
| **Language-ID embedding added to the acoustic-token stream** | VALL-E X, arXiv:2303.03926 (2023-03-07) §3.3, Table 6 | "we embed language IDs into dense vectors and add them to the embeddings of acoustic tokens." The input is a multilingual G2P (phonemes), not characters | Subjective accent score (1–5): en→zh 4.03 with LID, 2.35 without, 2.25 with the wrong LID; zh→en 4.10 / 2.98 / 2.55. The correct LID slightly *lowers* similarity to the source speaker (ASV 0.48 against 0.49–0.50) | [P] |
| **Language ID in the codec prefix** | Qwen3-TTS (§1.1) | One think-block token per utterance | No published ablation. Dialect IDs show that the token carries phonology [I] | [P] code / [I] |
| **Language-specific G2P front-ends** | GPT-SoVITS, Bert-VITS2 / Style-Bert-VITS2 (pinyin for zh, pyopenjtalk for ja, separate phone sets), VALL-E X | Homographs disappear by construction once the language of the text span is known | Accent (as opposed to reading) still leaks across languages: GPT-SoVITS #1054 reports that a Japanese reference speaking Chinese or English "会有很明显的口音" (has an obvious accent) | [P] |
| **Mixed character/phoneme input (polyphone control)** | CosyVoice 3 §2.3/§5.5 (pinyin and CMU phonemes in an expanded vocabulary); IndexTTS `<word\|reading>` (pinyin, CMU, kana) | Replace only the ambiguous unit | CosyVoice 3, Table 13: RepMono + CatPhn reaches 86.7% (zh) and 100% (en) correction. Qwen3-TTS has no official equivalent. Fish S2-Pro users in #1263 (2026-04-10) found per-character phoneme annotation "fixes pronunciation but makes speech very unnatural" | [P] |
| **Kanji homograph disambiguation inside Japanese** | Sarashina2.2-TTS (2026-06-24); Zhang, CAWL 2023 | Data scaling plus targeted synthesis for kanji polyphony | Japanese-internal readings only, not cross-lingual | [P] |
| **ZH–JA code-switching TTS papers** | — | — | I found **no paper specific to Mandarin–Japanese code-switching that handles shared characters**. The closest Mandarin–English work uses shared phone sets or separate encoders with language embeddings | [U] (negative search) |

**Takeaways for Q1 [I]:**
- Every system that fixes the homograph problem gives the model explicit language information at or below the token level.
- None of them relies on context alone.
- Qwen3-TTS offers only an utterance-level codec token, and our fine-tune never trained with it.
- Kana conversion of the *Japanese training text* is the one fix that needs no architecture change. It stops the fine-tune from reinforcing "Han character → Japanese reading".

---

## 3. Accent leakage in cross-lingual cloning and fine-tuning (Q2)

### 3.1 Training-time disentanglement

- **Zhang et al. 2019, "Learning to Speak Fluently in a Foreign Language"** (arXiv:1907.04448, Interspeech 2019) [P].
  - Uses phonemic input plus "an adversarial loss term to encourage the model to disentangle its representation of speaker identity," because speaker identity is "perfectly correlated with language in the training data."
  - English↔Spanish↔Mandarin cloning without bilingual data. The domain-adversarial speaker classifier with gradient reversal sits on the text encoder.
- **YourTTS** (arXiv:2112.02418): language embeddings plus a speaker-consistency loss.
  - It is the model the user's prompt calls "YourTTS-style". Its accent results are subjective only [S] (not re-read in this session).
- **Amazon Polyglot** (blog, 2021-02-03) [P].
  - A multi-speaker, multi-language model with language-ID and speaker embeddings on phoneme input. The English Alexa voice speaks Spanish.
  - In MUSHRA accent tests, "no statistically significant difference" from native Spanish recordings.
  - Phoneme mapping (the older method) gave "heavy foreign accents."
- **DSE-TTS** (arXiv:2306.14145, Interspeech 2023) [P abstract].
  - One speaker embedding goes to the acoustic model (speaking style); a second goes to the vocoder (timbre).
  - Beats SANE-TTS "especially in terms of nativeness."
  - Motivation: VQ acoustic tokens hold less speaker information than mel spectrograms.
  - A Qwen3-TTS analogue would be a native-style LM plus Kurisu timbre in a VC stage [I].
- **CrossAccent-TTS** (arXiv:2606.25403, 2026-06) [S, abstract via search].
  - An adversarial "Accent Suppression Module" removes accent from the speaker and style representations, with accent-intensity control.
  - Tested on Indic and L2-ARCTIC data.
- **X-Voice** (arXiv:2605.05611, 2026-05-07; code at github.com/sunnyxrxrx/X-Voice) [P].
  - F5-TTS extended to 30 languages with IPA input.
  - "Dual-level" language-ID injection (time-embedding MLP plus FiLM on text), motivated by "a lack of global constraints to fully decouple the speaker's timbre from their accents"; text-only LID still leaked accent.
  - Decoupled, scheduled CFG for the acoustic and linguistic terms.
  - **Nativeness is subjective only** (IMOS). Objective metrics are WER and SIM.

### 3.2 Inference-time control

- **MegaTTS 3** (arXiv:2502.18924v4, 2025-03-28) [P].
  - Multi-condition CFG with separate speaker (α_spk) and text (α_txt) guidance.
  - Training drops the prompt with p = 0.10, then drops the text with p = 0.5 when the prompt is dropped.
  - As α_txt rises, pronunciation goes from improper, to "the current speaker's accent," to "approaching the standard pronunciation of the target language."
  - Settings: accented α_spk 6.5 / α_txt 1.5; standard α_spk 2.0 / α_txt 5.0.
  - Evaluated on L2-ARCTIC with ASMOS (accent similarity) plus pitch-distribution moments. English only.
- **Closed-loop dual-reference guidance** (arXiv:2610.29351v2, 2026-10-22, VoxNexus) [P].
  - Training-free, on a frozen VoxCPM2.
  - Builds a *native-language anchor* once per speaker by converting native LibriTTS-R donors toward the target timbre with Seed-VC.
  - Mixes the two reference-conditioned predictions, v = (1−λ)·v_src + λ·v_anchor, with λ* = 0.33. Each emitted patch is fed back to both branches.
  - French→English: P(fr) from Whisper LID falls 0.093 → 0.025 (−73%). ECAPA similarity costs 0.042 (0.601 → 0.559). GOP rises 4.663 → 4.750. WER falls 0.016 → 0.009.
  - The VC cascade sounded more native in listening tests (32 vs 13) but lost more identity (ECAPA 0.468).
  - Spanish→English: P(es) −56%.
- **Activation steering for accent-neutralised zero-shot TTS** (arXiv:2603.05977, SLT 2026) [P abstract].
  - Layer-wise steering vectors come from activation differences between accented and native speech, added at inference.
  - The abstract names neither the base model nor the languages. Demo at accentsteer.github.io.
- **Accent Vector** (arXiv:2603.07534, 2026-03-08) [P].
  - Base model: XTTS-v2.
  - Method: LoRA (r16) task vector τ = θ_ft − θ_pre, from fine-tuning on native speech of another language under the *base* language ID; applied as θ_pre + α·τ with α in 0–1.
  - Accent is measured with Vox-Profile's accent classifier and VoxLingua107-ECAPA LID.
  - **Mandarin was the hardest target** (an English accent in Mandarin rose only 0.00 → 3.03%).
  - Larger α raised WER and lowered UTMOS.
  - For us [I]: the same weight arithmetic run in reverse is "scale the Kurisu fine-tune delta down for Mandarin".
- **Prompt and mode choice** [P].
  - **MiniMax-Speech** (arXiv:2505.07916):
    - Zero-shot cloning (speaker encoder only, no transcript) gave "significantly lower WER across all tested languages" than one-shot ICL.
    - ICL gave higher SIM. The authors say ICL is "constrained by the prosody of a specific text-audio prompt."
  - **VoxCPM #321** (maintainer, 2026-06-03): "Ultimate Cloning Mode" (transcript-conditioned) copies "pronunciation habits, and accent". Use regular mode when the target accent matters.
  - **F5-TTS #315**, owner: "Accent is just part of speech style… the current model design is actually transfer all style."

### 3.3 Cross-lingual results of current open models (all CER/WER; no accent metric)

| Model | ja→zh | zh→ja | Notes | Tag |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base | 3.43 | 8.40 | CV3-Eval, Table 7 | [P] |
| CosyVoice 3 (1.5B + DiffRO) | 3.05 | 7.08 | Kana front-end for Japanese; 0.5B without RL about 6.9 | [P] / [I] |
| CosyVoice 2 | 48.1 | 13.1 | The shared-character failure | [P] |
| IndexTTS-2.5 | — | — | Reports per-language results only. "test-ja" WER 9.95 / SS 0.833 | [P] |
| X-Voice | no ja pair | — | Its zh results are close to Qwen3-TTS's | [P] |
| Seed-TTS | — | — | No cross-lingual accent analysis found | [U] |

**Takeaways for Q2 [I]:**
- Speaker–language disentanglement is a mature idea (2019 onward), but it has always needed **native data in the target language from other speakers**, together with language and speaker IDs.
- The newest work (2026) is training-free guidance or steering, built on a native "anchor".
- Every paper that measures accent objectively uses either a **language-ID or accent-classifier posterior** or **GOP**. None uses Mandarin tone accuracy.

---

## 4. Data strategies (Q3)

| Strategy | Evidence | Measured effect | Fit for us | Tag |
|---|---|---|---|---|
| **Multilingual replay: native data from other speakers, with speaker and language IDs** | CosyVoice 3 §2.6.1/§5.8: an "auxiliary training dataset" of "studio-quality monolingual data from randomly-selected speakers covering all supported languages". "The speaker ID and the language ID of every utterance are specified in a natural language instruction." Amazon Polyglot; Zhang 2019; GPT-SoVITS owner on #1626: "fine tune the model using all languages you need to cross" [S, found by a sub-search]; GPT-SoVITS wiki: with only language A in the training set, "该模型不具备跨语种能力" (the model loses cross-lingual ability) [S] | CosyVoice 3: polyglot CER/WER under 4% for zh/en/de/es/fr/it/ru; ja 9%, ko about 6%. **No accent metric.** Amazon: accent not significantly different from native recordings (MUSHRA) | **High.** We have the Chinese dub (374 clips, 0.48 h) and the English dub (1,485 clips) as ready-made "other speakers". Add a few hours of clean public Mandarin (e.g. AISHELL-3, a multi-speaker Mandarin set) if 0.48 h is too thin [I] | [P] (CosyVoice 3, Amazon) |
| **Chinese dub as a separate speaker with `language=chinese`** | Same principle as above; PR #384 provides speaker IDs, PR #361 provides language IDs | None published for Qwen3-TTS | **High.** Same character, comparable acting style, native Mandarin. Risk: dub audio may contain music or effects; needs separation and DNSMOS filtering (as in the v2 data rules) [I] | [I] |
| **Synthetic native-Mandarin data in the target voice (TTS or dub → timbre-only VC → train as Kurisu + `chinese`)** | Piotrowski et al., arXiv:2309.08255 (ICONIP 2023): VC converts target-locale data into the target voice and trains a polyglot TTS, which "outperforms SOTA… large multilingual TTS". Ellinas et al., arXiv:2210.17264: same idea (**v2 withdrawn 2024-02-27**). Dual-reference paper: VC anchors from native donors | arXiv:2309.08255 reports gains but its abstract gives no accent numbers. In the 2026 dual-reference study the VC cascade sounded most native but lost identity (ECAPA 0.468 vs 0.601 direct) | **Medium-high.** Use a *timbre-only* converter so Mandarin tones and rhythm survive: Seed-VC v1 (GPL-3.0, archived 2025-11-21), not v2 `--convert-style`, which converts accent *toward the target speaker*; Vevo-**Timbre** ("style-preserved", CC-BY-NC-4.0), not Vevo-Voice or Vevo-Style; CosyVoice `inference_vc`; OpenVoice. Tag the converted rows with a separate speaker ID (`kurisu_vc`), or keep the share low, so VC artefacts do not leak into the JA voice [I] | [P] / [I] |
| **Self-training with accent filtering / RL** | GRPO and DPO in TTS (F5R-TTS, GLM-TTS, Magpie-TTS docs, KIT IWSLT 2026: GRPO with a CER+SSIM reward). Qwen3-TTS itself used DPO and GSPO post-training [P report §3.2] | No published accent-reward results (negative search). A 2025/26 prosody paper warns that GRPO on transcription rewards flattens prosody into monotone speech [S] | **Medium, later.** Rejection-sample many ZH generations and keep the top ones by P(zh) and tone accuracy, then DPO or a short SFT. Better as a second-round step once metrics exist [I] | [P] / [I] |
| **Phoneme or IPA input** | Qwen3 #323: reduced the Hindi accent but sounded "robotic" | Anecdotal | **Low.** Qwen BPE was never trained on IPA [I] | [P] |
| **Kana Japanese training text** (CosyVoice 3 recipe applied to the fine-tune) | CosyVoice 3 §5.2.2 | Fixed ja→zh in CosyVoice 2 → 3 (but that change was made in pre-training) | **Medium.** It removes the fine-tune's reinforcement of kanji → Japanese reading. Costs: JA inference needs G2P (pyopenjtalk); Qwen3 was pre-trained on kanji text, so kana-only input is out of distribution and may hurt JA prosody [I] | [P] / [I] |
| **Train longer on more data** | F5-TTS #315: a German FT "training longer mostly gets rid of the accent, but not completely". Qwen #27 user: "Training doesn't help" | Anecdotal | Not applicable; we have no more Kurisu Mandarin | [P] |

---

## 5. Inference-time mitigations (Q4)

| Mitigation | Rationale / evidence | Cost | Tag |
|---|---|---|---|
| **`language="Chinese"` on the fine-tuned model** | Language-ID tokens reduce accent in VALL-E X (4.03 vs 2.35 accent score) and in KIT IWSLT 2026 (Fish S2-Pro with tags: target-language ID probability 89.85% → 91.64%). Our fine-tune never saw the think-mode prefix, but tokens 2154 and 2055 are effectively unchanged base embeddings: they never appeared in the input, so they got no gradient, only negligible AdamW decay at lr 5e-6 × 240 steps. **Unknown whether the fine-tuned weights still respect them** | Zero (one flag) | [P] / [I] |
| **Route ZH/EN to the Base model, x-vector-only mode, with Kurisu's speaker embedding and `language="Chinese"`** | MiniMax and VoxCPM: transcript-free cloning copies less accent than ICL. Qwen supports `x_vector_only_mode` | Low: two models in memory (about 2 × 3.4 GB bf16) | [P] / [I] |
| **Route to Base ICL mode** | ICL "better preserves prosody" [P Qwen report §3.3], so it also copies accent (F5, VoxCPM, MiniMax). Sarashina: Qwen relies heavily on prompt acoustics. If used, pick a short, flat, low-emotion Kurisu clip, and test whether a *Mandarin* prompt in Kurisu's timbre (a VC'd dub line) beats a Japanese one | Low | [P] / [I] |
| **Weight interpolation θ_base + α·(θ_ft − θ_base) for non-JA text** | The full-FT analogue of a LoRA scale. Same arithmetic as Accent Vector (arXiv:2603.07534), where α trades accent against WER and UTMOS. Expect timbre to drop as α falls | Low: precompute 2–3 merged checkpoints, or keep two models | [P analogue] / [I] |
| **Speaker-vector interpolation**: blend Kurisu's x-vector with a native Mandarin female x-vector (e.g. the Chinese dub actress), weight 0.1–0.3 | Speaker embeddings carry language and accent information. A Qiita user concludes "訛りは、この埋め込みの中にあります" (the accent lives in this embedding) [S]. DSE-TTS motivation [P]. Costs similarity | Very low | [S] / [I] |
| **Language CFG (logit extrapolation)**: l = l_auto + γ(l_zh − l_auto), or with base logits as the "native" branch | AR-LM CFG precedent: Koel-TTS (arXiv:2502.05236) applies CFG to an LLM-TTS for transcript and speaker adherence. MegaTTS 3 shows text CFG pushes pronunciation toward the standard accent. Needs two forward passes and a custom sampling loop | Medium: about 1 day of code | [P] / [I] |
| **Dual-reference logit mixing (closed loop)**: run (a) the fine-tune or Base conditioned on Kurisu and (b) Base conditioned on a native-Mandarin anchor in Kurisu timbre (dub line → timbre-only VC); mix logits (1−λ)·a + λ·b, λ about 0.3, and feed each sampled token to both | Direct port of arXiv:2610.29351 (−73% source-LID mass at −0.04 ECAPA). For Qwen the code predictor (codebooks 1–15) must also be run or mixed | Medium-high | [P] / [I] |
| **VC cascade**: synthesize native Mandarin (Base with a native speaker, or CosyVoice 3), then timbre-only VC into Kurisu | Most native in the 2026 listening test, but the largest identity loss | Low-medium; adds latency | [P] |
| **Text normalisation for Mandarin input** | Avoid Japanese cues in the text: use Chinese punctuation (，；：“”) rather than 、「」; use Simplified forms (气 not 気); spell out digits in hanzi (Qwen #68/#151: digits follow the reference language) | Zero | [I] |
| **Inline tone-marked pinyin for drift-prone characters** | Community reports in #99 that `běi`-style works in Base; others failed in #273; Fish users found annotation unnatural | Zero, unreliable | [S] |
| **Instruction text ("请用标准普通话")** | Not available in Base. CosyVoice #1172 (2025-04-14): a Mandarin instruct still sounded foreign | — | [P] |

---

## 6. Measuring accent objectively (Q5)

**State of the art.**
- No open tool scores "Japanese-accented Mandarin" directly.
  - No public L1-identification model or corpus exists for Japanese learners of Mandarin [U, negative search].
  - CommonAccent covers en/es/de/it only. Vox-Profile's accent classifiers are English-only [S via tool survey].
- The papers that report accent objectively use one of four proxies:
  1. A **language-ID posterior**. Accent Vector uses VoxLingua107-ECAPA [P]; dual-reference guidance uses Whisper LID P(source) [P]; KIT IWSLT uses the target-language-ID probability [P].
  2. An **accent-classifier posterior or embedding cosine**. Vox-Profile [P]; CrossAccent-TTS uses GenAID embeddings [S].
  3. **Phone-level substitution or GOP.**
     - Amazon's *Phoneme Substitution Rate* (arXiv:2508.18006, MLSP 2025) is "the percentage of phonemes … substituted with a phoneme different from the ground truth". It uses a wav2vec2-xlsr-53 CTC phone recognizer and agreed with MUSHRA accent rankings on ">90%" of 50+ system pairs (de/en/es/fr) [P].
     - Dual-reference guidance reports GOP from wav2vec2-xls-r-300m phoneme posteriors [P].
  4. **PPG or formant distances to a native reference.** Zhong et al., arXiv:2505.14410 (Interspeech 2025), found them better aligned with accent similarity than WER, which has "significant limitations" [P abstract]. The tool survey adds that global F0 metrics did not track accent there (SRCC 0.11–0.46) [S].
- CosyVoice 3, IndexTTS-2.5 and Seed-TTS report no accent metric at all [P for CosyVoice 3 and IndexTTS-2.5].

**Building blocks that run offline on one GPU** (verified on 2026-10-28 unless tagged):

| Purpose | Tool | Tones? | Licence | Notes | Tag |
|---|---|---|---|---|---|
| Segmental + tone recognition *without an LM* | **charsiu** `charsiu/zh_xlsr_fc_10ms` (text-free, frame classifier) and `charsiu/zh_w2v2_tiny_fc_10ms` (forced aligner) | **Yes.** 210 labels: pinyin initials plus finals with tone 1–5 (tokenizer `charsiu/tokenizer_zh_pinyin`, G2P via G2pM) | Code MIT; weight licence not tagged | 16 kHz, 10 ms frames, xls-r-300m backbone, weights from 2021-12. **The best single tool for us** | [P] (config + `processors.py` checked) |
| Toned forced alignment | MFA Mandarin v3 dictionary (tone letters) | Yes | CC BY 4.0 | Its `phone_goodness` path is deprecated; not a real GOP | [S] |
| Segmental IPA | `facebook/wav2vec2-xlsr-53-espeak-cv-ft`, ZIPA, allosaurus, POWSM | No usable tones (espeak-cv-ft has no tone-3 token) | Apache / CC-BY / GPL | Usable for a PSR-style metric on segments only | [S] |
| Language ID ("Japaneseness") | `speechbrain/lang-id-voxlingua107-ecapa` | — | Apache-2.0 | zh and ja among 107 languages; used as the accent-leak proxy in arXiv:2603.07534 | [P] |
| Language ID (cross-check) | Whisper-large-v3 `detect_language` | — | Apache-2.0 (MIT code) | Used in arXiv:2610.29351 (faster-whisper) | [P] |
| Language ID (avoid) | `facebook/mms-lid-*` | — | **CC BY-NC 4.0** | Non-commercial | [S] |
| Character ASR + timestamps | Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B | No | Apache-2.0 | Keep for CER and alignment only. Its LM hides accent | [S] |
| Speech-LLM pronunciation judge | `xiaomi-research/TTS-PRISM-7B` (arXiv:2604.22225) | Its rubric covers tones and tone sandhi ("slight accent" = 4) | Apache-2.0 | HF 2026-04-14. Needs MiMo-Audio-Tokenizer. Not validated for L2 accent | [P] (existence, licence) / [S] (rubric) |
| Tone classifier | ToneNet, J-ToneNet, etc. | — | — | **No public weights found.** A small classifier trained on AISHELL-3 with toned alignments is easy [I] | [U] |
| Mandarin GOP / MDD | Kaldi `gop_speechocean762` (English); arXiv:2606.22022 (segmental vs tonal) | — | — | No released Mandarin GOP model; OMPAL corpus (French L1, CC BY 4.0) | [S] |

**What to use [I]:**
- **Charsiu's toned-pinyin frame classifier** gives the segmental/tonal split the problem needs: wrong syllable = homograph drift; right syllable, wrong tone = prosodic accent.
- **The VoxLingua107 P(ja)/P(zh) ratio** gives the global "Japaneseness" score that two 2026 papers already use.
- **Native and Japanese-accented anchors on the same sentences** calibrate both. Every one of these proxies is unvalidated for JA→ZH, so a small blind listening test stays in the loop (§9).

---

## 7. Evidence that this happens across models

| Repo / issue | Date | Symptom / statement | Tag |
|---|---|---|---|
| fishaudio/fish-speech #1263 | 2026-04-10 | S2-Pro voice cloned from Japanese: "shared CJK characters (漢字) are incorrectly pronounced in Japanese"; 十二月十五日 → じゅうにがつじゅうごにち. Phoneme annotation "makes speech very unnatural". Closed as stale with no fix; "我也遇到这个问题了" (I hit this too) 2026-07-29 | [P] |
| QwenAudio/CosyVoice #411 | 2024-09-20 | Japanese (Zundamon) prompt + `<\|zh\|>` text: "所有的汉字都是日语的读法" (every Han character read the Japanese way) | [P] |
| CosyVoice #139 | 2024-07-15 | "跨语种从日文生成中文外语味很重" (Japanese→Chinese cross-lingual output sounds heavily foreign). Collaborator: "related to training data, our cross-lingual data may not be enough" | [P] |
| CosyVoice #385 | 2024-09-12 | Japanese→Chinese output drifts into Cantonese. Collaborator: "the drawback of bpe tokenize… chinese and Cantonese have same character" | [P] |
| OpenVoice #258 | 2024-06-04 | "中文男性，感觉有浓重的日本口音" (a Chinese male voice with a heavy Japanese accent) | [S] (via sub-search) |
| GPT-SoVITS #1054; wiki | 2024-05 / wiki | A Japanese reference gives an obvious accent in Chinese and English. Wiki: large single-language fine-tunes wash out cross-lingual ability | [P] / [S] |
| F5-TTS #315, #240; VoxCPM #321, #289 | 2024-10 → 2026-06 | Maintainers: accent is transferred from the reference by design; use a native reference or non-ICL mode | [P] |
| Qwen3-TTS #134, #27, #323 | 2026 | See §1.3 | [P] |

Nobody in these threads measures accent with a metric; every judgment is by ear.

---

## 8. Ranked options for our setup (expected effect against cost) [I unless tagged]

These options apply to our setup: a full fine-tune at lr 5e-6 for 240 steps. One retrain is cheap (well under an hour on the existing RunPod setup). **Data preparation and evaluation are the real costs.**

| Rank | Option | Expected effect on accent | Cost | Main risk | Basis |
|---|---|---|---|---|---|
| **0** | **Diagnostic matrix + accent metrics (§9).** Run this first, because it decides between options 2–3 and 4–6 | — (decides the path) | About 0.5–1 day to script; minutes to run | None | — |
| **1** | **Free inference fixes on the current model.** Try `language="Chinese"` (think + 2055) instead of Auto. Normalise Mandarin input: Chinese punctuation (no 、「」), Simplified forms, digits spelled out in hanzi | Small to moderate, unknown. The language token is the mechanism Qwen itself uses for dialects; the fine-tune never saw the think prefix, so the result is uncertain | About 0. Just add these conditions to the probe set | The think prefix may destabilise the fine-tune (stop or timbre). Check the cap-hit rate | §1.1, VALL-E X, KIT |
| **2** | **Route ZH/EN to Base x-vector-only mode** with Kurisu's speaker embedding (`spk_id 3000` vector, or one computed from a neutral clip) and `language="Chinese"`. Fallback: Base ICL with a short neutral clip | Large *if* condition B in §9 is native-sounding. Expected to keep less timbre and style than the fine-tune | Low: second model in memory (about 3.4 GB bf16); app routing by script | Weaker Kurisu identity in ZH. ICL copies accent | MiniMax, VoxCPM #321, Qwen report §3.3 |
| **3** | **Weight interpolation** θ_base + α·Δ_ft for non-JA text (α ∈ {0.25, 0.5, 0.75}), combined with `language="Chinese"` | Moderate. A continuous dial between options 1 and 2 | Low: 3 merged checkpoints, or blend at load time | Monotonic accent-vs-timbre trade-off; pick α from the metric curve | Accent Vector (task-vector scaling) |
| **4** | **Retrain v3 with language IDs + multilingual replay** (port PR #361 + PR #384). Rows: <br>• Kurisu JA → `speaker=kurisu, language=japanese` <br>• Chinese dub → `speaker=kurisu_zh_dub, language=chinese` <br>• English dub → `speaker=kurisu_en_dub, language=english` <br>• optionally 2–5 h of clean native Mandarin from several public speakers → `language=chinese` <br>Replay at about 15–30% of samples. Infer with `speaker=kurisu, language=chinese` | **Large: the literature-backed fix.** It breaks the speaker↔language confound and makes the language token carry the phonology (VALL-E X, Amazon Polyglot, CosyVoice 3 §2.6.1). The dub gives native Mandarin in the same acting register | Medium. Code port (prefix width 5→6, every `8+n` offset shifts, multi-speaker ref mels, speaker IDs from 3000). `ft02_sft.py` already has #384's label-shift and text_projection fixes, so only its multi-speaker part is new. Clean the dub audio (separation + DNSMOS ≥ 3); 1–2 training runs | JA regression (watch JA CER and sim). Dub timbre leaking into Kurisu (watch sim on ZH). JA inference must then pass `language="Japanese"` consistently | [P] for the recipe elsewhere; [U] for Qwen3-TTS (nobody has reported results) |
| **5** | **Add synthetic native Mandarin in Kurisu's timbre** (on top of option 4). Take the Chinese-dub lines (and/or Base-generated native Mandarin), apply **timbre-only** VC to Kurisu (Seed-VC v1 without `--convert-style`, Vevo-Timbre, or CosyVoice `inference_vc`), and train them as `speaker=kurisu, language=chinese` (or a separate `kurisu_vc` speaker if artefacts appear) | Large on accent, if VC quality holds. It is the only data that says "this voice speaks native Mandarin" directly | Medium: VC run + QC (sim ≥ Kurisu threshold, P(zh) ≥ native, tone metric) | VC artefacts and buzz learned into the voice. Accent-converting VC modes would *re-import* the Japanese accent | arXiv:2309.08255, arXiv:2610.29351 anchors |
| **6** | **Freeze text embeddings / text_projection (and optionally the lower talker layers) during the JA fine-tune** | Small to moderate on shared-kanji readings. The fine-tune can no longer re-bind Han tokens to Japanese readings. The #323 reporter did the opposite (raised the text-embedding LR) to *strengthen* language adaptation | Very low: one flag. Combine with option 4 as an ablation | Slightly weaker JA reading fixes | [I]; GPT-SoVITS owner: layer freezing "may work, but I haven't do the experiment" [P] |
| **7** | **Kana Japanese training text** (CosyVoice 3 recipe): convert Kurisu's JA transcripts to kana, and run JA inference through pyopenjtalk G2P | Moderate on segmental homographs; little effect on tones | Medium. G2P in the app; may hurt JA prosody and readings (CosyVoice 3: JA CER about 9%) | JA quality, our primary goal, may regress | CosyVoice 3 §5.2.2 |
| **8** | **Inference-time guidance**: language CFG (l_auto + γ(l_zh − l_auto)), or closed-loop dual-reference logit mixing with a native-Mandarin anchor in Kurisu timbre (λ ≈ 0.3) | Potentially large (−73% source-LID mass in the 2026 paper) | Medium-high: custom sampling loop over talker + code predictor; 2× compute | Unproven for AR codec LMs; may destabilise stopping | MegaTTS 3, arXiv:2610.29351, Koel-TTS |
| **9** | **Accent-filtered self-training / DPO.** Sample N ZH outputs per text, rank by P(zh) + tone accuracy + sim, then train on the winners or on pairs | Moderate, incremental | Medium; needs §9 metrics first | Reward hacking: monotone prosody (a known GRPO failure) | GLM-TTS, F5R, Magpie docs |
| **10** | **VC cascade at inference** (native Mandarin TTS → timbre-only VC) | Most native accent | Low-medium; latency | Largest identity loss (ECAPA 0.468 vs 0.601 in arXiv:2610.29351) | [P] |
| — | IPA / inline pinyin, instruction text | Unreliable or robotic | — | — | #323, #99/#273, CosyVoice #1172 |

**Suggested sequence:**
- Run option 0, and add the option 1 conditions to it.
- If Base condition B sounds native, ship option 2 or 3 for ZH/EN right away.
- In parallel, build v3 with options 4 + 6, then add option 5 if the dub alone is not enough.
- Options 8 and 9 are research follow-ups.

---

## 9. Evaluation protocol for accent (automatic, single GPU)

### 9.1 Diagnostic matrix (generate first)

Same Mandarin texts, 3 seeds each, default sampling (temperature 0.9, top-k 50, repetition penalty 1.05):

| ID | Model / mode | Speaker | `language` | Purpose |
|---|---|---|---|---|
| A | Base, x-vector-only | Kurisu x-vector | Auto | Does the *x-vector alone* carry the accent? |
| B | Base, x-vector-only | Kurisu x-vector | Chinese | Effect of the language token on the base |
| C | Base, ICL | Kurisu JA clip (+ transcript) | Chinese | Effect of the reference prompt (#134 pattern) |
| D | Base, x-vector-only | Native Mandarin female (Chinese-dub actress clip) | Chinese | **Native anchor** (upper bound) |
| E | Kurisu v2 (current) | `kurisu` | Auto | Current behaviour |
| F | Kurisu v2 | `kurisu` | Chinese | Option 1 |
| G | v2 with α = 0.5 interpolation | `kurisu` | Chinese | Option 3 |
| H | **CustomVoice** `speaker="ono_anna"` (Qwen's own Japanese SFT voice) | ono_anna | **Japanese** (on Chinese text) | **Japanese-accent anchor**. This model *was* trained with language IDs, so the condition is in-distribution. Fallback: Base x-vector-only + Kurisu vector + `language="Japanese"` |
| J | CustomVoice `ono_anna` | ono_anna | Chinese | Does Qwen's own monolingual-SFT Japanese voice keep an accent in Mandarin? (Tests the card's "can speak any language" claim) |
| H2 (extra) | Kurisu v2 | `kurisu` | Japanese | Informational only. v2 never saw the think prefix, so do **not** use it for calibration |
| R | Real recordings: Chinese dub clips (+ optional AISHELL-3) | — | — | Ground-truth native (metric calibration; different texts) |

**How to read it:**
- If A ≈ E, the accent is already in Base + x-vector, so data fixes must target the speaker cue (options 4–5).
- If A and B are native but E is not, the fine-tune introduced it (options 2–4, 6).
- C − B gives the size of ICL accent copying.
- If J is accented, Qwen's own SFT recipe has the same weakness. If J is native, compare its training format with ours [I].

### 9.2 Probe text set (about 80 sentences)

1. **30 shared-character sentences.** Dense in characters whose Japanese and Mandarin readings differ: 日 月 年 时间 大学 先生 生活 问题 世界 研究 实验 东京.
2. **15 date, number and time sentences**, e.g. 十二月十五日, 3点20分, 1.5倍 (digits drifted to the reference language in #68/#151/#1263).
3. **15 simplified-only control sentences** with almost no shared forms (说 这 们 么 吗 还 让 给 对 时 见).
4. **10 tone-sandhi and neutral-tone sentences** (你好, 一个, 不是, 老鼠, 东西 neutral).
5. **10 long sentences** (40+ characters).

Keep each sentence ≤ 15 s. Generate references with pypinyin or g2pW, apply 3-3 / 一 / 不 sandhi rules, and accept tone 5 leniently. Make sure the sandhi tooling handles this [U].

### 9.3 Metrics (all offline; resample to 16 kHz)

| # | Metric | Tool | Captures |
|---|---|---|---|
| M0 | CER (keep, intelligibility only) | Qwen3-ASR-1.7B or Paraformer | Gross errors |
| M1 | **Japaneseness = mean log P(ja) − log P(zh)**, and the % of utterances with P(zh) < 0.9 | VoxLingua107-ECAPA; cross-check with Whisper-large-v3 `detect_language` | Global accent (the proxy used in arXiv:2603.07534 and 2610.29351) |
| M2 | **Toneless syllable error rate (SER)**: overall and on the **shared-character subset**; separate initial and final error rates | charsiu `zh_xlsr_fc_10ms`, text-free: frame argmax → collapse repeats/silence → syllables → Levenshtein against reference pinyin. **No LM** | Homograph drift and segmental accent (Japanese-style readings show up as whole-syllable substitutions) — a Mandarin PSR |
| M3 | **Tone error rate (TER)** on syllables whose toneless syllable is correct, plus a 5×5 tone confusion matrix | Same charsiu output | Tone flattening (tones 2/3/4 → 1, or tone 3 → 2) |
| M4 | **Per-syllable F0 contour distance** to anchor D on the same text: semitones, speaker z-normalised, DTW. Also the utterance pitch span | charsiu forced aligner `zh_w2v2_tiny_fc_10ms` for boundaries; RMVPE or pyworld for F0 | Prosodic accent. Per-syllable, *not* global F0 (global F0 failed to track accent in Zhong et al.) |
| M5 | Speaker similarity to Kurisu | Existing v2 sim metric (same encoder as the confirm-suite) | The cost side of every fix |
| M6 (optional) | TTS-PRISM-7B "Pronunciation Accuracy" 1–5; charsiu posteriorgram DTW distance to anchor D | TTS-PRISM; charsiu | Holistic checks |

**Runtime estimate:**
- Volume: 80 texts × 3 seeds × 10 conditions ≈ 2.4k clips.
- Models (load one at a time): ECAPA (about 20M params), Whisper-large-v3 (about 3 GB fp16), charsiu (300M), Qwen3-ASR (1.7B).
- Fits a 24 GB GPU; well under an hour [I].

### 9.4 Calibration and acceptance

1. **Metric validity.** A metric is kept only if it separates anchor D and set R (native) from anchor H (Japanese) with a large effect (Cohen's d > 1) and ranks D < E < H as expected. Drop any metric that fails.
2. **Human spot check.**
   - Before trusting the metrics for decisions, have 1–3 native Mandarin listeners rate nativeness (1–5) blind on 20 utterances × 4 systems (D, E, best candidate, H).
   - Report the Spearman correlation of M1–M4 with those ratings.
   - VALL-E X used this 1–5 accent scale [P].
3. **Acceptance rule for a ZH configuration [I, starting values]:**
   - M1 within 0.5 nats of D;
   - shared-subset SER ≤ 1.5× D's;
   - TER ≤ D's + 3 pp;
   - M5 drop ≤ 0.05 against E;
   - CER ≤ 0.07;
   - cap-hit rate unchanged.
4. **Regression guard.** Every candidate (especially options 4–7) must also pass the existing JA confirm-suite (JA CER 0.030 / sim 0.667 for v2).

---

## 10. Unverified or open

- Whether Qwen3-TTS fine-tuned with language IDs (PR #361) keeps native pronunciation. **No public results.** #323 was never followed up.
- Whether the "Aiden" SFT in the Qwen report used language IDs, and what language it was trained on.
- Whether `language="Chinese"` on an Auto-only fine-tune is stable. Not tested by anyone publicly.
- Base-model inline tone-marked pinyin: community reports conflict (#99, #273).
- CosyVoice 3 Table 7: the ja→zh values per model size come from a flattened table [I].
- CrossAccent-TTS details, YourTTS accent numbers, and the activation-steering base model were not read in full.
- No Japanese-L1 Mandarin accent corpus, L1 classifier or open Mandarin GOP model was found.
- Accent proxies (LID posterior, PSR) are validated only for European languages and English accents. Their validity for JA→ZH tones is untested, hence §9.4.

---

## Sources (accessed 2026-10-28)

All `arXiv:NNNN.NNNNN` entries resolve at `https://arxiv.org/abs/NNNN.NNNNN`. The submission and revision dates are given where each paper is first cited above.

**Qwen3-TTS**
- Tech report arXiv:2601.15621 (2026-01-22): https://arxiv.org/abs/2601.15621
- Code: https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py ; finetuning `dataset.py`, `sft_12hz.py`, `README.md`
- Configs: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/config.json ; https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- Issues: #134 https://github.com/QwenLM/Qwen3-TTS/issues/134 ; #323 …/issues/323 ; #27 …/issues/27 ; #151 …/issues/151 ; #68 …/issues/68 ; #99 …/issues/99 ; #273 …/issues/273 ; #240 …/issues/240 ; #248 …/issues/248
- PRs: #361 https://github.com/QwenLM/Qwen3-TTS/pull/361 ; #384 …/pull/384

**Homographs and front-ends**
- CosyVoice 3 arXiv:2505.17589 https://arxiv.org/abs/2505.17589
- IndexTTS-2.5 arXiv:2601.03888 https://arxiv.org/abs/2601.03888
- VALL-E X arXiv:2303.03926 https://arxiv.org/abs/2303.03926
- Sarashina2.2-TTS arXiv:2606.25369 https://arxiv.org/abs/2606.25369

**Accent and disentanglement**
- Zhang et al. 2019 arXiv:1907.04448
- DSE-TTS arXiv:2306.14145
- CrossAccent-TTS arXiv:2606.25403
- X-Voice arXiv:2605.05611 (https://github.com/sunnyxrxrx/X-Voice)
- MegaTTS 3 arXiv:2502.18924
- Closed-loop dual-reference guidance arXiv:2610.29351
- Activation steering arXiv:2603.05977
- Accent Vector arXiv:2603.07534
- MiniMax-Speech arXiv:2505.07916
- Koel-TTS arXiv:2502.05236
- KIT IWSLT 2026 arXiv:2606.07240
- Amazon Polyglot blog (2021-02-03) https://amazon.science/blog/english-language-alexa-voice-learns-to-speak-spanish
- Piotrowski et al. arXiv:2309.08255
- Ellinas et al. arXiv:2210.17264 (v2 withdrawn)

**VC tools**
- Seed-VC https://github.com/Plachtaa/seed-vc (archived 2025-11-21)
- Vevo https://huggingface.co/amphion/Vevo
- CosyVoice `inference_vc` https://github.com/FunAudioLLM/CosyVoice

**Other repos**
- fish-speech #1263 https://github.com/fishaudio/fish-speech/issues/1263
- CosyVoice #139, #385, #411, #1172 https://github.com/QwenAudio/CosyVoice/issues/
- VoxCPM #321 https://github.com/OpenBMB/VoxCPM/issues/321
- F5-TTS #315 https://github.com/SWivid/F5-TTS/issues/315
- GPT-SoVITS #1054, #1626, wiki https://github.com/RVC-Boss/GPT-SoVITS/wiki
- OpenVoice #258

**Metrics**
- charsiu https://github.com/lingjzhu/charsiu , https://huggingface.co/charsiu/zh_xlsr_fc_10ms
- VoxLingua107 ECAPA https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
- Amazon PSR arXiv:2508.18006
- Zhong et al. arXiv:2505.14410
- TTS-PRISM-7B https://huggingface.co/xiaomi-research/TTS-PRISM-7B (arXiv:2604.22225)

**Community**
- Qiita @kodomo-news (2026-09-17) https://qiita.com/kodomo-news/items/bcdd294a0a54358f9567
- Zenn fujinumagic (2026-06-18) https://zenn.dev/fujinumagic/articles/local-japanese-tts-voice-clone
