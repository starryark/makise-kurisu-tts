# Kurisu v3: research synthesis and plan (2026-10-28)

This document combines five parallel research reports. They are all in this folder:
- `sota_models_2026-10.md`
- `finetune_methods_2026-10.md`
- `labeling_sota_2026-10.md`
- `crosslingual_accent_2026-10.md`
- `qwen_internals_conditions.md`

It also draws on my own measurements of the dataset (`data_facts_v3.md`).

The claims that would change a decision were checked against primary sources by hand; see §6.

## 1. Answers

### 1.1 Is Qwen3-TTS still the state of the art for this job?

**Yes, for our combination of requirements. Nothing released from May to October 2026 replaces it.** That combination is:
- best-in-class Japanese after fine-tuning;
- Chinese and English in the same voice;
- an open fine-tuning path.

**What has not changed:**
- No new Qwen weights exist. The Base checkpoint dates from 2026-01-23, `qwen-tts` is still 0.1.1, and the 25 Hz models never shipped.
- Qwen3.5-Omni and Qwen-Audio-3.0-TTS are API or paper only.

**Each challenger fails one hard requirement:**

| Model | Why it doesn't replace Qwen |
|---|---|
| Irodori v4.1 | Japanese only |
| IndexTTS-2.5, FireRedTTS3, Ming-3.0 | No fine-tuning code |
| Fish S2 / Drama 3 | Fine-tuning code removed; licence limits |
| VoxCPM2, Confucius4-TTS, dots.tts | Weaker Japanese |

**Qwen's real weakness is your Chinese-accent complaint.** In the only human cross-lingual study (Confucius4-TTS paper, arXiv 2608.11650), Qwen3-TTS ranked 4.00/6 for en→zh pronunciation, against VoxCPM2 at 1.88. Caveats: zero-shot cloning, internal raters, competitor authors.

**Switch triggers:**
- After a v3 trained with language ids and native replay, native listeners still rate the Mandarin accent poorly, while a VoxCPM2 LoRA does clearly better.
- Qwen's Japanese reading error rate on held-out text is more than about 2 points worse than an Irodori LoRA's.
- Qwen releases new TTS weights.
- IndexTTS or Fish release fine-tuning code.

### 1.2 Is the fine-tuning approach still best practice?

**The training method is: keep it.** Full fine-tuning with fp32 master weights, lr 5e-6, about 1 epoch, using our bug-fixed trainer. No 2026 evidence beats it for the target voice at about 10 h. Upstream PR #278 and PR #384 are still unmerged, so the custom trainer is still required.

**The conditioning recipe is not.** v2 trained every row with:
- no language id (the `nothink` prefix, i.e. `language="Auto"`);
- only Japanese audio;
- every condition and event stripped out.

**That makes the Kurisu speaker slot the only cue the model has for "Japanese".** It is the most likely mechanism behind the accent (§1.4).

**Not recommended:**
- **Instruction text from Base.** Qwen collaborator wangxiongts (#25, 2026-01-26): "our base model does not currently support instruct."
- **Online RL.** Japanese CER already matches the real recordings, and RL tends to flatten prosody. Offline DPO comes last, if at all.

### 1.3 Why did I remove the bracketed text, and what is the SOTA labeling approach?

#### Why I removed them in v2

1. **Too few trustworthy examples.**
   - 143 of the 148 Japanese `[sigh]` labels came from an automatic classifier, and only 5 from people. `[gasp]` has 27 human labels and `[breath]` 16.
   - No game clip was ever reviewed by a person.
   - The anime reviews are themselves inconsistent: `[gasp]` is used for stammer inhales, and `[breath]` and `[gasp]` overlap.
2. **They don't tokenize as one unit.** In Qwen's tokenizer each bracket tag becomes 2–4 different token sequences depending on the preceding character (`[s`·`igh`·`]` vs `[`·`s`·`igh`·`]`). `[sigh]` and `[singing]` also share their first token.
   - The IaraTTS-SFT-v3 model card extended its vocabulary so tags are single tokens. According to the research agent's reading of that card, tags split by the tokenizer produced no audible events, while atomic tokens rendered about 50–60% of the time. I did not re-check those two figures myself.
3. **v1's endless generations clustered on tagged prompts.** That was in the r16 eval, and it is unverified whether that eval used greedy decoding.

#### What that reasoning got wrong

It was a conservative call for a one-shot run, not a finding that tags can't work. And deleting a tag does not delete the sigh from the audio.

**An event the text doesn't explain gets learned into the default voice as random noise.** The SOTA position, and your own planning doc's §2 ("annotate what is audible"), is that every audible event and condition must be explained by the input. The same applies to Amadeus and the phone clips: they must be labelled, not deleted.

#### What the field does (late 2026)

- **Point events (sigh, laugh, gasp):** inline tags at the event position.
  - Small fixed vocabularies use added atomic special tokens: CosyVoice 2/3, Higgs TTS 3, Chatterbox-Turbo, NVSpeech, and the NVVSpeech 2026 challenge winner.
  - Open plain-text vocabularies (Fish S2) only work at more than 10 million hours of data.
- **Whole-utterance style:** an instruction placed before the text. This only pays off at scale, and Qwen Base doesn't support it.
- **Recording channel (phone, device, radio):** the least modelled category.
  - The precedents are: a condition token, with "clean" as the default (Irodori 📞); a learned environment ID (Environment-Aware TTS); or DSP after generation (MiniMax `lofi_telephone`).
  - The settled principle: **the untagged default is clean, and every degraded clip carries its label.** Augmentation that is uncorrelated with speaker and content is what makes the label separable.
- **Labelers:**
  - No automatic labeler handles Japanese non-verbal sounds reliably. Sigh F1 is 0.45–0.75 and gasp 0.2–0.3; audio-LLM captioners invent events.
  - SOTA pipelines (NVSpeech, Fish S2) use automatic candidate generation plus human verification.
  - The strongest label source we have is the VN script itself: its kana interjections are the stage direction the actress performed. For example, there are about 80 laughs written as ふふっ/クスッ that currently have no tag.
- **Counts:** no paper gives a minimum per tag. Field evidence:
  - About 250–300 examples of a new tag can still be ignored (VoxCPM #292; IaraTTS).
  - The challenge winner added 1,000 diverse synthetic examples for its rarest tags.
  - Working rule: **≥150 verified examples to advertise a control; 50–149 train only, to explain the audio; under 50 merge into a family token.**

### 1.4 The Chinese "Japanese accent": has SOTA addressed it?

**Partly for wrong readings of shared characters, and not in any measured way for accent.**
- No paper evaluates a Japanese single-speaker voice speaking Mandarin with an accent metric.
- CosyVoice 3's authors say there is "no reliable objective method to evaluate the authenticity of accents".
- The problem is confirmed across models, each reported in issue threads:

| Model | Issue |
|---|---|
| Fish S2-Pro | #1263 |
| CosyVoice | #411, #139, #385 |
| OpenVoice | #258 |
| GPT-SoVITS | #1054 |
| Qwen3-TTS | #134, #323 |

**Levers with evidence:**

| Lever | Evidence |
|---|---|
| Explicit language conditioning | VALL-E X ablation (verified): native-accent score 4.03 with the language ID vs 2.35 without (en→zh, subjective 1–5 scale). IndexTTS-2.5 adds a per-token language embedding and reports homograph mispronunciations "effectively suppressed". CosyVoice 3 converts Japanese to kana so that kanji tokens mean only Chinese. |
| Native target-language data with speaker and language IDs | CosyVoice 2/3 multilingual SFT, VoxCPM (10–20% replay), Amazon Polyglot |
| Native speech converted into the target timbre (timbre-only VC) | Amazon VC-polyglot. Accent-converting VC modes re-import the accent. |
| Inference-time guidance toward native | Mega-TTS 3 accent intensity, language-contrastive / dual-reference guidance (2026). Unproven on AR codec LMs. |

**Evidence on our own disk:**
- v2's ASR transcripts show Japanese readings: 时间 heard as 只见 (Japanese *jikan*), 先 heard as 三 (Japanese *sen*).
- Base cloning from a Japanese reference with `Auto` spoke Japanese for 8/8 Chinese prompts.
- Qwen issue #323 (verified): a Bengali single-speaker fine-tune came out with a Hindi accent. A commenter attributes it to `dataset.py` hard-coding `nothink`, so the language field is ignored. The thread's suggested fix was never tested.

**How to measure it** (ASR error rate hides accent):
- **VoxLingua107 language-ID log-ratio**, P(ja) vs P(zh). Two 2026 papers already use it as an accent proxy.
- **Toneless syllable error rate and tone error rate** from charsiu's toned-pinyin frame classifier, which has no language model.
- **Calibration on anchors:**
  - native: the Chinese-dub clips, and Base with a native reference;
  - Japanese-accented: Qwen's own Japanese CustomVoice speaker forced to speak Chinese;
  - plus a short blind native-listener check.

## 2. Reconciled v3 design

The reports disagreed on one point. Here is how I resolved it.

### Conditions (Amadeus, phone)

**Use separate speaker rows rather than text tokens.**
- The internals and fine-tuning reports favour speaker rows; the labeling report favours text tokens.
- I choose speaker rows because:
  - it is the model's native global-condition mechanism;
  - the frozen speaker encoder's embedding already carries the channel, so the row starts with a strong prior rather than a random new token;
  - the default `kurisu` row stays clean by construction;
  - the Space only needs a speaker dropdown.
- Text tokens (`<amadeus>`, `<phone>`) stay as the ablation in case conditions need to combine.

**What a separate row means:**
- The talker weights are shared by every row. The 920 Amadeus clips train the same model and the same voice as everything else.
- Only the switch that selects the device sound is separate, so Amadeus is kept, not split off.

**Space wiring:** every new `spk_id` entry needs a matching `spk_is_dialect` entry in `config.json`, or `generate()` raises `KeyError`.

**Rows:**
- **`kurisu`** (3000): clean studio and anime speech.
- **`amadeus`** (3002): clips with an audible device EQ.
  - All 652 game Amadeus clips.
  - Anime Amadeus clips only where a spectral-shape detector, trained on game Amadeus vs game Kurisu, says the EQ is audible.
  - Clean-sounding anime Amadeus clips go to `kurisu`, because the label must describe the sound, not the story role.
- **`phone`** (3003): the 28 real anime phone clips, plus 300–500 diverse clean clips filtered with the *measured* show phone curve and jittered.
  - The real phone sound is a bass-cut, mid-boost EQ, not a 300–3400 Hz telephone band.
  - Compare blind against plain DSP on default output.
  - **DSP may well win.** A row trained mostly on our own augmentation mostly learns our own filter. That outcome is acceptable, not a failure.

### Events

New single tokens placed in the 260 unused embedding rows, so no resize is needed. Each is initialized from related words.

| Token | Covers | Count | Status |
|---|---|---|---|
| `<sigh>` | sigh, breath-out | 148 | Expose once 150+ verified |
| `<inhale>` | gasp + breath + sharp inhale | 79 | |
| `<laugh>` | — | 0 tagged, about 80 script candidates | |
| `<nv>` | catch-all for other audible non-verbal sounds | — | Trained, never advertised |

- **Kana rule:**
  - Articulated sounds keep the script kana with no tag.
  - Non-lexical sounds get the token, plus any kana the script already has, e.g. `<laugh>ふふっ`, `<sigh>はぁ……`.
- **Pause:** keep `……`. It is already one token, the model knows it, and it worked in v2.
- **Upsampling:** square-root, capped at 3×.
- **Never drop condition labels.**

### Language

- Kurisu Japanese rows get the `japanese` language id: `think, think_bos, 2058, think_eos, SPK, pad`.
- About 25% of rows keep the exact `nothink` layout so that `Auto` still works.
- Copy the layout from `generate()` exactly, with per-row prefix offsets. **Do not copy PR #361**: its Auto rows use a layout `generate()` never builds.
- Verify with an fp32 test asserting that trainer and `generate()` embeddings are equal.

### Native Chinese and English

- **About 15% of rows from other native speakers**, each with their own x-vector and language id. Sources:
  - AISHELL-3 (Apache-2.0) for Mandarin;
  - LibriTTS-R and/or the English dub (93 clean minutes) for English;
  - or Base self-generation from native references.
- **The Chinese dub is too small to matter alone.** Only 9.4 minutes is non-Amadeus Kurisu; the other 19.3 minutes is Amadeus.
- **Then, if needed, Kurisu-timbre native Mandarin**, made by timbre-only voice conversion of native speech.

### Selection

The v2 Japanese confirm suite stays as the regression guard. Added to it:
- Chinese and English accent metrics (language-ID log-ratio; syllable and tone error rates), calibrated on the anchors in §1.4;
- an Amadeus/phone leakage check: the default output's long-term spectrum stays within ±1 dB of v2's;
- tag gates: at least 60% realisation, at most 5% false triggers, and no increase in endless generations.

## 3. Experiment order

Each step changes one thing.

| Step | What | Cost |
|---|---|---|
| E0 | Inference-only diagnostics on existing weights:<br>• Base with Kurisu x-vector, Auto vs Chinese<br>• Base cloning with a Japanese reference<br>• Base with a native reference (anchor)<br>• Qwen's own Japanese speaker forced into Chinese (accent anchor)<br>• v2 Auto vs Chinese<br>• v2/Base weight blend.<br>Then build and calibrate the accent metrics. | about 1–2 h GPU; no training |
| E1 | Re-run v2 with 2 seeds (noise floor) | minutes each |
| E2 | E1 + language-id rows | |
| E3 | E2 + 15% native replay | |
| E3f | E3 with text embedding and projection frozen (± bottom half of layers) | |
| E4 | best + Kurisu-timbre native Mandarin, if E3 is not enough | VC setup |
| E5 | E1 + `amadeus`/`phone` speaker rows, vs DSP-only control | |
| E6 | E1 + event tokens, after verification | needs the labeling pass |
| Final | Combine what was adopted, 2 seeds, pre-registered selection, publish | |

E0 decides how to make the Kurisu-voiced Mandarin data for E4:
- If Base with an x-vector already sounds native, Base self-generation is enough.
- If not, voice conversion is required.

**Cost:**
- Compute is roughly 2–4× v2's ~$15. Each training run takes minutes; generation and evaluation take hours.
- Label verification is human time, not pod time: 4–7 h of listening if tags are to be exposed.

**Human checks that gate shipping:**
- **Mandarin accent.** Every accent metric here is unvalidated for Japanese-accented Mandarin. A blind rating is needed before any claim about v3's Chinese ships: 20 utterances × 4 systems, 1–5 scale, by you or a native speaker.
- **Tags.** The 60% realisation gate also needs a listener.

## 4. Decisions that are the user's

1. **Who verifies the event labels?**
   - SOTA practice is a person checking every candidate: about 1.5–2.5k clips, 4–7 h of listening. I cannot listen.
   - Alternative: rely on script kana (high precision, game only) plus the existing human tags. That gives fewer exposed tags.
   - I can build a review page (audio, spectrogram, accept/move/relabel) that saves decisions.
2. **What "sound effects" means.**
   - Non-verbal vocal sounds (sigh, gasp, laugh) are covered above.
   - Background sound effects (footsteps, ringtones, knocks; about 50 clips) are too few to learn. The SOTA route is mixing a sound library CapSpeech-style, in a later version.
3. **Native Mandarin replay data.** Use public corpora (AISHELL-3, other speakers), the Chinese dub as its own speaker, or both.
4. **Publishing.** Whether v3 replaces the published model if it wins, as v2 did, with v2 kept at a revision tag.
5. **Dataset repo.** Whether `starrydark/Kurisu_Voice` gets the new labels (a v3 label config) or they stay in the model repo.

## 5. Corrections to the earlier v2 reports

- **"Write non-verbal sounds as kana" was half right.** Kana is right for articulated sounds. Non-lexical ones need an atomic token, otherwise the event is unexplained.
- **The Sarashina2.2 paper's very poor Qwen Japanese numbers look like runaway generation under unknown settings**, not misread kanji. The CER was 63 even with Japanese prompts.
- **"Amadeus is ~18 dB brighter" was an oversimplification.**
  - Game Amadeus is about 16 dB in median tilt, with lower HNR. It is non-linear, so a single EQ can't reproduce it.
  - Anime Amadeus is mostly near-clean.
  - The phone clips are a bass-cut, mid-boost EQ.
- **Your planning doc cites arXiv 2604.16211 as "NVBench".** It is NVV-SuperBench; NV-Bench is arXiv 2603.15352.

## 6. Primary-source checks performed for this synthesis

| Claim | Result |
|---|---|
| VALL-E X language-ID ablation | **Confirmed**: 4.03 / 2.35 / 2.25 (correct / none / wrong ID), en→zh, subjective. arXiv 2303.03926 |
| Qwen #25 | **Confirmed**: "our base model does not currently support instruct" (wangxiongts, 2026-01-26) |
| Qwen #323 | **Confirmed**: Bengali, about 7 h, Hindi-like accent; `nothink` hard-coded per a commenter, whose status isn't clear. The fix was never tested, and the issue was closed by a bot. |
| IndexTTS-2.5 | **Confirmed**: token-level language embedding "effectively suppressed" homograph errors. arXiv 2601.03888, authors' own claim |
| Confucius4-TTS human evaluation | **Confirmed**: en→zh pronunciation rank Qwen 4.00 vs VoxCPM2 1.88. Internal, no rater details. arXiv 2608.11650 |
| IaraTTS card | **Confirmed**: vocabulary extended so tags are single tokens. The 50–60% rendering figure comes from the agent's reading of the card. |
| charsiu `zh_xlsr_fc_10ms` | **Exists**, but has no model card or licence on HF. Its label set (toned pinyin) comes from the agent's reading of the repo config; I did not re-check it. |
| PR #384 | **Confirmed**: open (2026-10-17), multi-speaker SFT plus label fixes, no language-id handling |
