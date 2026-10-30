# Japanese single-character TTS: where things stand in late Oct 2026

Your custom trainer is fine. It already fixes the loss bugs that the official Qwen3-TTS fine-tuning script still has, so those bugs did not cause your problems. The cause was the training settings plus the tags in your data. The ranking below would only change if Japanese naturalness counted for more than everything else.

**Source tags:**
- **[P]** I read it in a primary source: repo code, an issue, a README, an HF card or the HF API, or an arXiv paper.
- **[S]** Secondary: a blog, or note/zenn/qiita.
- **[I]** My own inference.
- **[L]** Your local files under `kurisu_qwen3tts/`.
- **[U]** I could not verify it.

## 0. Your current run

1. **Your trainer is correct where the official script is not.**
   - The official `finetuning/sft_12hz.py` on `main` still has three bugs [P]:
     - It shifts labels twice, so the main model (talker) learns to predict frame t+2 instead of t+1.
     - It never applies `text_projection`.
     - The code predictor (sub-talker) reads a hidden state that has already seen the frame it is predicting.
   - The fix is PR #278, open since 2026-03-29 and still not merged. Issue #371 (2026-09-16) reports a run that "could not stop generating."
   - With the official script, a user reported (in the PR #278 thread) 5 s sentences coming out 13–201 s long. With the fixes plus 2 epochs at lr 5e-6, all 44 test sentences ended normally, with CER 2.0% [P].
   - Your `ft02_sft.py` applies #278, #337, text_projection and the frozen speaker encoder (#351), with `label_alignment=fixed` [L].
2. **The run overfit badly.**
   - The talker's training loss (codec-0 cross-entropy) at epoch 10 was **0.06–0.27** [L].
   - The base model, scored with the correct alignment on held-out audio, gets talker CE 1.09 and sub-talker CE 4.94 [P, PR #278]. Those held-out numbers are the comparison I am using here.
   - A CE near 0.1 means the model memorised the training clips. That matches your validation minimum around epoch 1.5.
3. **The flat sub-talker loss (~4.0–4.2) is normal.** You are already below the base model's held-out 4.94 (Korean) and 5.38 (Spanish) [P]. My reading is that the residual codebooks hold fine detail with high irreducible entropy, and their loss weight is only 0.3 [I].
4. **The endless generations come from tags and unusual spacing, not from overfitting.**
   - In `eval/kurisu-lora-r16`, most probes ran to the length cap:

     | Epoch | Probes that hit the cap |
     |---|---|
     | 0 | 33 of 40 |
     | 1 | 30 of 40 |
     | 2 | 26 of 40 |
     | 3–5 | 19–23 of 40 |

   - The cap hits are almost all in prompts with tags (`tag_initial` 4/4, `tag_medial` 4/4, `tagged` 8–12/12) or with "、 " (comma plus an ASCII space). Normal sentences hit the cap 0–1 times out of 4 [L].
   - **Cap hits are worst at epochs 0–2, so choosing an earlier epoch will not fix stopping.**
     - The epoch you pick controls overfitting.
     - Fixing the tags and spacing controls stopping.
   - **Please confirm which eval you meant.** The r32 eval (12 probes, epochs 3–10, including tag prompts) had **0/12** cap hits. All the hits I found are in the r16 eval, which used the earlier data build, so the 2026-09-05 rebuild may already have fixed part of this. I also could not tell whether r16 ran with `-Deterministic` (greedy decoding); greedy decoding by itself is known to cause runaway generation [L/U].
   - In your current training set, `[pause]` appears 602 times. The other tags are rare: `[sigh]` 32, `[gasp]` 24, `[digital]` 17, `[breath]` 12. Of 8,353 rows, 1,543 contain spaces [L].
   - Qwen3-TTS has no explicit non-verbal support. The maintainer says such sounds come out only implicitly, from text like "hahaha" or "emmm" [P #9].
5. **Your eval script uses a prompt format the model never saw in training.**
   - Training (official and yours) always uses the `nothink` prompt format, which is what `language="Auto"` produces.
   - `language="Japanese"` inserts `[think, think_bos, lang_id, think_eos]` instead [P `generate()`, #323].
   - `ft03_eval.py` defaults to Japanese. `app.py` uses Auto [L].
6. **There are no newer packages or models to switch to.**
   - `qwen-tts` 0.1.1 (2026-02-06) is still the latest on PyPI [P].
   - The only official weights are the 12Hz checkpoints from 2026-01-21. The 25Hz models described in the paper were **never released** (people were still asking in #34 in Aug 2026) [P].
   - There is no open Qwen3.5-Omni or Qwen TTS successor.
   - Qwen-Audio-3.0-Gen-Preview (arXiv 2607.27011) is an audio-scene generation model with no published weights [P].

## 1. Ranked shortlist

The ranking weighs fit to all three goals and the cost of switching. **For Japanese naturalness alone, #2 and Style-Bert-VITS2 JP-Extra probably beat #1.**

### #1. Keep Qwen3-TTS-12Hz-1.7B-Base, with corrected settings and data

It is the only candidate that covers Japanese, cross-lingual and tags in one Apache-2.0 model you already deploy, and your failures have explanations you can fix.

**Strengths:**
- Its cross-lingual results from a Japanese prompt are among the best published: ja→en WER 3.04 and ja→zh CER 3.43, against CosyVoice3's 4.20 and 3.05 [P tech report, Table 7].
- On Japanese itself: WER 3.82 and speaker similarity (SIM) 0.788, against MiniMax 3.52/0.776 [P Table 6].

**Weaknesses:**
- **Japanese reading accuracy.** The independent Sarashina2.2-TTS paper (2026-06-24) measured Qwen3-TTS at:
  - JSUT CER 14.87, the worst of the five systems it tested.
  - Joyo-kanji Kana-CER 185.9 raw, or 21.7 with per-clip errors capped. The authors put this down to "frequent hallucinations."
  - CER up 328% when the voice prompt is not Japanese [P].
- **Open bugs:**
  - #55: the last Japanese word gets cut off. Unresolved [P].
  - #386 (2026-10-22): a periodic ~40 ms echo [P].
  - #343: after LoRA, short utterances and the first second or two of any utterance can switch timbre. Passing a speaker embedding from the base model at inference works around it [P].
- **Cross-lingual erosion.** Fine-tuning on a large single-language set washes out other languages; the GPT-SoVITS project documents this on its own wiki [P]. Separately, clones tend to keep the source accent (#134) [P].

### #2. Irodori-TTS v4.1-Small, fine-tuned for Japanese

**What it is:**
- MIT licence, 0.8B parameters, released 2026-08-11 [P HF API]. The repo has 1.5k stars and was last pushed 2026-10-25.
- A rectified-flow DiT on a Japanese 48 kHz audio codec, with a ModernBERT-ja text encoder.
- **It accepts Japanese text only.** For English or Chinese you would need a second engine (#1 or the Qwen Base model).

**Japanese accuracy** [P card]:
- JSUT standard CER 7.22 and sentence Kana-CER 3.43.
- Joyo Kana-CER 7.29.
- These were measured with a different setup from the Sarashina paper, so you cannot compare them directly with Qwen's 14.87 [I].
- With one short reference clip, speaker similarity was "modestly lower than v3."

**Non-verbal control is built in.** Emoji in the text map onto your tags [P EMOJI_ANNOTATIONS.md]:
- 😮‍💨 breath or sigh
- 😮 gasp
- ⏸️ pause or silence
- 🌬️ heavy breathing
- 🤭 chuckle
- 👂 whisper

**Community experience:**
- Japanese users treat it as the 2026 pick alongside SBV2 [S].
- An anime fine-tune exists [S].
- A note.com user ran a LoRA on about 120 clips and got the most faithful character voice of anything they tried. Removing the whisper clips stopped 👂 from working [S].
- No 10 h+ single-speaker results have been published [U].

**The model card asks users not to impersonate a voice without explicit consent** [P].

### #3. VoxCPM2 LoRA, as an all-in-one fallback

**What it is:** Apache-2.0, 2B parameters, released 2026-04-03 [P].

**It has the best-documented official fine-tuning of any candidate** [P docs]:
- Ready-made LoRA and full fine-tuning configs.
- An explicit stop loss (`loss/stop` = 1.0), which is relevant to your stopping problem.
- LoRA adapters you can swap at runtime.

**Japanese and cross-lingual numbers:**
- MiniMax-ja: CER 4.63, **SIM 0.828**, against Qwen's 0.788 [P README].
- CV3-ja: CER 5.96.
- Chinese prompt to Japanese text: 11.84, against Qwen's 36.09 [P, IndexTTS-2.5 paper].

**Limits:**
- No built-in tags.
- Needs about 20 GB of VRAM for LoRA and 40 GB for full fine-tuning.

### Not shortlisted

- **Style-Bert-VITS2 JP-Extra (v2.7.0, 2025-08-24).**
  - Still the proven Japanese single-character default. One paper reports MOS 4.37 against 4.38 for real recordings, from 10–15 minutes of data [P arXiv 2505.17320].
  - Japanese-only, no tags.
  - **Worth training as a cheap baseline to A/B against.**
- **GPT-SoVITS.**
  - The last formal release is v2ProPlus (2025-06-06).
  - V5 has existed only as code on a branch since 2026-10-05, and its few-shot fine-tuning is "not studied" [P].
  - Its own wiki says large single-language data wipes out cross-lingual ability [P].
- **Fish S2 Pro.**
  - Best Japanese accuracy: MiniMax-ja CER 2.76, SIM 0.796 [P].
  - Licensed for non-commercial use only.
  - The fine-tuning code was removed from `main` on 2026-10-13. Its docs say LoRA "will only learn the speaker's speech patterns and not the timbre" [P].
  - Drama 3 (2026-10-01) does worse on Japanese (5.02) and needs a separate licence for any hosted service [P].
- **IndexTTS-2.5.** Strong Japanese (5.30) and the best Chinese-to-Japanese result (6.38), but **no official fine-tuning** (#518) [P]. It could serve as a zero-shot English/Chinese engine [I].
- **CosyVoice3-0.5B-2512.**
  - Has built-in `[breath]`, `[sigh]` and `[laughter]` tags.
  - Full fine-tuning only, no LoRA.
  - Reached only about 9% Japanese CER in the paper's own experiment turning a one-language speaker multilingual [P].
- **Others set aside, with the main reason:**
  - Higgs TTS 3: no fine-tuning code, non-commercial.
  - MOSS-TTS 1.5: 8B parameters, CV3-ja 10.97.
  - Step-Audio-EditX: no licence file for the weights.
  - FireRedTTS-2/3, Chatterbox Multilingual V3, Kani-TTS, Orpheus: weak or no Japanese, or no official fine-tuning code.
  - Llasa: non-commercial. Spark-TTS: Chinese/English only. CSM: English only.
  - OmniVoice: CC-BY-NC.
  - Sarashina2.2-TTS: best Japanese similarity, but non-commercial and I found no fine-tuning code.

## 2. Recipe A: Qwen3-TTS, corrected

**Keep:** `ft02_sft.py` with `label_alignment=fixed`, text_projection on, the frozen speaker encoder, and `sub_talker_weight 0.3` (the official value) [P/L].

**Change one setting at a time, so a bad result doesn't hide the cause:**
- **(a)** Keep r32 / α64 and lower the learning rate to **3e-5**. Train **3 epochs** (about 780 steps at effective batch 32), with cosine decay, 3% warmup, and an eval every 100 steps.
- **(b)** Keep 1e-4 and stop at about epoch 1.5–2. Your existing `adapter-epoch-1` is the quickest thing to try.

**Full fine-tuning, if you have a bigger GPU:** keep master weights in fp32 and use lr **5e-6** for 1–2 epochs. That is the setting from the working run in the PR #278 thread [P].

**Fix the language mismatch:**
- **Simplest:** serve and evaluate only with `language="Auto"`.
- **Better for cross-lingual:** port the #323 change so Japanese rows train with the `japanese` language id. You can then pass `english` or `chinese` at inference [I].
  - The prompt prefix grows by one token, so the speaker slot (index 6) and every `8+n_text` offset in `collate` move by +1.
  - Copy the token order in `generate()` exactly.

**Fix the tags:**
- **Remove `[sigh]`, `[gasp]`, `[breath]` and `[digital]` until each has at least ~150–300 clean examples** [I]. Keep the audio and write the sound as kana, the way anime-whisper does: はぁ…, っ, ふふっ [P].
- Render `[pause]` as "……" or "、" instead [I].
- **Never put an ASCII space after Japanese punctuation** [L].
- If you keep bracket tags, add them as dedicated tokens rather than letting the tokenizer split them [I].

**Choosing a checkpoint:**
- Pick by anime-whisper kana-CER, speaker similarity, and the cap-hit rate on held-out text without tags.
- **Do not pick by validation loss.** One project's blind listening test picked epoch 2 over the lowest-loss epoch 7 [S].
- Also try lower LoRA scales at inference (0.3–1.0) [S instavar; re-check those values with the fixed loss].

**Cross-lingual:**
- Run fixed English and Chinese probes at every eval.
- Serve English and Chinese with the adapter at about 0.3–0.5 scale, or through Base voice-clone mode (ICL) [I]. In ICL mode, pick a reference clip whose ending won't leak into the output (#341) [P].

**Inference:**
- Keep the defaults: temperature 0.9, top-k 50, repetition penalty 1.05 [P].
- Set `max_new_tokens` to about 1.5–2× (mora count ÷ ~8 per second × 12.5 frames per second), and regenerate once if it hits the cap [I].
- If a micro-batch exceeds 8,192 frames, switch the code predictor's attention to eager, because flash-attention backward produces NaNs [P #378].

## 3. Recipe B: Irodori-TTS v4.1-Small

**Data and manifest:**
- Use `--normalize-db -16`; the codec was trained with it [P].
- `max_latent_steps` 750 (about 30 s) covers all your clips [P].
- Turn on `--text-normalize`.
- Each manifest row needs `{text, latent_path, speaker_id:"kurisu"}` [P].
- Your audio is 24 kHz and the codec is 48 kHz, so the output will be band-limited [I].

**Tags:** map `[sigh]`/`[breath]`→😮‍💨, `[gasp]`→😮 and `[pause]`→⏸️, with a few hundred examples per emoji [I].

**Training steps:**
1. Start with speaker inversion. It trains only 16 speaker tokens on a frozen model, so it is a near-free baseline [P].
2. Then LoRA with the defaults [P]:
   - r16 / α32 on `diffusion_attn`, saving the duration predictor too.
   - Muon optimiser, lr 1e-4, batch 8, the config's `wsd` schedule, a save every 1k steps.
   - At about 1,250 steps per epoch, evaluate at 2k, 4k and 6k steps [I].
3. If the timbre still underfits, do a full fine-tune at 2e-5–5e-5 for 1–2 epochs [I].

**Inference:** 40 sampling steps, with guidance scales of 3.0 for text, 3.0 for caption and 5.0 for speaker [P]. It fits easily in a ZeroGPU slot.

## 4. Dataset prep

- **Clip length:**
  - Keep 1–20 s. VoxCPM's docs say clips under 1 s are unstable [P].
  - Style-Bert-VITS2 uses 2–14 s [P].
  - You have 610 clips under 1 s. Keep only those with exact transcripts, and keep them to a few percent of the set [I].
- **Silence:**
  - 100–200 ms at the start.
  - **At most 200–300 ms at the end, never more than 0.5 s.** VoxCPM's FAQ says trailing silence over 0.5 s is "the most common cause" of generation that never stops [P].
  - Check that the 751 clips ending in "……" don't carry long breath or noise tails [L/I].
- **Loudness:** one target for the whole set, either −23 LUFS (Fish's default) or −20 dBFS with gain capped at ±3 dB (Emilia) [P].
- **Transcripts:**
  - Re-transcribe with litagin/anime-whisper (MIT, trained on 5.3k h of galgame audio, CER 13.0 against 16.5 for whisper-large-v3) [P].
  - Compare on kana, not kanji.
  - Review anything over ~10–15% error [I].
- **Game vs anime audio:**
  - Make the dry VN game audio the core of the set.
  - Accept source-separated anime clips only if DNSMOS OVRL is at least 3.0 (Emilia's threshold) and there is no audible bleed [P/I].
  - I found no primary study on training TTS with separated audio [U].
- **Split and dedup:**
  - Hold out 1–2%, balanced across length, emotion and tags.
  - Add a fixed probe list that includes English, Chinese and text without tags.
  - Dedup on normalised text plus near-identical audio [I].

## 5. Main risks

1. **Qwen may still misread Japanese after fine-tuning** (Sarashina paper). Test on unseen kanji-heavy text.
2. **Cross-lingual ability** may erode under #1, and #2 has none.
3. **Tags with under ~30 examples won't be learned** and will make stopping worse [L/I].
4. **Community trainers inherit the official bugs.** vspeech, EasyFinetuning and sruckh all use the unfixed objective [P].
5. **ZeroGPU hardware has changed.** It is now half an RTX Pro 6000 Blackwell (48 GB), or a full one (96 GB) at twice the quota. The default call limit is 60 s, torch 2.8–2.13 is supported, and `torch.compile` is not [P HF docs].
6. **Legal and ethical.** This is a public clone of a real voice actress. Japan's actors' union, Nippairen (日俳連), issued a guideline on 2024-10-15 that requires consent, compensation and control, and it backs the #NOMORE無断生成AI campaign [S news]. Irodori's card separately asks users not to impersonate voices without consent [P].
   - **Erratum (2026-10-29):** the guideline 「音声に関する生成AI利用ガイドライン」 was published on **2026-10-15**, not 2024-10-15; 2024-10-15 is the launch of #NOMORE無断生成AI. See `bangdream/research/recommendation_2026-10-29.md` §6.

## 6. Could not verify

- Any candidate's 10 h+ single-speaker fine-tuning metrics; none are published.
- Whether Irodori's −16 is LUFS or dBFS.
- Whether training with a Japanese language id preserves cross-lingual ability.
- Which Qwen variant the Sarashina paper evaluated.
- What "Hosted Service" means in §V of the Drama 3 licence.
- The licence on the Step-Audio-EditX weights.
- Whether the r16 eval used greedy decoding.
