# Kurisu v3: results (2026-10-29)

**Published.**
- Model: `starrydark/Kurisu_Qwen3_TTS`, commit `fa33568`. The weights' sha256 `db51266f…` was verified against the pod checkpoint.
- Earlier releases are tagged `v2` (`92a0836`) and `v1-epoch10`.
- Space: commit `1a5d3d9`, running. It now has voice and language dropdowns; the language is detected from the script.
- Blind Mandarin rating page (private): https://claude.ai/artifact/HiZNcoid5r433sg7kcNDjD
  - 16 sentences × 5 systems.
  - The answer key is local only: `kurisu_v2/v3/rating_key.json`.
  - Ratings are stored in the artifact's db, collection `ratings`.
- **Rating done (2026-10-29).** Full analysis: `rating_results.md`.
  - v3 scored 5.00, the same as the native anchor. v2 Chinese 3.94, v2 Auto 3.56, Japanese-read anchor 1.06.
  - v3 beat v2 Chinese on 9 of 16 sentences and lost on none (p = 0.004).
  - One rater; the scale is at ceiling.
  - Model card updated: model repo commit `df09626`, README only; the weights are unchanged from `fa33568`.
- **Training details published (2026-10-29), model repo commit `008b95c`.** Weights unchanged (sha256 `db51266f…`). The commit adds:
  - a full "Training details" section in the card;
  - `training_code/`: 32 scripts plus a README with the run order;
  - `training_logs/`: final-run training logs and the mix and selection logs;
  - `eval/listening_test.json`: aggregated scores, no clip ids or rater id.
- **Space commit `a5c8b92`** adds a "How v3 was trained" README section, an "About this model" accordion, and Mandarin, Amadeus and phone examples. It was pushed through the Hub API.
- **Dataset `starrydark/Kurisu_Voice` (2026-10-29): commits `8cdf0f8` and `c7a476e`.**
  - **Added `labels/kurisu_tts_v3/{train,validation,test}.jsonl`**, one row per Japanese clip (10,039 rows). Columns: `v3_text`, `v3_voice`, keep/drop decision and reason, `v3_trained`, tag evidence, the v2 decision, both ASR transcripts and their CER. Join key: `sha256`.
  - **The card has a new section about them.** The default config, audio and metadata are unchanged.
  - **The first commit added a `kurisu_tts_v3` config, which loaded empty.** `datasets` infers one builder per repo (audiofolder here), so the second commit removed it and documented loading the files with the `json` loader instead.
  - **Discovery pass (2026-10-29).** Research: Hub search, SEO/LLM discovery, and keyword/competitor surveys.
    - **Repos renamed** (old ids redirect with HTTP 307; likes, downloads and tags kept):
      - model → `starrydark/Makise_Kurisu_Qwen3_TTS` (card commit `26c9681`);
      - dataset → `starrydark/Makise_Kurisu_Voice` (card commit `a78b349`);
      - the Space keeps its id (commit `6676ca9`).
    - **Cards:** Makise Kurisu H1 and title, JA/ZH/KO alias line, Japanese and Chinese summaries, key-facts table, FAQ, a train-your-own guide, BibTeX, Qwen3-TTS paper link, character tags, `license_link`, and a 1200×630 thumbnail.
    - **Space:** `datasets:` link added.
    - **Collection:** https://huggingface.co/collections/starrydark/makise-kurisu-ai-voice-steins-gate (model, Space, dataset, paper).
    - **Result:** Hub id search for "makise" now finds the model (#3 of 17) and the dataset (#1). Full-text search lags behind edits.
  - **Data findings:**
    - In 163 clips, Qwen3-ASR echoed its context prompt instead of transcribing. `asr_qwen3` is null there, and those clips' mismatch decision rested on anime-whisper alone.
    - 266 `transcript_mismatch` rows are likely transcript errors; none has been reviewed by ear.
    - Two byte-identical take pairs: PHENO 01678/01694 and SGRE 00538/00551.

**Selected checkpoint.** `final_s20261028/checkpoint-step-343`: training seed 20261028, 1.0 epoch. The recipe:
- the v2 recipe;
- E3: Japanese language id, plus 15% native zh/en replay;
- E4: about 5% Kurisu-voiced pseudo-Mandarin from Base;
- E5b: `amadeus` and `phone` speaker rows;
- E6b: `[sigh]` and `[nv]` tokens, without upsampling.

## Final numbers (paired with v2, same texts and 4 sampling seeds)

| | v2 | v3 |
|---|---|---|
| JA CER (592 clips) | 0.032 | 0.030 |
| JA sim | 0.668 | 0.671 |
| zh M1 (language=Chinese for both) | −12.8 | −19.6 |
| zh shared-char SER | 0.170 | 0.060 |
| zh CER | 0.071 | 0.017 |
| en WER | 0.005 | 0.002 |
| runaways | 0 | 0 |

**Reference points.**
- Native anchor: M1 −22.6, SER 0.048.
- Base model with Kurisu's speaker vector: −19.9 / 0.057.
- v2 as served (Auto): −6.6 / 0.289.

**Fresh-seed re-test (26000):** A2 passes (ΔCER −0.0036, ΔSIM lower bound −0.000). A3′ passes (0.053 / 0.058).

**Device voices.** A5 projection 1.03–1.14 for `amadeus`, 1.16–1.31 for `phone` (in-sample). No leak into the default voice: A3′ ≤ 0.06.

**Tags** (detected at 5% FPR on the pick):
- `[sigh]` 13% vs 3% for twins.
- `[nv]` 14% vs 0%.
- Experimental.

## What did not work

- **E3f, freezing the text path:** no extra accent gain.
- **E5, a separate anime-Amadeus voice:** its x-vector cannot carry that sound. E0 had already shown this.
- **E6, tags upsampled ×3:** cost similarity.
- **E7, background-SFX tokens:** detected in 0–7% of outputs, and the default voice shifted.

## Amendments to the pre-registered rule

All were disclosed in `SELECTION_RULE.md` before the run each one governs.

| Amendment | Change |
|---|---|
| A | Statistically calibrated gates |
| B | E5b and E6b follow-ups |
| C | A3 replaced by A3′: the original A3 failed negative controls on training-seed drift |
| D | Matched-regime replication of E4: spread 0.96, gain 4.82 → adopted |

## Cost

- One Secure H100 pod, about 10.5 h at $3.99/h, so roughly $42.
- The ZeroGPU Space is covered by PRO.

## Next

1. ~~The user's blind rating~~: done. See `rating_results.md`. For a stronger accent claim, use more raters and a finer scale, such as a MUSHRA-style 0–100 slider with a voice-matched native anchor.
2. **If tags matter:** collect more verified takes (human listening). 81 sighs is below the ~150 at which a tag becomes a control.
3. **If the Mandarin accent still matters:** voice-conversion pseudo-data, or language-contrastive CFG at inference. Both are in `crosslingual_accent_2026-10.md`.
