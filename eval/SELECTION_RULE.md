# Checkpoint selection rule (pre-registered 2026-10-28, before any training result)

Scored by `pod/p06_eval.py` → `summary.json["select"]`, which covers only the untagged
Japanese probes: 24 held-out test-split lines (`ja_heldout`) + 20 novel sentences
(`ja_novel`). Tagged/control probes and EN/ZH probes are diagnostics, not criteria.

1. **Eligibility:** `runaway == 0` (no clip longer than 2.5x the expected duration)
   on the select set.
2. **Primary:** lowest `cer_mean` (reading-space CER of Qwen3-ASR vs intended text).
3. **Tie-break:** any eligible checkpoint within 0.01 CER of the best is a tie; among
   ties, highest `sim_mean` (ECAPA cosine to the centroid of real held-out Kurisu clips).
4. **Guard:** the winner's EN and ZH `cer_mean` must not exceed 1.5x those of the
   zero-shot Base-model clone; if it does, the next eligible checkpoint is taken and
   the trade-off reported.
5. Validation loss is logged and reported but is **not** a criterion.
7. **Amendment (made after the `real` baseline, before any trained checkpoint was
   scored):** the original runaway test (`dur_ratio > 2.5`) flagged 5/24 *real*
   held-out recordings, because acted speech is slow and pause-heavy (ratio p99
   4.9 over 536 real val/test takes). Runaway is now: hits the 400-frame cap
   (>= 31 s), or both > 5x the expected duration and > 6.5 s over it. All
   summaries were recomputed with the same code (`p06b_resummarize.py`).
8. **Confirmation (registered after the main-suite ranking, before running it):**
   step 6's listening check cannot be done by the agent, and main-suite
   differences (n=44) are within sampling noise. The checkpoints kept by pruning
   (top 2 per run) plus v1 are re-scored on suite `confirm`: the 128 held-out lines
   disjoint from the main suite + the 20 novel sentences, sampling seeds offset by
   +5000. Steps 1-4 are applied to `confirm` alone; that ranking is final. The user
   is sent A/B clips for the listening check.
6. The final pick is confirmed by listening to the select-set clips of the top two
   candidates; any override of steps 1-4 is reported with its reason.

---

# v3 rule (pre-registered 2026-10-28, before any v3 data build, metric run or training)

v3 adds capabilities v2 lacks, so v3 is accepted on non-inferiority for Japanese
plus a separate gate for each new capability. The thresholds below are fixed now and
are not tuned after results are seen.

V1. **Noise floor.** E1 re-runs the v2 recipe with 2 seeds on the v3 code path. Its
    seed-to-seed spread on the confirm suite defines "within noise" (Δ = |seed A − seed B|).

V2. **Japanese non-inferiority (required)**, on suite `confirm` with seed offset +5000:
    - runaway == 0;
    - CER ≤ v2 + max(0.005, Δ_CER);
    - SIM ≥ v2 − max(0.01, Δ_SIM).

    The v2 reference values are 0.030 / 0.667.

V3. **Default voice stays clean (required).** The `kurisu` row with no tags, on the
    confirm-suite texts, is compared with v2 on the same texts:
    - median long-term-average-spectrum difference ≤ 1 dB in every band from 150 Hz
      to 12 kHz;
    - median spectral tilt within ±1 dB.

V4. **Accent (ZH/EN).**
    - A metric is valid only if it separates the native anchors (D, R) from the
      Japanese-accent anchor (H) with Cohen's d > 1, and ranks D < E(v2) < H. A metric
      that fails is dropped and reported as dropped.
    - On the valid metrics, v3 must be at least neutral against v2. "Improved" is
      claimed only if it is better by more than the E1 noise floor.
    - Forced CER guard: ZH and EN CER ≤ 0.07 and no runaways.
    - **No accent-improvement claim ships until the user's blind rating**
      (20 utterances × 4 systems, 1–5).

V5. **Amadeus / phone rows.**
    - Fidelity: the row's output tilt and LTAS sit closer to its real takes than to
      clean Kurisu.
    - Each row must pass the JA CER guard: CER ≤ 0.07.

V6. **Event tokens.**
    - No tag reaches 150 human-verified examples (the user chose no listening). Every
      tag is therefore trained to explain the audio and documented as experimental,
      not advertised as a control.
    - A tag token is kept in the shipped tokenizer only if tagged prompts add no
      runaways and the tagged-probe CER stays ≤ 0.07.

V7. **Background-SFX arm.** It is isolated and can be cut.
    - It ships only if V2 and V3 still hold with it.
    - An audio tagger must detect the requested event in ≥ 50% of SFX-tagged outputs
      and in ≤ 5% of untagged outputs.
    - Otherwise v3 ships without it, and this is reported.

## Amendment A (2026-10-28, after code review, before any v3 data, metric or training result)

Independent reviews of the evaluation code and the statistics showed that V1–V7, as
first written, can be satisfied or failed by sampling noise alone. Each figure below
was computed from v2-era per-clip data, not from v3:
- Japanese CER: 71% of clips score 0, and the 5 worst carry 32% of the total. The
  minimum detectable difference at n=148 is about 0.020.
- 0.0297 is the best of 7 checkpoints, so it flatters v2 by about 0.007.
- A two-seed |Δ| ranges 0.001–0.013 (10th–90th percentile), so it is a random margin.
- "Every band ≤ 1 dB" fails an identical model 93% of the time.
- A duration-based tag proxy is meaningless: a model never trained on tags reads them
  aloud.

V1–V7 are therefore replaced by A1–A7. V8 still applies.

A1. **Reference and noise floor.**
    - v2 (`starrydark/Kurisu_Qwen3_TTS`, the published v2) is regenerated in the v3
      harness at sampling-seed bases 6000 and 11000. All v3 comparisons are paired
      against v2 at the same seeds.
    - The 6000 regeneration must reproduce v2's confirm CER (0.0297 ± ASR
      nondeterminism). If it does not, the harness is debugged before anything else
      runs.
    - E1a/E1b (v2 recipe, two training seeds) measure training-seed noise.

A2. **Japanese non-inferiority (required).**
    - Measured on the shipped Japanese mode:
      - `language="Japanese"` for checkpoints trained with language ids (the Space
        picks the language from the script);
      - `"Auto"` otherwise.
    - The 148 confirm texts are generated with S sampling seeds each: S = 2 for
      ablations, S = 4 for final candidates.
    - One-sided 90% paired bootstrap over texts (resampling texts, with seeds nested)
      against v2 on the same seeds. Pass if:
      - the bound on ΔCER ≤ +0.010;
      - the bound on ΔSIM ≥ −0.015;
      - runaway == 0.
    - Language-trained checkpoints must also keep `Auto` usable: mean ΔCER(Auto) ≤ +0.010.
    - The final pick is re-tested at fresh seeds (base 26000; S = 4 uses bases
      6000/11000/16000/21000). Clerical fix made while implementing, before any
      result: 21000 was already one of the S = 4 bases.
    - v2 is pinned to commit 92a08362cb2c95f7dfc8a7ecf4cc00db625f9d57 and generated at
      every seed base a candidate uses.
    - A de-duplicated sub-score, excluding confirm texts that have word-for-word or
      ≥ 0.85-similar twins in training (inherited from the dataset split), is reported
      alongside.

A3. **Default voice stays clean (required).**
    - Same texts, `kurisu` row, no tags, paired against v2.
    - LTAS in 1/3-octave bands, 160 Hz–10 kHz (nfft 4096), on active frames. Per band,
      the median over texts of the per-text difference.
    - Pass if:
      - the RMS over bands ≤ 1.0 dB;
      - no band exceeds 2.0 dB;
      - the median tilt difference is within ±1.0 dB.
    - If the v2-seed-6000 vs v2-seed-11000 null's 95th percentile exceeds a stated
      threshold, that percentile replaces the threshold. This is fixed now, before
      measurement.

A4. **Accent (ZH/EN).**
    - Anchors: native D (Base + native reference) and R (real dub); graded C (Base ICL
      from a Japanese take), J (Qwen's own Japanese CustomVoice speaker in Chinese),
      E/F (v2), and the dose-response G25/G50/G75.
    - A metric is valid if all of these hold:
      - d(D, C) > 1;
      - it is monotone over G25 → G50 → G75 → F;
      - its seed-to-seed per-text correlation is ≥ 0.5.
    - Metrics are reported per language and never pooled.
    - A Chinese accent gain counts only if:
      - it is beyond the A1 noise on a valid metric;
      - the shared-character syllable error rate is not worse;
      - zh SIM drops ≤ 0.03.
    - **English has no valid automatic accent metric** (smoke d = 0.06). No English
      accent claim is made.
    - No accent claim ships before the user's blind rating.

A5. **Amadeus / phone rows.** (For `phone`, the reference takes are also training takes:
    only 22 exist. Its A5 check is therefore in-sample, and is reported as such.)
    - Project (row output LTAS − kurisu output LTAS, same texts) onto (real row takes −
      real kurisu takes). Require ≥ 0.5.
    - JA CER ≤ 0.07 on that row.

A6. **Event tokens.**
    - Each tagged prompt is compared with its untagged twin.
    - CER on the words: Δ ≤ 0.05.
    - Runaways: no increase.
    - Detection rate of the token's AudioSet class (CED tagger: Breathing/Sigh for
      `[sigh]`, any of Breathing/Gasp/Laughter/Grunt for `[nv]`), at a threshold set to
      a 5% false-positive rate on the same checkpoint's untagged confirm outputs.
    - Reported, and documented as experimental whatever the rate.

A7. **Background SFX.**
    - Same detector rule (CED, the class's AudioSet label), ≥ 30 prompts per class.
    - A class ships only if detection ≥ 50% at 5% FPR, and A2 and A3 still hold.

A8. **Combination (replaces V8 where they differ).**
    - Final = adopted changes, 2 training seeds, checkpoints at 0.75 / 1.0 / 1.25 epochs.
    - Candidates must pass A2–A5 at S = 4.
    - Among passing checkpoints, rule steps 1–3 apply to the paired-bootstrap means.
    - Publish only if A2 and A3 pass.

A9. **Adoption rules.** These decide which ablations enter `final`. They were fixed
    before any ablation result. "Screen" means the A2 S=2 paired screen against v2 at
    matched seeds; "E1 spread" means the E1a/E1b difference on the same metric.
    - **E2 / E3** (language ids ± replay) adopt if all of these hold:
      - the A2 screen passes on the shipped mode, with mean Auto ΔCER ≤ +0.010;
      - zh M1 (language-ID log-ratio, valid per A4) improves on E1 by more than the E1
        spread;
      - the zh shared-character SER is not worse.
      - If both pass, E3 is preferred.
    - **E3f** (frozen text path) adopts over E3 if its zh M1 improves on E3 by more than
      the E1 spread and the A2 screen still passes.
    - **E4** (pseudo-Mandarin) runs only if the E0 gate below is met. It adopts if zh M1
      improves on its parent by more than the E1 spread, zh SIM drops ≤ 0.03, and the A2
      screen passes.
    - **E5** (device rows) adopts if:
      - A3 passes;
      - A5 ≥ 0.5 for `amadeus` and `amadeus_anime`;
      - each row's JA CER ≤ 0.07.
      - Phone's A5 is in-sample and only reported. If it fails, the phone row still ships
        as experimental, documented.
    - **E6** (event tokens) adopts if A6 ΔCER ≤ 0.05, runaways do not increase, and the
      A2 screen passes. Tags ship as experimental regardless.
    - **E7** (SFX) adopts per class if A7 ≥ 50% at 5% FPR, and A2/A3 hold for the run.
    - **No adopted change** means `final` is E1's recipe. In that case v3 is not
      published, because there is nothing to gain over v2.
A10. **E0 gate for the pseudo-Mandarin route.**
    - `pseudo`/E4 runs only if condition B's seed-averaged zh M1 lies at least halfway
      from F (v2, `language="Chinese"`) to D (native anchor): M1(B) ≤ (M1(F) + M1(D)) / 2.
    - M1 must be valid per A4.
    - Otherwise the route is skipped and the reason reported. The voice-conversion route
      is left for a later round.

A11. **Readings registered after the statistics review** (before any v3 result).
    - **A2 bootstrap:** two-stage. Resample texts, then seeds within each resampled
      text. On v2-era data this is conservative: an equal model passes the screen 83%
      of the time at S=2 and 93% at S=4. A true ΔCER of +0.010 passes 5–7%.
    - **A3 null:** the v2-vs-v2 null is scaled by 1/√S, because it compares single
      seeds while candidates are means over S. This is stricter than the literal
      wording, and matches the measured no-difference 95th percentiles.
    - **A4 fail rule:** a checkpoint fails A4 only if its zh M1 is worse than v2 by
      more than the E1 spread (paired mean over common texts and seeds), and only when
      M1 is valid. Without E1, A4 cannot fail. Other accent metrics are report-only.
      A4's monotonicity runs in a fixed direction: M1 and shared-character SER must
      increase G25 < G50 < G75 < F.
    - **A1:** v2 regenerated at seed base 6000 must match the archived v2 confirm
      outputs on per-clip duration (within 0.02 s) for at least 90% of clips.
      Otherwise the harness is investigated before any ranking.
    - **A5:** phone is report-only, as A9 already says. A phone failure never removes
      a checkpoint.
    - **A7:** "≥ 30 prompts" counts generations: 15 texts × S seeds.
    - **"E1"** in A9 is the per-text mean of E1a and E1b. Its spread is |E1a − E1b| on
      the same metric.

## Amendment B (2026-10-29, after the E1–E7 ablations, before E4, E5b, E6b or final)

**Disclosure.** These follow-ups were designed after seeing E5 and E6 results. Their
gates are fixed here, before they run.

- **A9 results so far:**
  - E3: ADOPT.
  - E3f, E5, E6, E7: REJECT.
  - E5 failed only on the `amadeus_anime` row's A5 (projection −0.23; `amadeus` 1.03,
    phone 1.15 in-sample).
  - E6 failed the A2 similarity screen (lower bound −0.035). Tag detection at 5% FPR
    was 22% for `[nv]` vs 7% for untagged twins, and 10% vs 3% for `[sigh]`.
  - E7: SFX detection 0–7%, and A3 failed. Background SFX are dropped from v3.

B1. **E5b.** The E1 data plus the `amadeus` (game) and `phone` rows with phone
    augmentation. The `amadeus_anime` row and its 251 clips are dropped.
    - Rationale, available before E5 ran: E0's x-vector probe showed the
      `amadeus_anime` vector cannot carry that sound (projection −0.43, against +1.04
      for `amadeus`).
    - Merging those clips into `amadeus` would label audio that does not sound like
      game Amadeus. Routing them to `kurisu` would leak a partial EQ.
    - **Adopts if:** A3 passes; A5(amadeus) ≥ 0.5; each row's JA CER ≤ 0.07; phone is
      report-only.
B2. **E6b.** E6 with one change: `--event_repeat 1` (no upsampling of event rows; ×3
    had added about 400 event-heavy rows, about 5% of training). Same gate as E6.
B3. **E4.** Parent E3. Gate as in A9. The actual pseudo share is reported if fewer rows
    than 6% pass p11's filters.
B4. **Final, and fallback.**
    - Final = the E1 recipe + every adopted change, at S=4, with gates A2–A5.
    - If that combined final fails A2 or A3, a reduced final with E3 (+ E4 if adopted)
      alone runs under the same gates.
    - If that also fails, v2 stays published.

## Amendment C (2026-10-29, after seeing ablation A3 values, before any A3′ value is computed)

**Disclosure.** Written after seeing the A3 statistics of E1a–E7 and E5b. The original A3
verdicts stay in the report next to the amended ones.

**Diagnosis.** A3's null was built from v2-vs-v2 *sampling-seed* differences and never
saw *training-seed* variation. On the pure negative controls (no device rows, no SFX):

| Run | rms / max dB | A3 |
|---|---|---|
| E1a | 0.87 / 1.95 | pass |
| E1b | 0.71 / 1.61 | pass |
| E3 | 1.07 / 2.21 | **fails** |
| E3f | 0.88 / 1.91 | pass |
| E6 | 1.15 / 2.53 | **fails** |

Two retrains of the v2 recipe already sit within 0.15 dB of the line. So A3 measures
generic retraining drift, not device-EQ leakage. As written, even the E3-only fallback
could not publish.

**A3′ replaces A3 in A8 and B4. A3 is kept as a reported statistic, not a gate.**
- **Default-voice shift:** `kurisu`-row LTAS minus v2 LTAS, on the same confirm texts
  and seeds, shipped mode.
- **Projection:** that shift is projected onto each real device direction, with the same
  normalisation as A5 (1.0 = the full device shift):
  - real `amadeus` takes − real `kurisu` takes;
  - real `phone` takes − real `kurisu` takes.
- **Threshold, fixed a priori:** ≤ 0.10 in both directions. That is, at most 10% of an
  Amadeus or phone EQ may leak into the default voice.
- This is the question A3 was meant to answer. Generic drift in similarity is already
  guarded by A2's SIM bound.

**Validity check before use.**
- A3′ is computed first on the negative controls E1a, E1b, E3 and E3f, and the values
  are reported.
- All four must be ≤ 0.10. If not, A3′ is unusable too, and that is reported.
- Only after that is A3′ computed for E5, E5b, E6b, E4 or final.
- The original A3 thresholds are NOT widened.

**B1 is applied as registered.** E5b has no `amadeus_anime` row, so only `amadeus`
needs A5 ≥ 0.5. p12's E5 rule, which also demands `amadeus_anime`, does not overrule B1.

## Amendment D (2026-10-29, after seeing E4's result, before E3b/E4b run)

**Disclosure.** Designed after seeing E4. Under A9 as written, E4 stands REJECTED: zh M1
improved on E3 by 4.07, against an E1 spread of 4.19.

**Diagnosis.** A9's noise floor (the E1a/E1b spread) came from non-language-trained runs
evaluated in Auto mode. In that mode v2 alone swings from M1 −6.6 to −13.0 just by
changing the language flag. The E3-vs-E4 comparison is in Chinese mode. Nobody knows the
matched-regime spread, because only one training seed of each exists.

**Replication.**
- **Runs:** E3b and E4b, at training seed 20261028, with the same data and flags as E3
  and E4. S = 2.
- **Matched spread:** max(|E3a − E3b|, |E4a − E4b|) on mean zh M1 (Chinese mode, paired
  texts and seeds).
- **E4 adopts only if all of these hold:**
  - the mean M1 gain of (E4a, E4b) over (E3a, E3b) exceeds the matched spread;
  - the shared-character SER is not worse (the guard, as in E3's rule; M1 stays the
    primary);
  - the zh SIM drop is ≤ 0.03;
  - the A2 screen and A3′ pass on both E4 seeds.
- **Pre-commitment:** if the gain does not exceed the matched spread, E4 is out of v3 and
  is reported as a v3.1 candidate. There will be no further amendment on this question.

V8. **Combination.**
    - Adopted changes are combined and trained with 2 seeds.
    - Checkpoint choice within the final run uses steps 1–4 above on `confirm`,
      restricted to checkpoints passing V2–V5.
    - Publish only if V2 and V3 pass. Otherwise v2 stays published and the result is
      reported.
