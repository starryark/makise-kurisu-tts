# Blind Mandarin accent rating: results (2026-10-29)

**Setup.**
- One rater (the repository owner), blind.
- 16 sentences × 5 systems = 80 clips, shuffled per sentence under letters A–E with hashed file names.
- Scale 1–5: 5 = native (母语水平), 1 = barely Mandarin.
- All 80 clips were rated; the store held one rater and no unknown clips.
- Data: `kurisu_v2/v3/rating_results_joined.json`. Script: `kurisu_v2/v3/rating_analysis.py`. The answer key `rating_key.json` stays local.

**Clip provenance.** Every system used its first sampling seed; none was cherry-picked.
- D, E, F and H: the E0 seed-1 clips.
- V3: the pick (`final_s20261028/checkpoint-step-343`), `language="Chinese"`, seed base 6000. This was the only mode at that seed.

## Per system

95% bootstrap CI over the 16 sentences (20,000 resamples).

| System | Mean | 95% CI | Rating counts (1/2/3/4/5) |
|---|---:|---|---|
| D: Base model cloning a native Mandarin reference voice (top anchor; synthetic, not a human recording) | 5.00 | [5.00, 5.00] | 0/0/0/0/16 |
| **V3**, language=Chinese | **5.00** | [5.00, 5.00] | 0/0/0/0/16 |
| F: v2, language=Chinese | 3.94 | [3.38, 4.50] | 0/3/2/4/7 |
| E: v2, Auto (as the old Space served it) | 3.56 | [2.88, 4.25] | 1/4/3/1/7 |
| H: Japanese-read floor anchor | 1.06 | [1.00, 1.19] | 15/1/0/0/0 |

## Paired comparisons

Per sentence; p-values from an exact sign test and a sign-flip permutation test.

| Comparison | Mean diff | 95% CI | Better / tie / worse | p (sign / perm) |
|---|---:|---|---|---|
| V3 − v2 Chinese | +1.06 | [+0.56, +1.62] | 9 / 7 / 0 | 0.004 / 0.004 |
| V3 − v2 Auto | +1.44 | [+0.75, +2.12] | 9 / 7 / 0 | 0.004 / 0.004 |
| v2 Chinese − v2 Auto | +0.38 | [−0.12, +0.94] | 6 / 7 / 3 | 0.51 / 0.29 |
| native ref − V3 | 0.00 | — | 0 / 16 / 0 | — |

## By category (mean rating)

| Category | n | v2 Auto | v2 Chinese | V3 | native |
|---|---:|---:|---:|---:|---:|
| Shared characters | 7 | 2.86 | 3.14 | 5.00 | 5.00 |
| Numbers | 3 | 2.67 | 3.67 | 5.00 | 5.00 |
| Control, sandhi, long | 6 | 4.83 | 5.00 | 5.00 | 5.00 |

The accent the user described was in the sentences dense with shared characters and numbers. Those are exactly the categories where v2 lost points and v3 did not.

## Agreement with the automatic metrics

Per clip, Spearman correlation with the human rating:

| Metric | All 80 clips | E/F/V3 only (48 clips, anchors excluded) |
|---|---:|---:|
| Shared-character SER | −0.84 | **−0.73** (p = 3e-9) |
| SER | −0.80 | −0.64 |
| M1 (LID log-ratio) | −0.72 | −0.33 (p = 0.02) |
| TER | −0.55 | −0.38 |

- Shared-character SER is the best predictor of the listener's judgement.
- M1 separates the anchors well but ranks the middle systems more weakly.

## Caveats

- **One rater, 16 sentences.** This is not a panel MOS.
- **Ceiling.** V3 and the native anchor both got 5 on every sentence, so the scale cannot resolve whether any accent remains. The automatic metrics still put V3 slightly short of native on M1 (−18.5 vs −23.5 on these clips).
- **The native anchor is synthetic:** the Base model cloning a native speaker, not a human recording.
- **The anchors are not voice-matched.** D is a different speaker and H is Qwen's Japanese voice. Only E, F and V3 share Kurisu's timbre, so the test was fully blind only among those three. "Same as the anchor" means the same rating, not indistinguishable audio. The v2 → v3 result is the within-timbre comparison and is not affected.
- **The human rating does not clearly confirm that telling v2 the language helps** (+0.38, CI includes 0), although the automatic metrics showed a large gap (shared SER 0.285 → 0.207 on these clips).
- **Timing.** The session lasted 7.4 min; the total audio is 6.6 min.

## Conclusion

- On this test, v3's Mandarin was judged as native as the native-voice anchor on all 16 sentences.
- It beat v2 on 9 of 16 sentences and was never worse (p = 0.004).
- The gain is concentrated on shared-character and number sentences, where the accent was.
