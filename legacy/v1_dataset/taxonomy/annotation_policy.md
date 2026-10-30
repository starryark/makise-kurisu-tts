# Annotation policy — Makise Kurisu / Amadeus Kurisu, Qwen3-TTS SFT

Written against `qwen3_tts_dataset_preparation_humanlike_emotional.md` (the guide) and
`qwen3_tts_local_nvv_labeling_pipeline_supplement.md` (the supplement). Where this
dataset departs from either, the reason is given.

---

## 1. What one speaker means here

`kurisu` and `amadeus_kurisu` are **one training speaker**. Amadeus is the same
performer playing the digitised copy of the character; the guide's single-speaker rule
(§33.1) is about one *voice*, and this is one voice. The distinction is kept in
`variant` on every master-manifest row so it can be sliced later, but it never reaches
the Qwen transcript and never changes `ref_audio`.

## 2. Source selection

| source | included | why |
|---|---|---|
| `voice_dataset_game/wavs/` | all | isolated studio voice track, tier A+B |
| `voice_dataset_game/wavs_quarantine/` | **all** | quarantined for being short and non-lexical (`……`), not for being dirty — these are the expressive takes the supplement §2 says to keep |
| `voice_dataset_clean/wavs/` | all | tier A+B anime dialogue |
| `voice_dataset_clean/wavs_quarantine/` | none | excluded by instruction |

Two clips were dropped as byte-identical duplicates of another clip (the scenario
bytecode references one voice file from two script points). They are listed in
`manifests/01_dropped_duplicates.jsonl`.

## 3. Transcript policy

The transcript that ships with each clip is **the lexical ground truth** and is never
replaced by ASR (supplement §10). It is normalised, not rewritten:

| rule | effect | clips |
|---|---|---|
| strip `{\anN}` | ASS subtitle positioning markup from the anime subs | 42 |
| strip leading `（紅莉栖）` | subtitle speaker label, not spoken | 4 |
| `BASE(かな)` → `かな` | ruby where the base is latin/digits: `ＳＥＲＮ(セルン)` → `セルン`, `ロト６(シックス)` → `ロトシックス`. The actor says the reading. `（仮）`/`（自称）` are kanji and **are** spoken, so they are left alone | 10 |
| full-width alnum → half-width | unifies `ＩＢＮ５１００` and `ＩＢＮ5100` into one string | 1073 |
| `─ — – − ` → `―` | one dash character for trailing/interrupted speech | 234 |
| `⁻²⁴` → `の-24乗` | matches how the parallel RE:BOOT take writes the same line | 1 |
| `β`→`ベータ`, `α`→`アルファ` | spoken that way in-universe (`β世界線` = ベータせかいせん) | 20 |

Punctuation is preserved exactly — `……`, `！`, `？`, `―`. It carries pause and prosody
and is **never** converted into a vocalisation tag (supplement §3).

Not normalised, flagged as `symbol_risk` for review instead: `∞ ℃ ″ ＃ ＊ ＝ ＠`. Guessing
their reading would be inventing transcript.

## 4. Tag inventory

See `family_to_qwen.yaml`. The Qwen-facing inventory is deliberately small and the
**shipped** subset is smaller still — a family only ships if enough verified examples
exist (guide §12). Families the classifier recognises but that are never exposed in v1:
`tsk`, `lipsmack`, `swallow`, `whistle`, `growl`, `mumble`, `other`.

Formatting is fixed: lowercase, square brackets, one spelling, e.g. `[sigh]`. Never
`[SIGH]`, `(sigh)`, `[sighing]`.

## 5. How a tag is earned

Nothing becomes a tag from text alone, and nothing becomes a tag from one model.

**Isolated clips** (transcript is purely non-lexical, e.g. `……`):

1. `laion/vocalburst-classifier-single` over the whole clip;
2. probabilities aggregated *by family* before ranking — no single fine class
   ("Contented Sigh") ever decides anything (supplement §7);
3. `mispeech/ced-base` votes independently on the same audio;
4. the family must clear the threshold **calibrated for that family** (stage 05) and
   CED must place it in its top 2.

**Inside spoken lines**, a burst additionally has to sit where the transcript does not
reach — `laion/vocalburst-locator` v2 and `Qwen3-ForcedAligner-0.6B` must agree that
there is an energetic, lexically unexplained region there. On this corpus the locator
alone fires mostly on short Japanese lexical interjections (え, うん, あっ) that are
already written in the transcript; tagging those would duplicate the text and teach the
model to bolt a sound onto a word it is already speaking.

**Written onomatopoeia** (`ふふ`, `はぁ……`, `きゃっ`) is **kept as text**, not converted.
Two reasons: the audio models agreed with the written prior on only 11 of 23 such
clips, so conversion would be guessing; and Japanese genuinely lexicalises these forms,
which is the case the guide (§4.2) says to leave as words. The audio evidence is
recorded in the master manifest either way.

## 6. Confidence and review states

`auto_accept` · `uncertain` · `reject`. Nothing is silently deleted (supplement §26).

* `uncertain` clips stay in `manifests/09_manual_review.jsonl` and
  `qwen/held_back_uncertain.jsonl`. They are ready to fold in after listening.
* `reject` is only for clips with no usable transcript at all.

Families that are always reviewed regardless of score: `moan`, `groan`, `grunt`, `hum`,
`mumble`, `growl` (supplement §16 — these are the distinctions annotators cannot make
reliably, and a wrong control token is worse than a missing one).

## 7. What has NOT been done

**No human has listened to any clip in this build.** The thresholds are calibrated
against a measured proxy — 400 ordinary-speech clips as negatives — which separates
"burst" from "ordinary speech" but cannot verify that a sigh is a sigh rather than a
gasp. Before training, work `reports/05_listening_queue.csv` top-down; it is ordered so
the riskiest decisions come first. The guide (§44) asks for exactly this on `[sigh]`,
`[laugh]`, `[moan]`, `[groan]`, `[giggle]`.

## 8. Audio

24 kHz mono PCM-16. Loudness aligned **per source group**, one gain per group, never
per clip — the four games and the anime differed by up to 11 dB, but the loudness
relationships *inside* each source are untouched (guide §9.5). No denoising, no gating,
no trimming: the sources are already isolated studio voice tracks with nothing to
remove, and the guide (§9.3/§9.4) is explicit that those steps destroy sigh tails and
soft breaths.
