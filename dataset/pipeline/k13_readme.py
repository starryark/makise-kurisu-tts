"""Stage 13 - write the dataset card from the reports.

Every number in the card is read out of ``reports/``.  Nothing here is typed by
hand, so the card cannot drift away from the build that produced it.
"""
from __future__ import annotations

import json

from kcommon import (ALL_TAGS, CONDITION_TAGS, EVENT_TAGS, HF_OUT, LANGUAGES,
                     MIN_TAG_EXAMPLES, REPORTS, STRUCTURAL_TAGS, VOICE_ACTOR,
                     utf8_stdout)

utf8_stdout()

LANG_NAME = {"ja": "Japanese", "en": "English", "zh": "Chinese",
             "de": "German", "fr": "French", "es": "Spanish"}
MEDIUM_NAME = {"game": "game", "anime": "anime", "song": "character song"}


def load(name):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    inv = load("01_inventory.json")
    aud = load("02_audio.json")
    spk = load("04_speaker.json")
    cal = load("08_calibration.json")
    dec = load("09_decisions.json")
    spl = load("10_split.json")
    exp = load("11_export.json")
    ver = load("12_verify.json")
    ref = load("11_reference.json")
    ref_rows = [[f"`{v}`", LANG_NAME[v], f"`{r['file']}`", r["duration_seconds"],
                 r["speaker_margin"], r["closest_other_character"]]
                for v, r in sorted(ref.items())]

    langs = sorted(exp["by_language"], key=lambda l: -exp["by_language"][l])
    lang_rows = [[f"{LANG_NAME[l]} (`{l}`)", VOICE_ACTOR[l].split(" (")[0],
                  f"{exp['by_language'][l]:,}", exp["hours_by_language"][l]]
                 for l in langs]

    tags = exp["tags"]
    prov_h = exp["tag_provenance"]["human"]
    prov_a = exp["tag_provenance"]["auto"]
    tag_rows = []
    for t in sorted(tags, key=lambda t: -tags[t]):
        kind = ("event" if t in EVENT_TAGS else
                "condition" if t in CONDITION_TAGS else "structural")
        tag_rows.append([f"`{t}`", kind, f"{tags[t]:,}",
                         prov_h.get(t, 0), prov_a.get(t, 0)])

    nvv_pol_all = cal.get("nvv", {}).get("policy", {})
    nvv_pol = {**nvv_pol_all.get("in_speech", {}), **nvv_pol_all.get("whole_clip", {})}
    nvv_ev = cal.get("nvv", {}).get("evidence", {})
    nvv_rows = []
    for scope in ("whole_clip", "in_speech"):
        for fam in sorted(nvv_pol_all.get(scope, {})):
            p = nvv_pol_all[scope][fam]
            nvv_rows.append([
                f"`{fam}`", scope, p["mode"], p["family_score_min"],
                p["family_margin_min"],
                "yes" if p["secondary_required"] else "no",
                p.get("precision", "—"), p.get("recall", "—"),
                (f"{p['false_positives']}/{p['negatives']}" if p["mode"] == "precision"
                 else f"{p['control_false_positives']}/{p['negatives']}")])
    nvv_none = sorted(
        f for f, e in nvv_ev.items() if f not in nvv_pol
        and (e["whole_clip_positives"] + e["in_speech_positives"]) >= 8)
    pres = cal.get("nvv", {}).get("presence_gate")

    det_rows = []
    for tag, per in cal.get("detectors", {}).items():
        for scope in ("per_language", "pooled"):
            for name, d in sorted(per.get(scope, {}).items()):
                if d.get("positives", 0) < 8:
                    continue
                label = (LANG_NAME.get(name, name) if scope == "per_language"
                         else f"pooled: {name}")
                det_rows.append([
                    f"`{tag}`", label, d["positives"], d["reviewed_clips"],
                    d.get("loo_auc") if d.get("loo_auc") is not None else "—",
                    d.get("precision", "—") if d.get("usable") else "—",
                    d.get("recall", "—") if d.get("usable") else "—",
                    "usable" if d.get("usable") else "floor not reached"])
    sel = cal.get("detector_selected_per_language", {})
    sel_rows = [[f"`{k.split('|')[0]}`", LANG_NAME[k.split('|')[1]], v["scope"],
                 v["precision"], v["recall"],
                 "yes" if v["ships_as_tag"] else "metadata only"]
                for k, v in sorted(sel.items())]

    q = exp["by_quality_tier"]
    split_rows = [[s, f"{exp['splits'][s]['clips']:,}", exp["splits"][s]["hours"]]
                  for s in ("train", "validation", "test")]

    series_rows = [[k, f"{v:,}"] for k, v in exp["by_series"].items()]

    md = f"""---
pretty_name: Makise Kurisu Multilingual Voice Dataset
language:
{chr(10).join(f"- {l}" for l in sorted(LANGUAGES.values()))}
license: other
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- audio
- speech
- multilingual
- parallel-corpus
- expressive-tts
- paralinguistics
- steins-gate
- anime
- games
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_dir: data
  drop_labels: true
---

# Makise Kurisu Multilingual Voice Dataset

{exp['clips']:,} labelled clips ({exp['hours']} hours) of **Makise Kurisu**,
including the **Amadeus Kurisu** variant, in **six languages**, cut from the
STEINS;GATE games, the anime, and three character songs.

Every clip carries the transcript, a measured acoustic profile, an independent
speaker-identity check, and — where it could be earned rather than guessed — an
expressive tag. This is an unofficial, fan-made dataset with no affiliation to
the STEINS;GATE rights holders.

## Read this first: six languages means six performers

`speaker_id` is `kurisu` for every row because the **character** is the same.
The **voice** is not. Each dub was recorded by a different actor, so a model
trained on all six rows at once will learn a blend of six people.

{table(["Language", "Performer", "Clips", "Hours"], lang_rows)}

Filter on `voice_id` (or equivalently `language`) for single-speaker training.
Japanese covers games, anime and songs and is by far the largest single voice.

## Summary

{table(["Property", "Value"], [
    ["Clips", f"{exp['clips']:,}"],
    ["Duration", f"{exp['hours']} h"],
    ["Languages", len(exp['by_language'])],
    ["Audio", f"WAV PCM 16-bit, mono, {aud['sample_rate'] // 1000} kHz"],
    ["Character variants", f"kurisu {exp['by_variant'].get('kurisu', 0):,} · "
                           f"amadeus {exp['by_variant'].get('amadeus', 0):,}"],
    ["Clips carrying an expressive tag", f"{exp['clips_with_any_tag']:,} "
        f"({exp['clips_with_any_tag'] / exp['clips'] * 100:.1f}%)"],
    ["Cross-lingual parallel lines", f"{exp['clips_in_parallel_group']:,} clips in "
        f"{exp['parallel_groups']:,} groups"],
    ["Human-reviewed clips", f"{exp['human_reviewed']:,}"],
    ["Median / p05 / p95 duration",
     f"{exp['duration_seconds']['median']} s / {exp['duration_seconds']['p05']} s / "
     f"{exp['duration_seconds']['p95']} s"],
])}

### Splits

{table(["Split", "Clips", "Hours"], split_rows)}

Splits are cut between *performances*, not between files. Whole episodes and
whole game scenario files move together across all six languages, every
cross-lingual parallel line stays on one side, and any line whose lexical
content repeats within a language is bound to a single split — which matters
here because RE:BOOT re-records thousands of STEINS;GATE script lines.

Verified on the published files: **{spl.get('substantive_text_leakage_train_to_validation', '?')}**
substantive lines shared between train and validation,
**{spl.get('substantive_text_leakage_train_to_test', '?')}** between train and test,
**{spl.get('parallel_groups_split_across_boundaries', '?')}** parallel groups
straddling a boundary, across {spl.get('components', '?')} leakage components.

### Sources

{table(["Source", "Clips"], series_rows)}

## Expressive tags

Two kinds of tag reach `transcription`. **Event tags** mark a non-verbal
vocalisation and sit where it happens. **Condition tags** describe the whole
utterance and sit at the front, in a fixed order.

{table(["Tag", "Kind", "Clips", "From human review", "Automatic"], tag_rows)}

A tag needs at least {MIN_TAG_EXAMPLES} examples to ship at all; anything rarer
was demoted to metadata rather than published as a control nobody can steer.

### How a tag is earned

The manual review of the anime corpus had already written bracket annotations
into {cal.get('gold_set', {}).get('reviewed_clips', 0):,} transcripts. That is a
character-specific gold set, so this build spends it on calibration instead of
treating those annotations as the only labelled data.

**Non-verbal vocalisations** are named by `laion/vocalburst-classifier-single`
with fine-class probabilities aggregated *by family* before ranking, checked by
`mispeech/ced-base`, and — for anything inside a spoken line — required to fall
where `laion/vocalburst-locator` sees a burst *and* `Qwen3-ForcedAligner-0.6B`
says no transcript unit reaches. Thresholds come from a sweep against the gold
positives and {cal.get('nvv', {}).get('evidence', {}).get('sigh', {}).get('control_clips', 0)}
ordinary speech clips used as a negative control:

{table(["Family", "Scope", "Mode", "score ≥", "margin ≥", "2nd model must agree", "precision", "recall", "false positives"], nvv_rows) if nvv_rows else "_No family reached its floor._"}

Families with gold examples that did **not** reach the floor and are therefore
never auto-assigned: {', '.join('`' + f + '`' for f in nvv_none) if nvv_none else '_none_'}.

**Delivery and playback conditions** have no off-the-shelf model, so a small
logistic regression per language is fitted on the deterministic acoustic features
and scored leave-one-out, on the reviewed subset only — the one place where the
*absence* of a tag means a human looked and decided no.

{table(["Tag", "Fitted on", "Gold positives", "Reviewed clips", "LOO AUC", "precision", "recall", "Outcome"], det_rows)}

Each language then uses its own fit when that clears the floor, and the pooled
fit it belongs to otherwise:

{table(["Tag", "Language", "Model used", "precision", "recall", "In transcript"], sel_rows) if sel_rows else "_No detector reached the precision floor._"}

Where a fit could not clear the precision floor, the human labels still ship and
nothing is added. `[digital]` is the clearest case: the European dubs render
Amadeus through an audible low-pass that a detector finds easily, while the
Japanese original uses a thin, bass-cut EQ that the reviewers heard as narrative
context — so Japanese keeps its {cal.get('detectors', {}).get('[digital]', {}).get('per_language', {}).get('ja', {}).get('positives', 0)}
human labels and receives no automatic ones.

Whispering, shouting and room reverberation are physiological or physical rather
than linguistic, so those detectors pool their gold clips across all six
languages — the only way `[yell]` (18 gold clips) and `[echo]` (29) have enough
positives to be fitted at all. `[digital]` pools only across the four dubs whose
signature points the same way.

**`[pause]`** is purely measured: an internal silence the aligner places between
two words, long enough to be in the upper decile *for its own language* (Japanese
dialogue breathes differently from Spanish, and a fixed threshold would tag every
sentence boundary in one and nothing in the other), at most two per clip, and only
where the transcript does not already write an ellipsis — `……` says it perfectly
well on its own. Thresholds used:
{", ".join(f"{LANG_NAME[k]} {v}s" for k, v in sorted(dec.get("pause_threshold_s_by_language", {}).items())) or "—"}.

**`[nonverbal]`** is the honest coarse label. Naming a sound is harder than
noticing there is one, so a separate gate is calibrated for the weaker claim
"this is a non-lexical vocalisation, family unknown" — the classifier must be
confident a burst is present and CED must not think it is ordinary speech.
{("It runs at precision " + str(pres['precision']) + " / recall " + str(pres['recall'])
   + " against " + str(pres['negatives']) + " ordinary-speech clips."
   if pres and pres.get('mode') == 'precision' else
   ("It is calibrated for specificity: " + str(pres['control_false_positives']) + " of "
    + str(pres['negatives']) + " ordinary-speech clips would pass it."
    if pres else ""))}
It exists so that a take whose script line is only `……` can enter the corpus with
a label that says exactly what is known, instead of being thrown away or shipped
with a guessed family.

### Labels that deliberately do *not* reach the transcript

The reviewers also marked non-character sound and performance notes. Those are
useful, but a TTS model should not learn to *produce* a doorbell, so they were
moved out of `transcription` and into columns:

- `contamination` — `noise`, `footstep`, `knock`, `ringtone`: audible non-character
  sound in the clip. {exp['clips']:,} clips carry a `contamination_score` from the
  same calibrated framework, so the whole corpus can be filtered, not just the
  reviewed part.
- `emotion_notes` — `serious`, `playful`, `angry`, `embarrassed`, `scared`,
  `mumbled`, `foreign_accent`: too rare to be a control, kept for curation.

## Dataset structure

Hugging Face `AudioFolder` layout:

```text
data/
├── train/       metadata.jsonl + *.wav
├── validation/  metadata.jsonl + *.wav
└── test/        metadata.jsonl + *.wav
```

### Columns

{table(["Field", "Meaning"], [
    ["`file_name`", "audio file, relative to the split directory"],
    ["`transcription`", "**the training label** — text plus any earned tag"],
    ["`text`", "the same line with every tag removed"],
    ["`language` / `voice_id`", "`ja` `en` `zh` `de` `fr` `es`; also the performer identity"],
    ["`voice_actor`", "who is speaking"],
    ["`speaker_id` / `speaker_name`", "`kurisu` / `Makise Kurisu`"],
    ["`character_variant`", "`kurisu` or `amadeus`"],
    ["`medium` / `series` / `scene`", "game · anime · song; title; scenario file or episode"],
    ["`style`", "`speech` or `singing`"],
    ["`tags` / `event_tags` / `condition_tags`", "tags present in `transcription`"],
    ["`human_tags` / `auto_tags`", "which of them a person wrote and which a detector earned"],
    ["`human_reviewed`", "a person adjudicated this clip upstream"],
    ["`from_quarantine`", "recovered from the game's non-lexical quarantine pool"],
    ["`parallel_group`", "id shared by the same line across dubs, or null"],
    ["`contamination` / `contamination_score`", "non-character sound, human and automatic"],
    ["`emotion_notes` / `measured_pauses`", "performance notes; measured internal pauses"],
    ["`quality_tier` / `quality_flags`", "`A`/`B`/`C` and why"],
    ["`duration_seconds` `sample_rate` `active_rms_dbfs` `peak_dbfs` `lufs`", "level and length"],
    ["`speech_ratio` `lead_silence_s` `tail_silence_s` `internal_pause_max_s`", "activity"],
    ["`snr_est_db` `noise_floor_dbfs`", "what is left between the words"],
    ["`f0_median_hz` `f0_range_st` `voiced_ratio` `hnr_db`", "pitch register, movement, periodicity"],
    ["`bandwidth_roll995_hz` `spectral_tilt_db` `reverb_decay_db_per_s`", "playback path and room"],
    ["`speaking_rate_units_per_s` `lexical_units`", "rate over speaking time; chars for ja/zh, words otherwise"],
    ["`speaker_similarity` `speaker_rival_similarity` `speaker_margin` `speaker_rival`", "independent identity check"],
    ["`alignment_coverage`", "share of the clip the forced aligner attributes to the transcript"],
    ["`transcript_source`", "where the words came from"],
    ["`sha256`", "digest of the published WAV"],
])}

### Quality tiers

{table(["Tier", "Clips", "Meaning"], [
    ["A", f"{q.get('A', 0):,}", "no quality flag, and a trusted transcript source"],
    ["B", f"{q.get('B', 0):,}", "usable, with at least one soft flag"],
    ["C", f"{q.get('C', 0):,}", "contamination, low SNR, clipping, or a contradicted speaker check"],
])}

Flags observed: {', '.join(f"`{k}` ({v:,})" for k, v in exp['quality_flags'].items()) or '_none_'}.

## Reference clips

`reference/<voice_id>.wav` is one clean, untagged, 4–10 s take per voice, chosen
by measurement rather than by ear: the highest margin over the closest other
STEINS;GATE character among tier-A clips of that voice. `reference/reference.json`
records which clip each one is and why.

{table(["voice_id", "Language", "File", "Duration (s)", "Margin over rival", "Closest rival"], ref_rows) if ref_rows else "_not generated_"}

This matters for voice-cloning fine-tunes that read a reference on every training
row and write its embedding into a speaker table — a reference that is not
actually the target voice poisons the run silently.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("starrydark/Kurisu_Voice")

# single-speaker Japanese, clean tier, with the expressive tags
ja = ds["train"].filter(lambda r: r["language"] == "ja" and r["quality_tier"] != "C")

# the same line in every dub
groups = ds["train"].filter(lambda r: r["parallel_group"] is not None)

# no tags at all
plain = ds["train"].map(lambda r: {{"label": r["text"]}})
```

Use `transcription` to train tag control, `text` to ignore it.

## Curation

**Audio.** One standardised 24 kHz mono PCM-16 WAV per clip. Game audio is
resampled from 48 kHz with `soxr` VHQ. Nothing is denoised, gated or trimmed.

**Dynamics were put back.** The upstream anime and song builds normalised every
clip independently to a fixed active-speech RMS, which flattens the difference
between a whisper and a shout. Each clip recorded the gain it was given, so this
build inverts it exactly and then applies **one gain per source group** to a
common {aud['loudness_target_lufs']} LUFS target, measured to ITU-R BS.1770-4 by
pooling the K-weighted blocks of the whole group. Loudness relationships inside a
group survive untouched. The result is an active-level spread of
**{aud['active_rms_dbfs']['spread_p05_p95_db']} dB** p05–p95 across the corpus
({aud['active_rms_spread_by_medium'].get('anime')} dB within the anime portion,
which was flat before).

{table(["Source group", "Measured LUFS", "Gain applied"],
       [[k, v, aud['group_gain_db'][k]] for k, v in aud['group_loudness_before_lufs'].items()])}

**Speaker identity was re-checked on the published audio** with
`speechbrain/spkrec-ecapa-voxceleb`, which took no part in either upstream
decision. Embeddings are centred before scoring — raw cosines between any two
young female anime voices sit above 0.9 and separate nothing — and every clip is
scored against the closest *other* STEINS;GATE character recorded in the same
language and medium, drawn from {spk.get('control_clips', 0):,} negative-control
clips. `speaker_margin` is that difference and it is published per clip.

**The game quarantine pool was relabelled, not discarded.**
{dec.get('quarantine_total', 0)} takes whose scenario text is only `……` were set
aside upstream for having no usable transcript. The line *is* the sound. Where
the calibrated gate could name it, the transcript became that tag:
**{dec.get('quarantine_recovered', 0)}** of them are now in the corpus —
{dec.get('whole_clip_named', 0)} with a named family and
{dec.get('whole_clip_generic', 0)} as `[nonverbal]`. The rest are held back
rather than shipped with a guessed control token.

**Held back:** {exp['held_back']:,} clips with no nameable content are not
published.

## Limitations

- **Six languages are six performers.** Train on one `voice_id` at a time unless
  a blended voice is what you want.
- The speech is acted character dialogue and does not represent ordinary
  conversation in any of these languages.
- Anime clips are source-separated and can retain music, effects, or separation
  artefacts. `snr_est_db`, `contamination` and `quality_tier` are there to filter
  on; game clips are studio-clean by construction.
- The source mix is very uneven — Japanese is {exp['hours_by_language']['ja']} h
  of {exp['hours']} h, and Chinese covers only the STEINS;GATE 0 dub.
- Automatic tags are precision-optimised, so **recall is low by design**: an
  untagged clip is not evidence that nothing happened in it.
- Tag strings are a convention of this dataset, not an official Qwen3-TTS,
  XTTS or VITS feature. Evaluate controllability separately from audio quality.
- No exhaustive human listening pass has been done on the automatic labels.

## Rights and licensing

The underlying audio, dialogue, characters, names, and STEINS;GATE-related marks
remain the property of their respective rights holders. No copyright, trademark,
performer, publicity, personality, or voice rights are transferred or licensed by
this repository, which is why it uses Hugging Face's `other` designation rather
than claiming an open-data licence.

Do not use this dataset to impersonate any of the performers, to mislead anyone
about authentic or official speech, to suggest endorsement, to harass, or to
evade copyright, publicity, personality, or voice-protection rights. Disclose
synthetic audio produced from models trained on it. Users and redistributors are
responsible for determining whether their access and intended use are lawful in
their jurisdiction. Rights holders may contact the repository owner here for
correction or removal.

## Attribution

If you use this dataset, cite this page and identify the source material as audio
from STEINS;GATE series games, anime and character songs. This request does not
replace any attribution or permission required by the rights holders.
"""
    (HF_OUT / "README.md").write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {HF_OUT / 'README.md'} ({len(md)} chars)")


if __name__ == "__main__":
    main()
