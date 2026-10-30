# Combined Makise Kurisu voice dataset — build

> **In this repository:** `pipeline/` here is `combined/pipeline/` of the original workspace. The upstream sources it reads (the game voice set, the six-language anime build and the songs) are not included. Set `KURISU_GAME_WORKSPACE` to the folder holding `voice_dataset_game`. The published result is [starrydark/Makise_Kurisu_Voice](https://huggingface.co/datasets/starrydark/Makise_Kurisu_Voice), formerly `starrydark/Kurisu_Voice`.

Builds one multilingual, expressively tagged Makise Kurisu / Amadeus Kurisu
corpus out of three upstream datasets, verifies it, and stages it for
`huggingface.co/datasets/starrydark/Kurisu_Voice`.

```
combined/
├── pipeline/          the build, one file per stage
├── manifests/         per-stage output, jsonl, one row per clip
├── reports/           every measurement the build made, json
├── audio/wavs/        the standardised 24 kHz mono WAVs
├── TAGGING.md         generated: what can be labelled without a human
└── README.md          this file
```

## What goes in

| Source | Language | What it is |
|---|---|---|
| `../../STEINS;GATEVoice/voice_dataset_game` | ja | Four MAGES./M2 games. Clip and transcript paired by the games' own scenario bytecode, so there is no alignment step and no ambiguity about who spoke. Tier A/B is speech; tier C is the quarantine pool — takes whose script line is only `……`. |
| `../datasets/<language>` | ja en zh de fr es | The six-language anime build: TV, movie, OVA, specials. Dual-ASR transcripts, ECAPA speaker verification, and a completed manual review whose bracket annotations become this build's gold set. |
| `../Cleaned_Songs/dataset` | ja | Three character songs, lyric-sheet transcripts. |

`voice_dataset_clean` — the older Japanese-only anime cut — is deliberately **not**
read. It covers the same episodes as `../datasets/japanese` with the
pre-correction timing, including the episode 10/13 label files later found to
contain episode 1 dialogue. Taking both would put two cuts of the same
performance in the corpus and leak them across the split boundary.

## Stages

| | | GPU | What it does |
|---|---|---|---|
| `k01` | `k01_inventory.py` | | Collect every Kurisu/Amadeus clip; normalise the text, then pull the reviewers' bracket annotations out of it |
| `k02` | `k02_audio.py` | | 24 kHz mono WAV; **undo** the upstream per-clip loudness normalisation; one BS.1770-4 gain per source group |
| `k03` | `k03_features.py` | | 16 deterministic acoustic features per clip |
| `k04` | `k04_speaker.py` | ✓ | ECAPA identity check against other STEINS;GATE characters, on the published audio |
| `k05` | `k05_locate.py` | ✓ | Where a non-verbal burst plausibly is |
| `k06` | `k06_align.py` | ✓ | Forced alignment in six languages: token spans and lexically unexplained gaps |
| `k07` | `k07_nvv.py` | ✓ | Name the sounds: vocal-burst classifier + CED, on whole clips, in-speech crops, gold clips and a negative control |
| `k08` | `k08_calibrate.py`, `k08_nvv.py` | | Measure every detector against the gold set; decide where the thresholds go |
| `k09` | `k09_decide.py` | | One decision per clip; the master manifest |
| `k10` | `k10_split.py` | | Cross-lingual line pairing; leakage-safe splits |
| `k11` | `k11_export.py` | | Write the Hugging Face repository and pick one reference clip per voice |
| `k12` | `k12_verify.py` | | Verify the files that will actually be pushed. Gates the upload |
| `k13` | `k13_readme.py` | | Generate the dataset card from `reports/` |
| `k15` | `k15_tagging_doc.py` | | Generate `TAGGING.md` from `reports/` |
| `k14` | `k14_push.py` | | Push. Refuses to run unless `k12` passed |

## Running it

```powershell
Set-Location '<workspace>\combined\pipeline'   # this repo's dataset\pipeline
$py = '..\..\.venv-curation\Scripts\python.exe'

# everything that has no output yet
& $py run_all.py

# only if something else is using the GPU: make the model stages wait for room
& $py run_all.py --need-vram 4.0

& $py run_all.py --from k09      # re-decide without re-running the models
& $py run_all.py --only k11      # rebuild the export
& $py k14_push.py --dry-run      # show what would change on the Hub
& $py k14_push.py                # push
```

Stages are skipped when their manifest already exists; `--force` ignores that.
`k04` falls back to the CPU when the card is busy and picks CUDA otherwise
(`KURISU_DEVICE=cuda|cpu` forces the choice). It batches by padded seconds of
audio rather than by clip count, which matters on any machine: ECAPA pads every
item in a batch to the longest one in it, so mixing a 24 s monologue with
half-second reactions multiplies the work by sixty.

## The three decisions worth knowing about

**Six languages means six performers.** `speaker_id` is `kurisu` on every row
because the character is the same; the voice is not. Every row carries
`voice_id` and `voice_actor`, and single-speaker training must filter on it.

**Dynamics were restored, not preserved.** The anime and song builds normalised
each clip independently to a fixed active-speech RMS, which flattens the
difference between a whisper and a shout. Each clip recorded the gain it was
given, so `k02` inverts it exactly and re-aligns with one gain per source group.
Active-level spread inside the anime portion goes from about 0 dB to 8 dB.

**Tags are earned, not asserted.** A tag reaches the transcript only if a human
wrote it, or a detector whose precision (or false-positive rate) was measured on
the human gold set says so. Where a language's fit cannot clear the floor, that
language keeps its human labels and receives no automatic ones — see `TAGGING.md`
for what that cost and what it bought.
