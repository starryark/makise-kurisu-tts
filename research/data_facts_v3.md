# Data facts for v3 (measured 2026-10-28 from `hf_dataset_upload/data/*/metadata.jsonl`)

## Japanese voice (Asami Imai): 10,039 clips

| Group | Clips |
|---|---|
| game-kurisu | 7,707 |
| anime-kurisu | 1,305 |
| game-amadeus | 652 |
| anime-amadeus | 294 |
| song-kurisu | 52 |
| anime-kurisu-[digital] (phone/device) | 28 |
| anime-amadeus-[digital] | 1 |

## Tag provenance, Japanese only

`human` means `human_tags` and `auto` means `auto_tags`; the two never overlap.

| Tag | Any | Human | Auto | By variant | By medium | Note |
|---|---|---|---|---|---|---|
| `[sigh]` | 148 | 5 | 143 | kurisu 125 / amadeus 23 | game 136 / anime 12 | almost entirely auto-labelled, unverified |
| `[gasp]` | 58 | 27 | 31 | kurisu 57 / amadeus 1 | game 29 / anime 29 | |
| `[breath]` | 16 | 16 | 0 | | all anime | |
| `[pause]` | 558 clips | 0 | 558 | kurisu 534 / amadeus 24 | | from measured pauses |
| `[digital]` | 29 | 29 | 0 | | all anime | phone calls in the anime |
| `[singing]` | 53 | 53 | — | | | |

- `emotion_notes` is nearly empty: 36 notes in total. `human_reviewed` covers 479 Japanese clips, all anime or song.
- **There are no trusted counts at the level a dedicated tag needs.** The user's own guide (§12) puts that at 75–200+ verified examples.
  - Only `[gasp]` (27 human labels) clears even the guide's 25-example "experimental" bar.
  - `[sigh]` would need verifying: 143 of its 148 labels are auto.

## Acoustic profile per group (p10 / p50 / p90)

| Group | spectral_tilt_db | HNR dB | rolloff99.5 Hz |
|---|---|---|---|
| game-kurisu | -20.8 / **-5.0** / 14.1 | 6.2 / **11.8** / 15.3 | 3.2k / 5.5k / 11.0k |
| game-amadeus | 0.6 / **+11.2** / 25.7 | 2.3 / **6.8** / 11.7 | 5.4k / 8.8k / 11.0k |
| anime-kurisu | -17.7 / **-6.6** / 12.3 | 5.0 / **10.4** / 13.9 | |
| anime-amadeus | -18.0 / **-2.7** / 18.3 | 5.0 / **9.1** / 12.4 | |
| anime-kurisu-digital (n=28) | -1.5 / **+17.3** / 33.2 | 0.3 / **3.3** / 7.1 | 4.6k / 8.2k / 10.9k |

### What the profiles show

- **Game Amadeus is a coherent, learnable acoustic condition.** Its median tilt is about 16 dB above game Kurisu, it is noisier (lower HNR) and brighter.
- **Anime Amadeus mostly sounds clean.** Its median is only about 4 dB above anime Kurisu; only its top tail is filtered. That fits the anime staging, where Amadeus is often shown face-to-face on a screen.
  - **So the narrative label `character_variant=amadeus` is not an acoustic label.** A condition tag derived from the variant alone would be wrong for many anime clips.
- **The `[digital]` phone clips are strongly processed but not narrowband.** Their rolloff is about 8 kHz, so this is not a classic 300–3400 Hz telephone band.
  - With n=28, the clips alone are too few to learn a phone condition. Augmenting clean clips with a matching device filter is the realistic route.

## Accent evidence already on disk (v2 confirm eval, Qwen3-ASR hypotheses)

| Prompt (ZH) | v2 ASR | Likely cause |
|---|---|---|
| 时间对于… | 只见对于… | 時間 read as Japanese *jikan*, heard as zhǐjiàn |
| 先提出假设 | 三D提出假设 | 先 read as Japanese *sen* |
| 记忆是以电信号… | 是一个是以电经过… | |

- **The two highlighted rows are direct traces of Japanese on-yomi leaking into Mandarin.** Japanese-shared characters get Japanese readings.
- **Base-model voice cloning (ICL) with a Japanese reference and `language=Auto` produced Japanese for 8/8 Chinese prompts** (CER 0.77–1.0). In Auto mode the language is inferred from the speaker prompt.
  - This is the extreme case of the same interference, and the strongest argument for explicit language-id conditioning in training and at inference.
