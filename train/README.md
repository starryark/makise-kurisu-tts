# Training code for Kurisu Qwen3-TTS v3

This is the exact code that built, trained, evaluated and selected this checkpoint. It
was run on one H100 80GB pod with `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`;
`bootstrap_v3.sh` installs the environment, and the work directory is `/workspace/kurisu`.

**Paths and tokens.**
- Scripts read paths from `ft_config.py` and their own argparse defaults.
- `label_v3.py` ran on a local machine; set `ROOT` at its top.
- The Hub token is read from `/workspace/hf/token` and is never printed.

**Repository names.** On 2026-10-29 the repos were renamed:
- `starrydark/Kurisu_Qwen3_TTS` became `starrydark/Makise_Kurisu_Qwen3_TTS`;
- `starrydark/Kurisu_Voice` became `starrydark/Makise_Kurisu_Voice`.

The old ids still redirect. The scripts keep the ids they ran with, which are also keys in the published evaluation tables.

## Run order

| Step | Script | What it does |
|---|---|---|
| 0 | `bootstrap_v3.sh` | Install pinned packages, prefetch models at pinned revisions, then run steps 1 and 4–7 |
| 1 | `p01_fetch.py` | Download `starrydark/Kurisu_Voice` |
| 2 | `p02_asr.py`, `p02b_utmos.py` | Qwen3-ASR transcripts and quality tiers. Run once for v2, reused by v3 (`asr.jsonl`) |
| 3 | `label_v3.py` | v3 labels: text normalisation, tags, speaker routing. Uses the v2 rules in `p03_build.py` |
| 4 | `p03_build_v3.py` | Apply the filters and audio policy, write `decisions.jsonl` and the speaker reference lists |
| 5 | `p03c_augment.py` | Phone-EQ augmentation (and the background-SFX experiment, not shipped) |
| 6 | `p05_replay.py` | AISHELL-3 and LibriTTS-R replay rows, one speaker vector each |
| 7 | `p04_codes_v3.py` | Encode all audio to 12 Hz codec tokens |
| 8 | `p10_e0.py` | E0 accent diagnosis (conditions A–J), run before any training |
| 9 | `p11_pseudo_zh.py` | Kurisu-voiced pseudo-Mandarin from the Base model, filtered |
| 10 | `mix_v3.py` | Build one experiment's train and validation mix |
| 11 | `train_sft.py` | Full fine-tune. The docstring lists the four upstream-trainer fixes |
| 12 | `p06_eval_v3.py` (+ `metrics.py`, `accent_metrics.py`, `ced_tagger.py`) | Generate and score the Japanese suite and the Mandarin/English probes, tags and voices |
| 13 | `p07_select_v3.py`, `p12_adopt.py`, `p13_a3prime.py` | Pre-registered gates, adoption verdicts, and the A3′ device-leak projection |

`queue_v3.sh` runs steps 8–13 as named stages: `e0`, `baselines`, `pseudo`, the E1–E7 ablations, `final` and `retest`.

The shipped model is the `final` stage's training seed 20261028, step 343. It was
trained with:

```
mix_v3.py --v2_ids v2_kept_ids.txt --language japanese --replay --replay_share 0.15 \
  --pseudo_share 0.06 --speakers kurisu,amadeus,phone --phone_aug --phone_repeat 4 \
  --text v3 --tag_only --event_repeat 1

train_sft.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --mode full --full_dtype fp32 \
  --optimizer adamw --lr 5e-6 --weight_decay 0.01 --epochs 2 --batch_size 16 \
  --effective_batch 32 --seed 20261028 --speakers kurisu=...,amadeus=...,phone=... \
  --aux_speakers replay/aux_speakers.json --p_auto_lang 0.25 \
  --new_tokens new_tokens_v3.json --stop_after 1.25 --save_at 0.75,1.0
```

## Checks

- **`test_equivalence.py`** builds a checkpoint with the trainer's own `save_checkpoint`. It then compares the trainer's input embeddings with what `generate_custom_voice` feeds the model, for every voice, language layout and tag case.
- **The selection rule and every amendment** are in `../eval/SELECTION_RULE.md`, and each decision is in `../eval/decisions.txt`.

## Data files here

- `v2_kept_ids.txt`: the v2 training rows that v3's default voice keeps.
- `suite_confirm.jsonl`: the Japanese confirmation suite.
- `probes_zh_v3.jsonl` and `probes_en_v3.jsonl`: the Mandarin and English probes.
- `e0_refs_src.json`: the E0 reference clips.
- `new_tokens_v3.json`: the tag tokens and their initialisation words.
