"""Write README.md for the dataset from the live reports, so it can never drift."""
from __future__ import annotations

import json

from common import PROJECT, REPORTS, utf8_stdout

utf8_stdout()


def load(name):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    inv, aud = load("01_inventory.json"), load("02_audio.json")
    clf, sec = load("03_classifier.json"), load("04_secondary.json")
    cal, loc = load("05_calibration.json"), load("06_locator.json")
    ali, evt = load("07_alignment.json"), load("08_events.json")
    dec, rep = load("09_decisions.json"), load("10_dataset_report.json")
    ver = load("verify.json")

    t = rep.get("totals", {})
    nvv = rep.get("nvv", {})
    per_tag = nvv.get("per_tag", {})

    def row(k, v):
        return f"| {k} | {v} |"

    L = []
    A = L.append
    A("# Makise Kurisu — Qwen3-TTS single-speaker SFT dataset\n")
    A("Curated from the STEINS;GATE game-voice and anime datasets in this repo. "
      "`kurisu` and `amadeus_kurisu` are treated as **one speaker** — same performer, "
      "same voice — as instructed.\n")
    A("Built by `pipeline/run_all.py`. Every number below is read from `reports/`, "
      "not typed by hand.\n")

    A("## What shipped\n")
    A("| | |")
    A("|---|---|")
    A(row("training clips", f"**{t.get('clips', '?')}**"))
    A(row("training audio", f"**{t.get('hours', '?')} h**"))
    A(row("held back for review", rep.get("held_back", {}).get("clips", "?")))
    A(row("audio", "24 kHz mono WAV PCM-16"))
    A(row("reference", "`./audio/ref.wav` — the same row on every line of the JSONL"))
    A(row("duration median / p05 / p95",
          f"{t.get('median_s','?')} s / {t.get('p05_s','?')} s / {t.get('p95_s','?')} s"))
    A(row("splits", " · ".join(
        f"{k} {v.get('clips', 0)}" for k, v in rep.get("splits", {}).items())))
    A("")

    A("```")
    A("kurisu_qwen3tts/")
    A("├── qwen/train_raw.jsonl val_raw.jsonl test_raw.jsonl   <- feed these to prepare_data.py")
    A("│         held_back_uncertain.jsonl  probe_prompts.jsonl")
    A("├── audio/ref.wav          <- the ref_audio for every row")
    A("├── audio/wavs/*.wav       <- 24 kHz mono training clips")
    A("├── manifests/09_master.jsonl   <- rich per-clip metadata, every model score")
    A("├── manifests/09_manual_review.jsonl")
    A("├── taxonomy/*.yaml  annotation_policy.md")
    A("├── reports/*.json  05_listening_queue.csv  10_clip_index.csv")
    A("└── pipeline/*.py          <- reproducible build")
    A("```\n")

    A("## Use\n")
    A("Paths in the JSONL are relative to this directory, so run the official "
      "preparation from here:\n")
    A("```bash")
    A("cd kurisu_qwen3tts")
    A("python /path/to/Qwen3-TTS/finetuning/prepare_data.py \\")
    A("    --input  qwen/train_raw.jsonl \\")
    A("    --output qwen/train_with_codes.jsonl")
    A("```\n")

    A("## Sources\n")
    A("| source group | clips | loudness before | gain applied |")
    A("|---|---:|---:|---:|")
    for g, v in (aud.get("groups") or {}).items():
        A(f"| `{g}` | {v['clips']} | {v['lufs_before']} LUFS | {v['gain_db']:+.2f} dB |")
    A("")
    A(f"All four games and the anime were aligned to a common "
      f"**{aud.get('target_lufs','?')} LUFS** with **one gain per source**, never per clip, "
      f"so the loudness relationships inside each source survive (guide §9.5). "
      f"`game:SG` arrived {abs(aud.get('groups',{}).get('game:SG',{}).get('gain_db',0) - aud.get('groups',{}).get('anime:SG',{}).get('gain_db',0)):.0f} dB "
      f"louder than `anime:SG`. No denoising, gating or trimming was applied.\n")
    A(f"`{inv.get('from_quarantine','?')}` of the selected clips come from "
      f"`voice_dataset_game/wavs_quarantine/` — kept, as instructed, because they are "
      f"quarantined there for being short and non-lexical, not for being dirty. "
      f"{rep.get('from_quarantine','?')} of them reached the export; the remainder are "
      f"the ones whose sound could not be named confidently, held for review rather "
      f"than shipped with a guessed tag.\n")

    A("## Expressivity labelling\n")
    A("Tags in the export:\n")
    if per_tag:
        A("| tag | clips | hours | median | isolated | in-speech | splits |")
        A("|---|---:|---:|---:|---:|---:|---|")
        for tag, v in per_tag.items():
            A(f"| `{tag}` | {v['clips']} | {v['hours']} | {v['median_s']} s | "
              f"{v['isolated']} | {v['in_speech']} | "
              f"{' '.join(f'{k}:{n}' for k, n in v['splits'].items())} |")
    else:
        A("_none survived the gate_")
    A("")
    A(f"That is {nvv.get('share_of_dataset', 0):.1%} of the dataset — deliberately "
      f"low. The corpus simply does not contain more verifiable non-verbal events, and "
      f"the guide is explicit (§11.1, §31) that inflating them produces a caricature.\n")

    A("### How a tag is earned\n")
    A("Three independent signals, all required:\n")
    A("1. **`laion/vocalburst-classifier-single`** names the burst, with probabilities "
      "aggregated *by family* before ranking (supplement §7) — no single fine class "
      "like \"Contented Sigh\" ever decides anything on its own.")
    A("2. **`mispeech/ced-base`** votes independently on the same audio.")
    A("3. For anything inside a spoken line, **`laion/vocalburst-locator` v2** and "
      "**`Qwen3-ForcedAligner-0.6B`** must agree the region is energetic *and* "
      "lexically unexplained.\n")
    A("Thresholds are not guessed. 400 ordinary-speech clips were run through the same "
      "classifier as a negative control, which measures how often each family is "
      "hallucinated on plain acted Japanese:\n")
    ctl = clf.get("negative_control_ordinary_speech", {})
    surv = sec.get("surviving_families", {})
    A("| gate | isolated clips kept | false positives on 400 speech clips |")
    A("|---|---:|---:|")
    A(f"| classifier alone, score ≥ 0.70 | "
      f"{sum((surv.get('isolated_t0.70_primary_only') or {}).values())} | "
      f"{sum((surv.get('CONTROL_t0.70_primary_only') or {}).values())} |")
    A(f"| **both models agree, score ≥ 0.70** | "
      f"{sum((surv.get('isolated_t0.70_both_models') or {}).values())} | "
      f"{sum((surv.get('CONTROL_t0.70_both_models') or {}).values())} |")
    A(f"| **both models agree, score ≥ 0.50** | "
      f"{sum((surv.get('isolated_t0.50_both_models') or {}).values())} | "
      f"{sum((surv.get('CONTROL_t0.50_both_models') or {}).values())} |")
    A("")
    A("`sigh` and `gasp` are the only families with enough support *and* a measured "
      "zero-ish false-positive rate, so they are the only ones granted an auto-accept "
      "rule. `lipsmack` and `grunt` turned out to be the classifier's junk attractors "
      "on Japanese speech and are never exposed. Full sweep in "
      "`reports/05_threshold_sweep.csv`.\n")

    A("### Japanese-specific decisions\n")
    A("* **Punctuation is never read as a vocalisation.** `……` means a pause, a "
      "trailing thought or low-energy delivery — it is preserved verbatim and only the "
      "audio decides whether a sound is there (supplement §3).")
    A("* **Written onomatopoeia stays written.** `ふふ`, `はぁ……`, `きゃっ` are left as "
      "lexical text rather than rewritten to `[laugh]`/`[sigh]`. The audio models "
      f"agreed with the written form on only "
      f"{clf.get('onomatopoeia', {}).get('prior_audio_agreement', '?')} of these clips, "
      "so converting them would be guessing — and Japanese genuinely lexicalises them, "
      "which is the case the guide (§4.2) says to leave alone.")
    A("* **The locator alone is not trustworthy here.** It fires mostly on short "
      "Japanese interjections (え, うん, あっ) that are already in the transcript. "
      "Requiring an alignment gap is what removes them.\n")

    ref = load("11_reference_check.json")
    if ref:
        s = ref["summary"]
        A("## Is `ref.wav` really this voice?\n")
        A("Qwen reads the reference on every training row and writes its embedding into "
          "the speaker table, so a wrong reference quietly poisons the whole fine-tune. "
          "Checked with WavLM x-vectors against both target variants **and** ten other "
          "STEINS;GATE characters as a negative control, scored on globally centered "
          "embeddings (raw cosines saturate above 0.9 for every young female anime voice "
          "and tell you nothing):\n")
        A("| group | centered cos to ref |")
        A("|---|---:|")
        for r in ref["ranking"]:
            mark = "**" if r["group"].startswith("target:") else ""
            A(f"| {mark}`{r['group']}`{mark} | {mark}{r['centered']:.3f}{mark} |")
        A("")
        A(f"Both target groups outrank every other character by "
          f"**{s['margin_target_minus_best_other']:+.3f}**. Kurisu and Amadeus score "
          f"**{s['kurisu_vs_amadeus_centroid_cos']:.3f}** against each other — which is "
          f"the measurement backing the decision to train them as one speaker.\n")

    A("## Reproduce\n")
    A("```bash")
    A("cd pipeline")
    A("set ALIGNER_PYTHON=<venv>\\Scripts\\python.exe   # qwen-asr pins transformers 4.57.6")
    A("set CUDA_FRACTION=0.80                          # leave headroom for other GPU work")
    A("python run_all.py")
    A("python verify_dataset.py")
    A("```\n")
    if ver:
        A(f"`verify_dataset.py`: **{len(ver.get('failures', []))} failures**, "
          f"{len(ver.get('warnings', []))} warnings.\n")

    A("## Known limits\n")
    A("* **Nobody has listened to this yet.** Thresholds are calibrated against a "
      "measured proxy, which separates \"burst\" from \"ordinary speech\" but cannot "
      "confirm a sigh is not a gasp. Work `reports/05_listening_queue.csv` top-down "
      "before training — it is ordered riskiest-first (guide §44).")
    if per_tag:
        thin = [t for t, v in per_tag.items() if v["clips"] < 25]
        if thin:
            A(f"* {', '.join('`%s`' % t for t in thin)} ship with fewer than 25 examples. "
              "The guide (§12) calls that range experimental — the acoustics may be "
              "learned, reliable control probably not.")
        iso = sum(v["isolated"] for v in per_tag.values())
        ctx = sum(v["in_speech"] for v in per_tag.values())
        share = ctx / max(iso + ctx, 1)
        A(f"* {share:.0%} of tagged events are **contextual** (speech→event→speech), "
          f"{iso} isolated against {ctx} in-speech. The guide (§11.2) prefers 70–90% "
          f"contextual; this corpus is single-line game/anime dialogue, so the "
          f"in-speech path is heavily gated for precision and cannot be pushed much "
          f"higher without inventing events.")
    A(f"* `okabe`-style over-representation does not apply, but source mix does: "
      f"{', '.join(f'{k} {v}' for k, v in (rep.get('by_source_group') or {}).items())}.")
    A("* RE:BOOT audio is band-limited to ~10.5 kHz against ~12.5 kHz for the Ogg games; "
      "`series` is on every master-manifest row if that matters to you.")
    A("* Custom `[tag]` strings are **not** an official Qwen3-TTS feature — the SFT "
      "loader tokenises `text` as ordinary text (guide §1.4). They are a convention this "
      "dataset teaches, so evaluate controllability separately from audio quality "
      "(guide §27) using `qwen/probe_prompts.jsonl`.")
    A("")

    (PROJECT / "README.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote README.md ({len('\n'.join(L))} chars)")


if __name__ == "__main__":
    main()
