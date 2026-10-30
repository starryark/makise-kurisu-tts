"""Stage 11 - is ref.wav actually this character, and is the corpus one voice?

The guide (§16) makes the reference audio load-bearing: the speaker encoder reads it
on every training row and the fine-tune writes its embedding into the speaker table.
So it is worth proving, not assuming, that the supplied ref_audio.wav is the same
voice as the training clips.

microsoft/wavlm-base-plus-sv (x-vector speaker verification) embeds:

  * ref.wav
  * a sample of `kurisu` clips
  * a sample of `amadeus_kurisu` clips
  * a sample of OTHER characters from the same source datasets - the negative control,
    which is what makes the similarity numbers mean anything

Writes reports/11_reference_check.json.
"""
from __future__ import annotations

import collections
import json
import random

import numpy as np
import soundfile as sf
import torch

import gpu
from common import (CLEAN, GAME, MANIFESTS, REF_OUT, REPORTS, SPEAKERS, read_jsonl,
                    utf8_stdout)
from vb_models import to_16k

utf8_stdout()

MODEL = "microsoft/wavlm-base-plus-sv"
N_PER_GROUP = 150
N_OTHER_SPEAKERS = 10
SEED = 20260821


def main() -> None:
    dev = gpu.configure()
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).to(dev).eval()

    @torch.no_grad()
    def embed(paths: list[str]) -> np.ndarray:
        out = []

        def run(chunk):
            waves = []
            for p in chunk:
                x, sr = sf.read(p, dtype="float32", always_2d=False)
                if x.ndim == 2:
                    x = x.mean(axis=1)
                w = to_16k(x, sr)[:16000 * 12]
                if len(w) < 16000:
                    w = np.pad(w, (0, 16000 - len(w)))
                waves.append(w)
            inp = fe(waves, sampling_rate=16000, padding=True, return_tensors="pt")
            e = model(**{k: v.to(dev) for k, v in inp.items()}).embeddings
            return list(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())

        out = gpu.run_batched(paths, run, batch=4, label="xvector", progress_every=200)
        return np.stack(out)

    rng = random.Random(SEED)
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    # only clips with real speech and a sane duration make good speaker evidence
    good = [r for r in rows if r["speech_status"] == "speech" and 1.5 <= r["dur"] <= 12]
    groups: dict[str, list[str]] = {}
    for variant in ("kurisu", "amadeus"):
        pool = [r["audio_abs"] for r in good if r["variant"] == variant]
        groups[f"target:{variant}"] = rng.sample(pool, min(N_PER_GROUP, len(pool)))

    # negative control: other characters, same recording chains
    others = collections.defaultdict(list)
    for root in (GAME, CLEAN):
        for m in read_jsonl(root / "metadata.jsonl"):
            if m["speaker_id"] in SPEAKERS or m["tier"] == "C":
                continue
            if not (1.5 <= m.get("dur", 0) <= 12):
                continue
            p = root / m["rel_path"]
            others[m["speaker_id"]].append(str(p).replace("\\", "/"))
    top_others = sorted(others, key=lambda k: -len(others[k]))[:N_OTHER_SPEAKERS]
    for spk in top_others:
        groups[f"other:{spk}"] = rng.sample(others[spk], min(60, len(others[spk])))

    print(f"embedding ref + {sum(len(v) for v in groups.values())} clips "
          f"across {len(groups)} groups")

    ref_raw = embed([str(REF_OUT)])[0]
    emb = {name: embed(paths) for name, paths in groups.items()}
    gpu.free()

    # Raw x-vector cosines saturate: every young female anime voice recorded through a
    # similar chain lands above 0.75, so a "0.96 to the target" number means little on
    # its own. Centering by the global mean removes that shared direction, which is
    # standard x-vector scoring practice and is what makes the margins interpretable.
    allE = np.concatenate([ref_raw[None, :]] + list(emb.values()))
    mu = allE.mean(0)

    def norm(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    ref_c = norm(ref_raw - mu)

    result, centroids = {}, {}
    for name, E in emb.items():
        Ec = norm(E - mu)
        c = norm(Ec.mean(0))
        centroids[name] = c
        raw, cen = E @ ref_raw, Ec @ ref_c
        result[name] = {
            "n": len(E),
            "cos_to_ref_mean": round(float(raw.mean()), 4),
            "cos_to_ref_median": round(float(np.median(raw)), 4),
            "cos_to_ref_p05": round(float(np.percentile(raw, 5)), 4),
            "cos_centroid_to_ref": round(float(norm(E.mean(0)) @ ref_raw), 4),
            "centered_cos_mean": round(float(cen.mean()), 4),
            "centered_cos_median": round(float(np.median(cen)), 4),
            "centered_centroid_to_ref": round(float(c @ ref_c), 4),
        }

    key = "centered_centroid_to_ref"
    tgt = {k: v[key] for k, v in result.items() if k.startswith("target:")}
    oth = {k: v[key] for k, v in result.items() if k.startswith("other:")}
    best_other = max(oth.items(), key=lambda kv: kv[1])
    margin = round(min(tgt.values()) - best_other[1], 4)
    ranked = sorted(result.items(), key=lambda kv: -kv[1][key])
    top2_are_targets = all(k.startswith("target:") for k, _ in ranked[:2])

    report = {
        "model": MODEL,
        "ref_audio": str(REF_OUT).replace("\\", "/"),
        "scoring": ("cosine on globally centered x-vectors; raw cosines kept for "
                    "reference but they saturate across similar voices"),
        "groups": result,
        "ranking": [{"group": k, "centered": v[key], "raw": v["cos_centroid_to_ref"]}
                    for k, v in ranked],
        "summary": {
            "ref_vs_kurisu": tgt["target:kurisu"],
            "ref_vs_amadeus": tgt["target:amadeus"],
            "best_other_character": list(best_other),
            "margin_target_minus_best_other": margin,
            "targets_rank_first_and_second": top2_are_targets,
            "kurisu_vs_amadeus_centroid_cos": round(
                float(centroids["target:kurisu"] @ centroids["target:amadeus"]), 4),
        },
        "verdict": (
            f"ref_audio matches the target voice: both target groups outrank every "
            f"other character, margin {margin:+.3f} over the nearest ({best_other[0]})"
            if top2_are_targets and margin > 0.05 else
            f"WARNING: ref_audio is not clearly closer to the target than to other "
            f"characters (margin {margin:+.3f} over {best_other[0]}) - listen before training"),
    }
    (REPORTS / "11_reference_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\n" + report["verdict"])


if __name__ == "__main__":
    main()
