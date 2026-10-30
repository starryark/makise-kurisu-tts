"""Stage 04 - an independent speaker check on the audio we are about to publish.

The upstream builds already verified speaker identity, but they did it on their
own intermediate audio with their own profiles.  This stage re-does it on the
exact WAV files that ship, with a model that was not part of either upstream
decision, and it scores every clip against a *negative control* rather than
against a threshold: how close is this clip to the Kurisu centroid compared with
the closest other STEINS;GATE character recorded in the same medium and language.

Two details matter for the numbers to mean anything.

* **Centre the embedding space.**  Raw ECAPA cosines between any two young
  female anime voices sit above 0.9 and separate nothing.  Subtracting the mean
  of the whole embedding population first turns a saturated 0.95-vs-0.93 into a
  usable 0.72-vs-0.31.
* **Compare inside one medium and one language.**  Dry studio game voice and
  source-separated anime dialogue have different channel colouring, and the six
  dubs are six different performers, so a centroid is built per
  ``(language, medium)`` and a clip is only ever compared with rivals from its
  own cell.
"""
from __future__ import annotations

import collections
import json
import os
import random
import subprocess

import numpy as np
import soundfile as sf
import soxr
import torch

from kcommon import (ANIME_ROOT, GAME_ROOT, LANGUAGES, MANIFESTS, PROJECT,
                     REPORTS, TARGET_SR, read_jsonl, utf8_stdout, write_json,
                     write_jsonl)

utf8_stdout()

EMB_SR = 16000
BATCH = 32
BUDGET_S = 120.0        # padded seconds of audio per batch
CONTROL_PER_SPEAKER = 120          # negative-control clips per rival character
MIN_CONTROL_CLIPS = 25
SEED = 20260904

GAME_CONTROL = ["okabe", "mayuri", "daru", "maho", "suzuha", "faris", "luka",
                "moeka", "kagari", "yuki", "tennouji", "nae", "leskinen",
                "reyes", "kaede", "akiha"]
ANIME_CONTROL = ["okabe_rintaro", "shiina_mayuri", "hashida_itaru",
                 "hiyajo_maho", "amane_suzuha", "faris_nyannyan",
                 "urushibara_ruka", "kiryu_moeka", "shiina_kagari",
                 "amane_yuki", "tennouji_yugo", "tennouji_nae",
                 "alexis_leskinen", "nakase_katsumi", "kurushima_kaede",
                 "judy_reyes", "amadeus_maho"]


def _free_vram_gib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        total, used = (float(x) for x in out.split(","))
        return (total - used) / 1024.0
    except Exception:
        return 0.0


def _dur(path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    except Exception:
        return 0.0


def load_16k(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != EMB_SR:
        x = soxr.resample(x, sr, EMB_SR, quality="HQ")
    return np.ascontiguousarray(x, dtype=np.float32)


class Ecapa:
    def __init__(self, device: str):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(PROJECT / "cache" / "ecapa"),
            run_opts={"device": device})

    @torch.no_grad()
    def embed(self, waves: list[np.ndarray]) -> np.ndarray:
        n = max(max(len(w) for w in waves), EMB_SR // 2)
        batch = np.zeros((len(waves), n), dtype=np.float32)
        lens = np.zeros(len(waves), dtype=np.float32)
        for i, w in enumerate(waves):
            batch[i, :len(w)] = w
            lens[i] = len(w) / n
        t = torch.from_numpy(batch).to(self.device)
        L = torch.from_numpy(lens).to(self.device)
        e = self.model.encode_batch(t, L).squeeze(1)
        return e.float().cpu().numpy()


def embed_paths(model: Ecapa, paths: list, tag: str) -> np.ndarray:
    """Embed in input order, but batch by length.

    Every item in a batch is zero-padded to the longest one in it, so mixing a
    0.4 s reaction with a 24 s monologue multiplies the work by sixty.  Sorting by
    length first and restoring the order afterwards costs nothing and removes it.
    """
    import time
    import torch as _torch
    durs = [_dur(p) for p in paths]
    order = sorted(range(len(paths)), key=lambda i: durs[i])
    res: list[np.ndarray | None] = [None] * len(paths)
    t0 = time.time()
    done = 0
    i = 0
    reported = 0
    while i < len(order):
        # A batch is padded to its longest item, so bound the *padded* work
        # instead of the item count: that keeps VRAM flat whether the batch is
        # forty half-second reactions or two twenty-second monologues, which
        # matters on a card shared with something else.
        longest = durs[order[i]]
        n = 1
        while (i + n < len(order) and n < BATCH
               and max(longest, durs[order[i + n]]) * (n + 1) <= BUDGET_S):
            longest = max(longest, durs[order[i + n]])
            n += 1
        idx = order[i:i + n]
        for attempt in range(4):
            try:
                emb = model.embed([load_16k(paths[j]) for j in idx])
                break
            except _torch.OutOfMemoryError:
                _torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                idx = idx[:max(1, len(idx) // 2)]
                n = len(idx)
        for k, j in enumerate(idx):
            res[j] = emb[k]
        i += n
        done += n
        if done - reported >= 1000:
            reported = done
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {tag}: {done}/{len(paths)}  {rate:.0f} clip/s  "
                  f"eta {(len(paths) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    return (np.stack([r for r in res]) if paths
            else np.zeros((0, 192), dtype=np.float32))


def control_paths(rng: random.Random) -> dict[tuple[str, str, str], list]:
    """(language, medium, speaker) -> sampled clip paths for rival characters."""
    out: dict[tuple[str, str, str], list] = {}
    game_by = collections.defaultdict(list)
    for r in read_jsonl(GAME_ROOT / "metadata.jsonl"):
        if r.get("speaker_id") in GAME_CONTROL and r.get("tier") in ("A", "B"):
            p = GAME_ROOT / r["rel_path"]
            if p.exists():
                game_by[r["speaker_id"]].append(p)
    for spk, ps in game_by.items():
        rng.shuffle(ps)
        if len(ps) >= MIN_CONTROL_CLIPS:
            out[("ja", "game", spk)] = ps[:CONTROL_PER_SPEAKER]

    for lang_dir, lang in LANGUAGES.items():
        root = ANIME_ROOT / lang_dir
        meta = root / "metadata.jsonl"
        if not meta.exists():
            continue
        by = collections.defaultdict(list)
        for r in read_jsonl(meta):
            if r.get("speaker") in ANIME_CONTROL:
                p = root / r["audio"].replace("/", "\\")
                if p.exists():
                    by[r["speaker"]].append(p)
        for spk, ps in by.items():
            rng.shuffle(ps)
            if len(ps) >= MIN_CONTROL_CLIPS:
                out[(lang, "anime", spk)] = ps[:CONTROL_PER_SPEAKER]
    return out


def trimmed_centroid(E: np.ndarray) -> np.ndarray:
    """Mean of the 80% of rows closest to the provisional mean."""
    if len(E) == 0:
        return np.zeros(E.shape[1] if E.ndim == 2 else 192, dtype=np.float32)
    c = E.mean(axis=0)
    c /= np.linalg.norm(c) + 1e-9
    s = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)) @ c
    keep = s >= np.percentile(s, 20)
    c2 = E[keep].mean(axis=0)
    return c2 / (np.linalg.norm(c2) + 1e-9)


def main() -> None:
    # This card is shared with other work.  ECAPA is small enough to run on the
    # CPU in minutes, so fall back rather than queue behind whatever else is on
    # the GPU; KURISU_DEVICE forces the choice either way.
    device = os.environ.get("KURISU_DEVICE") or (
        "cuda" if torch.cuda.is_available() and _free_vram_gib() >= 2.0 else "cpu")
    print(f"[device] {device} ({_free_vram_gib():.1f} GiB free)", flush=True)
    rng = random.Random(SEED)
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    model = Ecapa(device)

    target_paths = [PROJECT / r["audio"] for r in rows]
    Et = embed_paths(model, target_paths, "target")

    ctrl = control_paths(rng)
    ctrl_keys, ctrl_emb = [], []
    for key, paths in sorted(ctrl.items()):
        ctrl_keys.append((key, len(paths)))
        ctrl_emb.append(embed_paths(model, paths, f"control {key[0]}/{key[2]}"))
    Ec = np.concatenate(ctrl_emb, axis=0) if ctrl_emb else np.zeros((0, Et.shape[1]))

    # ---- centre the whole population, then normalise -------------------------
    allE = np.concatenate([Et, Ec], axis=0)
    mu = allE.mean(axis=0)
    Etc = Et - mu
    Etc /= np.linalg.norm(Etc, axis=1, keepdims=True) + 1e-9
    Ecc = Ec - mu
    if len(Ecc):
        Ecc /= np.linalg.norm(Ecc, axis=1, keepdims=True) + 1e-9

    # rival centroids per (language, medium, speaker)
    rival: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = collections.defaultdict(list)
    off = 0
    for (key, n) in ctrl_keys:
        lang, medium, spk = key
        rival[(lang, medium)].append((spk, trimmed_centroid(Ecc[off:off + n])))
        off += n

    # Kurisu centroid per (language, medium), seeded from the most trustworthy rows
    def trustworthy(r) -> bool:
        if r["medium"] == "game":
            return (r["upstream_tier"] == "A"
                    and r["upstream_scores"].get("label_confidence") == "high"
                    and r["duration"] >= 1.0)
        return (r["upstream_tier"] in ("strict", "manual") and r["duration"] >= 1.0)

    cells = collections.defaultdict(list)
    for i, r in enumerate(rows):
        cells[(r["language"], r["medium"])].append(i)

    centroid: dict[tuple[str, str], np.ndarray] = {}
    for cell, idx in cells.items():
        seed = [i for i in idx if trustworthy(rows[i])]
        if len(seed) < 20:
            seed = idx
        centroid[cell] = trimmed_centroid(Etc[seed])

    out, report_cells = [], {}
    for cell, idx in sorted(cells.items()):
        c = centroid[cell]
        rivals = rival.get(cell) or rival.get((cell[0], "anime")) or rival.get(("ja", "game"))
        rnames = [n for n, _ in rivals] if rivals else []
        R = np.stack([v for _, v in rivals]) if rivals else np.zeros((0, len(c)))
        sims = Etc[idx] @ c
        rs = Etc[idx] @ R.T if len(R) else np.zeros((len(idx), 1)) - 1.0
        best = rs.max(axis=1)
        bestn = [rnames[j] for j in rs.argmax(axis=1)] if rnames else [None] * len(idx)
        for k, i in enumerate(idx):
            out.append({
                "id": rows[i]["id"],
                "spk_cos": round(float(sims[k]), 4),
                "spk_rival_cos": round(float(best[k]), 4),
                "spk_margin": round(float(sims[k] - best[k]), 4),
                "spk_rival": bestn[k],
                "spk_cell": f"{cell[0]}:{cell[1]}",
            })
        report_cells[f"{cell[0]}:{cell[1]}"] = {
            "clips": len(idx),
            "rivals": rnames,
            "cos_p05": round(float(np.percentile(sims, 5)), 4),
            "cos_median": round(float(np.median(sims)), 4),
            "margin_p01": round(float(np.percentile(sims - best, 1)), 4),
            "margin_p05": round(float(np.percentile(sims - best, 5)), 4),
            "margin_median": round(float(np.median(sims - best)), 4),
            "clips_below_zero_margin": int((sims - best < 0).sum()),
        }

    write_jsonl(MANIFESTS / "04_speaker.jsonl", out)

    # cross-check: are the two variants and the two Japanese media the same voice?
    byv = collections.defaultdict(list)
    for i, r in enumerate(rows):
        byv[(r["language"], r["medium"], r["character_variant"])].append(i)
    pair = {}
    for a in sorted(byv):
        for b in sorted(byv):
            if a >= b or a[0] != b[0]:
                continue
            ca, cb = trimmed_centroid(Etc[byv[a]]), trimmed_centroid(Etc[byv[b]])
            pair[f"{'/'.join(a)} vs {'/'.join(b)}"] = round(float(ca @ cb), 4)

    write_json(REPORTS / "04_speaker.json", {
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "device": device,
        "target_clips": len(rows),
        "control_clips": int(len(Ec)),
        "control_speakers": {f"{k[0]}:{k[1]}:{k[2]}": n for k, n in ctrl_keys},
        "embedding_centred": True,
        "cells": report_cells,
        "variant_and_medium_similarity": pair,
    })
    print(json.dumps({"cells": report_cells,
                      "variant_and_medium_similarity": pair},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
