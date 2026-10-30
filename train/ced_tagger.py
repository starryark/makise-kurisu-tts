"""CED AudioSet tagger: event-token realisation (SELECTION_RULE A6) and background-SFX
detection (A7).

Model: mispeech/ced-base (Apache-2.0), pinned below. 527 AudioSet classes. The HF
head already applies the sigmoid, so `probs()` are probabilities - never sigmoid them
again. CED's label strings are the AudioSet display names cut at the first comma
("Walk, footsteps" -> "Walk"); `label_ids` accepts either form.

Scoring: mono, resampled to 16 kHz, cut into <= 10 s windows (5 s hop, the last one
aligned to the end); a label's clip score is its max over windows. CED itself would
mean-pool a long clip's chunks, which dilutes a short event.

Detection rule (A6/A7): a class fires on a clip when its score (max over the class's
labels) exceeds a threshold set to a 5% false-positive rate on the same checkpoint's
untagged confirm outputs (`fpr_threshold`).

Local check (2026-10-28, real Kurisu JA takes, event_tags not human-verified):
  [sigh] (n=120) vs untagged (n=150): Breathing AUC 0.95 (TPR 0.80 @ 5% FPR),
  Sigh 0.75, Gasp 0.80;  [gasp] (n=57): Gasp AUC 0.92 (TPR 0.54), Breathing 0.84.

CLI:  python ced_tagger.py --wav a.wav --wav b.wav [--classes sigh,nv,rain]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

CED_REPO = "mispeech/ced-base"
CED_REV = "db3e14a8db4c21b56b165261c39649741a900e7f"
SR = 16000
WIN_S, HOP_S = 10.0, 5.0

# detection class -> AudioSet display names (any of them)
EVENT_CLASSES: Dict[str, List[str]] = {
    "sigh": ["Breathing", "Sigh"],
    "nv": ["Breathing", "Gasp", "Laughter", "Grunt"],
    "inhale": ["Breathing", "Gasp"],
    "laugh": ["Laughter", "Giggle", "Chuckle, chortle"],
    "gasp": ["Gasp"],
}
# mirrors p03c_augment.AUDIOSET_LABELS (the classes the SFX augmentation mixes in)
SFX_CLASSES: Dict[str, List[str]] = {
    "footsteps": ["Walk, footsteps"], "knock": ["Knock"],
    "ringtone": ["Ringtone", "Telephone bell ringing"],
    "rain": ["Rain", "Rain on surface", "Raindrop"],
    "crowd": ["Crowd", "Chatter", "Hubbub, speech noise, speech babble"],
    "typing": ["Typing", "Computer keyboard"], "microwave": ["Microwave oven", "Beep, bleep"],
    "traffic": ["Traffic noise, roadway noise"], "wind": ["Wind", "Wind noise (microphone)"],
}


def token_class(token: str) -> str:
    """'[bg_rain]' -> 'rain', '[sigh]' -> 'sigh'."""
    t = token.strip("[]")
    return t[3:] if t.startswith("bg_") else t


def class_names(token: str) -> List[str]:
    """AudioSet names for a tag token; unknown classes fall back to labels whose name
    contains the class word (resolved against the model in `CED.label_ids`)."""
    c = token_class(token)
    table = SFX_CLASSES if token.strip("[]").startswith("bg_") else EVENT_CLASSES
    return table.get(c) or [f"~{c.replace('_', ' ')}"]


def _short(name: str) -> str:
    return name.split(",")[0].strip().lower()


class CED:
    def __init__(self, device: Optional[str] = None, repo: str = CED_REPO, rev: str = CED_REV):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(repo, revision=rev, trust_remote_code=True)
        self.model = AutoModelForAudioClassification.from_pretrained(
            repo, revision=rev, trust_remote_code=True).eval().to(self.device)
        id2 = {int(k): v for k, v in self.model.config.id2label.items()}
        self.labels = [id2[i] for i in range(len(id2))]
        self._by_short = {_short(l): i for i, l in enumerate(self.labels)}

    def label_ids(self, names: Iterable[str]) -> List[int]:
        out = []
        for n in names:
            if n.startswith("~"):                      # generic fallback
                w = n[1:].lower()
                out += [i for i, l in enumerate(self.labels) if w in l.lower()]
            elif _short(n) in self._by_short:
                out.append(self._by_short[_short(n)])
            else:
                raise KeyError(f"AudioSet label {n!r} not in {CED_REPO}")
        return sorted(set(out))

    def probs(self, path: str) -> np.ndarray:
        """[527] max-over-windows probabilities for one clip."""
        import soundfile as sf
        from scipy.signal import resample_poly
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            g = math.gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        win, hop = int(WIN_S * SR), int(HOP_S * SR)
        if len(x) < SR // 4:
            x = np.pad(x, (0, SR // 4 - len(x)))
        starts = [0] if len(x) <= win else sorted(set(list(range(0, len(x) - win, hop)) + [len(x) - win]))
        best = None
        for s in starts:
            seg = x[s:s + win]
            with self.torch.no_grad():
                feats = self.fe(seg, sampling_rate=SR, return_tensors="pt")
                p = self.model(**{k: v.to(self.device) for k, v in feats.items()}).logits[0]
            p = p.float().cpu().numpy()
            best = p if best is None else np.maximum(best, p)
        return best

    def score(self, paths: Sequence[str], names: Iterable[str]) -> List[Dict[str, float]]:
        """Per clip {AudioSet label: prob} for the requested labels."""
        ids = self.label_ids(names)
        out = []
        for p in paths:
            pr = self.probs(p)
            out.append({self.labels[i]: round(float(pr[i]), 5) for i in ids})
        return out


def all_names(tokens: Iterable[str]) -> List[str]:
    names = set()
    for t in tokens:
        names.update(class_names(t))
    return sorted(names)


def class_score(label_probs: Dict[str, float], token: str, ced_labels: Optional[Sequence[str]] = None) -> Optional[float]:
    """Max over the token's AudioSet labels present in `label_probs` (label strings as
    CED returns them)."""
    want = class_names(token)
    vals = []
    for lab, p in label_probs.items():
        for n in want:
            if (n.startswith("~") and n[1:].lower() in lab.lower()) or _short(n) == _short(lab):
                vals.append(p)
    return max(vals) if vals else None


def fpr_threshold(negatives: Sequence[float], fpr: float = 0.05) -> float:
    """Score above which at most `fpr` of the negatives fire (detection: score > thr)."""
    v = np.sort(np.asarray([x for x in negatives if x is not None], dtype=float))
    if not len(v):
        return float("nan")
    k = int(math.ceil((1 - fpr) * len(v))) - 1
    return float(v[min(max(k, 0), len(v) - 1)])


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", action="append", required=True)
    ap.add_argument("--classes", default="sigh,nv," + ",".join(SFX_CLASSES))
    a = ap.parse_args()
    toks = [f"[{c}]" if c in EVENT_CLASSES else f"[bg_{c}]" for c in a.classes.split(",") if c]
    ced = CED()
    for p, s in zip(a.wav, ced.score(a.wav, all_names(toks))):
        print(Path(p).name, json.dumps({token_class(t): class_score(s, t) for t in toks}))


if __name__ == "__main__":
    main()
