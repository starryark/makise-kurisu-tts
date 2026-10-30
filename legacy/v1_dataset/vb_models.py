"""Model wrappers for the local NVV labelling pipeline.

Two models, both from LAION:

* ``VocalBurstClassifier``  - laion/vocalburst-classifier-single. Names an isolated
  burst. Wrapped rather than used through the repo's own ``inference.py`` because
  that file predates the taxonomy patch: indices 79/80/81 now decode to the string
  ``"No burst"`` but its ``predict()`` still lets them into the top-k list. Here the
  whole ``{79, 80, 81, 82}`` set is treated as no-burst mass, and probabilities are
  aggregated into families before anything is decided (supplement s7).

* ``VocalBurstLocator``     - laion/vocalburst-locator, ``model_v2.pt`` with the v2
  post-processing values (threshold 0.50 / merge 0.10 / min-dur 0.10), not the v1-era
  defaults still hard-coded in the repo's inference script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

import gpu
from common import FINE_TO_FAMILY

CLS_REPO = "laion/vocalburst-classifier-single"
LOC_REPO = "laion/vocalburst-locator"
SR = 16000


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    w = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    w = torchaudio.functional.resample(w, sr, SR)
    return w.squeeze(0).numpy()


# ------------------------------------------------------------------ classifier

class _MLP(nn.Module):
    def __init__(self, d, h, depth, out):
        super().__init__()
        L = [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.0)]
        for _ in range(depth - 1):
            L += [nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.0)]
        L += [nn.Linear(h, out)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


class VocalBurstClassifier:
    #: indices whose label is not a real burst after the upstream taxonomy patch
    NO_BURST_IDX = (79, 80, 81, 82)

    def __init__(self, device: str | None = None):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        device = device or gpu.configure()
        d = Path(snapshot_download(CLS_REPO))
        self.repo_dir = d
        self.revision = d.name
        self.device = device
        self.cfg = json.loads((d / "config.json").read_text())
        self.classes = json.loads((d / "classes.json").read_text())
        self.vc = AutoModel.from_pretrained(
            self.cfg["embedder"], trust_remote_code=True).to(device).eval()
        self.mlp = _MLP(self.cfg["input_dim"], self.cfg["hidden"],
                        self.cfg["depth"], self.cfg["num_classes"]).to(device).eval()
        self.mlp.load_state_dict(torch.load(d / "model.pt", map_location=device))
        self.burst_idx = [i for i in range(len(self.classes))
                          if i not in self.NO_BURST_IDX]

    @torch.no_grad()
    def embed(self, waves16k: list[np.ndarray]) -> torch.Tensor:
        # The encoder zero-pads every item to a full 30 s window internally, so the
        # batch is padded to one common length here and nothing longer than 30 s is
        # ever sent (that would silently become several chunks per item).
        n = max(len(w) for w in waves16k)
        n = min(max(n, SR // 2), SR * 30)
        batch = np.zeros((len(waves16k), n), dtype=np.float32)
        for i, w in enumerate(waves16k):
            w = w[:n]
            batch[i, :len(w)] = w
        t = torch.from_numpy(batch).to(self.device)
        return self.vc.encode_waveform(t)

    @torch.no_grad()
    def probs(self, waves16k: list[np.ndarray]) -> np.ndarray:
        e = self.embed(waves16k)
        logit = self.mlp(e)
        return torch.softmax(logit, -1).float().cpu().numpy()

    def interpret(self, p: np.ndarray, topk: int = 5) -> dict:
        """One probability row -> fine top-k, family scores, margins."""
        p_nb = float(sum(p[i] for i in self.NO_BURST_IDX))
        fam: dict[str, float] = {}
        for i in self.burst_idx:
            f = FINE_TO_FAMILY.get(self.classes[i], "other")
            fam[f] = fam.get(f, 0.0) + float(p[i])
        ranked = sorted(fam.items(), key=lambda kv: -kv[1])
        order = sorted(self.burst_idx, key=lambda i: -p[i])[:topk]
        return {
            "p_no_burst": round(p_nb, 4),
            "top_classes": [{"label": self.classes[i], "index": i,
                             "probability": round(float(p[i]), 4)} for i in order],
            "family_scores": {k: round(v, 4) for k, v in ranked if v >= 1e-3},
            "family": ranked[0][0] if ranked else None,
            "family_score": round(ranked[0][1], 4) if ranked else 0.0,
            "family_margin": round(ranked[0][1] - ranked[1][1], 4) if len(ranked) > 1 else
            (round(ranked[0][1], 4) if ranked else 0.0),
        }

    def classify(self, waves16k: list[np.ndarray], topk: int = 5) -> list[dict]:
        return [self.interpret(row, topk) for row in self.probs(waves16k)]


# ------------------------------------------------------------------ locator

class _WhisperSegmenter(nn.Module):
    def __init__(self, whisper_id: str = "openai/whisper-small"):
        super().__init__()
        from transformers import WhisperModel
        self.whisper = WhisperModel.from_pretrained(whisper_id)
        d_model = self.whisper.config.d_model
        hidden = max(256, d_model // 2)
        self.proj = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.1))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3), nn.GELU(), nn.Dropout(0.1))
        self.out = nn.Linear(hidden, 1)

    def forward(self, input_features):
        enc = self.whisper.encoder(input_features=input_features).last_hidden_state
        h = self.proj(enc).permute(0, 2, 1)
        h = self.temporal(h).permute(0, 2, 1)
        return self.out(h).squeeze(-1)


class VocalBurstLocator:
    CLIP_S = 30.0
    FRAMES = 1500

    def __init__(self, device: str | None = None, checkpoint: str = "model_v2.pt"):
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        device = device or gpu.configure()
        ckpt = hf_hub_download(LOC_REPO, checkpoint)
        self.checkpoint = checkpoint
        self.device = device
        self.model = _WhisperSegmenter()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        self.model = self.model.to(device).eval()
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    @torch.no_grad()
    def frame_probs(self, waves16k: list[np.ndarray]) -> np.ndarray:
        n = int(self.CLIP_S * SR)
        batch = [np.pad(w[:n], (0, max(0, n - len(w[:n])))) for w in waves16k]
        feats = self.fe(batch, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device)
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(feats)
        else:
            logits = self.model(feats)
        return torch.sigmoid(logits).float().cpu().numpy()

    def events(self, probs: np.ndarray, dur_s: float, threshold: float = 0.50,
               merge_gap: float = 0.10, min_dur: float = 0.10) -> list[dict]:
        """v2 post-processing. Events are clipped to the real clip duration."""
        binary = probs > threshold
        raw, in_ev, start = [], False, 0
        for f, v in enumerate(binary):
            if v and not in_ev:
                start, in_ev = f, True
            elif not v and in_ev:
                raw.append((start, f))
                in_ev = False
        if in_ev:
            raw.append((start, len(binary)))
        if not raw:
            return []
        f2s = self.CLIP_S / self.FRAMES
        merged = [[raw[0][0], raw[0][1]]]
        for s, e in raw[1:]:
            if (s - merged[-1][1]) * f2s <= merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        out = []
        for fs, fe in merged:
            s, e = fs * f2s, fe * f2s
            if s >= dur_s:
                continue
            e = min(e, dur_s)
            if e - s < min_dur:
                continue
            out.append({"start_s": round(s, 3), "end_s": round(e, 3),
                        "score": round(float(probs[fs:fe].mean()), 3)})
        return out


# ------------------------------------------------------------------ secondary

#: AudioSet class ids (CED-base head) grouped into our families. Used only as an
#: independent second opinion (supplement s12) - never as the primary label.
CED_FAMILY_IDS = {
    "sigh": [26],
    "breath": [41, 45, 42, 43],
    "gasp": [44],
    "laugh": [16, 20, 17],
    "giggle": [18, 19],
    "chuckle": [21],
    "groan": [38],
    "moan": [38, 25],
    "grunt": [39],
    "sob": [22, 24],
    "cry": [22, 23, 25, 24],
    "scream": [14, 11, 8, 12],
    "cough": [47],
    "throat_clear": [48],
    "sniff": [50, 46],
    "hum": [37, 496],
    "whistle": [40, 402],
}
CED_SPEECH_IDS = [0, 1, 2, 3, 4, 5, 6, 15, 68]
CED_SING_IDS = [27, 32, 33, 34, 254, 266]


class CedSecondary:
    """mispeech/ced-base - generic AudioSet sanity check."""

    def __init__(self, device: str | None = None):
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

        device = device or gpu.configure()
        self.device = device
        self.model = AutoModelForAudioClassification.from_pretrained(
            "mispeech/ced-base", trust_remote_code=True).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(
            "mispeech/ced-base", trust_remote_code=True)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def scores(self, waves16k: list[np.ndarray]) -> list[dict]:
        out = []
        for w in waves16k:                      # CED takes variable length; one at a time
            if len(w) < SR // 10:
                w = np.pad(w, (0, SR // 10 - len(w)))
            inp = self.fe(w, sampling_rate=SR, return_tensors="pt")
            logits = self.model(**{k: v.to(self.device) for k, v in inp.items()}).logits
            p = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()
            fam = {k: float(max(p[i] for i in ids)) for k, ids in CED_FAMILY_IDS.items()}
            ranked = sorted(fam.items(), key=lambda kv: -kv[1])
            top_any = int(np.argmax(p))
            out.append({
                "family": ranked[0][0],
                "family_score": round(ranked[0][1], 4),
                "family_top3": [{"family": k, "score": round(v, 4)} for k, v in ranked[:3]],
                "speech_score": round(float(max(p[i] for i in CED_SPEECH_IDS)), 4),
                "singing_score": round(float(max(p[i] for i in CED_SING_IDS)), 4),
                "top_audioset": {"label": self.id2label[top_any],
                                 "score": round(float(p[top_any]), 4)},
            })
        return out

    @staticmethod
    def agrees(primary_family: str | None, ced: dict, rank: int = 2,
               floor: float = 0.05) -> bool:
        """True when the primary family is in CED's top-`rank` and not negligible."""
        if not primary_family:
            return False
        for e in ced["family_top3"][:rank]:
            if e["family"] == primary_family and e["score"] >= floor:
                return True
        return False
