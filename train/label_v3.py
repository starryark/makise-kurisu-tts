"""Kurisu v3 labeling pass: speaker rows, event tokens and background tokens per JA clip.

No human listening is involved: every label comes from a high-precision automatic rule,
and every rule is written out below. Audio files are only read, never written.

Run with the labeling venv (needs scikit-learn and pyopenjtalk-plus, the same g2p the
v2 pod used, so reading-space CER reproduces decisions.jsonl exactly):

    <scratchpad>/venv_labels/Scripts/python label_v3.py

Inputs
  hf_dataset_upload/data/{train,validation,test}/metadata.jsonl   (language == "ja")
  kurisu_v2/artifacts/build/decisions.jsonl                       (v2 per-clip decisions)
  kurisu_v2/artifacts/src/asr.jsonl                               (Qwen3-ASR + anime-whisper, all clips)
  kurisu_v2/pod/p03_build.py                                      (imported: reading/cer/cer_bar,
                                                                   train_text/normalise_text, speech_bounds)
Outputs (kurisu_v2/v3/labels/)
  labels_v3.jsonl, report.json, review_examples.md, audio_features.npz (LTAS + trimmed span cache)

Pipeline
  1. Drop logic mirrors p03 with the two device reasons removed:
       singing (style != speech, or a [singing] tag) -> tier_c -> no_lexical_text -> anime SNR
       below the game p05 floor -> transcript_mismatch (p03 cer_bar) -> no_audible_speech ->
       too_short / too_long.
     CER comes from decisions.jsonl where v2 computed it; otherwise it is computed here with
     p03.reading()/cer() (verified: 0 differences on all 8,595 v2-scored clips).
     Duration is p03's post-trim length (speech span + 0.12 s lead + 0.25 s tail pad), computed
     here for every clip with p03.speech_bounds.
     Exception: isolated non-lexical clips with an event (text = tokens + punctuation / NVV kana)
     are tag-only rows: exempt from no_lexical_text and the CER bar, duration floor 0.3 s on the
     same measure; tier C, SNR and singing still apply. Rows where both ASR systems hear >4 kana
     are flagged in notes, not dropped.
  2. speaker_row: [digital] -> phone; game amadeus -> amadeus; anime amadeus -> amadeus_anime
     if its LTAS classifier score (trained on SG0 anime Amadeus vs SG0 anime Kurisu, OOF inside
     SG0) is above the threshold that at most 10% of SG0 anime Kurisu reach, else kurisu;
     everything else -> kurisu.
  3. Text: [pause] -> "……" (v2 rule) -> isolated-clip tokens, or inline human tags, corroborated
     auto tags and game-script kana tokens -> v2 normalise_text -> background tokens in front.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import unicodedata
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(r"E:\STEINS;GATEVoice")
DATA = ROOT / "hf_dataset_upload" / "data"
V2 = ROOT / "kurisu_v2"
OUT = V2 / "v3" / "labels"
SPLITS = ("train", "validation", "test")

sys.dont_write_bytecode = True          # never write into kurisu_v2/pod/__pycache__
sys.path.insert(0, str(V2 / "pod"))
import p03_build as p03                 # noqa: E402  (v2 rules, imported rather than copied)

SR = p03.SR
PAD = p03.LEAD_PAD + p03.TAIL_PAD
TAGONLY_MIN_DUR = 0.3
SPEAKER_ROWS = ("kurisu", "amadeus", "amadeus_anime", "phone")
SEED = 0

# ---------------------------------------------------------------- tokens

EVENT_TOKENS = ("[sigh]", "[inhale]", "[laugh]", "[nv]")
BG_MAP = {"noise": "[bg_noise]", "footstep": "[bg_footsteps]",
          "ringtone": "[bg_ringtone]", "knock": "[bg_knock]"}       # dict order = fixed output order
TAG_TO_TOKEN = {"[sigh]": "[sigh]", "[gasp]": "[inhale]", "[breath]": "[inhale]"}
TOKEN_FAMILY = {"[sigh]": "sigh", "[inhale]": "inhale", "[laugh]": "laugh", "[nv]": "nv"}
FAMILY_TOKEN = {v: k for k, v in TOKEN_FAMILY.items()}
V3_TOKEN_RE = re.compile(r"\[(?:sigh|inhale|laugh|nv|bg_(?:noise|footsteps|ringtone|knock))\]")
ANY_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")

# ---------------------------------------------------------------- kana families

KANA = "ぁ-ゖァ-ヺー"
WORD = KANA + "一-鿿々〆A-Za-z0-9Ａ-Ｚａ-ｚ０-９"
LB, LA = rf"(?<![{WORD}])", rf"(?![{WORD}])"        # a kana unit must be a whole word

LAUGH_UNIT = (r"(?:[ふフ]{2,}|[うウ][ふフ]+|[あア][はハ]{2,}|[はハ]{3,}|[えエ][へヘ]{2,})[っッ]?"
              r"|(?:[くク][すス]){2,}|[くク][すス][っッ]|[ぷプ][っッ]")
SIGH_UNIT = r"[はハ][ぁあァア]+|[ふフ][ぅうゥウ]+[っッ]?|[はハふフ]ー+"
GASP_UNIT = r"[はハひヒ][っッ]|[きキ][ゃャ][っッ]"
NV_UNIT = (r"[うウ][ぅうゥウ]*[っッ]?|[くぐクグ][っッ]|[ぐグ][すス][っッ]?|[くク][しシ][ゅュ][っッ]?"
           r"|ゴホン|ごほん|[ごゴげゲ][ほホ][っッ]?|[むム][ぐグ][ぅゥっッ]?|[んン][ぐグ][っッ]"
           r"|[つツ][ぅゥ][っッ]|ぐぬぬ|[んン]+[っッ]|[っッ]")
FAMILY_UNIT = {"laugh": LAUGH_UNIT, "sigh": SIGH_UNIT, "inhale": GASP_UNIT}
FAMILY_RE = {f: re.compile(LB + "(" + u + ")" + LA) for f, u in FAMILY_UNIT.items()}
# report-only family for the anime-whisper diff: strain/sob/cough kana, without the bare っ
FAMILY_RE["nv_other"] = re.compile(LB + "(" + NV_UNIT.rsplit("|", 1)[0] + ")" + LA)
assert NV_UNIT.rsplit("|", 1)[1] == "[っッ]"
ISOLATED_CORE_RE = re.compile(
    "(?:" + "|".join(f"(?:{u})" for u in (LAUGH_UNIT, SIGH_UNIT, GASP_UNIT, NV_UNIT)) + ")*")
UNIT_RE = re.compile("|".join(f"(?P<{f}>{u})" for f, u in
                              (("laugh", LAUGH_UNIT), ("sigh", SIGH_UNIT), ("inhale", GASP_UNIT),
                               ("nv", NV_UNIT))))
PUNCT_CHARS = "\\s…‥―〜~、。，．！？!?・「」『』（）().,\\-—“”\"'"
PUNCT_RE = re.compile(f"[{PUNCT_CHARS}]+")
ELLIPSIS_START = re.compile(r"\s*(?:…|‥|―|\.\.\.|・・・)")
ARTICULATED_AFTER = re.compile(r"[\s…‥・.―ー〜~]*[？?！!]")
QUESTION_AFTER = re.compile(r"[\s…‥・.―ー〜~]*[？?]")
HM_RE = re.compile(LB + "([んン]+[っッ])" + LA)
CLAUSE_PREV = set("。！？!?…‥―、,．.「『（(") | {""}
PHRASE_END = re.compile(r"[。！？!?]+")
QWEN_CONTEXT_MARK = "鳳凰院凶真、椎名まゆり、橋田至"


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def strip_tags(text: str) -> str:
    return re.sub(r"\[[a-z_]+\]", "", text)


def sigh_articulated(text: str, end: int) -> bool:
    """はあ？ / はぁ……？ / ふぅ！ : a pronounced 'huh?/what!', not a breath."""
    return bool(ARTICULATED_AFTER.match(text, end))


def prev_char(text: str, start: int) -> str:
    """Previous visible character, skipping spaces and placeholder markers."""
    i = start - 1
    while i >= 0:
        c = text[i]
        if c.isspace():
            i -= 1
            continue
        if c == "\x01":                      # skip a whole placeholder \x00<n>\x01
            j = text.rfind("\x00", 0, i)
            i = j - 1
            continue
        return c
    return ""


def family_count(text: str, family: str) -> int:
    """Occurrences of a family's NVV kana in a hypothesis/reference (articulated sighs excluded)."""
    t = kata2hira(text)
    n = 0
    for m in FAMILY_RE[family].finditer(t):
        if family == "sigh" and sigh_articulated(t, m.end()):
            continue
        n += 1
    return n


# ---------------------------------------------------------------- loading

def load() -> tuple[list[dict], dict, dict]:
    meta = []
    for sp in SPLITS:
        for line in (DATA / sp / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            if r["language"] != "ja":
                continue
            r["split"] = sp
            r["id"] = Path(r["file_name"]).stem
            r["path"] = str(DATA / sp / r["file_name"])
            meta.append(r)
    dec = {}
    for line in (V2 / "artifacts" / "build" / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            d = json.loads(line)
            dec[d["id"]] = d
    asr = {}
    for line in (V2 / "artifacts" / "src" / "asr.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            a = json.loads(line)
            asr[a["id"]] = a
    ids = [r["id"] for r in meta]
    assert len(ids) == len(set(ids)) == 10039, len(ids)
    assert set(ids) == set(dec) == set(asr), "metadata / decisions / asr ids disagree"
    return meta, dec, asr


# ---------------------------------------------------------------- audio features

NFFT, HOP = 4096, 480
EDGES = np.geomspace(50.0, 12000.0, 33)                      # 32 log-spaced bands
_BAND = np.digitize(np.fft.rfftfreq(NFFT, 1 / SR), EDGES) - 1
_WIN = np.hanning(NFFT).astype(np.float32)


def audio_features(path: str) -> tuple[np.ndarray, float]:
    """(32-band normalised LTAS in dB over active frames, p03 trimmed span in seconds)."""
    w, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SR:
        raise SystemExit(f"{path}: {sr} Hz")
    w = w.mean(axis=-1) if w.ndim > 1 else w
    b = p03.speech_bounds(w)
    fade = int(0.005 * SR)
    span = (b[1] - b[0]) / SR if b is not None and (b[1] - b[0]) > 2 * fade else -1.0
    x = np.pad(w, (0, max(0, NFFT - len(w))))
    n = 1 + (len(x) - NFFT) // HOP
    frames = x[np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]] * _WIN
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    active = e > max(np.percentile(e, 95) - 30.0, -75.0)      # frames within 30 dB of the loud ones
    if active.sum() < 2:
        active[:] = True
    p = (np.abs(np.fft.rfft(frames[active], axis=1)) ** 2).mean(0)
    bands = np.array([p[_BAND == k].mean() if np.any(_BAND == k) else np.nan for k in range(32)])
    if np.isnan(bands).any():
        ok = ~np.isnan(bands)
        bands = np.interp(np.arange(32), np.where(ok)[0], bands[ok])
    bands = bands / bands.sum()                                 # each clip normalised to its own power
    return (10 * np.log10(bands + 1e-12)).astype(np.float32), float(span)


def get_audio_features(meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    cache = OUT / "audio_features.npz"
    ids = [r["id"] for r in meta]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["ids"]) == ids:
            return z["ltas"], z["span"]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(audio_features, [r["path"] for r in meta]))
    ltas = np.stack([a for a, _ in res])
    span = np.array([s for _, s in res], dtype=np.float64)
    np.savez_compressed(cache, ids=np.array(ids), ltas=ltas, span=span, band_edges_hz=EDGES)
    return ltas, span


# ---------------------------------------------------------------- classifier

def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=4, scoring="roc_auc", max_iter=5000))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def full_model(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=4, scoring="roc_auc", max_iter=5000)).fit(X, y)


def threshold_90(neg_scores: np.ndarray) -> float:
    """Smallest observed negative score with >= 90% of negatives strictly below it."""
    s = np.sort(neg_scores)
    k = math.ceil(0.9 * len(s))
    return float(s[min(k, len(s) - 1)])


def classify(meta: list[dict], ltas: np.ndarray) -> dict:
    """Routing model: SG0 anime Amadeus vs SG0 anime Kurisu (production-matched negatives).

    A model pooled over all anime learned the SG0-vs-2011 production difference (42% of SG0 Kurisu
    above its threshold vs 7% of 2011 Kurisu); it is kept below only as a reference.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    med = np.array([r["medium"] for r in meta])
    var = np.array([r["character_variant"] for r in meta])
    dig = np.array([("[digital]" in r["tags"]) or ("[digital]" in r["condition_tags"]) for r in meta])
    scene = np.array([r["scene"] or r["series"] for r in meta])
    series = np.array([r["series"] for r in meta])
    hnr = np.array([r["hnr_db"] for r in meta], dtype=float)
    ama = (var == "amadeus").astype(int)
    out: dict = {"features": "32-band LTAS (50 Hz-12 kHz, log-spaced, active frames within 30 dB of "
                             "the clip's p95 frame, normalised to the clip's own band power, dB)",
                 "model": "StandardScaler + LogisticRegressionCV(C in 0.01..3, inner 4-fold AUC)",
                 "cv": "5-fold StratifiedGroupKFold, groups = scene (anime episode / game script scene)",
                 "amadeus_score_note": "two scales: anime Amadeus rows carry the SG0-anime model's probability "
                                       "(out-of-fold for SG0 clips; full SG0 model for the out-of-training "
                                       "Cognitive Computing clips and the [digital] clip) - compare with "
                                       "anime_routing.threshold; game Amadeus rows carry the game model's OOF "
                                       "probability (diagnostic only; every game Amadeus clip -> amadeus)"}
    anime = (med == "anime") & ~dig
    sg0 = anime & (series == "Steins;Gate 0")
    game = med == "game"
    res = {}
    for name, mask in (("anime_sg0", sg0), ("game", game), ("anime_pooled_reference", anime)):
        X, y, g = ltas[mask], ama[mask], scene[mask]
        p = oof_scores(X, y, g)
        p_h = oof_scores(np.c_[X, hnr[mask]], y, g)
        res[name] = p
        out[f"{name}_amadeus_vs_kurisu"] = {
            "n_amadeus": int(y.sum()), "n_kurisu": int((y == 0).sum()), "n_groups": int(len(set(g))),
            "auc_oof_ltas": round(float(roc_auc_score(y, p)), 4),
            "auc_oof_ltas_plus_hnr": round(float(roc_auc_score(y, p_h)), 4)}
    out["anime_pooled_reference_amadeus_vs_kurisu"]["note"] = (
        "superseded routing model (all anime series pooled); confounded with production, see v1 report")

    # Scores for every anime clip on the SG0 scale: OOF inside SG0, full SG0 model outside it.
    sg0_model = full_model(ltas[sg0], ama[sg0])
    s_all = np.full(len(meta), np.nan)
    s_all[med == "anime"] = sg0_model.predict_proba(ltas[med == "anime"])[:, 1]
    s_all[sg0] = res["anime_sg0"]
    y0 = ama[sg0]
    thr = threshold_90(res["anime_sg0"][y0 == 0])
    am_rt = anime & (var == "amadeus")                       # routed population (non-[digital] anime Amadeus)
    up = am_rt & (s_all > thr)
    ku = anime & (var == "kurisu")
    out["anime_routing"] = {
        "threshold": round(thr, 4),
        "threshold_rule": "at most 10% of SG0 anime-Kurisu (non-[digital]) OOF scores lie at or above it; "
                          "anime Amadeus with score > threshold -> amadeus_anime, else kurisu",
        "sg0_kurisu_at_or_above_threshold": f"{int((s_all[sg0 & (var == 'kurisu')] >= thr).sum())}/"
                                            f"{int((sg0 & (var == 'kurisu')).sum())}",
        "anime_amadeus_to_amadeus_anime": int(up.sum()),
        "anime_amadeus_to_kurisu": int((am_rt & ~up).sum()),
        "by_series": {s: {"to_amadeus_anime": int((up & (series == s)).sum()),
                          "to_kurisu": int((am_rt & ~up & (series == s)).sum()),
                          "score_source": "OOF" if s == "Steins;Gate 0" else "full SG0 model (out of training)"}
                      for s in sorted(set(series[am_rt]))},
        "sanity_kurisu_above_threshold_by_series_not_rerouted": {
            s: f"{int((ku & (series == s) & (s_all > thr)).sum())}/{int((ku & (series == s)).sum())}"
            for s in sorted(set(series[ku]))},
        "sanity_note": "Kurisu clips outside SG0 are scored by the full SG0 model (never trained on them). "
                       "They stay speaker_row=kurisu whatever their score.",
    }
    # Cross-check: a model trained only on game Amadeus vs game Kurisu, applied to SG0 anime clips.
    game_model = full_model(ltas[game], ama[game])
    ga = game_model.predict_proba(ltas[sg0])[:, 1]
    out["anime_routing"]["game_model_cross_check"] = {
        "spearman_sg0_oof_vs_game_model_on_sg0_amadeus": round(float(spearmanr(res["anime_sg0"][y0 == 1],
                                                                                 ga[y0 == 1])[0]), 3),
        "game_model_auc_on_sg0_anime": round(float(roc_auc_score(y0, ga)), 4),
        "note": "the game Amadeus EQ (5-7 kHz peak) and the anime one (sub-bass cut, 1.5-4 kHz lift) differ, "
                "hence the separate amadeus_anime row",
    }
    score = {}
    for i, r in enumerate(meta):
        if r["character_variant"] != "amadeus":
            continue
        if r["medium"] == "anime":
            score[r["id"]] = float(s_all[i])
    gidx = np.where(game)[0]
    for j, i in enumerate(gidx):
        if var[i] == "amadeus":
            score[meta[i]["id"]] = float(res["game"][j])
    assert all(np.isfinite(v) for v in score.values()) and len(score) == int(ama.sum())
    return {"report": out, "score": score, "threshold": thr,
            "anime_ids_scored": {meta[i]["id"] for i in np.where(am_rt)[0]}}


# ---------------------------------------------------------------- text

class Ev:
    __slots__ = ("type", "source", "evidence")

    def __init__(self, type_: str, source: str, evidence: str):
        self.type, self.source, self.evidence = type_, source, evidence


def isolated_core(text_notags: str) -> str | None:
    """Kana core of a clip whose text is empty, punctuation, or only non-lexical kana units."""
    core = PUNCT_RE.sub("", text_notags)
    if not ISOLATED_CORE_RE.fullmatch(core):
        return None
    for m in FAMILY_RE["sigh"].finditer(text_notags):          # はあ？ is a word, not a breath
        if sigh_articulated(text_notags, m.end()):
            return None
    for m in HM_RE.finditer(text_notags):                       # んっ？ is an articulated "hm?"
        if QUESTION_AFTER.match(text_notags, m.end()):
            return None
    return core


def insert_family_tokens(text: str, families: tuple[str, ...], events: list, source: str,
                         cond, evidence_fmt: str) -> str:
    """Insert a placeholder before every qualifying kana unit (right to left to keep offsets)."""
    hits = []
    for fam in families:
        for m in FAMILY_RE[fam].finditer(text):
            if cond(fam, text, m):
                hits.append((m.start(), fam, m.group(1)))
    for start, fam, kana in sorted(hits, reverse=True):
        # already tagged with the same token right before this kana (e.g. a kept human/auto tag)?
        before = text[:start].rstrip()
        mm = re.search(r"\x00(\d+)\x01$", before)
        if mm and events[int(mm.group(1))] is not None and events[int(mm.group(1))].type == FAMILY_TOKEN[fam]:
            ev = events[int(mm.group(1))]
            if ev.source == "corroborated-auto" and source == "script-kana":
                ev.source = "script-kana"
                ev.evidence = evidence_fmt.format(kana=kana) + "; " + ev.evidence
            continue
        events.append(Ev(FAMILY_TOKEN[fam], source, evidence_fmt.format(kana=kana)))
        text = text[:start] + f"\x00{len(events) - 1}\x01" + text[start:]
    return text


def script_kana_cond(fam: str, text: str, m: re.Match) -> bool:
    if fam == "laugh":
        return True
    if fam == "sigh":
        if sigh_articulated(text, m.end()):
            return False
        return bool(ELLIPSIS_START.match(text, m.end())) or prev_char(text, m.start()) in CLAUSE_PREV
    if fam == "inhale":            # only when a pause (……/――) follows, never mid-breath speech
        return bool(ELLIPSIS_START.match(text, m.end()))
    return False


def build_text(r: dict, a: dict, is_game: bool, removed_auto: list) -> tuple[str, list[Ev], bool, list[str]]:
    """Returns (text_v3 without bg, events, isolated?, notes)."""
    notes: list[str] = []
    t = re.sub(r"[、。]?\[pause\]", "……", r["transcription"])          # v2 train_text rule, first
    plain = strip_tags(t)
    events: list[Ev | None] = []

    core = isolated_core(plain)
    if core is not None:
        # ---- isolated non-lexical clip: token(s) + the script's own punctuation/kana
        tags = [x for x in re.findall(r"\[[a-z_]+\]", t) if x in TAG_TO_TOKEN]
        body = plain.strip()
        if tags:
            toks = list(dict.fromkeys(TAG_TO_TOKEN[x] for x in tags))
            for x in dict.fromkeys(tags):
                origin = "human" if x in r["human_tags"] else "auto"
                events.append(Ev(TAG_TO_TOKEN[x], "isolated", f"isolated clip, {origin} tag {x}"))
            text = "".join(toks) + body
        else:
            fams = [m.lastgroup for m in UNIT_RE.finditer(core)]
            if any(f in FAMILY_UNIT for f in fams):
                text = insert_family_tokens(body, ("laugh", "sigh", "inhale"), events, "isolated",
                                            lambda f, s, m: True, "isolated clip, kana '{kana}'")
            else:
                events.append(Ev("[nv]", "isolated",
                                 f"isolated clip, no tag, kana '{core}'" if core else "isolated clip, no tag, no kana"))
                text = "\x000\x01" + body
        return finish(text, events), [e for e in events if e], True, notes

    # ---- lexical clip
    def sub(m: re.Match) -> str:
        tag = m.group(0)
        if tag not in TAG_TO_TOKEN:
            return ""                                                # [digital], [singing], ...
        origin = "human" if tag in r["human_tags"] else "auto"
        events.append(Ev(TAG_TO_TOKEN[tag], origin, f"{origin} tag {tag}"))
        return f"\x00{len(events) - 1}\x01"

    text = re.sub(r"\[[a-z_]+\]", sub, t)
    hyp = a.get("anime") or ""
    for i, ev in enumerate(events):
        if ev.source == "human":
            ev.evidence = f"human tag at its inline position ({ev.evidence.split()[-1]})"
            continue
        tag = ev.evidence.split()[-1]
        fam = TOKEN_FAMILY[ev.type]
        ph = f"\x00{i}\x01"
        pos = text.index(ph)
        # phrase = the sentence around the tag (bounded by 。！？)
        clean = re.sub(r"\x00\d+\x01", lambda mm: mm.group(0) if mm.group(0) == ph else "", text)
        cpos = clean.index(ph)
        starts = [0] + [m.end() for m in PHRASE_END.finditer(clean) if m.end() <= cpos]
        ends = [m.end() for m in PHRASE_END.finditer(clean) if m.start() >= cpos + len(ph)] + [len(clean)]
        lo, hi = max(starts), min(ends)
        kana_hits, artic_hits = [], []
        for m in FAMILY_RE[fam].finditer(clean):
            if lo <= m.start() < hi:
                if fam == "sigh" and sigh_articulated(clean, m.end()):
                    artic_hits.append(m.group(1))
                else:
                    kana_hits.append((abs(m.start() - cpos), m.group(1)))
        n_hyp, n_ref = family_count(hyp, fam), family_count(plain, fam)
        if kana_hits:
            kana = min(kana_hits)[1]
            # move the token to sit immediately before the nearest same-family kana
            text_wo = text.replace(ph, "")
            best = None
            for m in FAMILY_RE[fam].finditer(text_wo):
                if m.group(1) == kana:
                    d_ = abs(m.start() - pos)
                    if best is None or d_ < best[0]:
                        best = (d_, m.start())
            if best is not None:
                text = text_wo[:best[1]] + ph + text_wo[best[1]:]
            ev.source = "corroborated-auto"
            ev.evidence = f"auto tag {tag} + script kana '{kana}' in the same phrase (token placed before it)"
            notes.append(f"auto {tag} kept: script kana '{kana}' in phrase")
        elif n_hyp > n_ref:
            ev.source = "corroborated-auto"
            ev.evidence = (f"auto tag {tag} + anime-whisper hears {n_hyp} {fam} kana vs {n_ref} in the "
                           f"reference ('{hyp[:40]}')")
            notes.append(f"auto {tag} kept: anime-whisper {fam} kana not in reference")
            before, after = text[:pos], text[pos + len(ph):]
            if before and after and re.match(f"[{WORD}]", before[-1]) and re.match(f"[{WORD}]", after[0]):
                # never inside a word: move back to the previous punctuation/space boundary
                k = max([before.rfind(c) for c in "、。！？!?…‥― 　"] + [-1]) + 1
                text = before[:k] + ph + before[k:] + after
                notes.append(f"auto {tag} moved out of a word to the previous boundary")
        else:
            rule = ("R2_only_articulated_kana" if artic_hits else "R1_uncorroborated")
            why = (f"same-family kana in phrase is articulated ({'/'.join(artic_hits)})" if artic_hits else
                   f"no {fam} kana in the phrase and anime-whisper has no extra {fam} kana "
                   f"(hyp {n_hyp} vs ref {n_ref})")
            removed_auto.append({"id": r["id"], "tag": tag, "rule": rule, "why": why,
                                 "transcription": r["transcription"], "asr_anime": hyp})
            notes.append(f"auto {tag} removed ({rule}): {why}")
            events[i] = None
            text = text.replace(ph, "")

    if is_game:
        text = insert_family_tokens(text, ("laugh", "sigh", "inhale"), events, "script-kana",
                                    script_kana_cond, "game script kana '{kana}'")
    return finish(text, events), [e for e in events if e], False, notes


def finish(text: str, events: list) -> str:
    text = re.sub(r"\x00(\d+)\x01", lambda m: events[int(m.group(1))].type if events[int(m.group(1))] else "", text)
    text = p03.normalise_text(text)
    text = re.sub(r"\s*(\[(?:sigh|inhale|laugh|nv)\])\s*", r"\1", text)   # never a space next to a token
    text = re.sub(r"(\[(?:sigh|inhale|laugh|nv)\])\1+", r"\1", text)       # [inhale][inhale] -> [inhale]
    return text


def dedupe_events(text: str, events: list[Ev]) -> list[Ev]:
    """Keep one event per token occurrence in the final text (adjacent duplicates collapsed)."""
    counts = Counter(V3_TOKEN_RE.findall(text))
    out, seen = [], Counter()
    for e in events:
        if seen[e.type] < counts[e.type]:
            out.append(e)
            seen[e.type] += 1
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    import pyopenjtalk  # noqa: F401  (fail early if the venv is wrong)
    from importlib.metadata import version

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    meta, dec, asr = load()
    print(f"[load] {len(meta)} JA rows", flush=True)

    ltas, span = get_audio_features(meta)
    dur = np.where(span > 0, span + PAD, np.nan)
    print("[audio] features ready", flush=True)

    game_snr = sorted(r["snr_est_db"] for r in meta if r["medium"] == "game" and r.get("snr_est_db") is not None)
    snr_floor = game_snr[int(0.05 * len(game_snr))]
    assert abs(snr_floor - 28.09) < 1e-6, snr_floor

    clf = classify(meta, ltas)
    print("[clf] " + json.dumps({k: v for k, v in clf["report"].items() if "auc" in str(v)}, ensure_ascii=False)[:400], flush=True)

    rows, removed_auto = [], []
    cer_fresh = cer_checked = qwen_ctx = 0
    dur_check = []
    whisper_fire = Counter()
    whisper_fire_clips = 0
    degenerate_isolated = []
    asr_speech_flag = []
    for i, r in enumerate(meta):
        d, a = dec[r["id"]], asr[r["id"]]
        is_game = r["medium"] == "game"
        notes: list[str] = []
        if QWEN_CONTEXT_MARK in (a.get("qwen") or ""):
            qwen_ctx += 1
            notes.append("asr_qwen is the ASR context string (hallucination); only anime-whisper is informative")

        # ---- text and events
        text_ev, events, isolated, tnotes = build_text(r, a, is_game, removed_auto)
        notes += tnotes
        events = dedupe_events(text_ev, events)
        bg = "".join(BG_MAP[c] for c in BG_MAP if c in r["contamination"]) or None
        text_v3 = (bg or "") + text_ev
        text_plain = d.get("text_train") or p03.train_text(r["transcription"])

        if r["medium"] == "anime":
            plain = strip_tags(re.sub(r"[、。]?\[pause\]", "……", r["transcription"]))
            fired = [f for f in ("laugh", "sigh", "inhale", "nv_other")
                     if family_count(a.get("anime") or "", f) > family_count(plain, f)]
            whisper_fire.update(fired)
            whisper_fire_clips += bool(fired)

        # ---- CER (decisions.jsonl where v2 computed it, else fresh with p03.reading)
        if "cer_qwen" in d:
            cq, ca, nk, cer_src = d["cer_qwen"], d["cer_anime"], d["ref_kana"], "v2_decisions"
        else:
            ref = p03.reading(r["text"])
            cq = round(p03.cer(ref, p03.reading(a.get("qwen") or "")), 3)
            ca = round(p03.cer(ref, p03.reading(a.get("anime") or "")), 3)
            nk, cer_src = len(ref), "computed_v3"
            cer_fresh += 1

        # ---- drop logic: p03 order with the device reasons removed
        has_event = bool(events)
        # Isolated tag-only rows (text = event tokens + punctuation / NVV kana) are exempt from
        # no_lexical_text and the transcript CER bar (lead decision); tier C, SNR, duration and
        # singing still apply.
        exempt = isolated and has_event
        reason = None
        if r["style"] != "speech" or "[singing]" in r["tags"]:
            reason = "singing"
        elif r["quality_tier"] == "C":
            reason = "tier_c"
        if reason is None and not r["text"].strip() and not exempt:
            reason = "no_lexical_text"
        if (reason is None and r["medium"] == "anime" and r.get("snr_est_db") is not None
                and r["snr_est_db"] < snr_floor):
            reason = "anime_below_studio_snr"
        if reason is None and not exempt:
            cer_checked += 1
            if min(cq, ca) > p03.cer_bar(nk, r["duration_seconds"]):
                reason = "transcript_mismatch"
        if reason is None and exempt:
            if r["text"].strip() and min(cq, ca) > p03.cer_bar(nk, r["duration_seconds"]):
                degenerate_isolated.append(r["id"])
                notes.append(f"isolated tag-only row exempt from the CER bar (would fail: ref {nk} kana, "
                             f"cer {min(cq, ca)})")
            hq = 0 if QWEN_CONTEXT_MARK in (a.get("qwen") or "") else len(p03.reading(a.get("qwen") or ""))
            ha = len(p03.reading(a.get("anime") or ""))
            if min(hq, ha) > 4:
                asr_speech_flag.append(r["id"])
                notes.append(f"FLAG: tag-only row but both ASR hear >4 kana ({hq}/{ha}); kept per CER exemption")
        if reason is None:
            if not np.isfinite(dur[i]):
                reason = "no_audible_speech"
            elif dur[i] < (TAGONLY_MIN_DUR if (isolated and has_event) else p03.MIN_DUR):
                reason = "too_short"
            elif dur[i] > p03.MAX_DUR:
                reason = "too_long"
            elif dur[i] < p03.MIN_DUR:
                notes.append(f"isolated event clip kept below 0.6 s (dur {dur[i]:.2f} s; tag-only exception)")
        if "dur" in d and np.isfinite(dur[i]):
            dur_check.append(abs(d["dur"] - dur[i]))

        # ---- speaker row
        if "[digital]" in r["tags"] or "[digital]" in r["condition_tags"]:
            spk = "phone"
        elif r["character_variant"] == "amadeus" and r["medium"] == "game":
            spk = "amadeus"
        elif r["character_variant"] == "amadeus" and r["medium"] == "anime":
            spk = "amadeus_anime" if clf["score"][r["id"]] > clf["threshold"] else "kurisu"
            notes.append(f"anime Amadeus routed to {spk} (score {clf['score'][r['id']]:.3f} vs "
                         f"threshold {clf['threshold']:.3f})")
        else:
            spk = "kurisu"

        if bg:
            notes.append(f"background {bg} from contamination={r['contamination']}")
        tag_only = reason is None and isolated      # text_v3 = token(s) + punctuation / NVV kana only
        rows.append({
            "id": r["id"], "split": r["split"], "keep": reason is None, "reason": reason,
            "speaker_row": spk, "text_v3": text_v3, "text_plain": text_plain,
            "events": [{"type": e.type, "source": e.source, "evidence": e.evidence} for e in events],
            "bg": bg,
            "amadeus_score": round(clf["score"][r["id"]], 4) if r["character_variant"] == "amadeus" else None,
            "notes": notes,
            # audit fields
            "v2_kept": d["kept"], "v2_reason": d["reason"], "tag_only": tag_only,
            "isolated": isolated, "medium": r["medium"], "variant": r["character_variant"],
            "series": r["series"], "dur": None if not np.isfinite(dur[i]) else round(float(dur[i]), 3),
            "cer_qwen": cq, "cer_anime": ca, "ref_kana": nk, "cer_source": cer_src,
            "transcription": r["transcription"],
        })

    # ---------------------------------------------------------------- checks
    assert len(rows) == 10039 and len({x["id"] for x in rows}) == 10039
    allowed = set(EVENT_TOKENS) | set(BG_MAP.values())
    for x in rows:
        t = x["text_v3"]
        assert set(ANY_BRACKET_RE.findall(t)) <= allowed, (x["id"], t)
        assert "[" not in V3_TOKEN_RE.sub("", t) and "]" not in V3_TOKEN_RE.sub("", t), (x["id"], t)
        assert not re.search(r"\s\[|\]\s", t), (x["id"], t)
        assert "\x00" not in t and "\x01" not in t, x["id"]
        n_ev = sum(t.count(k) for k in EVENT_TOKENS)
        assert n_ev == len(x["events"]), (x["id"], t, x["events"])
    # rows v2 evaluated must get the same decision, except the documented changes
    changed = Counter()
    for x in rows:
        if x["v2_reason"] in ("device_voice_amadeus", "device_voice_phone"):
            continue
        if x["reason"] != x["v2_reason"]:
            changed[f"{x['v2_reason']} -> {x['reason']}"] += 1

    with (OUT / "labels_v3.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    report = make_report(rows, removed_auto, clf, snr_floor, changed, dict(
        cer_fresh=cer_fresh, cer_checked=cer_checked, qwen_ctx=qwen_ctx,
        dur_check=dur_check, whisper_fire=whisper_fire, whisper_fire_clips=whisper_fire_clips,
        degenerate_isolated=degenerate_isolated, asr_speech_flag=asr_speech_flag, pyopenjtalk=version("pyopenjtalk-plus")))
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_review(rows, removed_auto, clf)
    print(json.dumps({k: report[k] for k in ("kept", "kept_hours_by_speaker_row", "event_tokens_by_source",
                                               "bg_tokens", "tag_only_rows")}, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- reporting

def make_report(rows, removed_auto, clf, snr_floor, changed, x) -> dict:
    kept = [r for r in rows if r["keep"]]
    by_id = {r["id"]: r for r in rows}
    v2 = Counter(r["v2_reason"] for r in rows if not r["v2_kept"])
    v3 = Counter(r["reason"] for r in rows if not r["keep"])
    reasons = sorted(set(v2) | set(v3))

    def ev_table(sel, by=None):
        t = defaultdict(lambda: defaultdict(Counter) if by else Counter())
        for r in sel:
            for e in r["events"]:
                if by:
                    t[e["type"]][e["source"]][r[by]] += 1
                else:
                    t[e["type"]][e["source"]] += 1
        return {k: ({s: dict(c) for s, c in v.items()} if by else dict(v)) for k, v in sorted(t.items())}

    transitions = Counter(f"{r['v2_reason'] or 'kept'} -> {r['reason'] or 'kept'}" for r in rows
                          if (r["v2_reason"] or "kept") != (r["reason"] or "kept"))
    return {
        "generated": "2026-10-28",
        "script": str(Path(__file__).resolve()),
        "inputs": {"metadata": str(DATA), "decisions": str(V2 / "artifacts/build/decisions.jsonl"),
                   "asr": str(V2 / "artifacts/src/asr.jsonl"), "v2_rules_imported_from": str(V2 / "pod/p03_build.py")},
        "rows": len(rows),
        "kept": len(kept),
        "kept_hours": round(sum(r["dur"] for r in kept) / 3600, 3),
        "duration_measure": "p03 post-trim length: speech span (p03.speech_bounds) + 0.12 s lead + 0.25 s tail pad, "
                            "computed here for every clip; used for too_short/too_long, the 0.3 s tag-only floor "
                            "(effectively vacuous: padding alone is 0.37 s) and hours",
        "dur_vs_v2_max_abs_diff_s": round(max(x["dur_check"]), 4) if x["dur_check"] else None,
        "cer": {
            "option": "1 (computed locally)",
            "g2p": f"pyopenjtalk-plus {x['pyopenjtalk']} (same package as the v2 pod bootstrap)",
            "validation": "recomputing all 8,595 clips v2 CER-checked: 0 value differences, 0 decision flips",
            "rows_with_cer_computed_here": x["cer_fresh"],
            "rows_put_through_transcript_check": x["cer_checked"],
            "snr_floor_game_p05": snr_floor,
            "asr_qwen_context_hallucinations": x["qwen_ctx"],
        },
        "speaker_row_by_split_kept": {s: dict(Counter(r["split"] for r in kept if r["speaker_row"] == s))
                                      for s in SPEAKER_ROWS},
        "speaker_row_by_split_all_rows": {s: dict(Counter(r["split"] for r in rows if r["speaker_row"] == s))
                                          for s in SPEAKER_ROWS},
        "kept_hours_by_speaker_row": {s: round(sum(r["dur"] for r in kept if r["speaker_row"] == s) / 3600, 3)
                                      for s in SPEAKER_ROWS},
        "kept_by_speaker_row": dict(Counter(r["speaker_row"] for r in kept)),
        "event_tokens_by_source": ev_table(kept),
        "event_tokens_by_source_and_split": ev_table(kept, "split"),
        "event_tokens_by_source_and_speaker_row": ev_table(kept, "speaker_row"),
        "event_tokens_by_source_all_rows_incl_dropped": ev_table(rows),
        "kept_rows_with_any_event": sum(bool(r["events"]) for r in kept),
        "auto_tags": {
            "removed": len(removed_auto),
            "removed_by_rule_and_tag": {f"{k[0]} {k[1]}": v for k, v in
                                        sorted(Counter((a["rule"], a["tag"]) for a in removed_auto).items())},
            "removed_on_kept_rows": sum(1 for a in removed_auto if by_id[a["id"]]["keep"]),
            "kept_on_lexical_clips_corroborated_by_all_rows": dict(Counter(
                "script_kana" if "script kana" in e["evidence"] else "anime_whisper"
                for r in rows for e in r["events"]
                if "auto tag" in e["evidence"] and e["source"] != "isolated")),
            "auto_tags_on_isolated_clips_trusted_all_rows": sum(
                1 for r in rows for e in r["events"] if e["source"] == "isolated" and "auto tag" in e["evidence"]),
            "list": [{k: a[k] for k in ("id", "tag", "rule", "why")} for a in removed_auto],
        },
        "anime_whisper_diff": {
            "definition": "anime clips where asr_anime has more family kana than the reference (articulated はぁ？ "
                          "excluded; nv_other = strain/sob/cough kana such as うっ くっ ぐすっ, bare っ excluded). "
                          "Used only to corroborate auto tags; never creates a tag",
            "clips_firing": x["whisper_fire_clips"],
            "fires_by_family": dict(x["whisper_fire"]),
            "anime_clips": sum(1 for r in rows if r["medium"] == "anime"),
        },
        "bg_tokens": {
            "assigned_all_rows": dict(Counter(t for r in rows if r["bg"] for t in V3_TOKEN_RE.findall(r["bg"]))),
            "kept": dict(Counter(t for r in kept if r["bg"] for t in V3_TOKEN_RE.findall(r["bg"]))),
            "drop_reasons_of_bg_rows": dict(Counter(r["reason"] for r in rows if r["bg"] and not r["keep"])),
        },
        "classifier": clf["report"],
        "drop_reasons_v2_vs_v3": {k: {"v2": v2.get(k, 0), "v3": v3.get(k, 0)} for k in reasons},
        "decision_transitions_v2_to_v3": dict(sorted(transitions.items())),
        "v2_evaluated_rows_with_changed_reason": dict(changed),
        "tag_only_rows": {
            "kept_total": sum(r["tag_only"] for r in kept),
            "rescued_from_v2_no_lexical_text": sum(1 for r in kept if r["v2_reason"] == "no_lexical_text"),
            "rescued_from_v2_too_short": sum(1 for r in kept if r["v2_reason"] == "too_short"),
            "rescued_from_v2_transcript_mismatch": sum(1 for r in kept if r["v2_reason"] == "transcript_mismatch"),
            "from_v2_device_reasons": sum(1 for r in kept if r["tag_only"] and r["v2_reason"] in
                                          ("device_voice_amadeus", "device_voice_phone")),
            "already_kept_by_v2": sum(1 for r in kept if r["tag_only"] and r["v2_kept"]),
            "by_speaker_row": dict(Counter(r["speaker_row"] for r in kept if r["tag_only"])),
        },
        "isolated_rows": {"all": sum(r["isolated"] for r in rows), "kept": sum(r["isolated"] for r in kept)},
        "isolated_exempt_from_cer_that_would_fail_it": {
            "count": len(x["degenerate_isolated"]),
            "kept": sum(1 for i in x["degenerate_isolated"] if by_id[i]["keep"]),
            "with_bracket_tag": sum(1 for i in x["degenerate_isolated"]
                                    if any("tag [" in e["evidence"] for e in by_id[i]["events"])),
            "by_token": dict(Counter(e["type"] for i in x["degenerate_isolated"] for e in by_id[i]["events"])),
            "note": "isolated non-lexical clips (e.g. ……っ, はあ……) with a non-empty script text that v2's CER bar "
                    "would fail (reference reading of 0-2 kana vs Qwen3-ASR うん/あ). Exempt by lead decision; "
                    "tier C / SNR / duration / singing still applied.",
            "ids": x["degenerate_isolated"],
        },
        "tag_only_asr_speech_flags": {
            "count": len(x["asr_speech_flag"]),
            "kept": sum(1 for i in x["asr_speech_flag"] if by_id[i]["keep"]),
            "note": "isolated tag-only rows where both ASR systems output >4 kana (Qwen context hallucinations "
                    "counted as 0). Flagged in notes, not dropped.",
            "ids": x["asr_speech_flag"],
        },
        "text_checks": "passed: 10,039 unique rows; every bracket in text_v3 is one of [sigh] [inhale] [laugh] [nv] "
                       "[bg_*]; no space next to a token; token count == len(events)",
    }


def write_review(rows, removed_auto, clf) -> None:
    rng = random.Random(SEED)
    L = ["# v3 labels: review examples", "",
         "Generated by `kurisu_v2/v3/label_v3.py` on 2026-10-28. Random samples use seed 0 and are drawn from "
         "**kept** rows unless marked otherwise.", ""]
    groups = defaultdict(list)
    n_events = Counter()
    for r in rows:
        if not r["keep"]:
            continue
        seen = set()
        for e in r["events"]:
            key = (e["source"], e["type"])
            n_events[key] += 1
            if key not in seen:                     # one example per row, even with two events
                groups[key].append((r, e))
                seen.add(key)
    L.append("## 1. Event examples per source and family (up to 15 rows each)")
    for key in sorted(groups):
        items = groups[key]
        pick = rng.sample(items, min(15, len(items)))
        L += ["", f"### {key[0]} / {key[1]} ({n_events[key]} kept events in {len(items)} rows)", "",
              "| id | original transcription | text_v3 | evidence |", "|---|---|---|---|"]
        for r, e in pick:
            L.append(f"| `{r['id']}` | {md(r['transcription'])} | {md(r['text_v3'])} | {md(e['evidence'])} |")

    L += ["", "## 2. Rules that removed auto tags", "",
          "- **R1_uncorroborated**: the tag's family (sigh for `[sigh]`, inhale for `[gasp]`) has no kana in the "
          "same sentence (bounded by 。！？), and anime-whisper does not hear more kana of that family than the "
          "reference contains.",
          "- **R2_only_articulated_kana**: the only same-family kana in the sentence is articulated "
          "(はあ？ / はぁ！ = 'huh?/what!'), which is a word, not a breath.",
          "- Kana families: laugh = ふふ(っ), うふふ, あはは, ははは, えへへ, くすっ, くすくす, ぷっ; "
          "sigh = はぁ/はあ, ふぅ/ふう(っ), はー, ふー (not followed by ？/！); inhale = はっ, ひっ, きゃっ. "
          "Every unit must be a whole word (no kana, kanji or Latin letter directly before or after it). "
          "すぅ and ふっ are deliberately excluded as ambiguous.", ""]
    by_rule = defaultdict(list)
    for a in removed_auto:
        by_rule[a["rule"]].append(a)
    for rule, items in sorted(by_rule.items()):
        L += [f"### {rule}: {len(items)} tags removed", "",
              "| id | tag | transcription | asr_anime | why |", "|---|---|---|---|---|"]
        for a in items:
            L.append(f"| `{a['id']}` | `{a['tag']}` | {md(a['transcription'])} | {md(a['asr_anime'])} | {md(a['why'])} |")
        L.append("")

    thr = clf["threshold"]
    an = [r for r in rows if r["medium"] == "anime" and r["variant"] == "amadeus" and r["id"] in clf["anime_ids_scored"]]
    above = sorted([r for r in an if r["amadeus_score"] > thr], key=lambda r: r["amadeus_score"])[:10]
    below = sorted([r for r in an if r["amadeus_score"] <= thr], key=lambda r: -r["amadeus_score"])[:10]
    L += ["## 3. Anime Amadeus clips nearest the routing threshold", "",
          f"Threshold {thr:.4f}: at most 10% of SG0 anime-Kurisu out-of-fold scores reach it (model trained on SG0 "
          "anime Amadeus vs SG0 anime Kurisu; Cognitive Computing clips scored by the full SG0 model). "
          f"{sum(r['amadeus_score'] > thr for r in an)} anime Amadeus clips are above it (row `amadeus_anime`), "
          f"{sum(r['amadeus_score'] <= thr for r in an)} below (row `kurisu`). The 10 closest on each side, "
          "all rows including dropped ones:", ""]
    for name, sel in (("Above (routed to amadeus_anime)", above), ("Below (routed to kurisu)", below)):
        L += [f"### {name}", "", "| id | score | keep | reason | series | text_v3 |", "|---|---|---|---|---|---|"]
        for r in sel:
            L.append(f"| `{r['id']}` | {r['amadeus_score']:.4f} | {r['keep']} | {r['reason'] or ''} | "
                     f"{r['series']} | {md(r['text_v3'])} |")
        L.append("")
    (OUT / "review_examples.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def md(s) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
