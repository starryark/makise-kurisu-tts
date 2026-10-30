"""Automatic accent metrics for cross-lingual (Mandarin / English) TTS output.

See research/crosslingual_accent_2026-10.md sections 6 and 9.

  M1  language-ID log-ratio (VoxLingua107 ECAPA, speechbrain):
        zh clips: lid_ratio = log P(ja) - log P(zh); en clips: log P(ja) - log P(en);
        ja clips: log P(ja) - log P(zh) (reported for reference only).
        Also P(target) and the top-1 language. Higher lid_ratio = more Japanese.
  M2  toneless syllable error rate (SER), Mandarin only. charsiu text-free frame
        classifier (no LM): frame argmax -> collapse runs -> drop silence ->
        initial+final syllables -> Levenshtein against pypinyin reference.
        Also SER on the shared-character subset (characters with a Japanese kanji
        counterpart whose reading differs), initial / final error rates.
  M3  tone error rate (TER) over aligned syllables whose toneless syllable
        matches, plus a 5x5 tone confusion matrix (rows = reference tone after
        sandhi, cols = recognised tone).
  M5  speaker similarity: optional hook (pass embed_fn + ref_embedding).

Calibration: calibrate({group: [values]}) -> Cohen's d and ordering check.

CLI:
  python accent_metrics.py --wav-dir DIR --manifest rows.jsonl [--out-dir DIR]
  rows: {"file": "x.wav", "text": "...", "lang": "zh"|"en"|"ja",
         optional "pid", "category", "group", "shared_positions": [char idx]}
  writes OUT/scored_accent.jsonl and OUT/summary_accent.json

Runtime target: Python 3.11, torch 2.8, transformers 4.57.3 (pinned by
qwen-tts 0.1.1), speechbrain >= 1.1.2 (1.1.0 has a Windows-only lazy-import bug),
pypinyin >= 0.48, scipy, soundfile. Model revisions are pinned below.
Environment overrides: ACCENT_HF_CACHE (HF cache dir), ACCENT_LID_DIR,
ACCENT_CHARSIU_DIR, ACCENT_CHARSIU_TOK_DIR (local snapshot dirs, offline use).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Pinned models
# ---------------------------------------------------------------------------
LID_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
LID_REV = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"          # 2024-11-27
CHARSIU_REPO = "charsiu/zh_xlsr_fc_10ms"
CHARSIU_REV = "2d274d10cce1dd66818c0706abbc4cc20d4ef710"      # 2021-12-16
CHARSIU_TOK_REPO = "charsiu/tokenizer_zh_pinyin"
CHARSIU_TOK_REV = "a370e90191d18be4035783a7b98c509666e5c058"  # 2021-12-16

# Exact strings from the model's label_encoder.txt (verified 2026-10-28):
# 'en: English' => 20, 'ja: Japanese' => 45, 'zh: Chinese' => 106 (107 labels).
LID_LABELS = {"ja": "ja: Japanese", "zh": "zh: Chinese", "en": "en: English"}
LID_TARGET_VS = {"zh": "zh", "en": "en", "ja": "zh"}  # denominator of the ratio

SR = 16000
_models: Dict[str, object] = {}


def _cache_dir() -> Optional[str]:
    return os.environ.get("ACCENT_HF_CACHE") or None


def _device(device: Optional[str] = None) -> str:
    if device:
        return device
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def load_audio_16k(path: str) -> np.ndarray:
    """Mono float32 at 16 kHz (polyphase resampling; no torchaudio dependency)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


# ---------------------------------------------------------------------------
# M1: language ID
# ---------------------------------------------------------------------------
def _lid_model(device: Optional[str] = None):
    key = "lid"
    if key in _models:
        return _models[key]
    from huggingface_hub import snapshot_download
    from speechbrain.inference.classifiers import EncoderClassifier
    local = os.environ.get("ACCENT_LID_DIR") or snapshot_download(
        LID_REPO, revision=LID_REV, cache_dir=_cache_dir(),
        allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"])
    kwargs = {}
    if platform.system() == "Windows":
        # copying avoids symlink failures on Windows. Not elsewhere: speechbrain 1.0.3's
        # COPY path raises on the optional custom.py this repo does not have, while the
        # default symlink path tolerates it.
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
    savedir = Path(_cache_dir() or Path.home() / ".cache") / ("sb_lid_" + LID_REV[:8])
    # pretrained_path is overridden so the pretrainer reads the pinned snapshot
    # rather than re-fetching the files from the hub's default branch.
    clf = EncoderClassifier.from_hparams(
        source=str(local), savedir=str(savedir), run_opts={"device": _device(device)},
        overrides={"pretrained_path": str(local).replace("\\", "/")}, **kwargs)
    clf.eval()
    lab2ind = clf.hparams.label_encoder.lab2ind
    missing = [v for v in LID_LABELS.values() if v not in lab2ind]
    if missing:
        raise RuntimeError(f"LID labels {missing} not in label encoder; e.g. {list(lab2ind)[:5]}")
    _models[key] = (clf, {k: int(lab2ind[v]) for k, v in LID_LABELS.items()},
                    {int(i): l for l, i in lab2ind.items()})
    return _models[key]


def lid_scores(wav16: np.ndarray, lang: str, device: Optional[str] = None) -> dict:
    import torch
    clf, idx, ind2lab = _lid_model(device)
    with torch.no_grad():
        logp, _, top, _ = clf.classify_batch(torch.from_numpy(wav16).float()[None])
    logp = logp[0].float().cpu().numpy()  # log posteriors (sum exp = 1)
    vs = LID_TARGET_VS.get(lang, "zh")
    tgt = lang if lang in idx else "zh"
    top_lab = ind2lab[int(top[0])]
    return {
        "lid_ratio": round(float(logp[idx["ja"]] - logp[idx[vs]]), 4),
        "lid_ratio_vs": vs,
        "logp_ja": round(float(logp[idx["ja"]]), 4),
        "logp_zh": round(float(logp[idx["zh"]]), 4),
        "logp_en": round(float(logp[idx["en"]]), 4),
        "p_target": round(float(np.exp(logp[idx[tgt]])), 5),
        "lid_top1": top_lab.split(":")[0],
        "lid_top1_label": top_lab,
    }


# ---------------------------------------------------------------------------
# M2/M3: charsiu frame classifier
# ---------------------------------------------------------------------------
def _charsiu_class():
    """Wav2Vec2ForFrameClassification, vendored from charsiu.

    Source: https://github.com/lingjzhu/charsiu  src/models.py  (commit 13a69f2,
    2022-09-18). MIT License, Copyright (c) 2021 jzhu. This is charsiu's own class
    (Wav2Vec2ForCTC backbone + lm_head applied per 10 ms frame, no CTC blank); it
    is NOT transformers' Wav2Vec2ForAudioFrameClassification. Training-only loss
    code is omitted.
    """
    from transformers import Wav2Vec2ForCTC
    from transformers.modeling_outputs import CausalLMOutput

    class Wav2Vec2ForFrameClassification(Wav2Vec2ForCTC):
        def forward(self, input_values, attention_mask=None, output_attentions=None,
                    output_hidden_states=None, return_dict=None, labels=None):
            outputs = self.wav2vec2(input_values, attention_mask=attention_mask,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=True)
            hidden_states = self.dropout(outputs[0])
            logits = self.lm_head(hidden_states)
            return CausalLMOutput(loss=None, logits=logits,
                                  hidden_states=outputs.hidden_states,
                                  attentions=outputs.attentions)

    return Wav2Vec2ForFrameClassification


def _charsiu(device: Optional[str] = None):
    if "charsiu" in _models:
        return _models["charsiu"]
    from transformers import Wav2Vec2CTCTokenizer
    cls = _charsiu_class()
    src = os.environ.get("ACCENT_CHARSIU_DIR") or CHARSIU_REPO
    rev = None if os.environ.get("ACCENT_CHARSIU_DIR") else CHARSIU_REV
    model, info = cls.from_pretrained(src, revision=rev, cache_dir=_cache_dir(),
                                      output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"charsiu weights incomplete: {info}")
    model = model.eval().to(_device(device))
    tsrc = os.environ.get("ACCENT_CHARSIU_TOK_DIR") or CHARSIU_TOK_REPO
    trev = None if os.environ.get("ACCENT_CHARSIU_TOK_DIR") else CHARSIU_TOK_REV
    tok = Wav2Vec2CTCTokenizer.from_pretrained(tsrc, revision=trev, cache_dir=_cache_dir())
    # vocab.json has 210 entries (ids 0..209); the tokenizer also adds <s>/</s>
    # as extra special tokens (ids 210, 211) that the model never outputs.
    nv = model.config.vocab_size
    id2tok = {i: tok.convert_ids_to_tokens(i) for i in range(nv)}
    if len(set(id2tok.values())) != nv or id2tok[1] != "[SIL]" or id2tok[0] != "[PAD]":
        raise RuntimeError("charsiu tokenizer / model vocab mismatch")
    _models["charsiu"] = (model, id2tok)
    return _models["charsiu"]


def charsiu_logits(wav16: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """[frames, 210] logits at 10 ms. Normalisation = charsiu's feature extractor
    (do_normalize=True: zero mean / unit variance per utterance)."""
    import torch
    model, _ = _charsiu(device)
    x = (wav16 - wav16.mean()) / np.sqrt(wav16.var() + 1e-7)
    with torch.no_grad():
        inp = torch.from_numpy(x.astype(np.float32))[None].to(next(model.parameters()).device)
        logits = model(inp).logits[0].float().cpu().numpy()
    return logits


# charsiu label inventory (tokenizer_zh_pinyin vocab.json, 210 entries):
# [PAD]=0 [SIL]=1 [UNK]=2, 21 initials, 'rr' (r-colouring), and 37 finals x tones 1-5.
INITIALS = ("b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
            "zh", "ch", "sh", "r", "z", "c", "s")
FINALS = ("a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian",
          "iang", "iao", "ie", "ii", "iii", "in", "ing", "iong", "iou", "o", "ong", "ou",
          "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v", "van", "ve", "vn")
_FINAL_RE = re.compile(r"^([a-z]+)([1-5])$")


def decode_frames(ids: Sequence[int], id2tok: Dict[int, str], min_run: int = 2,
                  sil_boundary: int = 4, keep_orphans: bool = False) -> List[dict]:
    """Frame labels -> syllables [{ini, fin, tone, start, end}] (frames of 10 ms).

    1. run-length encode the argmax labels;
    2. drop non-silence runs shorter than `min_run` frames (flicker) and merge
       equal neighbours that the flicker had split;
    3. [SIL]/[PAD]/[UNK] runs separate syllables; a run of >= `sil_boundary`
       frames also flushes a dangling initial;
    4. initial + final -> syllable; a final without initial -> zero-initial
       syllable; 'rr' after a zero-initial 'e' gives 'er', otherwise it is
       dropped (r-colouring). An initial with no final ("orphan", fin='') is
       dropped unless keep_orphans.
    Defaults (min_run=2, orphans dropped) gave the lowest SER on 111 native
    Chinese-dub clips (min_run 1/2/3: 0.322/0.269/0.317; dropping orphans
    0.269 -> 0.242) and lowered SER for every TTS group too.
    Known limitation: there is no blank label, so two adjacent syllables with the
    same toned final and a zero initial (记忆 ji4 yi4) merge into one run.
    """
    runs: List[List] = []  # [label, start, length]
    for t, i in enumerate(ids):
        lab = id2tok[int(i)]
        if runs and runs[-1][0] == lab:
            runs[-1][2] += 1
        else:
            runs.append([lab, t, 1])
    if min_run > 1:
        kept: List[List] = []
        for r in runs:
            is_sil = r[0] in ("[SIL]", "[PAD]", "[UNK]")
            if not is_sil and r[2] < min_run:
                continue
            if kept and kept[-1][0] == r[0]:
                kept[-1][2] = r[1] + r[2] - kept[-1][1]
            else:
                kept.append(list(r))
        runs = kept
    syls: List[dict] = []
    pend = None  # pending initial run
    for lab, st, n in runs:
        if lab in ("[SIL]", "[PAD]", "[UNK]"):
            if n >= sil_boundary and pend is not None:
                syls.append({"ini": pend[0], "fin": "", "tone": 0, "start": pend[1], "end": pend[1] + pend[2]})
                pend = None
            continue
        if lab in INITIALS:
            if pend is not None:
                syls.append({"ini": pend[0], "fin": "", "tone": 0, "start": pend[1], "end": pend[1] + pend[2]})
            pend = (lab, st, n)
            continue
        if lab == "rr":
            if syls and syls[-1]["ini"] == "" and syls[-1]["fin"] == "e" and pend is None:
                syls[-1]["fin"] = "er"
                syls[-1]["end"] = st + n
            continue
        m = _FINAL_RE.match(lab)
        if not m:
            continue
        fin, tone = m.group(1), int(m.group(2))
        if pend is not None:
            syls.append({"ini": pend[0], "fin": fin, "tone": tone, "start": pend[1], "end": st + n})
            pend = None
        else:
            syls.append({"ini": "", "fin": fin, "tone": tone, "start": st, "end": st + n})
    if pend is not None:
        syls.append({"ini": pend[0], "fin": "", "tone": 0, "start": pend[1], "end": pend[1] + pend[2]})
    if not keep_orphans:
        syls = [s for s in syls if s["fin"]]
    return syls


def count_orphans(ids: Sequence[int], id2tok: Dict[int, str], min_run: int = 2) -> int:
    return sum(1 for s in decode_frames(ids, id2tok, min_run=min_run, keep_orphans=True) if not s["fin"])


# ---------------------------------------------------------------------------
# Reference pinyin
# ---------------------------------------------------------------------------
_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_TAG_RE = re.compile(r"\[[^\]]*\]")
_DIG = "零一二三四五六七八九"
_ZERO_INI = {
    "yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iou", "yan": "ian", "yin": "in",
    "yang": "iang", "ying": "ing", "yong": "iong", "yo": "o", "yu": "v", "yue": "ve",
    "yuan": "van", "yun": "vn", "wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "ui",
    "wan": "uan", "wen": "un", "wang": "uang", "er": "er", "r": "er",
    # syllabic nasals (嗯 n2, 呣 m2) have no charsiu label; nearest final
    "n": "en", "ng": "eng", "m": "en", "hm": "en", "hng": "eng",
}


def pinyin_to_charsiu(py: str) -> List[Tuple[str, str]]:
    """Toneless pypinyin (TONE3 body, 'v' for ü) -> [(initial, final)] in charsiu's
    inventory, following charsiu processors.py (transform_dict / rhyme_mapping)."""
    s = py.lower().replace("ü", "v").replace("u:", "v")
    if s == "weng":            # charsiu splits ueng into u + eng
        return [("", "u"), ("", "eng")]
    if s in _ZERO_INI:
        return [("", _ZERO_INI[s])]
    if s[:2] in ("zh", "ch", "sh"):
        ini, rest = s[:2], s[2:]
    elif s[:1] in INITIALS:
        ini, rest = s[:1], s[1:]
    else:
        return [("", s)]
    if rest == "i" and ini in ("zh", "ch", "sh", "r"):
        rest = "iii"
    elif rest == "i" and ini in ("z", "c", "s"):
        rest = "ii"
    elif ini in ("j", "q", "x") and rest.startswith("u"):
        rest = "v" + rest[1:]
    if rest == "iu":
        rest = "iou"
    elif rest == "ue":
        rest = "ve"
    return [(ini, rest)]


def _int_to_zh(n: int) -> str:
    if n == 0:
        return "零"

    def below_10k(x: int) -> str:
        out, zero = "", False
        for p, u in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            d = x // p % 10
            if d == 0:
                if out:
                    zero = True
            else:
                if zero:
                    out += "零"
                    zero = False
                out += _DIG[d] + u
        return out

    yi, rem = divmod(n, 10 ** 8)
    wan, rem = divmod(rem, 10 ** 4)
    s = ""
    if yi:
        s += below_10k(yi) + "亿"
    if wan:
        if yi and wan < 1000:
            s += "零"
        s += below_10k(wan) + "万"
    if rem:
        if (yi or wan) and rem < 1000:
            s += "零"
        s += below_10k(rem)
    if s.startswith("一十"):
        s = s[1:]
    return s


def _digits(s: str) -> str:
    return "".join(_DIG[int(c)] for c in s)


_CHAR_SUB = {"℃": "度"}  # NFKC would give "°C" (Latin C)
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)(%)|(\d{1,2}):(\d{2})|(\d+)\.(\d+)|(\d+)")
_LIANG_NEXT = set("个次天位种本张只件倍台块条辆名小")  # 2小时 -> 两小时


def _num_reading(m: "re.Match", nxt: str) -> str:
    if m.group(2):  # percent
        a = m.group(1)
        body = _num_reading(_NUM_RE.match(a), "")
        return "百分之" + body
    if m.group(3) is not None:  # 3:20
        h, mi = int(m.group(3)), m.group(4)
        out = ("两" if h == 2 else _int_to_zh(h)) + "点"
        if mi != "00":
            out += ("零" + _DIG[int(mi[1])]) if mi[0] == "0" else _int_to_zh(int(mi))
            out += "分"
        return out
    if m.group(5) is not None:  # decimal
        return _int_to_zh(int(m.group(5))) + "点" + _digits(m.group(6))
    s = m.group(7)
    if nxt == "年" and len(s) == 4:
        return _digits(s)
    if (len(s) > 1 and s[0] == "0") or len(s) > 12:
        return _digits(s)
    if s == "2" and (nxt in _LIANG_NEXT or nxt == "点"):
        return "两"
    return _int_to_zh(int(s))


def normalize_zh(text: str) -> Tuple[List[str], List[int], List[bool], bool]:
    """Text -> (chars, source index per char, from_number flag, has_latin).

    Strips [tags], applies NFKC per character (indices keep pointing into the
    original string) and spells digits as hanzi the way they are read
    (3点20分 -> 三点二十分, 1.5倍 -> 一点五倍, 2026年 -> 二零二六年, 50% -> 百分之五十).
    """
    chars: List[str] = []
    src: List[int] = []
    i = 0
    tag_spans = [(m.start(), m.end()) for m in _TAG_RE.finditer(text)]
    skip = set()
    for a, b in tag_spans:
        skip.update(range(a, b))
    for i, ch in enumerate(text):
        if i in skip:
            continue
        for c in unicodedata.normalize("NFKC", _CHAR_SUB.get(ch, ch)):
            chars.append(c)
            src.append(i)
    s = "".join(chars)
    out_c: List[str] = []
    out_i: List[int] = []
    out_n: List[bool] = []
    pos = 0
    for m in _NUM_RE.finditer(s):
        for k in range(pos, m.start()):
            out_c.append(s[k]); out_i.append(src[k]); out_n.append(False)
        nxt = s[m.end()] if m.end() < len(s) else ""
        rd = _num_reading(m, nxt)
        for k, c in enumerate(rd):
            # map each spoken char to a source char of the number token
            j = m.start() + min(k * (m.end() - m.start()) // max(len(rd), 1), m.end() - m.start() - 1)
            out_c.append(c); out_i.append(src[j]); out_n.append(True)
        pos = m.end()
    for k in range(pos, len(s)):
        out_c.append(s[k]); out_i.append(src[k]); out_n.append(False)
    has_latin = any(("a" <= c.lower() <= "z") for c in out_c)
    return out_c, out_i, out_n, has_latin


_NUMERAL_CHARS = set("零一二三四五六七八九十百千万亿两")
# colloquial readings accepted as correct toneless syllables
_ALT_READINGS = {"谁": ("shei", "shui"), "这": ("zhe", "zhei"), "那": ("na", "nei"), "哪": ("na", "nei")}


def _yi_tone(chars: List[str], tones: List[int], k: int, is_num: List[bool]) -> int:
    prev = chars[k - 1] if k > 0 else ""
    nxt = chars[k + 1] if k + 1 < len(chars) else ""
    if k + 1 >= len(chars):
        return 1                                  # phrase-final
    if prev == "第":
        return 1                                  # ordinal
    if prev and prev == nxt and prev not in _NUMERAL_CHARS and (k < 2 or chars[k - 2] != "一"):
        return 5                                  # 看一看 (not 一步一步)
    if nxt in "百千万亿":
        return 2 if tones[k + 1] == 4 else 4      # 一百 yi4 bai3, 一万 yi2 wan4
    if is_num[k] or prev in _NUMERAL_CHARS or nxt in _NUMERAL_CHARS:
        return 1                                  # digit reading: 十一, 一点五, 二零一六
    if nxt in "月号" and k == 0:
        return 1                                  # 一月 (January)
    t = tones[k + 1]
    return 2 if t in (4, 5) else 4


_DU_PREV = set("首成京古国建定迁故旧帝陪")      # 首都 成都 京都 ... : du1
_DU_NEXT = set("市城督")                          # 都市 都城 都督 : du1
_DI_PREV = set("块片这那此该每")                  # 这块地 那片地 : noun di4


def _context_fix(pc: List[str], bodies: List[Optional[str]], lex: List[int]) -> None:
    """In-place corrections of two pypinyin phrase-segmentation errors found on the v3
    probes (review 2026-10-28):
    - adverbial 地 (认真地 不断地 乖乖地 一步一步地) came out di4; it is de5 unless 地
      forms a dictionary word with a neighbour (地方 当地 土地) or ends the phrase (noun);
    - 大家都会 was segmented as 都会 'metropolis' (du1); 都 is dou1 except in place-name
      compounds (首都 成都 京都 / 都市 都城)."""
    try:
        from pypinyin.constants import PHRASES_DICT
    except Exception:                                   # pragma: no cover
        PHRASES_DICT = {}
    n = len(pc)
    for j, c in enumerate(pc):
        if c == "地" and bodies[j] == "di" and 0 < j < n - 1:
            if (pc[j - 1] + c not in PHRASES_DICT and c + pc[j + 1] not in PHRASES_DICT
                    and pc[j - 1] not in _DI_PREV):
                bodies[j], lex[j] = "de", 5
        elif c == "都" and bodies[j] == "du":
            prev_du = j > 0 and pc[j - 1] in _DU_PREV
            next_du = j + 1 < n and pc[j + 1] in _DU_NEXT
            if not (prev_du or next_du):
                bodies[j], lex[j] = "dou", 1


def zh_reference(text: str, accept_lexical: bool = False) -> Tuple[List[dict], dict]:
    """Reference syllables for a Mandarin sentence.

    Each item: {char, pos (index in `text`), ini, fin, tone (after sandhi),
    lex (lexical tone), ok (accepted tones)}.

    Tone sandhi is applied here per phrase (punctuation splits phrases):
      不 -> 2 before tone 4, else 4;  一 -> 2 before tone 4, 4 before 1/2/3,
      1 when final, ordinal or part of a digit string, 5 in V一V;
      3-3: in a run of n >= 2 third tones the first n-1 become 2. In runs of
      n >= 3 the outcome depends on prosodic grouping (老|保管 vs 保管|好), so
      every non-final syllable of such a run also accepts tone 3 in `ok`.
    Colloquial alternatives (谁 shei/shui, 这 zhe/zhei, 那/哪 na/nei) are
    accepted for the toneless syllable via `alts`.
    pypinyin >= 0.48 has `tone_sandhi=True`, but it works per segmented word and
    misses cross-word 3-3 chains and several 一 cases, so it is not used.
    `ok` holds the sandhi tone and, if accept_lexical, also the lexical tone.
    charsiu's training labels came from G2pM (no sandhi), but on 111 native
    Chinese-dub clips it reported the surface tone: 3-3 sandhi syllables came out
    tone 2 in 45/56 aligned cases (tone 3 in 6), 不 before tone 4 as bu2 in 19/23.
    So the default scores against the sandhi tone (accept_lexical=False).
    """
    from pypinyin import Style, lazy_pinyin
    chars, src, isnum, has_latin = normalize_zh(text)
    out: List[dict] = []
    phrase: List[int] = []
    n_unknown = 0

    def flush(idx: List[int]):
        nonlocal n_unknown
        if not idx:
            return
        han = "".join(chars[k] for k in idx)
        pys = lazy_pinyin(han, style=Style.TONE3, neutral_tone_with_five=True)
        if len(pys) != len(idx):
            pys = [lazy_pinyin(chars[k], style=Style.TONE3, neutral_tone_with_five=True)[0] for k in idx]
        bodies, lex = [], []
        for p in pys:
            m = re.match(r"^([a-zü:]+)([1-5])$", p)
            if m:
                bodies.append(m.group(1)); lex.append(int(m.group(2)))
            else:
                bodies.append(None); lex.append(0)
        pc = [chars[k] for k in idx]
        pn = [isnum[k] for k in idx]
        _context_fix(pc, bodies, lex)
        lexical = list(lex)
        for j, c in enumerate(pc):
            if c == "一":
                lexical[j] = 1
            elif c == "不":
                lexical[j] = 4
        tones = list(lexical)
        for j, c in enumerate(pc):
            if c == "不":
                if j + 1 < len(pc) and j > 0 and pc[j - 1] == pc[j + 1]:
                    tones[j] = 5                       # 好不好
                elif j + 1 < len(pc):
                    tones[j] = 2 if lexical[j + 1] == 4 else 4
            elif c == "一":
                tones[j] = _yi_tone(pc, lexical, j, pn)
        j = 0
        loose = set()
        while j < len(tones):                          # 3-3 sandhi chains
            if tones[j] == 3:
                e = j
                while e + 1 < len(tones) and tones[e + 1] == 3:
                    e += 1
                for q in range(j, e):
                    tones[q] = 2
                    if e - j >= 2:
                        loose.add(q)
                j = e + 1
            else:
                j += 1
        for j, k in enumerate(idx):
            if bodies[j] is None:
                n_unknown += 1
                continue
            ok = {tones[j]}
            if accept_lexical:
                ok.add(lexical[j])
            if j in loose:
                ok.add(3)
            alt_bodies = [bodies[j]] + [a for a in _ALT_READINGS.get(chars[k], ()) if a != bodies[j]]
            pieces = pinyin_to_charsiu(bodies[j])
            for ini, fin in pieces:
                alts = sorted({_syl_str(*pinyin_to_charsiu(a)[0]) for a in alt_bodies})                     if len(pieces) == 1 else [_syl_str(ini, fin)]
                out.append({"char": chars[k], "pos": src[k], "ini": ini, "fin": fin,
                            "tone": tones[j], "lex": lexical[j], "ok": sorted(ok), "alts": alts})

    for k, c in enumerate(chars):
        if _HAN_RE.match(c):
            phrase.append(k)
        else:
            flush(phrase)
            phrase = []
    flush(phrase)
    # adjacent same-final zero-initial pairs merge in the frame decoder
    merge_risk = sum(1 for a, b in zip(out, out[1:])
                     if b["ini"] == "" and a["fin"] == b["fin"] and a["tone"] == b["tone"])
    return out, {"has_latin": has_latin, "n_unknown": n_unknown, "merge_risk": merge_risk}


# ---------------------------------------------------------------------------
# Shared-character heuristic (used when a row has no shared_positions)
# ---------------------------------------------------------------------------
# A character is "shared" when it has a commonly used Japanese kanji counterpart
# (same, traditional or shinjitai form: 时=時, 东=東, 验=験) whose Japanese reading
# differs from Mandarin. NOT_SHARED lists Chinese-only function characters and
# characters without common Japanese use (我 是 只 are marginal in modern Japanese),
# plus near-homophones (三 san, 安 an, 看 kan ...). Grammatical particles
# 的/了/着/过/地/得 are treated as not shared. This is a heuristic; probe rows
# carry explicit shared_positions generated with the same rule.
NOT_SHARED = set(
    "的了着过地得吗呢吧啊呀嗯哦噢嘛啦哇哈呵喂哎唉欸诶咦哟呐呗哼嘿喔咯嘞"
    "么们这那哪谁你您她它咱俺啥怎咋俩些我是只"
    "很也还把跟吃喝找拿玩聊帮搞弄吵闹乖傻笨棒酷帅呆挺甭瞧瞅扔挤躲趟啰嗦"
    "三安看他太在拉卡沙巴"
)


def default_shared_positions(text: str) -> List[int]:
    """Indices of shared hanzi, plus digits and the . % : inside numbers (digits
    are read in the reference language, Qwen #68/#151, fish-speech #1263)."""
    t = unicodedata.normalize("NFKC", text)
    if len(t) != len(text):
        t = "".join(unicodedata.normalize("NFKC", c)[:1] or c for c in text)
    out = []
    for i, c in enumerate(text):
        n = t[i]
        if _HAN_RE.match(c) and c not in NOT_SHARED:
            out.append(i)
        elif n.isdigit():
            out.append(i)
        elif n in ".%:" and ((i > 0 and t[i - 1].isdigit()) or (i + 1 < len(t) and t[i + 1].isdigit())):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Alignment and scoring
# ---------------------------------------------------------------------------
def align(ref: Sequence, hyp: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Levenshtein alignment -> [(op, i_ref, j_hyp)], op in M S D I (-1 = none).
    A ref item may be a str or a set/frozenset of acceptable strings."""
    ref = [r if isinstance(r, (set, frozenset)) else frozenset([r]) for r in ref]
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if hyp[j - 1] in ref[i - 1] else 1
            d[i, j] = min(d[i - 1, j - 1] + c, d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if hyp[j - 1] in ref[i - 1] else 1):
            ops.append(("M" if hyp[j - 1] in ref[i - 1] else "S", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", i - 1, -1)); i -= 1
        else:
            ops.append(("I", -1, j - 1)); j -= 1
    return ops[::-1]


def _syl_str(ini: str, fin: str) -> str:
    return f"{ini}-{fin}"


def score_zh(ref: List[dict], hyp: List[dict], shared_positions: Optional[Iterable[int]] = None) -> dict:
    r_s = [frozenset(r.get("alts") or [_syl_str(r["ini"], r["fin"])]) for r in ref]
    h_s = [_syl_str(h["ini"], h["fin"]) for h in hyp]
    ops = align(r_s, h_s)
    S = sum(o[0] == "S" for o in ops)
    D = sum(o[0] == "D" for o in ops)
    I = sum(o[0] == "I" for o in ops)
    N = len(r_s)
    shared = set(shared_positions) if shared_positions is not None else None
    sh_n = sh_err = 0
    ini_err = fin_err = n_pairs = 0
    conf = np.zeros((5, 5), dtype=int)
    n_tone = tone_err = tone_err_strict = 0
    for op, i, j in ops:
        if i >= 0 and shared is not None and ref[i]["pos"] in shared:
            sh_n += 1
            sh_err += op in ("S", "D")
        if op in ("M", "S"):
            n_pairs += 1
            if op == "S":
                ini_err += ref[i]["ini"] != hyp[j]["ini"]
                fin_err += ref[i]["fin"] != hyp[j]["fin"]
        if op == "M":
            rt, ht = ref[i]["tone"], hyp[j]["tone"]
            if not (1 <= ht <= 5):
                continue
            n_tone += 1
            conf[rt - 1, ht - 1] += 1
            lenient_ok = ht in ref[i]["ok"] or ht == 5 or rt == 5
            tone_err += not lenient_ok
            tone_err_strict += ht != rt
    return {
        "n_ref_syl": N, "n_hyp_syl": len(h_s),
        "ser": round((S + D + I) / N, 4) if N else None,
        "ser_S": S, "ser_D": D, "ser_I": I,
        "n_shared": sh_n if shared is not None else None,
        "shared_err": sh_err if shared is not None else None,
        "ser_shared": round(sh_err / sh_n, 4) if shared is not None and sh_n else None,
        "ini_err": ini_err, "fin_err": fin_err, "n_pairs": n_pairs,
        "n_tone": n_tone, "tone_err": tone_err, "tone_err_strict": tone_err_strict,
        "ter": round(tone_err / n_tone, 4) if n_tone else None,
        "ter_strict": round(tone_err_strict / n_tone, 4) if n_tone else None,
        "tone_confusion": conf.tolist(),
        "align": " ".join(o[0] for o in ops),
    }


def fmt_ref(ref: List[dict]) -> str:
    return " ".join(f"{r['ini']}{r['fin']}{r['tone']}" for r in ref)


def fmt_hyp(hyp: List[dict]) -> str:
    return " ".join(f"{h['ini']}{h['fin']}{h['tone'] or ''}" for h in hyp)


# ---------------------------------------------------------------------------
# M5 hook
# ---------------------------------------------------------------------------
def speaker_similarity(paths: Sequence[str], embed_fn: Callable[[List[str]], np.ndarray],
                       ref_embedding: np.ndarray) -> List[float]:
    """Cosine of embed_fn(paths) rows against ref_embedding (e.g. metrics.embed and
    metrics.centroid from the v2 confirm-suite, so M5 matches that encoder)."""
    e = np.asarray(embed_fn(list(paths)), dtype=np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    r = np.asarray(ref_embedding, dtype=np.float64)
    r = r / (np.linalg.norm(r) + 1e-9)
    return [round(float(v), 4) for v in e @ r]


# ---------------------------------------------------------------------------
# Per-clip and per-directory scoring
# ---------------------------------------------------------------------------
def score_clip(path: str, text: str, lang: str, shared_positions: Optional[Iterable[int]] = None,
               device: Optional[str] = None, do_lid: bool = True, do_zh: bool = True,
               min_run: int = 2, accept_lexical: bool = False) -> dict:
    wav = load_audio_16k(path)
    out: dict = {"dur": round(len(wav) / SR, 3)}
    t0 = time.perf_counter()
    if do_lid:
        out.update(lid_scores(wav, lang, device))
    t1 = time.perf_counter()
    if do_zh and lang == "zh":
        _, id2tok = _charsiu(device)
        logits = charsiu_logits(wav, device)
        ids = logits.argmax(-1)
        hyp = decode_frames(ids, id2tok, min_run=min_run)
        out["n_orphan_initials"] = count_orphans(ids, id2tok, min_run=min_run)
        ref, info = zh_reference(text, accept_lexical=accept_lexical)
        if shared_positions is None:
            shared_positions = default_shared_positions(text)
            out["shared_source"] = "heuristic"
        else:
            out["shared_source"] = "provided"
        out.update(score_zh(ref, hyp, shared_positions))
        out.update(info)
        out["ref_pinyin"] = fmt_ref(ref)
        out["hyp_pinyin"] = fmt_hyp(hyp)
        # M2/M3 are invalid when the reference cannot cover what is spoken
        out["m23_valid"] = bool(ref) and not info["has_latin"] and info["n_unknown"] == 0
    t2 = time.perf_counter()
    out["t_lid"] = round(t1 - t0, 4)
    out["t_zh"] = round(t2 - t1, 4)
    return out


def _mean_sd(v: List[float]) -> dict:
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": len(a), "mean": round(float(a.mean()), 4),
            "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}


def aggregate(rows: List[dict]) -> dict:
    out: dict = {"n": len(rows)}
    lid = [r for r in rows if "lid_ratio" in r]
    if lid:
        out["lid_ratio"] = _mean_sd([r["lid_ratio"] for r in lid])
        out["p_target"] = _mean_sd([r["p_target"] for r in lid])
        out["frac_p_target_lt_0.9"] = round(float(np.mean([r["p_target"] < 0.9 for r in lid])), 4)
        tops: Dict[str, int] = {}
        for r in lid:
            tops[r["lid_top1"]] = tops.get(r["lid_top1"], 0) + 1
        out["lid_top1"] = dict(sorted(tops.items(), key=lambda kv: -kv[1]))
    zh = [r for r in rows if r.get("m23_valid")]
    if zh:
        N = sum(r["n_ref_syl"] for r in zh)
        E = sum(r["ser_S"] + r["ser_D"] + r["ser_I"] for r in zh)
        out["n_m23"] = len(zh)
        out["ser_micro"] = round(E / N, 4) if N else None
        out["ser"] = _mean_sd([r["ser"] for r in zh])
        shn = sum(r["n_shared"] or 0 for r in zh)
        she = sum(r["shared_err"] or 0 for r in zh)
        out["ser_shared_micro"] = round(she / shn, 4) if shn else None
        out["ser_shared"] = _mean_sd([r["ser_shared"] for r in zh])
        npairs = sum(r["n_pairs"] for r in zh)
        out["ini_err_micro"] = round(sum(r["ini_err"] for r in zh) / npairs, 4) if npairs else None
        out["fin_err_micro"] = round(sum(r["fin_err"] for r in zh) / npairs, 4) if npairs else None
        nt = sum(r["n_tone"] for r in zh)
        out["ter_micro"] = round(sum(r["tone_err"] for r in zh) / nt, 4) if nt else None
        out["ter_strict_micro"] = round(sum(r["tone_err_strict"] for r in zh) / nt, 4) if nt else None
        out["ter"] = _mean_sd([r["ter"] for r in zh])
        out["ter_strict"] = _mean_sd([r["ter_strict"] for r in zh])
        out["tone_confusion"] = np.sum([r["tone_confusion"] for r in zh], axis=0).tolist()
        out["merge_risk_syl"] = int(sum(r.get("merge_risk", 0) for r in zh))
    t = [r.get("t_lid", 0) + r.get("t_zh", 0) for r in rows]
    if t:
        out["sec_per_clip"] = round(float(np.mean(t)), 4)
    return out


def score_rows(rows: List[dict], wav_dir: Path, device: Optional[str] = None,
               do_lid: bool = True, do_zh: bool = True, min_run: int = 2,
               accept_lexical: bool = False, embed_fn: Optional[Callable] = None,
               ref_embedding: Optional[np.ndarray] = None, verbose: bool = True) -> List[dict]:
    scored = []
    for k, r in enumerate(rows):
        p = Path(r["file"])
        p = p if p.is_absolute() else wav_dir / p
        if not p.is_file():
            if verbose:
                print(f"[accent] missing {p}")
            continue
        res = score_clip(str(p), r.get("text", ""), r["lang"], r.get("shared_positions"),
                         device=device, do_lid=do_lid, do_zh=do_zh, min_run=min_run,
                         accept_lexical=accept_lexical)
        row = dict(r)
        row.update(res)
        scored.append(row)
        if verbose and (k + 1) % 25 == 0:
            print(f"[accent] {k + 1}/{len(rows)}")
    if embed_fn is not None and ref_embedding is not None and scored:
        paths = [str(Path(r["file"]) if Path(r["file"]).is_absolute() else wav_dir / r["file"]) for r in scored]
        for r, s in zip(scored, speaker_similarity(paths, embed_fn, ref_embedding)):
            r["sim"] = s
    return scored


def summarize(scored: List[dict], group_key: str = "group") -> dict:
    summary: dict = {"all": aggregate(scored)}
    for key, prefix in ((group_key, "group"), ("lang", "lang"), ("category", "category")):
        vals = sorted({str(r.get(key)) for r in scored if r.get(key) is not None})
        for v in vals:
            summary[f"{prefix}:{v}"] = aggregate([r for r in scored if str(r.get(key)) == v])
    # LID ratios are logP(ja)-logP(zh) on zh rows but logP(ja)-logP(en) on en rows, so a
    # group pooled over languages is not interpretable; report group x language too.
    pairs = sorted({(str(r.get(group_key)), str(r.get("lang"))) for r in scored
                    if r.get(group_key) is not None and r.get("lang") is not None})
    for g, l in pairs:
        summary[f"{prefix_gl(group_key)}:{g}:lang:{l}"] = aggregate(
            [r for r in scored if str(r.get(group_key)) == g and str(r.get("lang")) == l])
    return summary


def prefix_gl(group_key: str) -> str:
    return "group" if group_key == "group" else group_key


def score_dir(wav_dir: Path, manifest: Path, out_dir: Optional[Path] = None, **kw) -> dict:
    wav_dir, manifest = Path(wav_dir), Path(manifest)
    out_dir = Path(out_dir) if out_dir else wav_dir
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = score_rows(rows, wav_dir, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scored_accent.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf-8")
    summary = summarize(scored)
    summary["config"] = config_info(kw.get("min_run", 2), kw.get("accept_lexical", False))
    (out_dir / "summary_accent.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    return summary


def config_info(min_run: int = 2, accept_lexical: bool = False) -> dict:
    return {"lid": f"{LID_REPO}@{LID_REV[:8]}", "lid_labels": LID_LABELS,
            "charsiu": f"{CHARSIU_REPO}@{CHARSIU_REV[:8]}",
            "charsiu_tokenizer": f"{CHARSIU_TOK_REPO}@{CHARSIU_TOK_REV[:8]}",
            "min_run": min_run, "accept_lexical": accept_lexical,
            "ter_rule": "ter: hyp == sandhi tone (or lexical if accept_lexical) or either side tone 5; ter_strict: hyp == sandhi tone",
            "lid_ratio_sign": "log P(ja) - log P(zh|en); higher = more Japanese"}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """(mean(a) - mean(b)) / pooled sd (ddof=1)."""
    a = np.asarray([x for x in a if x is not None], dtype=float)
    b = np.asarray([x for x in b if x is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def calibrate(groups: Dict[str, List[float]], native: Sequence[str] = ("D", "R"),
              accented: Sequence[str] = ("H",), order: Sequence[str] = ("D", "E", "H"),
              higher_is_more_accented: bool = True, d_min: float = 1.0) -> dict:
    """Validity check for one metric (section 9.4).

    groups: {name: per-clip values}. d is reported as (accented - native) / pooled
    sd after flipping the sign for metrics where lower = more accented (e.g.
    p_target), so d > 0 always means "the Japanese anchor looks more Japanese".
    Passes if every native-vs-accented d > d_min and the group means follow
    `order` (least -> most accented), using only the groups present.
    """
    sgn = 1.0 if higher_is_more_accented else -1.0
    stats = {g: _mean_sd(list(v)) for g, v in groups.items()}
    d = {}
    for a in accented:
        for n in native:
            if a in groups and n in groups:
                d[f"{a}_vs_{n}"] = round(sgn * cohens_d(groups[a], groups[n]), 3)
    present = [g for g in order if g in groups and stats[g].get("n", 0) > 0]
    means = [sgn * stats[g]["mean"] for g in present]
    ordering_ok = all(x < y for x, y in zip(means, means[1:])) if len(means) > 1 else None
    d_ok = bool(d) and all((v == v) and v > d_min for v in d.values())
    return {"stats": stats, "d": d, "order_checked": present, "ordering_ok": ordering_ok,
            "d_ok": d_ok, "passes": bool(d_ok and ordering_ok is not False),
            "sign": "d = (accented - native)/pooled sd, oriented so > 0 = accented anchor more Japanese"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path, help="jsonl rows {file, text, lang, ...}")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-lid", action="store_true")
    ap.add_argument("--no-zh", action="store_true")
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--accept-lexical", action="store_true",
                    help="also accept the lexical (pre-sandhi) tone on sandhi syllables")
    a = ap.parse_args(argv)
    s = score_dir(a.wav_dir, a.manifest, a.out_dir, device=a.device, do_lid=not a.no_lid,
                  do_zh=not a.no_zh, min_run=a.min_run, accept_lexical=a.accept_lexical)
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
