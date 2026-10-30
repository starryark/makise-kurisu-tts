"""Native Mandarin + English replay rows for the v3 fine-tune (recipe rank 2).

v3 mixes ~15% rows of native Chinese/English from *other* speakers into the Kurisu
Japanese fine-tune. Each row has its own speaker (the trainer averages x-vectors over
the clips listed in aux_speakers.json) and a language id, so "language id => native
phonetics" is learned independently of the speaker and the Base model's ZH/EN is kept
(research/finetune_methods_2026-10.md 4.6 + 9.2, crosslingual_accent_2026-10.md 8 opt. 4).
Sources, licences and how they were verified: kurisu_v2/v3/replay_notes.md.

Mandarin  AISHELL-3 (OpenSLR SLR93, Apache-2.0) from a per-file Hugging Face mirror
          pinned to a commit whose wavs and transcripts were checked byte-for-byte
          against the OpenSLR tarball. Only the clips used are fetched (parallel HTTP),
          each checked against the mirror's LFS sha256. Speakers come from the train
          subset (it carries prosody labels), age groups B/C (14-40 y), northern accent
          by default. AISHELL-3 recording gain varies by speaker (median active loudness
          -42..-22 LUFS), so 4 probe clips screen out speakers below -33 LUFS, who would
          land under -23 even at p03's +9 dB clamp (keeping ZH rows level with JA/EN rather
          than tying "quiet" to the language id). Text is the character tier of
          content.txt (pinyin dropped); prosody-label "$" (intonation-phrase boundary)
          becomes "，", and "。" ends the sentence ("？" after 吗).
English   LibriTTS-R (OpenSLR SLR141, CC BY 4.0) from the HF parquet conversion, pinned.
          Speakers whose rows fill a whole parquet row group are found from the footer
          statistics, so only those ~30 MB row groups are read. Gender comes from the
          corpus' speakers.tsv; samples the corpus authors list as failed restorations
          are excluded. A 1.5x speaker pool is ranked by pitch spread (F0 IQR) to prefer
          livelier readers; dialogue lines (quotes, !, ?) go first within a speaker.
Audio     decode -> mono -> 24 kHz (polyphase) -> p03_build.normalise_audio, imported
          rather than copied, so edges (0.12 s lead / 0.25 s tail, 5 ms fades) and level
          (-23 LUFS, gain clamped to [-6, +9] dB, peak -1 dBFS) are exactly v2's.
Drops     clipped source (a run of >= 3 samples at full scale), too quiet (< -35 LUFS
          before gain, i.e. it would land > 3 dB under target even at the +9 dB clamp), no
          speech, outside 2-12 s after trim and pad, implausible speaking rate (zh 1.5-8
          hanzi/s, en 1-5 words/s), bad transcript characters, all-caps English.
Re-runs   decisions.jsonl caches every clip's outcome; existing wavs are reused, known
          drops are not refetched, and the selection walk is deterministic. A re-run
          with the same flags rewrites identical rows from the caches without network.

Outputs in $KURISU_WORK/replay/: wavs/, replay_rows.jsonl, aux_speakers.json,
report.json, decisions.jsonl, cache/ (metadata, LibriTTS-R raw wavs of pooled speakers).
Network: anonymous only (no token is read or sent).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p03_build import (GAIN_MAX, GAIN_MIN, LEAD_PAD, PEAK_MAX, SR, TAIL_PAD,  # noqa: E402
                       TARGET_LUFS, normalise_audio, speech_bounds)

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
OUT = WORK / "replay"
WAVS = OUT / "wavs"
CACHE = OUT / "cache"

ZH_REPOS = {  # repo -> pinned commit (both verified, see replay_notes.md)
    "shenyunhang/AISHELL-3": "e1af0fa23933585f1844b3c07b71006f86589bab",  # all 88,035 wavs
    "AISHELL/AISHELL-3": "f20d5db4a31fe779ef07bb1af4ea92da5c786622",      # owner's upload, subset
}
EN_REPO, EN_REV = "mythicinfinity/libritts_r", "0d1718db351207d5a493135a62d5ed8cc4733830"
OPENSLR_141 = ("https://openslr.trmal.net/resources/141/", "https://openslr.elda.org/resources/141/",
               "https://www.openslr.org/resources/141/", "https://openslr.magicdatatech.com/resources/141/")
EN_DOC_MD5 = {  # OpenSLR md5sum.txt; the elda mirror serves a doc.tar.gz (d39d...) whose
    # speakers.tsv/SPEAKERS.txt/LICENSE.txt are identical and only NOTE.txt + macOS files differ
    "doc.tar.gz": ("dad90f1b357625ce93d695b930977f7f", "d39d343743b31ac5d70eadea6987ee67"),
    "libritts_r_failed_speech_restoration_examples.tar.gz": ("43b616c18f317893cdf113af4d968de8",),
}
LICENSE = {"chinese": "Apache-2.0", "english": "CC-BY-4.0"}

MIN_DUR, MAX_DUR = 2.0, 12.0            # after trim + pad, i.e. what trains
QUIET_LUFS = TARGET_LUFS - GAIN_MAX - 3.0  # -35: would land > 3 dB under target even at the +9 dB clamp
SPK_PROBE, SPK_MIN_LUFS = 4, QUIET_LUFS + 2.0  # AISHELL-3 speaker level screen (median of 4 clips)
CLIP_LEVEL, CLIP_RUN = 0.999, 3
RATE = {"chinese": (1.5, 8.0), "english": (1.0, 5.0)}   # hanzi/s, words/s of speech
VAL_FRAC = 0.03
AUX_MAX = 20
ZH_AGES = ("B", "C")                     # AISHELL-3 age groups 14-25, 26-40
TRANSIENT = ("download_failed", "error")  # not persisted, retried on the next run
# cached decisions / speaker metadata are reused only under the same rules (bump "v1" when the
# text rules change)
RULES = hashlib.sha1(repr(("v1", MIN_DUR, MAX_DUR, QUIET_LUFS, CLIP_LEVEL, CLIP_RUN, RATE, LEAD_PAD, TAIL_PAD,
                           TARGET_LUFS, GAIN_MIN, GAIN_MAX, PEAK_MAX)).encode()).hexdigest()[:10]

CJK = re.compile(r"^[一-鿿]+$")
PINYIN = re.compile(r"^[a-z]+[1-5]$")
DIGIT_LATIN = re.compile(r"[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]")
ZH_NUMERALS = set("零一二三四五六七八九十百千万亿点两幺")
ZH_TEXT_OK = re.compile(r"^[一-鿿][一-鿿，]*[。？]$")
EN_TEXT_BAD = re.compile(r"[0-9\[\]{}<>_*#/\\|~@^=+]")
EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
EXPRESSIVE = re.compile(r"[!?\"“”]")

_log_lock = threading.Lock()
_bytes = Counter()


def log(msg: str) -> None:
    with _log_lock:
        print(f"[replay {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- network (anonymous)

_tls = threading.local()


def _session():
    s = getattr(_tls, "session", None)
    if s is None:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = "kurisu-replay/1.0"
        _tls.session = s
    return s


def http(method: str, url: str, *, json_body=None, tries: int = 7, timeout: float = 120.0,
         tag: str = "other") -> bytes:
    import requests
    last = ""
    for k in range(tries):
        try:
            r = _session().request(method, url, json=json_body, timeout=timeout)
            if r.status_code == 200:
                _bytes[tag] += len(r.content)
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
            wait = r.headers.get("Retry-After", "")
            time.sleep(float(wait) if wait.isdigit() else min(60.0, 2.0 ** k))
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(min(60.0, 2.0 ** k))
    raise RuntimeError(f"{method} {url}: {last}")


def hf_resolve(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def cached_get(urls, dest: Path, tag: str) -> bytes:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    err = None
    for u in urls:
        try:
            data = http("GET", u, tag=tag)
            break
        except RuntimeError as e:
            err = e
    else:
        raise RuntimeError(f"all mirrors failed for {dest.name}: {err}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return data


def hf_paths_info(repo: str, rev: str, paths: list[str], workers: int) -> dict[str, list]:
    """path -> [size, lfs sha256] for the paths that exist at that revision."""
    url = f"https://huggingface.co/api/datasets/{repo}/paths-info/{rev}"
    chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]

    def one(chunk):
        return json.loads(http("POST", url, json_body={"paths": chunk, "expand": False}, tag="api"))

    out = {}
    with ThreadPoolExecutor(min(8, workers, max(1, len(chunks)))) as ex:
        for res in ex.map(one, chunks):
            for x in res:
                if x.get("type") == "file":
                    out[x["path"]] = [int(x["size"]), (x.get("lfs") or {}).get("oid")]
    return out


# ---------------------------------------------------------------- audio

_mtls = threading.local()


def meter():
    m = getattr(_mtls, "meter", None)
    if m is None:
        import pyloudnorm
        m = _mtls.meter = pyloudnorm.Meter(SR)
    return m


def clipped_runs(x: np.ndarray) -> int:
    m = np.abs(x) >= CLIP_LEVEL
    if not m.any():
        return 0
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int(np.sum((ends - starts) >= CLIP_RUN))


def to_24k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def loudness(core: np.ndarray) -> float:
    """Integrated loudness of the active-speech span, measured the way p03 measures it."""
    n = int(0.4 * SR)
    x = core if len(core) >= n else np.tile(core, n // max(1, len(core)) + 1)
    try:
        v = meter().integrated_loudness(x)
    except Exception:
        v = float("-inf")
    if not np.isfinite(v):
        v = 20 * np.log10(np.sqrt(np.mean(core ** 2)) + 1e-9) - 3.0
    return float(v)


def f0_stats(core: np.ndarray) -> tuple[float | None, float | None]:
    """(median F0 in Hz, IQR in semitones) over the loud frames of a 24 kHz span."""
    if len(core) < SR // 2:
        return None, None
    import librosa
    f0 = librosa.yin(core, fmin=60.0, fmax=600.0, sr=SR, frame_length=1200, hop_length=240)
    rms = librosa.feature.rms(y=core, frame_length=1200, hop_length=240)[0]
    n = min(len(f0), len(rms))
    db = 20 * np.log10(rms[:n] + 1e-9)
    keep = db > np.percentile(db, 95) - 20.0
    if keep.sum() < 20:
        return None, None
    st = 12 * np.log2(f0[:n][keep] / 440.0)
    q25, q50, q75 = np.percentile(st, [25, 50, 75])
    return float(440.0 * 2 ** (q50 / 12)), float(q75 - q25)


def process(raw: bytes, lang: str, stem: str, units: int) -> dict:
    """Decode, gate, normalise exactly as p03 does, write wavs/<stem>.wav."""
    rec = {"kept": False, "reason": None}
    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        rec["reason"] = "undecodable"
        return rec
    x = x.mean(axis=1)
    rec["raw_dur"] = round(len(x) / sr, 3)
    if clipped_runs(x):
        rec["reason"] = "clipped"
        return rec
    y = to_24k(x, sr)
    b = speech_bounds(y)
    if b is None:
        rec["reason"] = "no_audible_speech"
        return rec
    lin = loudness(y[b[0]:b[1]])
    rec["lufs_in"] = round(lin, 2)
    if lin < QUIET_LUFS:
        rec["reason"] = "too_quiet"
        return rec
    out, gain_db = normalise_audio(y, meter())
    if out is None:
        rec["reason"] = "no_audible_speech"
        return rec
    dur = len(out) / SR
    rec.update(dur=round(dur, 3), gain_db=round(gain_db, 2))
    if dur < MIN_DUR or dur > MAX_DUR:
        rec["reason"] = "too_short" if dur < MIN_DUR else "too_long"
        return rec
    rate = units / max(1e-3, dur - LEAD_PAD - TAIL_PAD)
    rec["rate"] = round(rate, 2)
    lo, hi = RATE[lang]
    if not lo <= rate <= hi:
        rec["reason"] = "rate_implausible"
        return rec
    core = out[int(LEAD_PAD * SR): len(out) - int(TAIL_PAD * SR)]
    f0, iqr = f0_stats(core)
    rec.update(f0_hz=None if f0 is None else round(f0, 1), f0_iqr_st=None if iqr is None else round(iqr, 2))
    path = WAVS / f"{stem}.wav"
    tmp = path.with_name(path.name + ".part")
    sf.write(str(tmp), out, SR, subtype="PCM_16", format="WAV")
    tmp.replace(path)
    rec["kept"] = True
    return rec


# ---------------------------------------------------------------- selection

def order_pool(names, tier, tag: str) -> list:
    """Seeded shuffle, then a stable sort by preference tier."""
    names = sorted(names)
    random.Random(tag).shuffle(names)
    return sorted(names, key=tier)


def quotas(n: int, female_ratio: float) -> tuple[int, int]:
    nf = int(math.floor(n * female_ratio + 0.5))
    return nf, n - nf


def walk(speakers, cands, decisions, target_s: float, cap: int):
    """Round-robin over speakers until the hours target is met.

    Decided clips count with their real duration, undecided ones with an estimate and are
    returned as work. When nothing is undecided the selection is final, and the same
    decisions always give the same selection (this is what makes re-runs reproducible).
    """
    sel = {s: [] for s in speakers}
    todo = []
    idx = dict.fromkeys(speakers, 0)
    n = dict.fromkeys(speakers, 0)
    cum = 0.0
    active = list(speakers) if cap > 0 else []
    while active and cum < target_s:
        nxt = []
        for s in active:
            if cum >= target_s:
                break
            lst = cands[s]
            while idx[s] < len(lst):
                c = lst[idx[s]]
                idx[s] += 1
                d = decisions.get(c["key"])
                if d is not None and not d["kept"]:
                    continue
                if d is not None and (WAVS / f"{c['stem']}.wav").exists():
                    sel[s].append(d)
                    cum += d["dur"]
                else:
                    todo.append(c)
                    cum += c["est"]
                n[s] += 1
                if n[s] < cap:
                    nxt.append(s)
                break
        active = nxt
    return sel, todo


class Decisions:
    def __init__(self, path: Path):
        self.path = path
        self.d: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("rules") == RULES:
                        self.d[r["key"]] = r

    def get(self, key):
        return self.d.get(key)

    def put(self, rec: dict) -> None:
        rec["rules"] = RULES
        with self.lock:
            self.d[rec["key"]] = rec
            if rec.get("reason") not in TRANSIENT:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_batch(lang: str, todo: list[dict], task, decisions: Decisions, workers: int) -> None:
    def run(c):
        try:
            rec = task(c)
        except Exception as e:  # keep the pool alive; retried next run
            rec = {"kept": False, "reason": "error", "error": repr(e)[:300]}
        rec.update(key=c["key"], lang=lang, speaker=c["spk_out"], stem=c["stem"], text=c["text"])
        decisions.put(rec)

    if todo:
        with ThreadPoolExecutor(max(1, min(workers, len(todo)))) as ex:
            list(ex.map(run, todo))


def fill(lang: str, speakers, cands, target_s: float, cap: int, task, decisions: Decisions,
         workers: int) -> dict[str, list[dict]]:
    for _ in range(100):
        sel, todo = walk(speakers, cands, decisions, target_s, cap)
        if not todo:
            got = sum(r["dur"] for v in sel.values() for r in v)
            log(f"{lang}: {sum(map(len, sel.values()))} clips, {got / 3600:.3f} h (target {target_s / 3600:.3f} h)")
            return sel
        log(f"{lang}: processing {len(todo)} clips")
        process_batch(lang, todo, task, decisions, workers)
    raise RuntimeError(f"{lang}: selection did not converge")


# ---------------------------------------------------------------- Mandarin (AISHELL-3)

def zh_parse_content(text: str) -> dict[str, tuple[str | None, str | None]]:
    """utt -> (hanzi, None) or (None, drop reason). Lines are 'X.wav<TAB>字 zi4 字 zi4 ...'."""
    out = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        name, body = line.split("\t", 1)
        utt = name.strip()
        utt = utt[:-4] if utt.endswith(".wav") else utt
        toks = body.split()
        han, py = toks[0::2], toks[1::2]
        if not toks or len(han) != len(py) or not all(PINYIN.match(p) for p in py):
            out[utt] = (None, "zh_transcript_unparsable")
        elif any(DIGIT_LATIN.search(t) for t in han):
            out[utt] = (None, "zh_digits_latin")
        elif not all(CJK.match(t) for t in han):
            out[utt] = (None, "zh_non_hanzi")
        else:
            out[utt] = ("".join(han), None)   # erhua tokens ("玩儿") are kept whole
    return out


def zh_parse_labels(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        p = line.strip().split("|")
        if len(p) == 3 and not line.startswith("#"):
            out[p[0]] = p[2]
    return out


def zh_text(hanzi: str, label: str | None) -> str:
    """Full-width punctuation: '，' at labelled intonation-phrase boundaries, final '。'/'？'."""
    body = hanzi
    if label and re.sub(r"[%$]", "", label) == hanzi:
        body = re.sub(r"\$+", "，", label.replace("%", "").rstrip("$")).strip("，")
    return body + ("？" if body.endswith("吗") else "。")


def run_zh(args, decisions: Decisions, stats: dict) -> dict:
    repo = args.zh_repo
    rev = ZH_REPOS[repo]
    meta = CACHE / "zh_meta" / repo.replace("/", "__")
    get = lambda p: cached_get([hf_resolve(repo, rev, p)], meta / p, "zh_meta").decode("utf-8")  # noqa: E731
    info = {}
    for line in get("spk-info.txt").splitlines():
        p = line.split()
        if len(p) == 4 and not line.startswith("#"):
            info[p[0]] = {"age_group": p[1], "gender": p[2], "accent": p[3]}
    content = zh_parse_content(get("train/content.txt"))
    labels = zh_parse_labels(get("train/label_train-set.txt"))
    by_spk = defaultdict(list)
    for utt in content:
        by_spk[utt[:7]].append(utt)
    log(f"zh: {len(content)} train utterances, {len(by_spk)} speakers, {len(labels)} prosody labels")

    def tier(s):
        i = info[s]
        return (i["age_group"] not in ZH_AGES, args.zh_accent != "any" and i["accent"] != args.zh_accent,
                i["age_group"] != "B")

    def task(c):
        try:
            raw = http("GET", hf_resolve(repo, rev, c["path"]), tag="zh_audio")
        except RuntimeError:
            return {"kept": False, "reason": "download_failed"}
        if c["oid"] and hashlib.sha256(raw).hexdigest() != c["oid"]:
            return {"kept": False, "reason": "download_failed", "error": "sha256 mismatch"}
        return process(raw, "chinese", c["stem"], c["units"])

    def clip_list(s: str, drops: Counter) -> list[dict]:
        lst = []
        for u in sorted(by_spk[s]):
            hanzi, reason = content[u]
            path = f"train/wav/{s}/{u}.wav"
            size, oid = sizes.get(path, [None, None])
            est = (size - 44) / 88200.0 if size else 0.0     # 44.1 kHz PCM16 mono
            if reason is None and len(hanzi) < 4:
                reason = "text_too_short"
            if reason is None and size is None:
                reason = "missing_in_mirror"
            if reason is None and (est < MIN_DUR or est > MAX_DUR + 2.0):
                reason = "too_short" if est < MIN_DUR else "too_long"
            if reason is None:
                text = zh_text(hanzi, labels.get(u))
                reason = None if ZH_TEXT_OK.match(text) else "zh_text_invalid"
            if reason:
                drops[reason] += 1
                continue
            lst.append({"key": f"zh/{u}", "spk_out": f"zh_{s}", "stem": f"zh_{s}_{u}", "text": text,
                        "units": len(hanzi), "est": max(MIN_DUR, est - 0.05), "path": path, "oid": oid,
                        "numeric": sum(ch in ZH_NUMERALS for ch in hanzi) > 0.5 * len(hanzi)})
        # est: trim + 0.37 s pad leaves AISHELL-3 clips ~0.0 s shorter (median -0.05 s, full run).
        # Tried last: bare number strings (stock prices, phone digits: ~6% of AISHELL-3 prompts, not
        # sentences), then clips under ~2.8 s raw (they often trim below 2 s). At most 3x the cap is tried.
        return _order_clips(lst, lambda c: (c["numeric"], c["est"] < 2.75), f"{args.seed}:zh:{s}")[:3 * cap]

    sizes_path = meta / f"sizes_{rev[:8]}.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    pre, screen = Counter(), {}
    cands, chosen = {}, []
    cap = args.max_clips_per_speaker
    nf, nm = quotas(args.zh_speakers, args.female_ratio)
    min_cands = min(cap, 10)
    for gender, quota in (("female", nf), ("male", nm)):
        pool = order_pool([s for s in by_spk if info.get(s, {}).get("gender") == gender], tier,
                          f"{args.seed}:zh:{gender}")
        got, i = [], 0
        while len(got) < quota and i < len(pool):
            batch = pool[i:i + math.ceil(1.5 * (quota - len(got))) + 1]
            i += len(batch)
            need = [f"train/wav/{s}/{u}.wav" for s in batch for u in by_spk[s]
                    if f"train/wav/{s}/{u}.wav" not in sizes]
            if need:
                found = hf_paths_info(repo, rev, need, args.workers)
                sizes.update({p: found.get(p, [None, None]) for p in need})
                sizes_path.write_text(json.dumps(sizes))
            lists, drops = {}, {}
            for s in batch:
                drops[s] = Counter()
                lists[s] = clip_list(s, drops[s])
            # Level screen: AISHELL-3 recording gain varies by speaker (median active loudness
            # -42..-22 LUFS). Speakers whose first clips sit under the quiet gate would lose most
            # clips and land below -23 even at the +9 dB clamp, so they are skipped up front.
            probe = [c for s in batch if len(lists[s]) >= min_cands for c in lists[s][:SPK_PROBE]
                     if decisions.get(c["key"]) is None]
            if probe:
                log(f"zh: level probe, {len(probe)} clips from {len(batch)} {gender} speakers")
                process_batch("chinese", probe, task, decisions, args.workers)
            for s in batch:
                if len(got) >= quota:
                    break
                if len(lists[s]) < min_cands:
                    log(f"zh: skip {s}: only {len(lists[s])} usable clips")
                    continue
                lv = [d["lufs_in"] for d in (decisions.get(c["key"]) for c in lists[s][:SPK_PROBE])
                      if d is not None and d.get("lufs_in") is not None]
                screen[f"zh_{s}"] = round(float(np.median(lv)), 1) if lv else None
                if not lv or np.median(lv) < SPK_MIN_LUFS:
                    log(f"zh: skip {s}: recorded too quietly (probe median {screen[f'zh_{s}']} LUFS)")
                    continue
                cands[s] = lists[s]
                pre.update(drops[s])
                got.append(s)
        if len(got) < quota:
            log(f"zh: WARNING only {len(got)}/{quota} {gender} speakers available")
        chosen += got

    sel = fill("chinese", chosen, cands, args.zh_hours * 3600, cap, task, decisions, args.workers)
    spk_meta = {f"zh_{s}": {"language": "chinese", "source_speaker": s, **info[s]} for s in chosen}
    stats["chinese"] = {"prefilter_drops": dict(pre),
                        "speaker_level_screen (probe median LUFS; skipped below %.0f)" % SPK_MIN_LUFS: screen}
    return {"sel": {f"zh_{s}": v for s, v in sel.items()}, "spk_meta": spk_meta,
            "source": f"AISHELL-3 (OpenSLR SLR93) via hf:{repo}@{rev[:7]}", "cands": cands}


def _order_clips(lst: list[dict], late, tag: str) -> list[dict]:
    """Seeded shuffle of a speaker's clips, then a stable sort by `late(c)` (False/smaller first)."""
    lst = sorted(lst, key=lambda c: c["key"])
    random.Random(tag).shuffle(lst)
    return sorted(lst, key=late)


# ---------------------------------------------------------------- English (LibriTTS-R)

def _tar_members(data: bytes) -> dict[str, bytes]:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf:
            base = m.name.rsplit("/", 1)[-1]
            if m.isfile() and not base.startswith("._"):
                out[base] = tf.extractfile(m).read()
    return out


def en_metadata() -> tuple[dict[str, str], set[str], dict]:
    meta = CACHE / "en_meta"
    check = {}
    blobs = {}
    for name in EN_DOC_MD5:
        blobs[name] = cached_get([u + name for u in OPENSLR_141], meta / name, "en_meta")
        md5 = hashlib.md5(blobs[name]).hexdigest()
        check[name] = {"md5": md5, "known": md5 in EN_DOC_MD5[name]}
        if md5 not in EN_DOC_MD5[name]:
            log(f"en: WARNING {name} md5 {md5} is not a known version {EN_DOC_MD5[name]}")
    gender = {}
    for line in _tar_members(blobs["doc.tar.gz"])["speakers.tsv"].decode("utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0].strip().isdigit():
            gender[p[0].strip()] = {"F": "female", "M": "male"}.get(p[1].strip().upper(), "unknown")
    bad = set()
    for name, b in _tar_members(blobs["libritts_r_failed_speech_restoration_examples.tar.gz"]).items():
        if name.endswith("_bad_sample_list.txt"):
            for line in b.decode("utf-8").splitlines():
                if line.strip().endswith(".wav"):
                    bad.add(line.strip().rsplit("/", 1)[-1][:-4])
    return gender, bad, check


def en_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def en_text_problem(text: str) -> str | None:
    if not EN_WORD.search(text):
        return "no_text"
    if EN_TEXT_BAD.search(text):
        return "en_bad_chars"
    letters = [c for c in text if c.isalpha()]
    if sum(c.isupper() for c in letters) > 0.5 * len(letters):
        return "en_all_caps"          # headings; a TTS LM may spell them out
    if len(EN_WORD.findall(text)) < 3:
        return "text_too_short"
    return None


def en_index(fs, splits: list[str], workers: int) -> dict[str, list]:
    """speaker -> [[shard, row_group, rows]] for row groups that hold only that speaker."""
    path = CACHE / f"en_index_{EN_REV[:8]}_{'+'.join(splits)}.json"
    if path.exists():
        return json.loads(path.read_text())
    import pyarrow.parquet as pq
    shards = []
    for split in splits:
        shards += sorted(fs.glob(f"datasets/{EN_REPO}@{EN_REV}/data/{split}/*.parquet"))

    def footer(shard):
        with fs.open(shard, "rb") as f:
            md = pq.ParquetFile(f).metadata
        j = [md.row_group(0).column(k).path_in_schema for k in range(md.num_columns)].index("speaker_id")
        res = []
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            st = rg.column(j).statistics
            if st is not None and st.has_min_max and st.min == st.max and rg.num_rows >= 40:
                v = st.min.decode("utf-8") if isinstance(st.min, bytes) else str(st.min)  # older pyarrow
                res.append((v, [shard, i, rg.num_rows]))
        return res

    idx = defaultdict(list)
    with ThreadPoolExecutor(min(16, workers)) as ex:
        for res in ex.map(footer, shards):
            for spk, item in res:
                idx[spk].append(item)
    path.write_text(json.dumps(idx))
    log(f"en: indexed {len(shards)} shards, {len(idx)} speakers with single-speaker row groups")
    return idx


def en_load_speaker(fs, spk: str, rgs: list, bad: set, need: int, seed: int) -> dict:
    """Read the speaker's row group(s) once, cache usable raw wavs + metadata + F0 stats."""
    mp = CACHE / "en_spk" / f"{spk}.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("rules") == RULES and (m.get("need", 0) >= need or len(m["rows"]) >= need):
            return m
    import pyarrow.parquet as pq
    rows, pre = [], Counter()
    raw_dir = CACHE / "en_raw" / spk
    raw_dir.mkdir(parents=True, exist_ok=True)
    for shard, rg, _ in rgs[:2]:
        with fs.open(shard, "rb") as f:
            tab = pq.ParquetFile(f).read_row_group(rg, columns=["audio", "text_normalized", "id"])
        _bytes["en_parquet"] += tab.nbytes
        for r in tab.to_pylist():
            utt, text = r["id"], en_norm(r["text_normalized"])
            raw = (r.get("audio") or {}).get("bytes") or b""
            reason = "en_restoration_failed" if utt in bad else en_text_problem(text)
            est = 0.0
            if reason is None:
                try:
                    inf = sf.info(io.BytesIO(raw))
                    est = inf.frames / inf.samplerate
                except Exception:
                    reason = "undecodable"
            if reason is None and (est < MIN_DUR or est > MAX_DUR + 1.0):
                reason = "too_short" if est < MIN_DUR else "too_long"
            if reason:
                pre[reason] += 1
                continue
            (raw_dir / f"{utt}.wav").write_bytes(raw)
            rows.append({"id": utt, "text": text, "est": round(est, 3)})
        if len(rows) >= need:
            break
    rows = _order_clips([dict(r, key=r["id"]) for r in rows], lambda r: EXPRESSIVE.search(r["text"]) is None,
                        f"{seed}:en:{spk}")
    f0s = []
    for r in rows[:12]:
        x, sr = sf.read(str(raw_dir / f"{r['id']}.wav"), dtype="float32", always_2d=True)
        y = to_24k(x.mean(axis=1), sr)
        b = speech_bounds(y)
        if b is not None:
            f0, iqr = f0_stats(y[b[0]:b[1]])
            if iqr is not None:
                f0s.append((f0, iqr))
    out = {"speaker": spk, "rules": RULES, "need": need, "rows": rows, "prefilter_drops": dict(pre),
           "row_groups": rgs[:2],
           "f0_hz": float(np.median([a for a, _ in f0s])) if f0s else None,
           "f0_iqr_st": float(np.median([b for _, b in f0s])) if f0s else None}
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def run_en(args, decisions: Decisions, stats: dict) -> dict:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=False)
    gender, bad, doc_check = en_metadata()
    idx = en_index(fs, args.en_splits.split(","), args.workers)
    nf, nm = quotas(args.en_speakers, args.female_ratio)
    cap = args.max_clips_per_speaker
    # rows a speaker should offer: its share of the hours at ~4 s/clip, x1.5 for drops (a 2nd row group
    # is read only below that)
    need = min(cap, math.ceil(1.5 * args.en_hours * 3600 / max(1, args.en_speakers) / 4.0))
    min_rows = min(cap, 10)
    chosen, ranking, metas = [], [], {}
    for g, quota in (("female", nf), ("male", nm)):
        pool_all = order_pool([s for s in idx if gender.get(s) == g], lambda s: 0, f"{args.seed}:en:{g}")
        size = max(quota + 1, math.ceil(quota * args.en_pool)) if quota else 0
        while True:
            pool = pool_all[:size]
            todo = [s for s in pool if s not in metas]
            if todo:
                log(f"en: loading {len(todo)} {g} pool speakers (row groups unless cached)")
                with ThreadPoolExecutor(min(8, args.workers, len(todo))) as ex:
                    for s, m in zip(todo, ex.map(lambda s: en_load_speaker(fs, s, idx[s], bad, need, args.seed),
                                                 todo)):
                        metas[s] = m
            usable = [s for s in pool if len(metas[s]["rows"]) >= min_rows and metas[s]["f0_iqr_st"] is not None]
            if len(usable) >= quota or size >= len(pool_all):
                break
            size += quota - len(usable) + 1
        ranked = sorted(usable, key=lambda s: (-metas[s]["f0_iqr_st"], s))
        pick = ranked[:quota]
        if len(pick) < quota:
            log(f"en: WARNING only {len(pick)}/{quota} {g} speakers available")
        for s in pool:
            m = metas[s]
            ranking.append({"speaker": f"en_{s}", "gender": g, "usable_rows": len(m["rows"]),
                            "f0_hz": None if m["f0_hz"] is None else round(m["f0_hz"], 1),
                            "f0_iqr_st": None if m["f0_iqr_st"] is None else round(m["f0_iqr_st"], 2),
                            "selected": s in pick})
        chosen += pick

    cands, pre = {}, Counter()
    for s in chosen:
        pre.update(metas[s]["prefilter_drops"])
        cands[s] = [{"key": f"en/{r['id']}", "spk_out": f"en_{s}", "stem": f"en_{s}_{r['id']}", "text": r["text"],
                     "units": len(EN_WORD.findall(r["text"])), "est": max(MIN_DUR, r["est"] + 0.3),  # pads add ~0.36 s
                     "raw": str(CACHE / "en_raw" / s / f"{r['id']}.wav")} for r in metas[s]["rows"]]

    def task(c):
        return process(Path(c["raw"]).read_bytes(), "english", c["stem"], c["units"])

    sel = fill("english", chosen, cands, args.en_hours * 3600, cap, task, decisions, args.workers)
    spk_meta = {f"en_{s}": {"language": "english", "source_speaker": s, "gender": gender[s],
                            "pool_f0_iqr_st": metas[s]["f0_iqr_st"]} for s in chosen}
    stats["english"] = {"prefilter_drops": dict(pre), "pool_ranking": ranking, "doc_check": doc_check,
                        "failed_restoration_list": len(bad), "pool_examined": len(metas)}
    return {"sel": {f"en_{s}": v for s, v in sel.items()}, "spk_meta": spk_meta,
            "source": f"LibriTTS-R (OpenSLR SLR141) via hf:{EN_REPO}@{EN_REV[:7]}", "cands": cands}


# ---------------------------------------------------------------- outputs

def u01(seed: int, stem: str) -> float:
    return int(hashlib.sha1(f"{seed}:{stem}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


def write_outputs(args, results: dict, decisions: Decisions, stats: dict, t0: float) -> dict:
    rows, aux, spk_rep = [], {}, {}
    for lang, res in results.items():
        for spk, recs in sorted(res["sel"].items()):
            recs = sorted(recs, key=lambda r: r["stem"])
            if not recs:
                continue
            u = {r["stem"]: u01(args.seed, r["stem"]) for r in recs}
            val = {r["stem"] for r in recs if u[r["stem"]] < VAL_FRAC}
            if len(val) == len(recs):          # at least one train clip per speaker
                val.discard(max(val, key=u.get))
            spk_rows = []
            for r in recs:
                spk_rows.append({"id": r["stem"], "audio": str(WAVS / f"{r['stem']}.wav"), "text": r["text"],
                                 "language": lang, "speaker": spk,
                                 "split": "val" if r["stem"] in val else "train", "dur": r["dur"],
                                 "source": res["source"], "license": LICENSE[lang]})
            rows += spk_rows
            train = [x for x in spk_rows if x["split"] == "train"]
            aux[spk] = [x["audio"] for x in sorted(train, key=lambda x: (not 3.0 <= x["dur"] <= 8.0, x["id"]))][:AUX_MAX]
            f0 = [r["f0_hz"] for r in recs if r.get("f0_hz")]
            iqr = [r["f0_iqr_st"] for r in recs if r.get("f0_iqr_st") is not None]
            spk_rep[spk] = {**res["spk_meta"][spk], "clips": len(recs), "hours": round(sum(r["dur"] for r in recs) / 3600, 4),
                            "train": len(train), "val": len(recs) - len(train),
                            "f0_median_hz": round(float(np.median(f0)), 1) if f0 else None,
                            "f0_iqr_st": round(float(np.median(iqr)), 2) if iqr else None}

    (OUT / "replay_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")
    (OUT / "aux_speakers.json").write_text(json.dumps(aux, indent=1, ensure_ascii=False), encoding="utf-8")

    langs = {}
    for lang, res in results.items():
        lr = [r for r in rows if r["language"] == lang]
        durs = sorted(r["dur"] for r in lr)
        keys = {c["key"] for v in res["cands"].values() for c in v}
        processed = [d for k, d in decisions.d.items() if k in keys]
        by_g = defaultdict(lambda: [0, 0.0, 0])
        for spk, m in spk_rep.items():
            if m["language"] == lang:
                g = by_g[m["gender"]]
                g[0] += 1
                g[1] += m["hours"]
                g[2] += m["clips"]
        langs[lang] = {
            "target_hours": getattr(args, "zh_hours" if lang == "chinese" else "en_hours"),
            "clips": len(lr), "hours": round(sum(durs) / 3600, 4),
            "train": sum(r["split"] == "train" for r in lr), "val": sum(r["split"] == "val" for r in lr),
            "speakers": len({r["speaker"] for r in lr}),
            "by_gender": {g: {"speakers": v[0], "clips": v[2], "hours": round(v[1], 4)} for g, v in sorted(by_g.items())},
            "dur_mean": round(float(np.mean(durs)), 3) if durs else None,
            "dur_p10_p50_p90": [round(float(np.percentile(durs, q)), 2) for q in (10, 50, 90)] if durs else None,
            "texts_with_internal_comma": sum("，" in r["text"] for r in lr) if lang == "chinese" else None,
            "drops": {"prefilter (metadata, all examined utterances of selected speakers)":
                      stats.get(lang, {}).get("prefilter_drops", {}),
                      "audio (clips fetched and processed)":
                      dict(Counter(d["reason"] for d in processed if not d["kept"]))},
            "processed_kept_unused": sum(1 for d in processed if d["kept"]) - len(lr),
            "source": res["source"], "license": LICENSE[lang],
        }
        extra = {k: v for k, v in stats.get(lang, {}).items() if k != "prefilter_drops"}
        langs[lang].update(extra)
    report = {
        "config": {k: v for k, v in vars(args).items()},
        "zh_revision": ZH_REPOS[args.zh_repo], "en_repo": EN_REPO, "en_revision": EN_REV,
        "rows": len(rows), "hours": round(sum(r["dur"] for r in rows) / 3600, 4),
        "languages": langs,
        "speakers": spk_rep,
        "processing": {"sr": SR, "lead_pad_s": LEAD_PAD, "tail_pad_s": TAIL_PAD, "target_lufs": TARGET_LUFS,
                       "gain_clamp_db": [GAIN_MIN, GAIN_MAX], "peak_max": round(PEAK_MAX, 4),
                       "dur_range_s": [MIN_DUR, MAX_DUR], "quiet_lufs": QUIET_LUFS,
                       "clip": {"level": CLIP_LEVEL, "run": CLIP_RUN}, "rate_ranges": RATE,
                       "val_frac": VAL_FRAC, "normalise": "p03_build.normalise_audio (imported)"},
        "downloaded_bytes_this_run": dict(_bytes),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zh_hours", type=float, default=1.5)
    ap.add_argument("--en_hours", type=float, default=0.8)
    ap.add_argument("--zh_speakers", type=int, default=30)
    ap.add_argument("--en_speakers", type=int, default=25)
    ap.add_argument("--female_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_clips_per_speaker", type=int, default=50,
                    help="cap per speaker; clips are added round-robin until the hours target is met")
    ap.add_argument("--langs", default="zh,en", help="comma list of zh,en")
    ap.add_argument("--zh_repo", default="shenyunhang/AISHELL-3", choices=sorted(ZH_REPOS))
    ap.add_argument("--zh_accent", default="north", choices=["north", "south", "any"],
                    help="preferred AISHELL-3 accent label; others are used only if too few")
    ap.add_argument("--en_splits", default="train.clean.100", help="comma list of LibriTTS-R splits")
    ap.add_argument("--en_pool", type=float, default=1.5, help="speaker pool size / speakers kept (F0 ranking)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, WAVS, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    decisions = Decisions(OUT / "decisions.jsonl")
    stats, results = {}, {}
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if "zh" in langs:
        results["chinese"] = run_zh(args, decisions, stats)
    if "en" in langs:
        results["english"] = run_en(args, decisions, stats)
    report = write_outputs(args, results, decisions, stats, t0)
    short = {k: {x: v[x] for x in ("clips", "hours", "train", "val", "speakers", "by_gender", "drops")}
             for k, v in report["languages"].items()}
    print(json.dumps({"rows": report["rows"], "hours": report["hours"], "languages": short,
                      "downloaded_bytes_this_run": report["downloaded_bytes_this_run"],
                      "elapsed_s": report["elapsed_s"]}, indent=2, ensure_ascii=True))
    short_of = [f"{k}: {v['hours']:.3f} h of {v['target_hours']} h" for k, v in report["languages"].items()
                if v["hours"] < 0.9 * v["target_hours"]]
    if short_of:   # outputs are written, but a pod chain must notice
        raise SystemExit("replay shortfall (more than 10% under target): " + "; ".join(short_of))


if __name__ == "__main__":
    main()
