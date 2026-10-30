"""Shared config, paths, taxonomy and the text normaliser for the Kurisu Qwen3-TTS build.

Every stage imports from here so the taxonomy and the normalisation rules have exactly
one definition. Nothing in this file touches the GPU.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- paths

PROJECT = Path(__file__).resolve().parent.parent          # kurisu_qwen3tts/
SRC_ROOT = PROJECT.parent                                  # STEINS;GATEVoice/
GAME = SRC_ROOT / "voice_dataset_game"
CLEAN = SRC_ROOT / "voice_dataset_clean"
REF_SRC = SRC_ROOT / "ref_audio.wav"

AUDIO = PROJECT / "audio"
WAVS = AUDIO / "wavs"
REF_OUT = AUDIO / "ref.wav"
MANIFESTS = PROJECT / "manifests"
TAXONOMY = PROJECT / "taxonomy"
CACHE = PROJECT / "cache"
REPORTS = PROJECT / "reports"
QWEN = PROJECT / "qwen"

for _d in (AUDIO, WAVS, MANIFESTS, TAXONOMY, CACHE, REPORTS, QWEN):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- selection

SPEAKERS = ("kurisu", "amadeus_kurisu")

# The user's selection rule, encoded once:
#   game  -> wavs/ AND wavs_quarantine/   (all of it; quarantine is short expressive takes)
#   clean -> wavs/ only                   (wavs_quarantine/ there is messy)
SOURCE_RULES = {
    "voice_dataset_game": {"wavs": True, "wavs_quarantine": True},
    "voice_dataset_clean": {"wavs": True, "wavs_quarantine": False},
}

TARGET_SR = 24000          # official Qwen3-TTS dataset.py asserts 24 kHz on ref audio
TARGET_LUFS = -23.0        # per-source integrated-loudness target (EBU R128); see s02
PEAK_CEILING_DBFS = -1.0   # after gain, no source may exceed this

# ---------------------------------------------------------------- taxonomy

# Qwen-facing inline tags. Deliberately small (supplement s4). Anything finer lives
# in the master manifest only.
QWEN_TAGS = [
    "[sigh]", "[breath]", "[gasp]",
    "[laugh]", "[giggle]", "[chuckle]",
    "[moan]", "[groan]", "[grunt]",
    "[sob]", "[cry]", "[scream]",
    "[cough]", "[throat_clear]", "[sniff]", "[yawn]", "[hum]",
]

# LAION vocalburst coarse group  ->  our canonical family.
# The classifier emits 83 fine classes in 17 coarse groups; we collapse per supplement s6.
GROUP_TO_FAMILY = {
    "sighs": "sigh",
    "breathing": "breath",
    "gasps_and_inhales": "gasp",
    "laughter": None,             # split by fine label, see FINE_TO_FAMILY
    "groans_and_moans": None,     # split moan vs groan by fine label
    "grunts": "grunt",
    "crying_and_distress": None,  # split sob vs cry
    "screams_and_shrieks": "scream",
    "coughing_and_sneezing": None,
    "humming": "hum",
    "throat_and_vocal_sounds": None,
    "tongue_clicks": "tsk",
    "mouth_and_lip_sounds": "lipsmack",
    "eating_and_drinking": "swallow",
    "whistling": "whistle",
    "hand_and_body_sounds": None,
    "no_burst": "no_burst",
}

FINE_TO_FAMILY = {
    # laughter
    "Breathy Giggle": "giggle", "Childlike Giggle": "giggle", "Nervous Giggle": "giggle",
    "Snorting Giggle": "giggle", "Snicker": "giggle",
    "Chuckle": "chuckle",
    "Cackle": "laugh", "Guffaw": "laugh",
    # sighs
    "Contented Sigh": "sigh", "Exasperated Sigh": "sigh", "Relief Sigh": "sigh",
    "Wistful Sigh": "sigh",
    # breathing
    "Deep Breath": "breath", "Deep Breathing": "breath", "Fast Breathing": "breath",
    "Heavy Breathing": "breath", "Normal Breathing": "breath", "Panting": "breath",
    "Slow Breathing": "breath",
    # gasps
    "Fearful Gasp": "gasp", "Sharp Inhale": "gasp", "Surprised Gasp": "gasp",
    # groans / moans
    "Exhausted Groan": "groan", "Frustrated Groan": "groan",
    "Pain Moan": "moan", "Pleasure Moan": "moan",
    # grunts
    "Affirmative Grunt": "grunt", "Displeased Grunt": "grunt", "Effort Grunt": "grunt",
    # crying
    "Convulsive Sob": "sob", "Quiet Sob": "sob", "Sobs": "sob",
    "Mournful Wail": "cry", "Trembling Whimper": "cry",
    # throat / vocal
    "Ahem": "throat_clear", "Clears Throat": "throat_clear",
    "Growl": "growl", "Gurgling": "other", "Hiss": "other",
    "Low Mumble": "mumble", "Whispered Mumble": "mumble",
    # humming
    "Humming": "hum", "Purr": "hum", "Resonant Hum": "hum", "Soft Hum": "hum",
    # coughing group
    "Cough": "cough", "Coughing": "cough", "Hiccup": "other", "Hiccups": "other",
    "Sniff": "sniff", "Snort": "sniff", "Yawn": "yawn",
    # screams
    "Scream": "scream", "Shriek": "scream",
    # whistling
    "Person Whistling Playfully": "whistle", "Person Whistling to Get Attention": "whistle",
    "Sharp Whistle": "whistle", "Soft Whistle": "whistle", "Wolf Whistle": "whistle",
    # mouth / lip
    "Blowing a Kiss": "lipsmack", "Kissing Noises": "lipsmack", "Kissing Sounds": "lipsmack",
    "Lip Smack": "lipsmack", "Smack One's Lips": "lipsmack", "Smacks Lips": "lipsmack",
    "Licking Sound": "lipsmack", "Spitting": "other", "Sucking Noise": "lipsmack",
    # tongue clicks
    "Click One's Tongue": "tsk", "Clicks Tongue": "tsk", "Tongue Click": "tsk", "Tsk": "tsk",
    # eating / drinking
    "Chewing Noises": "other", "Drinking Noises": "other", "Gulps": "swallow",
    "Nervous Gulp": "swallow", "Slurping Noises": "other", "Swallows": "swallow",
    # hand / body
    "Finger Snaps": "other",
    # the three patched-out classes and the real gate
    "No burst": "no_burst", "no_burst": "no_burst",
}

FAMILY_TO_QWEN = {
    "sigh": "[sigh]", "breath": "[breath]", "gasp": "[gasp]",
    "laugh": "[laugh]", "giggle": "[giggle]", "chuckle": "[chuckle]",
    "moan": "[moan]", "groan": "[groan]", "grunt": "[grunt]",
    "sob": "[sob]", "cry": "[cry]", "scream": "[scream]",
    "cough": "[cough]", "throat_clear": "[throat_clear]", "sniff": "[sniff]",
    "yawn": "[yawn]", "hum": "[hum]",
    # families we recognise but deliberately do NOT expose to Qwen in v1
    "tsk": None, "lipsmack": None, "swallow": None, "whistle": None,
    "growl": None, "mumble": None, "other": None, "no_burst": None,
}

# Families routed to manual review even at high confidence (supplement s16/s28).
HARD_FAMILIES = {"moan", "groan", "grunt", "hum", "mumble", "growl", "other"}

# Families that may auto-accept at the easier threshold when a second model agrees.
EASY_FAMILIES = {"sigh", "laugh", "giggle", "gasp", "cough", "yawn", "throat_clear", "scream"}

# ---------------------------------------------------------------- japanese NVV priors
#
# The transcripts are conventionally punctuated Japanese game/anime script text, so a
# vocalisation is often already written as onomatopoeia. We do not trust the text alone
# (supplement s3) but we do use it as a PRIOR that must AGREE with the audio model
# before an isolated clip is rewritten into a tag.

ONOMATOPOEIA_PRIOR = [
    # (regex, expected families, note)
    (r"^[っッ…。！？\s]*(?:ふふ+|フフ+|うふふ|"
     r"あはは|アハハ|ははは|ハハハ|えへへ|"
     r"ヘヘ|くすっ?|クスッ?|ウフフッ?)"
     r"[っッー…。！？\s]*$", {"laugh", "giggle", "chuckle"}, "laugh_onomatopoeia"),
    (r"^[…\s]*(?:はぁ|ハァ|はあ|ふぅ|フゥ|ふう)"
     r"[っッー…。\s]*$", {"sigh", "breath"}, "sigh_onomatopoeia"),
    (r"^[…\s]*(?:はっ|ハッ|ひっ|ひゃっ?|きゃぁ?)"
     r"[っッ…。！？\s]*$", {"gasp", "breath"}, "gasp_onomatopoeia"),
    (r"^[…\s]*(?:んっ|うっ|ぐっ|くぅ|うぐ|ううっ)"
     r"[っッー…。！？\s]*$", {"groan", "moan", "grunt"}, "strain_onomatopoeia"),
]

# Text that carries no lexical content at all -> a pure isolated-NVV candidate.
# Includes the music cue '♪〜' (the score's way of writing "she hums"), which is
# notation rather than something to read aloud. The small tsu 'っ' is deliberately
# NOT here: it is ordinary Japanese orthography for a glottal catch and a TTS model
# can render it, so lines like 'っ！' keep their transcript.
NONLEXICAL_RE = re.compile(
    r"^[…。、！？\s―─—．・“”♪〜~]*$")

# ---------------------------------------------------------------- text normalisation

_ASS_TAG = re.compile(r"\{[^{}]*\}")                       # {\an8} subtitle override blocks
_SPEAKER_PREFIX = re.compile(
    "^\\s*[（(]\\s*(?:アマデウス)?\\s*"
    "(?:牧瀬)?紅莉栖\\s*[）)]\\s*")  # （紅莉栖） subtitle speaker label
# BASE(KANA) ruby where BASE ends in latin/digits -> keep only the spoken reading
_RUBY_LATIN = re.compile(
    r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]+"
    r"[（(]([ぁ-ゟァ-ヿー・]+)[）)]")
_WIDE_ALNUM = re.compile(r"[Ａ-Ｚａ-ｚ０-９．／％]+")
_DASHES = re.compile(r"[─—–－−]")   # ─ — – － − -> ―
_WAVE = re.compile(r"[～]")                            # ～ -> 〜
_HALFKATA_DOT = re.compile(r"[･]")                    # ･ -> ・
_SUPER = str.maketrans({"⁰": "0", "¹": "1", "²": "2", "³": "3",
                        "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
                        "⁸": "8", "⁹": "9"})
_SUPER_RUN = re.compile(r"⁻?[⁰¹²³⁴-⁹]+")

# Evidence-based symbol expansions. Kept tiny and documented; every one of these is a
# word the actor demonstrably speaks (see reports/text_normalisation.md).
SYMBOL_EXPANSIONS = {
    "β": "ベータ",      # β -> ベータ
    "α": "アルファ",  # α -> アルファ
}
# Glyphs a Japanese TTS front-end may not verbalise; kept verbatim but flagged.
RISK_SYMBOLS = "∞℃″＃＊＝＠△▼□×＄＞＜"


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Return (normalised text, list of applied/flagged rule names)."""
    flags: list[str] = []
    t = text

    if _ASS_TAG.search(t):
        t = _ASS_TAG.sub("", t)
        flags.append("stripped_ass_markup")
    if _SPEAKER_PREFIX.search(t):
        t = _SPEAKER_PREFIX.sub("", t)
        flags.append("stripped_speaker_label")
    if _RUBY_LATIN.search(t):
        t = _RUBY_LATIN.sub(lambda m: m.group(1), t)
        flags.append("ruby_reading_kept")

    # 10^-24 style superscript runs -> spoken form ("10の-24乗"), matching how the
    # parallel SGRE take writes the same line.
    def _super(m: re.Match) -> str:
        s = m.group(0)
        neg = s.startswith("⁻")
        digits = s.lstrip("⁻").translate(_SUPER)
        return "の" + ("-" if neg else "") + digits + "乗"

    if _SUPER_RUN.search(t):
        t = _SUPER_RUN.sub(_super, t)
        flags.append("superscript_expanded")

    for sym, rep in SYMBOL_EXPANSIONS.items():
        if sym in t:
            t = t.replace(sym, rep)
            flags.append("symbol_expanded")

    if _WIDE_ALNUM.search(t):
        t = _WIDE_ALNUM.sub(lambda m: unicodedata.normalize("NFKC", m.group()), t)
        flags.append("halfwidth_alnum")
    if _DASHES.search(t):
        t = _DASHES.sub("―", t)
    t = _WAVE.sub("〜", t)
    t = _HALFKATA_DOT.sub("・", t)

    t = re.sub(r"[ 　]+", " ", t).strip()

    if any(c in t for c in RISK_SYMBOLS):
        flags.append("symbol_risk")
    if not t:
        flags.append("empty_after_normalisation")
    return t, sorted(set(flags))


def text_prior(text: str) -> tuple[set[str] | None, str | None]:
    """Onomatopoeia prior for an isolated clip: (expected families, rule name)."""
    for pat, fams, name in ONOMATOPOEIA_PRIOR:
        if re.match(pat, text):
            return fams, name
    return None, None


def is_nonlexical(text: str) -> bool:
    return bool(NONLEXICAL_RE.match(text))


# ---------------------------------------------------------------- io helpers

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


#: A transcript is only "memorisable" - and therefore a leakage risk across splits -
#: if it carries real lexical content. Control tokens like "[sigh]" are SUPPOSED to
#: appear on both sides of a split: that is what makes them testable. Strip the tags
#: and the whitespace, and judge what is left.
MIN_SUBSTANTIVE_CHARS = 6
_TAG_RE = re.compile(r"\[[a-z_]+\]")


def leak_key(text: str) -> str | None:
    """Key for train/eval overlap checks, or None when the line is not substantive."""
    core = _TAG_RE.sub("", text or "")
    core = re.sub(r"\s+", "", core)
    return core if len(core) >= MIN_SUBSTANTIVE_CHARS else None
