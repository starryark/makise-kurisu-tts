"""Shared configuration, taxonomy and helpers for the combined multilingual
Makise Kurisu voice dataset.

One definition of every path, every source rule and every tag string lives here.
Nothing in this module touches the GPU or reads audio.

Design notes that the rest of the pipeline depends on
--------------------------------------------------------
* ``kurisu`` and ``amadeus_kurisu`` are the same *character* and, inside one
  language, the same *performer*.  Across languages they are NOT the same
  performer -- each dub has its own actor -- so every row carries ``voice_id``
  and consumers that need a single voice must filter on it.
* The Qwen-facing transcript field is ``transcription``.  It is generated, never
  hand-edited.  ``text`` is the same line with every tag removed.
* A tag string only reaches ``transcription`` when it is in ``SHIPPED_TAGS``
  *and* the clip earned it either from human review or from a detector whose
  precision was measured on the human gold set.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------- paths

PIPELINE = Path(__file__).resolve().parent
PROJECT = PIPELINE.parent                          # combined/
WORKSPACE = PROJECT.parent                         # New STEINS DataSet/
# the game-voice workspace (voice_dataset_game, kurisu_qwen3tts); override with KURISU_GAME_WORKSPACE
LEGACY = Path(os.environ.get("KURISU_GAME_WORKSPACE", WORKSPACE.parent / "STEINS;GATEVoice"))

GAME_ROOT = LEGACY / "voice_dataset_game"
ANIME_ROOT = WORKSPACE / "datasets"
SONG_ROOT = WORKSPACE / "Cleaned_Songs" / "dataset"
LEGACY_PIPELINE = LEGACY / "kurisu_qwen3tts" / "pipeline"
HF_OUT = LEGACY / "hf_dataset_upload"

AUDIO = PROJECT / "audio"
WAVS = AUDIO / "wavs"
MANIFESTS = PROJECT / "manifests"
REPORTS = PROJECT / "reports"
CACHE = PROJECT / "cache"

for _d in (AUDIO, WAVS, MANIFESTS, REPORTS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ selection

#: Character-level speaker ids as they appear in each upstream dataset.
GAME_SPEAKERS = ("kurisu", "amadeus_kurisu")
ANIME_SPEAKERS = ("makise_kurisu", "amadeus_kurisu")

#: upstream speaker id -> the character variant we publish
VARIANT = {
    "kurisu": "kurisu",
    "makise_kurisu": "kurisu",
    "amadeus_kurisu": "amadeus",
}

LANGUAGES = {
    "japanese": "ja",
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "french": "fr",
    "spanish": "es",
}

#: Each dub is a different performer.  This is the single most important fact a
#: consumer needs, so it is a first-class column rather than a README footnote.
VOICE_ACTOR = {
    "ja": "Asami Imai",              # games, anime, character songs
    "en": "Trina Nishimura",
    "zh": "Chinese dub cast (uncredited in the supplied media)",
    "de": "German dub cast (uncredited in the supplied media)",
    "fr": "French dub cast (uncredited in the supplied media)",
    "es": "Spanish dub cast (uncredited in the supplied media)",
}

TARGET_SR = 24000            # official Qwen3-TTS dataset loader asserts 24 kHz
PEAK_CEILING_DBFS = -1.0     # after group gain no clip may exceed this
GROUP_LUFS_TARGET = -26.0    # every source group is aligned here, one gain per group

# ------------------------------------------------------------------- taxonomy
#
# Two kinds of tag reach the transcript.
#
#   event tags     a non-verbal vocalisation that happens at a point in time.
#                  Placed inline, at the position it occurs.
#   condition tags a property of the whole utterance (how it was delivered, or
#                  what it was played back through).  Placed as a prefix.
#
# Everything else the human reviewers wrote is preserved, but as metadata --
# a control token that appears on five clips out of fourteen thousand is not a
# control, it is noise (guide s12).

EVENT_TAGS = [
    "[sigh]", "[breath]", "[gasp]",
    "[laugh]", "[giggle]", "[chuckle]",
    "[moan]", "[groan]", "[grunt]",
    "[sob]", "[cry]", "[scream]",
    "[cough]", "[throat_clear]", "[sniff]", "[yawn]", "[hum]",
]
CONDITION_TAGS = ["[whisper]", "[yell]", "[digital]", "[echo]", "[singing]"]
STRUCTURAL_TAGS = ["[pause]"]

#: Last resort for a take that is audibly a non-lexical vocalisation but whose
#: family no gate can name.  It says exactly what is known and nothing more,
#: which is the difference between an honest coarse label and a guessed fine one
#: -- and it is what lets most of the game's quarantine pool, whose script line
#: is only "......", enter the corpus at all instead of being thrown away.
GENERIC_NVV_TAG = "[nonverbal]"

ALL_TAGS = EVENT_TAGS + CONDITION_TAGS + STRUCTURAL_TAGS + [GENERIC_NVV_TAG]

#: Minimum shipped occurrences for a tag to stay in the transcript at all.
MIN_TAG_EXAMPLES = 30
#: 30-79 ships but is flagged experimental in the report (guide s12).
EXPERIMENTAL_BELOW = 80

# --------------------------------------------------- human annotation vocabulary
#
# What the manual reviewers actually wrote, normalised.  Left column is every
# spelling observed in the upstream metadata; right column says where it goes.
#
#   ("event", tag)       -> inline NVV tag, kept where the reviewer put it
#   ("condition", tag)   -> whole-clip condition tag, moved to the prefix
#   ("pause", slot)      -> structural; becomes measured pause metadata
#   ("contamination", k) -> non-character sound in the clip; leaves the transcript
#   ("emotion", k)       -> performance note; metadata only
#   ("drop", None)       -> removed, carries no information we can act on

HUMAN_TAG_MAP = {
    # --- non-verbal vocalisations -------------------------------------------
    "sigh": ("event", "[sigh]"),
    "breath": ("event", "[breath]"),
    "breathe": ("event", "[breath]"),
    "gasp": ("event", "[gasp]"),
    "laugh": ("event", "[laugh]"),
    "laughing": ("event", "[laugh]"),
    "laughter": ("event", "[laugh]"),
    "giggle": ("event", "[giggle]"),
    "chuckle": ("event", "[chuckle]"),
    "moan": ("event", "[moan]"),
    "groan": ("event", "[groan]"),
    "grunt": ("event", "[grunt]"),
    "sob": ("event", "[sob]"),
    "cry": ("event", "[cry]"),
    "crying": ("event", "[cry]"),
    "scream": ("event", "[scream]"),
    "cough": ("event", "[cough]"),
    "throat_clear": ("event", "[throat_clear]"),
    "sniff": ("event", "[sniff]"),
    "yawn": ("event", "[yawn]"),
    "hum": ("event", "[hum]"),
    "sneeze": ("event", "[cough]"),        # one instance; nearest shipped family
    # --- delivery / playback conditions -------------------------------------
    "whisper": ("condition", "[whisper]"),
    "whispering": ("condition", "[whisper]"),
    "yell": ("condition", "[yell]"),
    "shout": ("condition", "[yell]"),
    "singing": ("condition", "[singing]"),
    "sing": ("condition", "[singing]"),
    "digital": ("condition", "[digital]"),
    "echo": ("condition", "[echo]"),
    "not digital": ("drop", None),          # explicit negative, not a control
    # --- structural / timing -------------------------------------------------
    "quiet": ("pause", "internal"),
    "pause": ("pause", "internal"),
    "silence": ("pause", "lead"),
    "stop": ("pause", "trail"),
    "fade": ("pause", "trail"),
    # --- sound that is not the character -------------------------------------
    "noise": ("contamination", "noise"),
    "footstep": ("contamination", "footstep"),
    "knock": ("contamination", "knock"),
    "ringtone": ("contamination", "ringtone"),
    # --- performance notes ----------------------------------------------------
    "serious": ("emotion", "serious"),
    "playful": ("emotion", "playful"),
    "teasing": ("emotion", "playful"),
    "angry": ("emotion", "angry"),
    "embarrassed": ("emotion", "embarrassed"),
    "scared": ("emotion", "scared"),
    "mumble": ("emotion", "mumbled"),
    "accent": ("emotion", "foreign_accent"),
    "voice crack": ("emotion", "voice_crack"),
    "vomit": ("drop", None),
}

TAG_RE = re.compile(r"\[[^\[\]]{1,24}\]")


def split_human_tags(text: str) -> tuple[str, list[tuple[int, str]], dict]:
    """Pull every bracketed annotation out of a manually reviewed transcript.

    Returns ``(plain_text, events, notes)`` where

    * ``plain_text`` is the line with every bracket removed and whitespace tidied,
    * ``events`` is ``[(offset_in_plain_text, canonical_event_tag), ...]``,
    * ``notes`` collects the condition / pause / contamination / emotion labels
      and the raw strings that were not recognised.
    """
    notes = {
        "conditions": [],
        "pauses": [],
        "contamination": [],
        "emotion": [],
        "unmapped": [],
        "raw": [],
    }
    events: list[tuple[int, str]] = []
    out: list[str] = []
    pos = 0
    last = 0
    for m in TAG_RE.finditer(text or ""):
        out.append(text[last:m.start()])
        pos += m.start() - last
        last = m.end()
        raw = m.group(0)
        notes["raw"].append(raw)
        key = raw[1:-1].strip().lower().replace("-", "_").replace("  ", " ")
        kind = HUMAN_TAG_MAP.get(key)
        if kind is None:
            key2 = key.replace(" ", "_")
            kind = HUMAN_TAG_MAP.get(key2)
        if kind is None:
            notes["unmapped"].append(raw)
            continue
        what, value = kind
        if what == "event":
            events.append((pos, value))
        elif what == "condition":
            notes["conditions"].append(value)
        elif what == "pause":
            notes["pauses"].append(value)
        elif what == "contamination":
            notes["contamination"].append(value)
        elif what == "emotion":
            notes["emotion"].append(value)
    out.append(text[last:])
    plain = "".join(out)

    # Re-tidy whitespace, and keep the event offsets pointing at the same word.
    fixed: list[tuple[int, str]] = []
    for off, tag in events:
        left = plain[:off]
        trimmed = re.sub(r"[ \u3000]+", " ", left).lstrip()
        fixed.append((len(trimmed), tag))
    plain = re.sub(r"[ \u3000]+", " ", plain).strip()
    for k in ("conditions", "pauses", "contamination", "emotion"):
        notes[k] = sorted(set(notes[k]))
    # clamp offsets after the strip
    fixed = [(min(max(o, 0), len(plain)), t) for o, t in fixed]
    return plain, fixed, notes


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub("", text or "")).strip()


# ------------------------------------------------------- vocal-burst taxonomy
#
# Imported from the previous single-language build so the two datasets share one
# definition of what a "family" is.  Kept here verbatim rather than imported so
# this pipeline has no import-time dependency on the older tree.

FINE_TO_FAMILY = {
    "Breathy Giggle": "giggle", "Childlike Giggle": "giggle", "Nervous Giggle": "giggle",
    "Snorting Giggle": "giggle", "Snicker": "giggle",
    "Chuckle": "chuckle",
    "Cackle": "laugh", "Guffaw": "laugh",
    "Contented Sigh": "sigh", "Exasperated Sigh": "sigh", "Relief Sigh": "sigh",
    "Wistful Sigh": "sigh",
    "Deep Breath": "breath", "Deep Breathing": "breath", "Fast Breathing": "breath",
    "Heavy Breathing": "breath", "Normal Breathing": "breath", "Panting": "breath",
    "Slow Breathing": "breath",
    "Fearful Gasp": "gasp", "Sharp Inhale": "gasp", "Surprised Gasp": "gasp",
    "Exhausted Groan": "groan", "Frustrated Groan": "groan",
    "Pain Moan": "moan", "Pleasure Moan": "moan",
    "Affirmative Grunt": "grunt", "Displeased Grunt": "grunt", "Effort Grunt": "grunt",
    "Convulsive Sob": "sob", "Quiet Sob": "sob", "Sobs": "sob",
    "Mournful Wail": "cry", "Trembling Whimper": "cry",
    "Ahem": "throat_clear", "Clears Throat": "throat_clear",
    "Growl": "growl", "Gurgling": "other", "Hiss": "other",
    "Low Mumble": "mumble", "Whispered Mumble": "mumble",
    "Humming": "hum", "Purr": "hum", "Resonant Hum": "hum", "Soft Hum": "hum",
    "Cough": "cough", "Coughing": "cough", "Hiccup": "other", "Hiccups": "other",
    "Sniff": "sniff", "Snort": "sniff", "Yawn": "yawn",
    "Scream": "scream", "Shriek": "scream",
    "Person Whistling Playfully": "whistle",
    "Person Whistling to Get Attention": "whistle",
    "Sharp Whistle": "whistle", "Soft Whistle": "whistle", "Wolf Whistle": "whistle",
    "Blowing a Kiss": "lipsmack", "Kissing Noises": "lipsmack",
    "Kissing Sounds": "lipsmack", "Lip Smack": "lipsmack",
    "Smack One's Lips": "lipsmack", "Smacks Lips": "lipsmack",
    "Licking Sound": "lipsmack", "Spitting": "other", "Sucking Noise": "lipsmack",
    "Click One's Tongue": "tsk", "Clicks Tongue": "tsk", "Tongue Click": "tsk",
    "Tsk": "tsk",
    "Chewing Noises": "other", "Drinking Noises": "other", "Gulps": "swallow",
    "Nervous Gulp": "swallow", "Slurping Noises": "other", "Swallows": "swallow",
    "Finger Snaps": "other",
    "No burst": "no_burst", "no_burst": "no_burst",
}

FAMILY_TO_TAG = {
    "sigh": "[sigh]", "breath": "[breath]", "gasp": "[gasp]",
    "laugh": "[laugh]", "giggle": "[giggle]", "chuckle": "[chuckle]",
    "moan": "[moan]", "groan": "[groan]", "grunt": "[grunt]",
    "sob": "[sob]", "cry": "[cry]", "scream": "[scream]",
    "cough": "[cough]", "throat_clear": "[throat_clear]", "sniff": "[sniff]",
    "yawn": "[yawn]", "hum": "[hum]",
    "tsk": None, "lipsmack": None, "swallow": None, "whistle": None,
    "growl": None, "mumble": None, "other": None, "no_burst": None,
}
TAG_TO_FAMILY = {v: k for k, v in FAMILY_TO_TAG.items() if v}

# ------------------------------------------------------- text handling per language

#: Scripts without inter-word spaces: rate has to be measured in characters.
CJK_LANGS = {"ja", "zh"}

_ASS_TAG = re.compile(r"\{[^{}]*\}")
_SPEAKER_PREFIX = re.compile(
    r"^\s*[（(]\s*(?:アマデウス)?\s*(?:牧瀬)?紅莉栖\s*[）)]\s*")
_RUBY_LATIN = re.compile(
    r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]+[（(]([ぁ-ゟァ-ヿー・]+)[）)]")
_WIDE_ALNUM = re.compile(r"[Ａ-Ｚａ-ｚ０-９．／％]+")
_DASHES = re.compile(r"[─—–－−]")
_SUPER = str.maketrans({"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"})
_SUPER_RUN = re.compile(r"⁻?[⁰¹²³⁴-⁹]+")
SYMBOL_EXPANSIONS_JA = {"β": "ベータ", "α": "アルファ"}
RISK_SYMBOLS = "∞℃″＃＊＝＠△▼□×＄＞＜"

#: A line with no lexical content at all -> a candidate pure non-verbal clip.
NONLEXICAL_RE = re.compile(r"^[…。、！？\s―─—．・“”♪〜~\.\!\?,\-–—'\"]*$")


def normalize_text(text: str, lang: str) -> tuple[str, list[str]]:
    """Light, language-aware normalisation.  Returns (text, applied rule names)."""
    flags: list[str] = []
    t = text or ""

    if _ASS_TAG.search(t):
        t = _ASS_TAG.sub("", t)
        flags.append("stripped_ass_markup")
    if lang == "ja" and _SPEAKER_PREFIX.search(t):
        t = _SPEAKER_PREFIX.sub("", t)
        flags.append("stripped_speaker_label")
    if lang == "ja" and _RUBY_LATIN.search(t):
        t = _RUBY_LATIN.sub(lambda m: m.group(1), t)
        flags.append("ruby_reading_kept")

    def _super(m: re.Match) -> str:
        s = m.group(0)
        neg = s.startswith("⁻")
        return "の" + ("-" if neg else "") + s.lstrip("⁻").translate(_SUPER) + "乗"

    if lang == "ja" and _SUPER_RUN.search(t):
        t = _SUPER_RUN.sub(_super, t)
        flags.append("superscript_expanded")
    if lang == "ja":
        for sym, rep in SYMBOL_EXPANSIONS_JA.items():
            if sym in t:
                t = t.replace(sym, rep)
                flags.append("symbol_expanded")

    if _WIDE_ALNUM.search(t):
        t = _WIDE_ALNUM.sub(lambda m: unicodedata.normalize("NFKC", m.group()), t)
        flags.append("halfwidth_alnum")
    t = _DASHES.sub("―" if lang in CJK_LANGS else "-", t)
    t = t.replace("～", "〜").replace("･", "・")
    t = re.sub(r"[ \u3000]+", " ", t).strip()

    if any(c in t for c in RISK_SYMBOLS):
        flags.append("symbol_risk")
    if not t:
        flags.append("empty_after_normalisation")
    return t, sorted(set(flags))


def is_nonlexical(text: str) -> bool:
    return bool(NONLEXICAL_RE.match(text or ""))


def lexical_units(text: str, lang: str) -> int:
    """Characters for CJK, words elsewhere -- the denominator for speaking rate."""
    core = strip_tags(text)
    if lang in CJK_LANGS:
        return len(re.sub(r"[\s…。、！？，,\.\!\?\-―〜・“”\"']", "", core))
    return len([w for w in re.split(r"[\s\u3000]+", core) if re.search(r"\w", w)])


MIN_SUBSTANTIVE_UNITS = 4
def leak_key(text: str, lang: str) -> str | None:
    """Key for train/eval overlap checks.

    Tags are supposed to appear on both sides of a split -- that is what makes
    them testable -- so only the lexical remainder counts.
    """
    core = strip_tags(text)
    core = re.sub(r"[\s\u3000]+", "", core).lower()
    if lexical_units(text, lang) < MIN_SUBSTANTIVE_UNITS:
        return None
    return core


# ------------------------------------------------------------------ io helpers

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


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
