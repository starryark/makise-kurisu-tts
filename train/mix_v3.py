"""Compose one experiment's train/val sets from the coded v3 sources.

Each flag switches on exactly one v3 change, so the runs in SELECTION_RULE.md (v3)
differ from their parent in one thing:

  E1  --v2_ids                                   v2's clips, plain text, Auto prefix
  E2  E1 + --language japanese                   language-id rows (p_auto in trainer)
  E3  E2 + --replay                              native zh/en replay rows
  E5  E1 + --speakers kurisu,amadeus,phone --phone_aug
  E6  E1 + --text v3 --tag_only                  event tokens (+ isolated events)
  E7  E6 + --sfx                                 background-SFX augmentation
  E4  E3 + --pseudo_share 0.06                   Kurisu-voiced native Mandarin (p11)
Writes <out>/train_with_codes.jsonl, <out>/val_with_codes.jsonl and <out>/mix.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

WORK = Path(os.environ.get("KURISU_WORK", "/workspace/kurisu"))
Q = WORK / "qwen_v3"
TOKEN = re.compile(r"\[[a-z_]+\]")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2_ids", default=None, help="v2_kept_ids.txt: restrict JA rows to v2's clips")
    ap.add_argument("--speakers", default="kurisu")
    ap.add_argument("--text", choices=["plain", "v3"], default="plain")
    ap.add_argument("--tag_only", action="store_true", help="keep isolated-event takes (v3 text)")
    ap.add_argument("--language", choices=["auto", "japanese"], default="auto")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--replay_share", type=float, default=0.15,
                    help="replay rows as a share of the final train set (capped by availability)")
    ap.add_argument("--pseudo_share", type=float, default=0.0,
                    help="Kurisu-voiced native Mandarin rows (p11) as a share of the final train set")
    ap.add_argument("--phone_aug", action="store_true")
    ap.add_argument("--sfx", action="store_true")
    ap.add_argument("--event_repeat", type=int, default=1,
                    help="repeat train rows containing an event token this many times")
    ap.add_argument("--phone_repeat", type=int, default=1,
                    help="repeat the real phone takes this many times")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.tag_only and args.text != "v3":
        raise SystemExit("--tag_only needs --text v3 (plain text has no event tokens)")

    speakers = {s.strip() for s in args.speakers.split(",")}
    v2 = None
    if args.v2_ids:
        v2 = set(Path(args.v2_ids).read_text(encoding="utf-8").split())

    def ja_rows(split: str) -> list[dict]:
        out = []
        for r in read(Q / f"ja_{split}_with_codes.jsonl"):
            if r["speaker"] not in speakers:
                continue
            in_v2 = v2 is not None and r["id"] in v2
            if r.get("tag_only") and not in_v2 and not args.tag_only:
                continue
            if v2 is not None and r["speaker"] == "kurisu" and not r.get("tag_only") and not in_v2:
                # --v2_ids pins the *default* voice to v2's clips (every one of them,
                # tag-only or not), so an ablation that adds speakers or isolated events
                # changes only that
                continue
            text = r["text"] if args.text == "v3" else r["text_plain"]
            if not text.strip():
                continue
            out.append({**r, "text": text, "language": args.language})
        return out

    def aug_rows(split: str) -> list[dict]:
        out = []
        for r in read(Q / f"aug_{split}_with_codes.jsonl"):
            kind = r.get("augment", "")
            if (kind.startswith("phone") and args.phone_aug and "phone" in speakers) or \
               (kind == "sfx" and args.sfx):
                text = r["text"] if args.text == "v3" or kind == "sfx" else TOKEN.sub("", r["text"])
                out.append({**r, "text": text, "language": args.language})
        return out

    rng = random.Random(args.seed)
    train = ja_rows("train")
    aug = aug_rows("train")
    if (args.phone_aug or args.sfx) and not aug:
        raise SystemExit("augmentation requested but qwen_v3/aug_train_with_codes.jsonl has no matching rows")
    if args.phone_aug and not any(r.get("augment", "").startswith("phone") for r in aug):
        raise SystemExit("--phone_aug: no phone augment rows")
    if args.sfx and not any(r.get("augment") == "sfx" for r in aug):
        raise SystemExit("--sfx: no sfx augment rows")
    events = re.compile(r"\[(sigh|nv|inhale|laugh)\]")
    extra = []
    for r in train:
        if args.event_repeat > 1 and events.search(r["text"]):
            extra += [r] * (args.event_repeat - 1)
        if args.phone_repeat > 1 and r["speaker"] == "phone":
            extra += [r] * (args.phone_repeat - 1)
    train += extra + aug
    val = ja_rows("val")
    if args.replay:
        pool = read(Q / "replay_train_with_codes.jsonl")
        if not pool:
            raise SystemExit("--replay: qwen_v3/replay_train_with_codes.jsonl is missing or empty")
        want = int(round(args.replay_share / (1 - args.replay_share) * len(train)))
        rng.shuffle(pool)
        train += pool[:want]
        val += read(Q / "replay_val_with_codes.jsonl")
        if want > len(pool):
            raise SystemExit(f"--replay: asked {want} rows, only {len(pool)} available")
    if args.pseudo_share > 0:
        pool = read(Q / "pseudo_train_with_codes.jsonl")
        want = int(round(args.pseudo_share / (1 - args.pseudo_share) * len(train)))
        if not pool:
            raise SystemExit("--pseudo_share: qwen_v3/pseudo_train_with_codes.jsonl is missing or empty")
        if want > len(pool):
            print(f"[mix] pseudo asked {want}, using all {len(pool)}")
        rng.shuffle(pool)
        train += pool[:want]
        val += read(Q / "pseudo_val_with_codes.jsonl")
    rng.shuffle(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (out / f"{name}_with_codes.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {
        "args": vars(args),
        "train": len(train), "val": len(val),
        "train_hours": round(sum(r["dur"] for r in train) / 3600, 3),
        "speakers": dict(Counter(r["speaker"] for r in train).most_common(12)),
        "n_speakers": len({r["speaker"] for r in train}),
        "languages": dict(Counter(r["language"] for r in train)),
        "tokens": dict(Counter(t for r in train for t in TOKEN.findall(r["text"]))),
        "augment": dict(Counter(r.get("augment") for r in train if r.get("augment"))),
        "repeated_rows": len(extra),
    }
    (out / "mix.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
