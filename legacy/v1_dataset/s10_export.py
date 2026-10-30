"""Stage 10 - splits, Qwen export, probe set, dataset report.

Splits hold out whole **scenes** (a scenario file in the games, an episode in the
anime), so a validation clip never shares a recording session with a training clip.
On top of that, scenes that share a substantive line are welded into one component
before assignment - SG and RE:BOOT re-recorded thousands of the same lines, and
without that step the same sentence would appear on both sides of the split.

Exports exactly the three fields the official trainer reads (audio / text /
ref_audio); everything else stays in the master manifest.
"""
from __future__ import annotations

import collections
import csv
import json
import random
import statistics

from common import (FAMILY_TO_QWEN, MANIFESTS, QWEN, REPORTS, leak_key,
                    read_jsonl, utf8_stdout, write_jsonl)

utf8_stdout()

REF_REL = "./audio/ref.wav"
TARGETS = {"train": 0.90, "val": 0.05, "test": 0.05}
SEED = 20260821

TAGS = sorted({t for t in FAMILY_TO_QWEN.values() if t})


def tags_in(text: str) -> list[str]:
    return [t for t in TAGS if t in text]


class Union:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> None:
    master = list(read_jsonl(MANIFESTS / "09_master.jsonl"))
    usable = [m for m in master if m["status"] == "auto_accept" and m["qwen_text"]]
    held = [m for m in master if m["status"] != "auto_accept"]
    print(f"{len(usable)} usable / {len(master)} clips  ({len(held)} held back)")

    # ---- scene components
    u = Union()
    for m in usable:
        u.find((m["source_group"], m["scene"]))
    by_text = collections.defaultdict(list)
    for m in usable:
        k = leak_key(m["text"])
        if k:
            by_text[k].append((m["source_group"], m["scene"]))
    for scenes in by_text.values():
        for s in scenes[1:]:
            u.join(scenes[0], s)

    comps = collections.defaultdict(list)
    for m in usable:
        comps[u.find((m["source_group"], m["scene"]))].append(m)

    # ---- greedy assignment, balancing clip count and tagged-clip count
    items = sorted(comps.values(), key=lambda c: -len(c))
    rng = random.Random(SEED)
    rng.shuffle(items)
    items.sort(key=lambda c: -len(c))

    n_tot = len(usable)
    n_tag = sum(1 for m in usable if tags_in(m["qwen_text"]))
    want = {k: {"clips": v * n_tot, "tagged": v * n_tag} for k, v in TARGETS.items()}
    have = {k: {"clips": 0, "tagged": 0} for k in TARGETS}
    assign = {}
    for comp in items:
        ct = sum(1 for m in comp if tags_in(m["qwen_text"]))

        def deficit(k):
            d = (want[k]["clips"] - have[k]["clips"]) / max(want[k]["clips"], 1)
            if ct and want[k]["tagged"]:
                d += 1.5 * (want[k]["tagged"] - have[k]["tagged"]) / want[k]["tagged"]
            return d

        k = max(TARGETS, key=deficit)
        have[k]["clips"] += len(comp)
        have[k]["tagged"] += ct
        for m in comp:
            assign[m["id"]] = k

    for m in usable:
        m["split"] = assign[m["id"]]
    for m in held:
        m["split"] = None
    write_jsonl(MANIFESTS / "09_master.jsonl", master)

    # ---- Qwen export
    counts = {}
    for split in TARGETS:
        rows = [m for m in usable if m["split"] == split]
        rows.sort(key=lambda m: m["id"])
        counts[split] = write_jsonl(QWEN / f"{split}_raw.jsonl", [
            {"audio": m["audio"], "text": m["qwen_text"], "ref_audio": REF_REL}
            for m in rows])
    write_jsonl(QWEN / "held_back_uncertain.jsonl", [
        {"audio": m["audio"], "text": m["qwen_text"], "ref_audio": REF_REL,
         "_status": m["status"], "_reason": m["decision_reason"]}
        for m in held if m["qwen_text"]])

    # ---- probe set (guide s26): tagged prompt, same text untagged, negative control
    probe = []
    tagged = [m for m in usable if tags_in(m["qwen_text"]) and m["split"] != "train"]
    for m in tagged[:40]:
        probe.append({"kind": "tagged", "text": m["qwen_text"], "from": m["id"]})
        probe.append({"kind": "untagged_same",
                      "text": " ".join(m["qwen_text"].replace(
                          tags_in(m["qwen_text"])[0], " ").split()) or "……",
                      "from": m["id"]})
    plain = [m for m in usable
             if m["split"] == "test" and not tags_in(m["qwen_text"])
             and 8 <= len(m["text"]) <= 40]
    for m in plain[:40]:
        probe.append({"kind": "negative_control", "text": m["text"], "from": m["id"]})
    for t in sorted({t for m in usable for t in tags_in(m["qwen_text"])}):
        probe += [{"kind": "isolated_tag", "text": t, "from": None},
                  {"kind": "tag_initial", "text": f"{t} そう、そういうこと。", "from": None},
                  {"kind": "tag_medial", "text": f"それは違う。{t} もう一度説明するわ。", "from": None},
                  {"kind": "tag_final", "text": f"もういいわ、勝手にして。{t}", "from": None}]
    write_jsonl(QWEN / "probe_prompts.jsonl", probe)

    # ---- report
    def stats(rows):
        d = sorted(m["dur"] for m in rows)
        if not d:
            return {}
        return {"clips": len(rows), "hours": round(sum(d) / 3600, 3),
                "median_s": round(statistics.median(d), 2),
                "p05_s": round(d[int(0.05 * len(d))], 2),
                "p95_s": round(d[int(0.95 * len(d))], 2),
                "min_s": d[0], "max_s": d[-1]}

    tag_rows = collections.defaultdict(list)
    for m in usable:
        for t in tags_in(m["qwen_text"]):
            tag_rows[t].append(m)

    dur_bucket = collections.Counter()
    for m in usable:
        d = m["dur"]
        dur_bucket["<1s" if d < 1 else "1-2s" if d < 2 else "2-4s" if d < 4
                    else "4-8s" if d < 8 else "8-15s" if d < 15 else ">15s"] += 1

    train_texts = {leak_key(m["text"]) for m in usable if m["split"] == "train"}
    train_texts.discard(None)
    leak = sum(1 for m in usable if m["split"] != "train"
               and leak_key(m["text"]) in train_texts)

    report = {
        "totals": stats(usable),
        "held_back": {"clips": len(held),
                      "by_status": dict(collections.Counter(m["status"] for m in held))},
        "splits": {k: stats([m for m in usable if m["split"] == k]) for k in TARGETS},
        "split_scene_components": len(comps),
        "substantive_text_leakage_train_to_eval": leak,
        "by_speaker_variant": dict(collections.Counter(m["variant"] for m in usable)),
        "by_source_group": dict(collections.Counter(m["source_group"] for m in usable)),
        "by_speech_status": dict(collections.Counter(m["speech_status"] for m in usable)),
        "from_quarantine": sum(1 for m in usable if m["from_quarantine"]),
        "duration_buckets": dict(dur_bucket),
        "nvv": {
            "clips_with_tag": sum(1 for m in usable if tags_in(m["qwen_text"])),
            "share_of_dataset": round(
                sum(1 for m in usable if tags_in(m["qwen_text"])) / len(usable), 4),
            "per_tag": {t: {"clips": len(v), "hours": round(sum(x["dur"] for x in v) / 3600, 3),
                            "median_s": round(statistics.median([x["dur"] for x in v]), 2),
                            "splits": dict(collections.Counter(x["split"] for x in v)),
                            "isolated": sum(1 for x in v
                                            if x["speech_status"] == "isolated_nvv_candidate"),
                            "in_speech": sum(1 for x in v for e in x["events"]
                                             if e.get("scope") == "in_speech" and e.get("accepted"))}
                        for t, v in sorted(tag_rows.items())},
        },
        "audio": {"sample_rate": 24000, "channels": 1, "format": "WAV PCM_16",
                  "ref_audio": REF_REL,
                  "same_ref_on_every_row": True},
        "export": {k: counts[k] for k in counts},
        "probe_prompts": len(probe),
    }
    (REPORTS / "10_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(REPORTS / "10_clip_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "status", "variant", "source_group", "scene",
                    "dur", "speech_status", "tags", "qwen_text"])
        for m in master:
            w.writerow([m["id"], m.get("split") or "", m["status"], m["variant"],
                        m["source_group"], m["scene"], m["dur"], m["speech_status"],
                        "|".join(tags_in(m["qwen_text"])), m["qwen_text"]])

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
