"""Stage 10 - cross-lingual line pairing, and splits that cannot leak.

Two things happen here.

**Parallel groups.**  Every dub of an episode is the same performance in another
language, and each build recorded its cue times on its own programme timeline
after an independently measured per-episode offset.  Recovering the pairing is
therefore a one-parameter problem per (language, episode): find the shift that
maximises total overlap between that language's Kurisu cues and the Japanese
ones, then link cues that overlap once shifted.  Nobody has to label anything.
The result is a genuine six-way parallel corpus of the same character saying the
same line, which is worth having on its own and is also the thing a naive split
would leak.

**Splits.**  A split boundary has to cut between *performances*, not between
files.  Two relations are unioned before anything is assigned:

* same episode or same game scenario file, across every language -- adjacent
  lines share a scene, a mood and a room, and the dubs share the script;
* same parallel group -- the French take of a line cannot be in train while the
  Spanish take of the same line is in test;

and then any line whose lexical content repeats within a language is pulled onto
one side afterwards -- RE:BOOT re-records thousands of STEINS;GATE script lines,
and an identical sentence on both sides of the boundary is a memorisation test,
not a generalisation test.  That repair runs after assignment rather than as a
third union, because unioning it cascades scene into scene until 55% of the
corpus is one indivisible component.

Whole components are then packed into train/validation/test to hit the target
proportions within every (language, medium) stratum, so no split ends up missing
a language.  Tag strings are deliberately *not* part of the leakage key: a
control token is supposed to appear on both sides, that is what makes it testable.
"""
from __future__ import annotations

import collections
import json
import random

import numpy as np

from kcommon import (ALL_TAGS, MANIFESTS, REPORTS, leak_key, read_jsonl,
                     utf8_stdout, write_json, write_jsonl)

utf8_stdout()

SEED = 20260905
TARGET = {"train": 0.90, "validation": 0.05, "test": 0.05}
OFFSET_RANGE_S = 40.0
OFFSET_STEP_S = 0.05
MIN_PAIR_OVERLAP = 0.40      # of the shorter cue


class Union:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def best_offset(target: list[tuple[float, float]],
                ref: list[tuple[float, float]]) -> tuple[float, float]:
    """Shift that maximises total overlap of ``target`` onto ``ref``."""
    if not target or not ref:
        return 0.0, 0.0
    # coarse grid first, then refine around the winner
    def score(d):
        tot = 0.0
        j = 0
        ts = sorted((s + d, e + d) for s, e in target)
        for s, e in ts:
            while j < len(ref) and ref[j][1] < s:
                j += 1
            k = j
            while k < len(ref) and ref[k][0] < e:
                tot += max(0.0, min(e, ref[k][1]) - max(s, ref[k][0]))
                k += 1
        return tot

    ref = sorted(ref)
    coarse = np.arange(-OFFSET_RANGE_S, OFFSET_RANGE_S + 1e-9, 0.25)
    b = max(coarse, key=score)
    fine = np.arange(b - 0.30, b + 0.30 + 1e-9, OFFSET_STEP_S)
    b = max(fine, key=score)
    return float(round(b, 3)), float(round(score(b), 2))


def main() -> None:
    rng = random.Random(SEED)
    rows = list(read_jsonl(MANIFESTS / "09_master.jsonl"))
    by_id = {r["id"]: r for r in rows}

    # ---------------------------------------------------- parallel groups
    anime = [r for r in rows if r["medium"] == "anime"
             and r.get("reference_start") is not None]
    by_scene_lang = collections.defaultdict(list)
    for r in anime:
        by_scene_lang[(r["scene"], r["language"])].append(r)

    offsets, pair_u = {}, Union()
    pairs_made = 0
    for scene in sorted({s for s, _ in by_scene_lang}):
        ref_rows = by_scene_lang.get((scene, "ja"), [])
        ref = [(r["reference_start"], r["reference_end"]) for r in ref_rows]
        for lang in ("en", "zh", "de", "fr", "es"):
            tgt_rows = by_scene_lang.get((scene, lang), [])
            if not tgt_rows or not ref:
                continue
            tgt = [(r["reference_start"], r["reference_end"]) for r in tgt_rows]
            d, sc = best_offset(tgt, ref)
            offsets[f"{scene}|{lang}"] = {"offset_s": d, "overlap_s": sc,
                                          "target_cues": len(tgt), "ref_cues": len(ref)}
            for r in tgt_rows:
                s, e = r["reference_start"] + d, r["reference_end"] + d
                best, bid = 0.0, None
                for q in ref_rows:
                    o = max(0.0, min(e, q["reference_end"]) - max(s, q["reference_start"]))
                    shorter = min(e - s, q["reference_end"] - q["reference_start"])
                    if shorter > 0 and o / shorter > best:
                        best, bid = o / shorter, q["id"]
                if bid and best >= MIN_PAIR_OVERLAP:
                    pair_u.union(bid, r["id"])
                    pairs_made += 1

    pgroups = collections.defaultdict(list)
    for r in anime:
        root = pair_u.find(r["id"]) if r["id"] in pair_u.p else None
        if root is not None:
            pgroups[root].append(r["id"])
    pgid = {}
    for n, (root, ids) in enumerate(sorted(pgroups.items())):
        if len(ids) < 2:
            continue
        for i in ids:
            pgid[i] = f"pg_{n:05d}"

    # ---------------------------------------------------- leakage components
    u = Union()
    for r in rows:
        u.find(r["id"])
    scene_key = collections.defaultdict(list)
    for r in rows:
        scene_key[(r["medium"], r["scene"])].append(r["id"])
    for ids in scene_key.values():
        for i in ids[1:]:
            u.union(ids[0], i)
    for ids in pgroups.values():
        for i in ids[1:]:
            u.union(ids[0], i)
    # Repeated lines are handled *after* assignment, not by unioning them into the
    # components.  RE:BOOT re-records so much of the STEINS;GATE script that
    # linking identical text to identical text and scene to scene cascades into a
    # single 7,722-clip component -- 55% of the corpus in one indivisible lump,
    # which makes a balanced split impossible.  Assign on scenes and parallel
    # groups, then move any repeated line that straddles a boundary.
    text_key = collections.defaultdict(list)
    for r in rows:
        k = leak_key(r["text"], r["language"])
        if k:
            text_key[(r["language"], k)].append(r["id"])
    dup_groups = [ids for ids in text_key.values() if len(ids) > 1]
    dup_links = sum(len(ids) - 1 for ids in dup_groups)

    comp = collections.defaultdict(list)
    for r in rows:
        comp[u.find(r["id"])].append(r["id"])

    # ---------------------------------------------------- assignment
    shipped = {r["id"] for r in rows if r["status"] == "ok"}
    strata = sorted({(by_id[i]["language"], by_id[i]["medium"]) for i in shipped})
    quota = {(s, sp): TARGET[sp] * sum(1 for i in shipped
                                       if (by_id[i]["language"], by_id[i]["medium"]) == s)
             for s in strata for sp in TARGET}
    have = collections.Counter()

    comps = sorted(comp.values(), key=lambda ids: -sum(1 for i in ids if i in shipped))
    rng.shuffle(comps)
    comps.sort(key=lambda ids: -sum(1 for i in ids if i in shipped))

    assign = {}
    for ids in comps:
        mine = [i for i in ids if i in shipped]
        if not mine:
            for i in ids:
                assign[i] = None
            continue
        cnt = collections.Counter((by_id[i]["language"], by_id[i]["medium"])
                                  for i in mine)
        # place the component where it leaves the largest relative deficit filled
        best, bestsp = None, "train"
        for sp in ("test", "validation", "train"):
            deficit = sum(max(0.0, quota[(s, sp)] - have[(s, sp)]) for s in cnt)
            overflow = sum(max(0.0, have[(s, sp)] + cnt[s] - quota[(s, sp)])
                           for s in cnt)
            score = deficit - 1.6 * overflow
            if best is None or score > best:
                best, bestsp = score, sp
        for i in ids:
            assign[i] = bestsp if i in shipped else None
        for s, n in cnt.items():
            have[(s, bestsp)] += n

    # ---- repeated-line repair ------------------------------------------------
    # A line whose lexical content appears on both sides of a boundary is a
    # memorisation test, not a generalisation test.  Whole repeated groups move to
    # train when any member is already there, and otherwise collapse onto the
    # evaluation split that holds most of them.
    # Moving by lexical group cuts across parallel groups, which component
    # assignment had already placed on one side of every boundary: pulling the
    # Japanese take of a repeated line into train leaves its five translations
    # behind, which is precisely the leak the parallel grouping exists to
    # prevent.  So the two repairs are alternated to a fixed point rather than
    # run once.  This is not a third union -- the components are untouched; only
    # the assignment moves, and "train" is absorbing, so it settles quickly.
    pg_members = collections.defaultdict(list)
    for cid, g in pgid.items():
        pg_members[g].append(cid)

    moved = collections.Counter()

    def collapse(groups) -> int:
        n = 0
        for ids in groups:
            places = [assign[i] for i in ids if i in shipped and assign.get(i)]
            if len(set(places)) <= 1:
                continue
            # Fixed precedence rather than "the split holding most of them".
            # A majority rule has no fixed point once the two repairs alternate:
            # a group with no train member can be pulled to validation by one
            # pass and back to test by the next, forever.  Ordering the splits
            # makes every move monotone, so the loop provably settles -- and the
            # ping-pong was invisible to k12, which compares train against each
            # evaluation split but never validation against test.
            target = next(s for s in ("train", "validation", "test") if s in places)
            for i in ids:
                if i in shipped and assign.get(i) and assign[i] != target:
                    moved[(assign[i], target)] += 1
                    assign[i] = target
                    n += 1
        return n

    passes = 0
    for passes in range(1, 51):
        if collapse(dup_groups) + collapse(pg_members.values()) == 0:
            break

    out = []
    for r in rows:
        m = dict(r)
        m["parallel_group"] = pgid.get(r["id"])
        m["leak_component"] = u.find(r["id"])
        m["split"] = assign.get(r["id"]) if r["status"] == "ok" else None
        out.append(m)
    write_jsonl(MANIFESTS / "10_split.jsonl", out)

    # ---------------------------------------------------- verification
    def keys(split):
        return {leak_key(m["text"], m["language"]) for m in out
                if m["split"] == split and leak_key(m["text"], m["language"])}
    tr = keys("train")
    leak_val = len(tr & keys("validation"))
    leak_test = len(tr & keys("test"))
    pg_split = collections.defaultdict(set)
    for m in out:
        if m["parallel_group"] and m["split"]:
            pg_split[m["parallel_group"]].add(m["split"])
    pg_broken = sum(1 for v in pg_split.values() if len(v) > 1)

    per = collections.defaultdict(collections.Counter)
    hours = collections.defaultdict(float)
    for m in out:
        if not m["split"]:
            continue
        per[m["split"]][f"{m['language']}:{m['medium']}"] += 1
        hours[m["split"]] += m["duration"]
    tagsplit = collections.defaultdict(collections.Counter)
    for m in out:
        if not m["split"]:
            continue
        for t in ALL_TAGS:
            if t in m["transcription"]:
                tagsplit[t][m["split"]] += 1

    report = {
        "clips": len(out),
        "shipped": len(shipped),
        "components": len(comp),
        "component_size": {
            "max": max(len(v) for v in comp.values()),
            "median": int(np.median([len(v) for v in comp.values()])),
        },
        "duplicate_text_groups": len(dup_groups),
        "duplicate_text_links": dup_links,
        "clips_moved_by_repeated_line_repair": {f"{a}->{b}": n
                                                for (a, b), n in moved.items()},
        "repair_passes_to_fixed_point": passes,
        "parallel_groups": len(set(pgid.values())),
        "clips_in_a_parallel_group": len(pgid),
        "parallel_pairs_linked": pairs_made,
        "parallel_group_size": dict(collections.Counter(
            collections.Counter(pgid.values()).values())),
        "measured_offsets_sample": dict(list(sorted(offsets.items()))[:12]),
        "splits": {k: dict(v) for k, v in per.items()},
        "split_clips": {k: sum(v.values()) for k, v in per.items()},
        "split_hours": {k: round(v / 3600.0, 3) for k, v in hours.items()},
        "substantive_text_leakage_train_to_validation": leak_val,
        "substantive_text_leakage_train_to_test": leak_test,
        "parallel_groups_split_across_boundaries": pg_broken,
        "tag_presence_by_split": {t: dict(c) for t, c in sorted(tagsplit.items())},
    }
    write_json(REPORTS / "10_split.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
