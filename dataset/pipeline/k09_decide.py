"""Stage 09 - one decision per clip, and the master manifest.

Everything upstream is merged here into ``manifests/09_master.jsonl``: the
normalised transcript, the audio facts, every model score, and the one field a
trainer actually reads -- ``transcription``.

Policy, in the order it is applied
----------------------------------
1. **A human label always wins.**  Every bracket a reviewer wrote is kept: event
   tags stay where they were written, condition tags move to a fixed prefix
   position so the string form is consistent, and the labels that are not
   controls -- ``[noise]``, ``[footstep]``, ``[serious]`` -- leave the transcript
   for metadata, because a token that appears on seven clips out of fourteen
   thousand is not something a model can be steered with.
2. **A clip whose script line says nothing becomes its sound, or it is held
   back.**  The game quarantine pool is 272 takes whose scenario text is only
   ``……``: the line *is* a breath, a gasp, a small reaction.  When the calibrated
   gate names the sound, the transcript becomes that tag and the clip enters the
   corpus.  When it cannot, the clip is held back rather than shipped with a
   guessed control token -- a wrong tag is worse than a missing one.
3. **An automatic event tag has to survive three independent signals**: the
   locator says a burst is here, the aligner says no transcript unit reaches it,
   and the classifier plus CED agree on what it is at the operating point stage
   08 measured.
4. **An automatic condition tag has to clear its own language's measured
   threshold**, and the rate it fires at in each language/medium cell is compared
   with the rate humans marked it at.  A cell that fires far above the human
   prevalence is suppressed and reported, because that is what a domain shift
   looks like from the inside.
5. **A tag with too little support is demoted** rather than shipped as a control
   nobody can steer.
"""
from __future__ import annotations

import collections
import json
import pickle
import re

import numpy as np

from kcommon import (ALL_TAGS, CJK_LANGS, CONDITION_TAGS, EVENT_TAGS,
                     EXPERIMENTAL_BELOW,
                     FAMILY_TO_TAG, GENERIC_NVV_TAG, MANIFESTS,
                     MIN_TAG_EXAMPLES, NONLEXICAL_RE, REPORTS, TAG_TO_FAMILY,
                     read_jsonl,
                     utf8_stdout, write_json, write_jsonl)

utf8_stdout()

PAUSE_MIN_S = 0.60          # floor; the real threshold is measured per language
PAUSE_PERCENTILE = 90       # a tagged pause is a long one *for this language*
PAUSE_MAX_PER_CLIP = 2      # a transcript full of controls is not a controllable one
PAUSE_CONTEXT = 8           # chars either side searched for a written ellipsis
WRITTEN_PAUSE = "…‥"        # only an explicit ellipsis counts as already-written
CONDITION_ORDER = ["[digital]", "[echo]", "[singing]", "[whisper]", "[yell]"]
PREVALENCE_TOLERANCE = 3.0  # cell fires at most this multiple of the human rate
PUNCT = "…。、！？ 　―─—．・“”,.!?;:"


def token_offsets(text: str, tokens: list[dict]) -> list[tuple[int, int]]:
    offs, cur = [], 0
    for t in tokens:
        s = t["text"]
        i = text.find(s, cur) if s else -1
        if i < 0:
            offs.append((cur, cur))
        else:
            offs.append((i, i + len(s)))
            cur = i + len(s)
    return offs


def insertion_index(text: str, offs, after_token: int | None) -> int:
    """Where a tag string goes. Punctuation is kept, never replaced."""
    if after_token is None:
        return 0
    end = offs[after_token][1] if after_token < len(offs) else len(text)
    while end < len(text) and text[end] in PUNCT:
        end += 1
    return end


def splice(text: str, insertions: list[tuple[int, str]], spaced: bool = True) -> str:
    """Insert tag strings at the given offsets.

    ``spaced`` is False for Japanese and Chinese.  Those scripts have no
    inter-word spaces, so padding a tag with them leaves an orphan space behind
    once the tags are stripped -- and then ``strip_tags(transcription) == text``,
    the invariant that lets a consumer drop the tags and recover the plain line,
    no longer holds.  A tag written flush against the surrounding characters is
    also how the human reviewers wrote theirs in the first place.
    """
    out = text
    for at, tag in sorted(insertions, key=lambda t: -t[0]):
        at = min(max(at, 0), len(out))
        left, right = out[:at], out[at:]
        if spaced:
            out = (left.rstrip() + (" " if left.strip() else "") + tag
                   + (" " if right.strip() else "") + right.lstrip())
        else:
            out = left + tag + right
    return re.sub(r"\s+", " ", out).strip()


def main() -> None:
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    feats = {f["id"]: f for f in read_jsonl(MANIFESTS / "03_features.jsonl")}
    spk = {s["id"]: s for s in read_jsonl(MANIFESTS / "04_speaker.jsonl")}
    loc = {o["id"]: o for o in read_jsonl(MANIFESTS / "05_locator.jsonl")}
    ali = {o["id"]: o for o in read_jsonl(MANIFESTS / "06_alignment.jsonl")}
    _nvv = json.loads((MANIFESTS / "08_nvv_policy.json").read_text("utf-8"))
    policy = _nvv["policy"]
    presence = _nvv.get("presence_gate")
    with open(MANIFESTS / "08_condition_models.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models, chosen = bundle["models"], bundle["chosen"]
    with open(MANIFESTS / "08_condition_features.pkl", "rb") as fh:
        cfeat = pickle.load(fh)

    obs = collections.defaultdict(list)
    for o in read_jsonl(MANIFESTS / "07_nvv.jsonl"):
        obs[o["id"]].append(o)

    # A pause worth a control token is a long one *for its language*.  Japanese
    # dialogue breathes differently from Spanish, and a fixed threshold would tag
    # every sentence boundary in one language and nothing at all in another, so
    # the bar is the language's own upper decile of internal silences.
    gaps_by_lang = collections.defaultdict(list)
    for r in rows:
        A = ali.get(r["id"])
        if not A or not A.get("ok"):
            continue
        for g in A.get("gaps", []):
            if g["kind"] == "internal" and not g["energetic"]:
                gaps_by_lang[r["language"]].append(g["end_s"] - g["start_s"])
    pause_threshold = {
        l: round(max(PAUSE_MIN_S, float(np.percentile(v, PAUSE_PERCENTILE))), 3)
        for l, v in gaps_by_lang.items() if v}

    def gate(o, scope: str) -> tuple[bool, str]:
        """Apply the rule calibrated for this scope. Whole-clip and in-speech
        verdicts come from different distributions and never share a threshold."""
        v, s = o["vocalburst"], o.get("secondary") or {}
        fam = v["family"]
        pol = policy.get(scope, {}).get(fam)
        if not pol:
            return False, f"no calibrated {scope} rule for '{fam}'"
        if v["family_score"] < pol["family_score_min"]:
            return False, (f"family_score {v['family_score']:.2f} < "
                           f"{pol['family_score_min']}")
        if v["family_margin"] < pol["family_margin_min"]:
            return False, (f"family_margin {v['family_margin']:.2f} < "
                           f"{pol['family_margin_min']}")
        if pol["secondary_required"] and not s.get("agrees_top2"):
            return False, "CED second opinion disagrees"
        if pol["mode"] == "precision":
            return True, (f"calibrated {scope} gate passed "
                          f"(precision {pol['precision']}, recall {pol['recall']})")
        return True, (f"calibrated {scope} gate passed (specificity mode: "
                      f"{pol['control_false_positives']}/{pol['negatives']} "
                      f"false positives on the negative control)")

    def has_burst(o) -> tuple[bool, str]:
        """The weaker claim: something non-lexical is here, family unknown."""
        if not presence:
            return False, "no calibrated presence gate"
        v, s2 = o["vocalburst"], o.get("secondary") or {}
        if v["p_no_burst"] > presence["p_no_burst_max"]:
            return False, (f"p_no_burst {v['p_no_burst']:.2f} > "
                           f"{presence['p_no_burst_max']}")
        if s2.get("speech_score", 1.0) > presence["speech_score_max"]:
            return False, (f"CED speech score {s2.get('speech_score')} > "
                           f"{presence['speech_score_max']}")
        if presence.get("mode") == "precision":
            return True, (f"presence gate passed (precision {presence['precision']}, "
                          f"recall {presence['recall']})")
        return True, (f"presence gate passed (specificity mode: "
                      f"{presence['control_false_positives']}/"
                      f"{presence['negatives']} false positives on ordinary speech)")

    # ---- automatic condition probabilities ----------------------------------
    auto_cond: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for key, sel in chosen.items():
        tag, lang = key.split("|")
        clf, names, _thr = models[sel["model_key"]]
        ids = [r["id"] for r in rows if r["language"] == lang and r["style"] == "speech"]
        if not ids:
            continue
        X = np.asarray([[cfeat[c][n] for n in names] for c in ids], dtype=float)
        p = clf.predict_proba(X)[:, 1]
        for cid, pi in zip(ids, p):
            auto_cond[cid][tag] = float(pi)

    # ---- prevalence guard, per (tag, language, medium) ----------------------
    #
    # Every fit was made on reviewed anime clips.  Most of the corpus is dry
    # studio game voice, which is a different channel, and a detector that has
    # never seen that channel can drift badly on it without any warning from its
    # own confidence.  Comparing the rate a tag fires at in each cell with the
    # rate humans marked it at is the cheapest available shift detector: a cell
    # that fires far above the reviewed prevalence is refusing to transfer, so it
    # is suppressed and reported rather than shipped.
    fired, total = collections.Counter(), collections.Counter()
    for r in rows:
        for key, sel in chosen.items():
            tag, lang = key.split("|")
            if r["language"] != lang or r["style"] != "speech":
                continue
            cell = (tag, lang, r["medium"])
            total[cell] += 1
            if auto_cond[r["id"]].get(tag, 0.0) >= sel["threshold"]:
                fired[cell] += 1
    # Independent sanity check on a label no detector ever saw: Amadeus Kurisu is
    # the variant that is heard through a device, so a working [digital] detector
    # should fire overwhelmingly on Amadeus clips.  For the languages the model
    # was fitted out of, this is the only validation available.
    variant_check = {}
    for key, sel in chosen.items():
        tag, lang = key.split("|")
        pool = [r for r in rows if r["language"] == lang and r["style"] == "speech"]
        if not pool:
            continue
        hit = [r for r in pool if auto_cond[r["id"]].get(tag, 0.0) >= sel["threshold"]]
        base = sum(1 for r in pool if r["character_variant"] == "amadeus") / len(pool)
        got = (sum(1 for r in hit if r["character_variant"] == "amadeus") / len(hit)
               if hit else None)
        variant_check[key] = {
            "fired": len(hit), "clips": len(pool),
            "amadeus_share_of_fired": None if got is None else round(got, 4),
            "amadeus_share_of_language": round(base, 4),
            "enrichment": None if not got or not base else round(got / base, 2),
            "fitted_on": sel.get("fitted_on"),
            "out_of_group": sel.get("out_of_group", False)}

    suppressed, cell_rates = {}, {}
    for cell, n in total.items():
        tag, lang, medium = cell
        gold = chosen[f"{tag}|{lang}"]["gold_prevalence"]
        rate = fired[cell] / max(n, 1)
        cell_rates[f"{tag}|{lang}|{medium}"] = {
            "clips": n, "fired": fired[cell], "rate": round(rate, 4),
            "gold_prevalence": gold}
        if gold and rate > PREVALENCE_TOLERANCE * gold:
            suppressed[f"{tag}|{lang}|{medium}"] = {
                "rate": round(rate, 4), "gold_prevalence": gold,
                "reason": f"fires at {rate / gold:.1f}x the reviewed prevalence"}
    # An out-of-group prediction that shows no variant enrichment has not
    # transferred, whatever its own confidence says.
    for key, v in variant_check.items():
        tag, lang = key.split("|")
        if not (tag == "[digital]" and v["out_of_group"]):
            continue
        if v["enrichment"] is None or v["enrichment"] < 1.5:
            for medium in {r["medium"] for r in rows if r["language"] == lang}:
                suppressed[f"{tag}|{lang}|{medium}"] = {
                    "rate": None, "gold_prevalence": None,
                    "reason": (f"out-of-group prediction shows no Amadeus enrichment "
                               f"({v['enrichment']})")}

    # ---- per-clip decision ---------------------------------------------------
    out = []
    for r in rows:
        m = dict(r)
        f = feats[r["id"]]
        m["features"] = {k: v for k, v in f.items() if k != "id"}
        m["speaker_check"] = {k: v for k, v in spk.get(r["id"], {}).items() if k != "id"}
        A = ali.get(r["id"])
        m["alignment"] = None if not A else {
            "ok": A["ok"], "n_tokens": A.get("n_tokens"),
            "coverage": A.get("coverage"),
            "gaps": [g for g in A.get("gaps", []) if g["energetic"]],
        }

        text = r["text"]
        # A line of pure punctuation carries no lexical content.  The game
        # scenario writes a wordless take as "……", so testing `not text` never
        # fires on the quarantine pool and the whole-clip naming path below would
        # be dead code -- the take would ship with an ellipsis as its target
        # text, which is the punctuation-inferred label the design forbids.
        lexical = bool(text) and not NONLEXICAL_RE.match(text)
        events_out, insertions, reasons = [], [], []
        status = "ok"

        # 1. human event tags, verbatim positions
        for e in r["human_events"]:
            insertions.append((e["offset"], e["tag"]))
            events_out.append({"tag": e["tag"], "source": "human",
                               "scope": "in_speech" if lexical else "whole_clip"})

        # 2. whole-clip candidates
        whole = [o for o in obs.get(r["id"], []) if o["scope"] in ("whole_clip", "gold")]
        if not lexical:
            if r["human_events"]:
                text = ""                       # the tags are the whole line
                reasons.append("human-named non-lexical take")
            elif whole:
                ok, why = gate(whole[0], "whole_clip")
                fam = whole[0]["vocalburst"]["family"]
                tag = FAMILY_TO_TAG.get(fam)
                events_out.append({"tag": tag, "family": fam, "source": "auto",
                                   "scope": "whole_clip", "accepted": bool(ok and tag),
                                   "reason": why,
                                   "vocalburst": whole[0]["vocalburst"],
                                   "secondary": whole[0]["secondary"]})
                if ok and tag:
                    text = ""               # the tag replaces the ellipsis
                    insertions.append((0, tag))
                    reasons.append(f"non-lexical take named '{fam}': {why}")
                else:
                    ok2, why2 = has_burst(whole[0])
                    if ok2:
                        text = ""
                        insertions.append((0, GENERIC_NVV_TAG))
                        events_out[-1]["accepted_generic"] = True
                        reasons.append(
                            f"non-lexical take, family not named ({why}); "
                            f"kept as {GENERIC_NVV_TAG}: {why2}")
                    else:
                        status = "uncertain"
                        reasons.append(
                            f"non-lexical take, no nameable sound: {why}; {why2}")
            else:
                status = "uncertain"
                reasons.append("no transcript and no classification")

        # 3. automatic in-speech events
        if text and A and A.get("ok"):
            offs = token_offsets(text, A["tokens"])
            spans = [(t["start_s"], t["end_s"]) for t in A["tokens"]]
            for o in obs.get(r["id"], []):
                if o["scope"] != "in_speech":
                    continue
                ok, why = gate(o, "in_speech")
                fam = o["vocalburst"]["family"]
                tag = FAMILY_TO_TAG.get(fam)
                after = None
                for i, (s, e) in enumerate(spans):
                    if e <= o["start_s"]:
                        after = i
                at = insertion_index(text, offs, after)
                rec = {"tag": tag, "family": fam, "source": "auto",
                       "scope": "in_speech", "start_s": o["start_s"],
                       "end_s": o["end_s"], "gap_kind": o.get("gap_kind"),
                       "locator_score": o.get("locator_score"),
                       "accepted": bool(ok and tag), "reason": why,
                       "insert_at": at,
                       "vocalburst": o["vocalburst"], "secondary": o["secondary"]}
                events_out.append(rec)
                if ok and tag and not any(
                        abs(hi - at) <= 4 and ht == tag for hi, ht in insertions):
                    insertions.append((at, tag))

        # 4. measured pauses the transcript does not already write
        pause_spans = []
        if text and A and A.get("ok"):
            offs = token_offsets(text, A["tokens"])
            spans = [(t["start_s"], t["end_s"]) for t in A["tokens"]]
            thr = pause_threshold.get(r["language"], PAUSE_MIN_S)
            for g in sorted(A.get("gaps", []),
                            key=lambda g: -(g["end_s"] - g["start_s"])):
                if len(pause_spans) >= PAUSE_MAX_PER_CLIP:
                    break
                if g["kind"] != "internal" or g["energetic"]:
                    continue
                if g["end_s"] - g["start_s"] < thr:
                    continue
                after = None
                for i, (s, e) in enumerate(spans):
                    if e <= g["start_s"]:
                        after = i
                at = insertion_index(text, offs, after)
                window = text[max(0, at - PAUSE_CONTEXT): at + PAUSE_CONTEXT]
                if any(ch in window for ch in WRITTEN_PAUSE):
                    continue                       # '……' already says it
                pause_spans.append({"at": at, "duration_s": round(
                    g["end_s"] - g["start_s"], 3)})
                insertions.append((at, "[pause]"))
        m["measured_pauses"] = pause_spans

        # 5. condition tags: human first, then calibrated automatic
        conditions = list(r["human_notes"]["conditions"])
        cond_src = {c: "human" for c in conditions}
        if r["style"] == "singing" and "[singing]" not in conditions:
            conditions.append("[singing]")
            cond_src["[singing]"] = "source"
        for key, sel in chosen.items():
            tag, lang = key.split("|")
            if r["language"] != lang or r["style"] != "speech":
                continue
            if not sel["ships_as_tag"]:
                continue
            if f"{tag}|{lang}|{r['medium']}" in suppressed:
                continue
            p = auto_cond[r["id"]].get(tag, 0.0)
            if p >= sel["threshold"] and tag not in conditions:
                conditions.append(tag)
                cond_src[tag] = "auto"
        if "[whisper]" in conditions and "[yell]" in conditions:
            pw = auto_cond[r["id"]].get("[whisper]", 0.0)
            py = auto_cond[r["id"]].get("[yell]", 0.0)
            drop = "[yell]" if pw >= py else "[whisper]"
            if cond_src.get(drop) == "auto":
                conditions.remove(drop)
        conditions = [c for c in CONDITION_ORDER if c in conditions]

        m["condition_tags"] = conditions
        m["condition_tag_source"] = cond_src
        m["condition_scores"] = {k: round(v, 4)
                                 for k, v in auto_cond.get(r["id"], {}).items()}
        m["contamination"] = sorted(set(r["human_notes"]["contamination"]))
        m["contamination_score"] = round(auto_cond.get(r["id"], {}).get("noise", 0.0), 4)
        m["emotion_notes"] = sorted(set(r["human_notes"]["emotion"]))
        m["pause_notes"] = sorted(set(r["human_notes"]["pauses"]))
        m["events"] = events_out

        # A named non-lexical take has had its ellipsis cleared above, so the
        # published plain text has to follow -- ``text`` is the transcription
        # with the tags removed, and k12 enforces exactly that.
        m["text"] = text
        body = splice(text, insertions,
                      spaced=r["language"] not in CJK_LANGS)
        m["transcription"] = (" ".join(conditions) + (" " if body else "") + body
                              if conditions else body)
        m["status"] = status
        m["decision_reason"] = "; ".join(reasons) or "verbatim transcript"
        out.append(m)

    # ---- demote tags with too little support --------------------------------
    counts = collections.Counter()
    for m in out:
        if m["status"] != "ok":
            continue
        for t in ALL_TAGS:
            if t in m["transcription"]:
                counts[t] += 1
    demoted = {t for t, n in counts.items() if n < MIN_TAG_EXAMPLES}
    for m in out:
        for t in demoted:
            if t not in m["transcription"]:
                continue
            if m["transcription"].strip() == t:
                m["status"] = "uncertain"
                m["decision_reason"] += (f"; demoted: only {counts[t]} examples of {t} "
                                         f"(< {MIN_TAG_EXAMPLES})")
            else:
                # A removed tag leaves a word boundary behind in a spaced script
                # and nothing at all in Japanese or Chinese, where inserting one
                # would break strip_tags(transcription) == text.
                sep = "" if m["language"] in CJK_LANGS else " "
                m["transcription"] = re.sub(
                    r"\s+", " ", m["transcription"].replace(t, sep)).strip()
                m["decision_reason"] += f"; stripped rare tag {t}"
                m["condition_tags"] = [c for c in m["condition_tags"] if c != t]

    for m in out:
        if m["status"] == "ok" and not m["transcription"]:
            m["status"] = "uncertain"
            m["decision_reason"] += "; empty transcription"

    write_jsonl(MANIFESTS / "09_master.jsonl", out)

    shipped = [m for m in out if m["status"] == "ok"]
    final = collections.Counter()
    by_source = collections.Counter()
    for m in shipped:
        for t in ALL_TAGS:
            if t in m["transcription"]:
                final[t] += 1
                by_source[(t, m["condition_tag_source"].get(t, "auto"))] += 1
    report = {
        "clips": len(out),
        "shipped": len(shipped),
        "held_back_uncertain": sum(1 for m in out if m["status"] == "uncertain"),
        "tags_shipped": dict(final.most_common()),
        "tags_experimental": {t: n for t, n in final.items()
                              if MIN_TAG_EXAMPLES <= n < EXPERIMENTAL_BELOW},
        "tags_demoted_too_rare": {t: counts[t] for t in sorted(demoted)},
        "condition_tag_provenance": {f"{t}:{s}": n for (t, s), n in
                                     sorted(by_source.items())},
        "clips_with_any_tag": sum(1 for m in shipped
                                  if any(t in m["transcription"] for t in ALL_TAGS)),
        "auto_in_speech_events_accepted": sum(
            1 for m in out for e in m["events"]
            if e.get("scope") == "in_speech" and e.get("source") == "auto"
            and e.get("accepted")),
        "whole_clip_generic": sum(
            1 for m in out for e in m["events"]
            if e.get("scope") == "whole_clip" and e.get("accepted_generic")),
        "whole_clip_named": sum(
            1 for m in out for e in m["events"]
            if e.get("scope") == "whole_clip" and e.get("source") == "auto"
            and e.get("accepted")),
        "quarantine_recovered": sum(1 for m in shipped if m["from_quarantine"]),
        "quarantine_total": sum(1 for m in out if m["from_quarantine"]),
        "measured_pause_tags": sum(len(m["measured_pauses"]) for m in shipped),
        "pause_threshold_s_by_language": pause_threshold,
        "contamination_flagged": sum(1 for m in shipped if m["contamination"]),
        "suppressed_cells": suppressed,
        "variant_enrichment_check": variant_check,
        "detector_fire_rates": cell_rates,
        "min_tag_examples": MIN_TAG_EXAMPLES,
    }
    write_json(REPORTS / "09_decisions.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "detector_fire_rates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
