"""Stage 09 - one decision per clip, and the master manifest.

Merges everything upstream into ``manifests/09_master.jsonl``: normalised transcript,
audio facts, every raw model score, and the single field the trainer will actually
read - ``qwen_text``.

Policy, in the order it is applied:

* purely non-lexical clip (``……``) -> becomes its tag when the calibrated gate from
  stage 05 clears it, otherwise it is held out as ``uncertain``: a clip whose
  transcript says nothing and whose sound cannot be named is unusable supervision,
  and a wrong control token is worse than a missing one (supplement s39).
* onomatopoeia-only clip (``ふふ``, ``はぁ……``) -> the written form is KEPT. The audio
  models agreed with the written prior on only 11 of 23 of these, so rewriting them
  into tags would be guessing; Japanese also genuinely lexicalises these, which is
  exactly the case the guide says to leave as words (guide s4.2).
* spoken line -> transcript verbatim, plus any tag stage 08 both named and placed.
* a tag with too few accepted examples is demoted rather than shipped as a control
  nobody can steer (guide s12).
"""
from __future__ import annotations

import collections
import json

from common import (FAMILY_TO_QWEN, MANIFESTS, REPORTS, read_jsonl, utf8_stdout,
                    write_jsonl)

utf8_stdout()

MIN_TAG_EXAMPLES = 10          # below this a tag is not shipped at all
EXPERIMENTAL_BELOW = 25        # 10-24 ships but is reported as experimental (guide s12)


def main() -> None:
    rows = list(read_jsonl(MANIFESTS / "02_audio.jsonl"))
    iso = {r["id"]: r for r in read_jsonl(MANIFESTS / "03_isolated_classified.jsonl")}
    policy = json.loads((REPORTS / "05_calibration.json").read_text(encoding="utf-8"))["policy"]
    events = collections.defaultdict(list)
    ev_path = MANIFESTS / "08_events.jsonl"
    if ev_path.exists():
        for e in read_jsonl(ev_path):
            events[e["id"]].append(e)

    def gate(entry) -> tuple[bool, str]:
        v, s = entry["vocalburst"], entry.get("secondary", {})
        fam = v["family"]
        pol = policy.get(fam, {}).get("auto_accept")
        if not pol:
            return False, f"no calibrated auto-accept rule for '{fam}'"
        if v["family_score"] < pol["family_score_min"]:
            return False, (f"family_score {v['family_score']:.2f} < "
                           f"{pol['family_score_min']}")
        if v["family_margin"] < pol["family_margin_min"]:
            return False, (f"family_margin {v['family_margin']:.2f} < "
                           f"{pol['family_margin_min']}")
        if pol["secondary_required"] and not s.get("agrees_top2"):
            return False, "CED second opinion disagrees"
        return True, "calibrated gate passed (both models)"

    # ---- pass 1: propose
    proposed = []
    for r in rows:
        m = dict(r)
        m["events"] = []
        status, reason, qwen_text = "auto_accept", "verbatim transcript", r["text"]

        if not r["text"]:
            status, reason, qwen_text = "reject", "empty transcript after normalisation", ""
        elif "symbol_risk" in r["text_flags"] and set(r["text"]) <= set("△▼×□$><!?＄＆＞＜！？"):
            status, reason, qwen_text = "reject", "transcript is garbled glyphs, not speech", ""

        elif r["speech_status"] == "isolated_nvv_candidate":
            e = iso.get(r["id"])
            if not e:
                status, reason = "uncertain", "not classified"
            else:
                ok, why = gate(e)
                fam = e["vocalburst"]["family"]
                tag = FAMILY_TO_QWEN.get(fam)
                m["events"] = [{"scope": "whole_clip", "family": fam,
                                "start_s": 0.0, "end_s": r["dur"],
                                "vocalburst": e["vocalburst"],
                                "secondary": e.get("secondary")}]
                if ok and tag:
                    status, reason, qwen_text = "auto_accept", f"isolated {fam}: {why}", tag
                elif ok and not tag:
                    status, reason = "uncertain", f"'{fam}' is not exposed as a Qwen tag in v1"
                else:
                    status, reason = "uncertain", f"isolated {fam}: {why}"

        elif r["speech_status"] == "onomatopoeia_candidate":
            e = iso.get(r["id"])
            agree = bool(e and e["text_prior"]
                         and e["vocalburst"]["family"] in e["text_prior"])
            reason = ("written onomatopoeia kept as lexical text"
                      + ("; audio agrees with the written form" if agree
                         else "; audio model disagrees, text still wins"))
            if e:
                m["events"] = [{"scope": "whole_clip_prior_check",
                                "family": e["vocalburst"]["family"],
                                "prior_agrees": agree,
                                "vocalburst": e["vocalburst"],
                                "secondary": e.get("secondary")}]

        # in-speech events (may apply to speech and onomatopoeia clips alike)
        accepted = []
        for ev in sorted(events.get(r["id"], []), key=lambda e: -e["insert_at"]):
            ok, why = gate(ev)
            fam = ev["vocalburst"]["family"]
            tag = FAMILY_TO_QWEN.get(fam)
            rec = {"scope": "in_speech", "family": fam,
                   "start_s": ev["start_s"], "end_s": ev["end_s"],
                   "gap_kind": ev["gap_kind"], "locator_score": ev["locator_score"],
                   "insert_at": ev["insert_at"],
                   "accepted": bool(ok and tag), "reason": why,
                   "vocalburst": ev["vocalburst"], "secondary": ev["secondary"]}
            m["events"].append(rec)
            if ok and tag and status != "reject":
                accepted.append((ev["insert_at"], tag))
        for at, tag in accepted:                       # descending offset: safe to splice
            at = min(max(at, 0), len(qwen_text))
            left, right = qwen_text[:at], qwen_text[at:]
            qwen_text = (left.rstrip() + (" " if left.strip() else "") + tag
                         + (" " if right.strip() else "") + right.lstrip())
        if accepted:
            reason += f"; +{len(accepted)} placed in-speech tag(s)"

        m["status"], m["decision_reason"], m["qwen_text"] = status, reason, qwen_text
        proposed.append(m)

    # ---- pass 2: demote tags with too little support (guide s12)
    tag_counts = collections.Counter()
    for m in proposed:
        for t in set(FAMILY_TO_QWEN.values()):
            if t and t in m["qwen_text"]:
                tag_counts[t] += 1
    demoted = {t for t, n in tag_counts.items() if n < MIN_TAG_EXAMPLES}
    experimental = {t: n for t, n in tag_counts.items()
                    if MIN_TAG_EXAMPLES <= n < EXPERIMENTAL_BELOW}

    for m in proposed:
        for t in list(demoted):
            if t in m["qwen_text"]:
                if m["qwen_text"].strip() == t:            # the tag was the whole clip
                    m["status"] = "uncertain"
                    m["decision_reason"] += (f"; demoted: only {tag_counts[t]} examples of "
                                             f"{t} (< {MIN_TAG_EXAMPLES})")
                else:
                    m["qwen_text"] = " ".join(m["qwen_text"].replace(t, " ").split())
                    m["decision_reason"] += f"; stripped rare tag {t}"

    write_jsonl(MANIFESTS / "09_master.jsonl", proposed)
    write_jsonl(MANIFESTS / "09_manual_review.jsonl",
                [m for m in proposed if m["status"] == "uncertain"])

    final_tags = collections.Counter()
    for m in proposed:
        if m["status"] != "auto_accept":
            continue
        for t in set(FAMILY_TO_QWEN.values()):
            if t and t in m["qwen_text"]:
                final_tags[t] += 1

    st = collections.Counter(m["status"] for m in proposed)
    report = {
        "clips": len(proposed),
        "status": dict(st),
        "tag_counts_shipped": dict(final_tags.most_common()),
        "tags_demoted_too_rare": {t: tag_counts[t] for t in sorted(demoted)},
        "tags_experimental_10_24_examples": experimental,
        "min_tag_examples": MIN_TAG_EXAMPLES,
        "clips_with_any_tag": sum(1 for m in proposed if m["status"] == "auto_accept"
                                  and any(t and t in m["qwen_text"]
                                          for t in set(FAMILY_TO_QWEN.values()))),
        "isolated_accepted": sum(1 for m in proposed
                                 if m["speech_status"] == "isolated_nvv_candidate"
                                 and m["status"] == "auto_accept"),
        "isolated_uncertain": sum(1 for m in proposed
                                  if m["speech_status"] == "isolated_nvv_candidate"
                                  and m["status"] == "uncertain"),
        "in_speech_tags_placed": sum(
            1 for m in proposed for e in m["events"]
            if e.get("scope") == "in_speech" and e.get("accepted")),
        "rejected": [{"id": m["id"], "reason": m["decision_reason"]}
                     for m in proposed if m["status"] == "reject"],
    }
    (REPORTS / "09_decisions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
