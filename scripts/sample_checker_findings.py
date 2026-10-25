#!/usr/bin/env python3
"""Sample the log-only checkers' findings into a precision-labelling sheet.

Checkers (all log-only, no measured precision): the post-answer review gate,
the grounding check, the personal-claim check.  Reads
``logs/turn_records.jsonl`` + ``daemon_debug*.log`` (read-only), emits one
JSONL sheet per checker into ``--out`` (a directory), stratified round-robin
across (day, verdict kind) with a fixed ``--seed``.  Test-probe turns are
excluded.

Privacy: the sheets are LOCAL-ONLY and contain conversation excerpts.  Item
contents are never printed; stdout carries counts only.  No network, no model
calls.  class: BC-58 (unmeasured guard).

What the sources do NOT hold (reported per item in ``context_gaps``): the
reply text, the claim span and the checker's evidence are not in the turn
records or debug logs (they live in the corpus / the checker's memory), so an
item is ``judgeable`` only when a span-like field (claim text, issues list or
redacted correction) was recoverable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKERS = ("review", "grounding", "personal_claim")
DEFAULT_N = 40
QUERY_MAX = 300
EVIDENCE_MAX = 600
SPAN_MAX = 400
JOIN_WINDOW_S = 180

_TEST_SPAN_RE = re.compile(r"\[test\]", re.IGNORECASE)
_LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[(\w+)\] \[([^\]]*)\] (.*)$")
_LOG_MARKERS = {
    "review": ("[REVIEW GATE] Response failed", "[REVIEW GATE] Response passed"),
    "grounding": ("[Grounding] LOG-ONLY flagged", "[GroundingCheck] Flagged verdict suppressed"),
    "personal_claim": ("[PersonalClaim] kinds:",),
}


def bound(text, limit):
    """Truncate to ``limit`` chars, marking the cut."""
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _parse_ts(value):
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt.replace(tzinfo=None)  # records and logs are both local wall time


def load_turn_records(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("ts"):
                rows.append(rec)
    return rows


def is_test_probe(rec):
    if rec.get("test_probe") or rec.get("test_env"):
        return True
    return _TEST_SPAN_RE.search(str(rec.get("query", ""))) is not None


def load_log_events(paths):
    """{checker: [(datetime, message)]} for the marker lines, sorted by time."""
    events = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _LOG_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                msg = m.group(4)
                for checker, markers in _LOG_MARKERS.items():
                    if any(mk in msg for mk in markers):
                        ts = _parse_ts(m.group(1).replace(" ", "T"))
                        if ts:
                            events[checker].append((ts, msg))
    for lst in events.values():
        lst.sort(key=lambda e: e[0])
    return events


def _nearest(events, ts, used, window=JOIN_WINDOW_S):
    best, best_gap = None, None
    for i, (ets, _msg) in enumerate(events):
        if i in used:
            continue
        gap = abs((ets - ts).total_seconds())
        if gap <= window and (best_gap is None or gap < best_gap):
            best, best_gap = i, gap
    return best


def _plan_excerpt(rec):
    plan = rec.get("response_plan")
    if plan in (None, "", {}):
        return ""
    return bound(plan if isinstance(plan, str) else json.dumps(plan, sort_keys=True), EVIDENCE_MAX)


def _base(checker, rec, kind, verdict, reason, span, evidence):
    span = bound(span, SPAN_MAX)
    gaps = ["reply_text_not_persisted", "checker_evidence_not_persisted"]
    if not span:
        gaps.append("claim_span_not_recoverable")
    ts = str(rec["ts"])
    return {
        "id": f"{checker}:{ts}", "checker": checker, "timestamp": ts, "day": ts[:10],
        "kind": kind, "verdict": verdict, "reason": bound(reason, SPAN_MAX),
        "claim_span": span, "evidence_excerpt": bound(evidence, EVIDENCE_MAX),
        "reply_excerpt": "", "query_excerpt": bound(rec.get("query", ""), QUERY_MAX),
        "response_len": rec.get("response_len"), "context_gaps": gaps,
        "judgeable": bool(span), "label": "", "note": "",
    }


def candidates(checker, records, events):
    """Candidate findings for one checker, one per fired turn record."""
    used, out = set(), []
    evs = events.get(checker, [])
    for rec in records:
        ts = _parse_ts(rec["ts"])
        if checker == "review":
            if not rec.get("review_fired"):
                continue
            passed = bool(rec.get("review_passed"))
            reason = ""
            if ts is not None:
                i = _nearest(evs, ts, used)
                if i is not None:
                    used.add(i)
                    msg = evs[i][1]
                    reason = msg.split("issues=", 1)[1] if "issues=" in msg else ""
            conf = rec.get("review_confidence")
            out.append(_base(checker, rec, "pass" if passed else "fail",
                             {"passed": passed, "confidence": conf}, reason, reason,
                             _plan_excerpt(rec)))
        elif checker == "grounding":
            if rec.get("grounding_flagged") is None:
                continue
            flagged = bool(rec.get("grounding_flagged"))
            correction = rec.get("grounding_verdict") or ""
            if not correction and flagged and ts is not None:
                i = _nearest(evs, ts, used)
                if i is not None:
                    used.add(i)
                    correction = evs[i][1].split("): ", 1)[-1]
            kind = ("flagged_with_text" if correction else "flagged_no_text") if flagged else "not_flagged"
            out.append(_base(checker, rec, kind,
                             {"flagged": flagged, "confidence": rec.get("grounding_confidence")},
                             rec.get("grounding_status") or "", correction, ""))
        else:
            status = rec.get("personal_claim_status")
            if status != "checked" or rec.get("personal_claim_reason") != "ok":
                continue
            counts = {k: int(rec.get(f"personal_claim_{k}_count") or 0)
                      for k in ("supported", "contradicted", "insufficient")}
            dominant = next((k for k in ("contradicted", "insufficient", "supported") if counts[k]), None)
            kind = "has_" + dominant if dominant else "no_findings"
            out.append(_base(checker, rec, kind,
                             {"status": status, **counts,
                              "candidates": rec.get("personal_claim_candidate_count"),
                              "delivery": rec.get("personal_claim_delivery")},
                             rec.get("personal_claim_reason") or "", "", ""))
    return out


def load_findings(path):
    """Rows of logs/checker_findings.jsonl grouped by checker ({} if absent)."""
    out = defaultdict(list)
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("checker") in CHECKERS and not row.get("test_env"):
                out[row["checker"]].append(row)
    return out


def finding_candidates(checker, rows):
    """Sheet items built from first-class finding lines (full context)."""
    out = []
    for row in rows:
        if _TEST_SPAN_RE.search(str(row.get("query_excerpt", ""))):
            continue
        ts = str(row.get("ts", ""))
        span = bound(row.get("claim_span"), SPAN_MAX)
        verdict = str(row.get("verdict", ""))
        out.append({
            "id": f"{checker}:{ts}:{len(out)}", "checker": checker, "timestamp": ts, "day": ts[:10],
            "kind": verdict.split(":")[0] if checker != "grounding" else verdict,
            "verdict": verdict, "reason": bound(row.get("reason"), SPAN_MAX), "claim_span": span,
            "evidence_excerpt": bound(row.get("evidence_excerpt"), EVIDENCE_MAX),
            "reply_excerpt": bound(row.get("reply_excerpt"), EVIDENCE_MAX),
            "query_excerpt": bound(row.get("query_excerpt"), QUERY_MAX),
            "response_len": None, "context_gaps": [] if span else ["claim_span_empty"],
            "judgeable": bool(span), "source": "checker_findings", "label": "", "note": "",
        })
    return out


def stratified_sample(items, n, seed):
    """Round-robin across (day, kind) strata, seeded; deterministic for a seed."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for it in items:
        strata[(it["day"], it["kind"])].append(it)
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    order = keys[:]
    rng.shuffle(order)
    picked = []
    while len(picked) < n and any(strata[k] for k in order):
        for k in order:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return sorted(picked, key=lambda it: it["timestamp"])


def build_sheets(records, events, n=DEFAULT_N, seed=0, findings=None):
    """Prefer first-class finding lines per checker; fall back to turn records + logs."""
    records = [r for r in records if not is_test_probe(r)]
    out = {}
    for checker in CHECKERS:
        if findings and findings.get(checker):
            pool = finding_candidates(checker, findings[checker])
        else:
            pool = candidates(checker, records, events)
        out[checker] = (len(pool), stratified_sample(pool, n, f"{seed}:{checker}"))
    return out


def default_log_paths(root=REPO_ROOT):
    return sorted(glob.glob(os.path.join(root, "daemon_debug*.log")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--turn-records", default=os.path.join(REPO_ROOT, "logs", "turn_records.jsonl"))
    ap.add_argument("--findings", default=os.path.join(REPO_ROOT, "logs", "checker_findings.jsonl"),
                    help="first-class finding lines (preferred per checker when present)")
    ap.add_argument("--debug-logs", nargs="*", default=None,
                    help="debug log files (default: repo-root daemon_debug*.log)")
    ap.add_argument("--out", required=True, help="output directory (one <checker>.jsonl each)")
    ap.add_argument("-n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args(argv)
    log_paths = args.debug_logs if args.debug_logs is not None else default_log_paths()
    records = load_turn_records(args.turn_records) if os.path.isfile(args.turn_records) else []
    findings = load_findings(args.findings)
    events = load_log_events(log_paths)
    os.makedirs(args.out, exist_ok=True)
    for checker, (pool, sample) in build_sheets(records, events, args.n, args.seed, findings).items():
        path = os.path.join(args.out, f"{checker}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for it in sample:
                fh.write(json.dumps(it, sort_keys=True) + "\n")
        strata = Counter(f"{it['day']}/{it['kind']}" for it in sample)
        print(f"{checker}: pool={pool} sampled={len(sample)} "
              f"unjudgeable={sum(1 for it in sample if not it['judgeable'])} "
              f"kinds={dict(Counter(it['kind'] for it in sample))} strata={len(strata)} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
