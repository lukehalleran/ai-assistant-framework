"""
Hand-label intent classifications from turn telemetry → confusion matrix.

NOTE (2026-08-03): hand-labeling is the OPTIONAL AUDIT path, not the required
workflow — `scripts/auto_label_intents.py` LLM-labels the backlog (with
optional dual-model agreement filtering) into the same labels file, and this
tool's --report reads both. Use this interactively only to spot-check the
LLM's labels or to adjudicate its disagreement slice.

Reads logs/turn_records.jsonl (one line per completed turn: query, predicted
intent, confidence, source). Interactive mode shows each unlabeled turn and
asks for the TRUE intent; labels append to data/intent_labels.jsonl keyed by
the record's timestamp, so sessions are resume-safe — already-labeled and
skipped records are never re-asked.

Usage:
    python scripts/label_intents.py                # label unlabeled records
    python scripts/label_intents.py --limit 50     # label at most 50 this run
    python scripts/label_intents.py --report       # confusion matrix + stats

Keys: 1-9 pick an intent, Enter = agree with prediction, s = skip (never
re-asked), q = quit (progress saved).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RECORDS = Path("logs/turn_records.jsonl")
LABELS = Path("data/intent_labels.jsonl")

INTENTS = [
    "factual_recall", "temporal_recall", "emotional_support", "casual_social",
    "technical_help", "creative_exploration", "meta_conversational",
    "project_work", "general",
]


def _load_records():
    if not RECORDS.exists():
        print(f"No telemetry at {RECORDS}")
        return []
    out = []
    with open(RECORDS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts") and r.get("query"):
                out.append(r)
    return out


def _load_labels():
    labels = {}
    if LABELS.exists():
        with open(LABELS) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    labels[d["ts"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue
    return labels


def _append_label(entry):
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS, "a") as f:
        f.write(json.dumps(entry) + "\n")


def label(limit=None):
    records = _load_records()
    labels = _load_labels()
    todo = [r for r in records if r["ts"] not in labels]
    print(f"{len(records)} records, {len(labels)} labeled/skipped, {len(todo)} to go.\n")
    menu = "  ".join(f"{i+1}={name}" for i, name in enumerate(INTENTS))

    done = 0
    for r in todo:
        if limit and done >= limit:
            break
        pred = r.get("intent", "?")
        conf = r.get("intent_confidence", 0)
        tone = r.get("tone_level", "?")
        print("-" * 78)
        print(f"Q: {r['query'][:200]}")
        print(f"predicted: {pred}@{conf:.2f}  tone={tone}  mode={r.get('mode')}")
        print(menu)
        try:
            ans = input("true intent [Enter=agree / 1-9 / s=skip / q=quit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped; progress saved.")
            return
        if ans == "q":
            break
        if ans == "s":
            _append_label({"ts": r["ts"], "query": r["query"], "predicted": pred,
                           "true": None, "skipped": True})
            continue
        if ans == "":
            true = pred
        elif ans.isdigit() and 1 <= int(ans) <= len(INTENTS):
            true = INTENTS[int(ans) - 1]
        else:
            print("  unrecognized — skipping this record for now (not saved)")
            continue
        _append_label({"ts": r["ts"], "query": r["query"], "predicted": pred,
                       "true": true, "confidence": conf, "tone": tone})
        done += 1
    print(f"\nLabeled {done} this session. Run --report for the matrix.")


def report():
    labels = [d for d in _load_labels().values()
              if d.get("true") and not d.get("skipped")]
    if not labels:
        print("No labels yet — run without --report first.")
        return
    n_right = sum(1 for d in labels if d["predicted"] == d["true"])
    print(f"{len(labels)} labeled; accuracy {n_right}/{len(labels)} "
          f"({100*n_right/len(labels):.0f}%)\n")

    confusion = defaultdict(Counter)
    for d in labels:
        confusion[d["true"]][d["predicted"]] += 1

    names = sorted({d["true"] for d in labels} | {d["predicted"] for d in labels})
    short = {n: n[:12] for n in names}
    header = "true / pred"
    print(f"{header:<14}" + "".join(f"{short[n]:>13}" for n in names))
    for t in names:
        row = confusion.get(t, Counter())
        print(f"{short[t]:<14}" + "".join(f"{row.get(p, 0):>13}" for p in names))

    print("\nper-true-intent recall:")
    for t in names:
        row = confusion.get(t, Counter())
        total = sum(row.values())
        if total:
            print(f"  {t:<20} {row.get(t, 0)}/{total} "
                  f"({100*row.get(t, 0)/total:.0f}%)")
    print("\nmost common confusions:")
    pairs = Counter()
    for t, row in confusion.items():
        for p, c in row.items():
            if p != t:
                pairs[(t, p)] += c
    for (t, p), c in pairs.most_common(8):
        print(f"  true={t} → predicted={p}: {c}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--report", action="store_true", help="Confusion matrix from labels")
    ap.add_argument("--limit", type=int, default=None, help="Max records this session")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        label(limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
