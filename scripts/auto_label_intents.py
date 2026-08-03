"""
Auto-label intent classifications from turn telemetry via LLM (no hand-labeling).

Replaces bulk hand-labeling as the path to the intent confusion matrix
(scripts/label_intents.py stays as the optional AUDIT tool — spot-check the
LLM's labels or the disagreement slice, never the required workflow).

Reads logs/turn_records.jsonl and labels each unlabeled record with an LLM
(strict single-label from the 9 IntentTypes + definitions). With --verify, a
second model labels independently and only AGREEMENTS are kept as trusted
labels; disagreements are written with "uncertain": true (excluded from
label_intents.py --report accuracy math by leaving "true" null, but preserved
for review).

Output: data/intent_labels.jsonl — same schema as the manual tool, plus
"labeler". Resume-safe: already-labeled ts are skipped.

Usage (needs OPENAI_API_KEY; run with Daemon up or down — read-only + API):
    python scripts/auto_label_intents.py --limit 50            # first batch
    python scripts/auto_label_intents.py                       # all remaining
    python scripts/auto_label_intents.py --verify kimi-3       # dual-model
    python scripts/label_intents.py --report                   # matrix
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RECORDS = Path("logs/turn_records.jsonl")
LABELS = Path("data/intent_labels.jsonl")

INTENTS = [
    "factual_recall", "temporal_recall", "emotional_support", "casual_social",
    "technical_help", "creative_exploration", "meta_conversational",
    "project_work", "general",
]

PROMPT = """Classify the intent of this user message sent to a personal AI companion. Respond with ONLY one label from:

factual_recall: asking to recall a stored personal fact ("what's my cat's name")
temporal_recall: asking about past events/timing ("what did we discuss last week")
emotional_support: venting, distress, sadness, seeking comfort — even without a question
casual_social: greetings, small talk, banter, brief acknowledgments
technical_help: debugging, code, tools, how-to for tech problems
creative_exploration: writing, brainstorming, imagination, hypotheticals
meta_conversational: questions about the AI itself (its memory, model, behavior)
project_work: working on the user's own project (this codebase, features, fixes)
general: anything else — knowledge questions, advice, mixed/unclear

Message: "{query}"

Label:"""


def _load_labeled_ts():
    done = set()
    if LABELS.exists():
        for line in LABELS.read_text().splitlines():
            try:
                done.add(json.loads(line)["ts"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _parse(raw: str):
    low = (raw or "").strip().lower()
    for name in INTENTS:
        if name in low:
            return name
    return None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None,
                    help="Labeling model (default: active model)")
    ap.add_argument("--verify", metavar="MODEL", default=None,
                    help="Second model; only agreements become trusted labels")
    args = ap.parse_args()

    from models.model_manager import ModelManager
    mm = ModelManager()

    records = []
    for line in RECORDS.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("ts") and r.get("query"):
                records.append(r)
        except json.JSONDecodeError:
            continue
    done = _load_labeled_ts()
    todo = [r for r in records if r["ts"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(records)} records, {len(done)} labeled, labeling {len(todo)} now")

    async def _label_one(query, model):
        raw = await mm.generate_once(
            PROMPT.format(query=query.replace('"', "'")[:400]),
            model_name=model,
            system_prompt="You are a strict single-label classifier.",
            max_tokens=8,
            temperature=0.0,
            disable_reasoning=True,
        )
        return _parse(raw if isinstance(raw, str) else "")

    n_ok = n_uncertain = n_fail = 0
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS, "a") as out:
        for i, r in enumerate(todo, 1):
            try:
                label = await _label_one(r["query"], args.model)
                verify = await _label_one(r["query"], args.verify) if args.verify else label
            except Exception as e:
                print(f"  [{i}] API error, stopping (resume-safe): {e}")
                break
            entry = {
                "ts": r["ts"], "query": r["query"],
                "predicted": r.get("intent", "?"),
                "confidence": r.get("intent_confidence"),
                "tone": r.get("tone_level"),
                "labeler": f"llm:{args.model or 'active'}"
                           + (f"+{args.verify}" if args.verify else ""),
            }
            if label and label == verify:
                entry["true"] = label
                n_ok += 1
            elif label and verify and label != verify:
                entry.update({"true": None, "uncertain": True,
                              "label_a": label, "label_b": verify})
                n_uncertain += 1
            else:
                entry.update({"true": None, "uncertain": True})
                n_fail += 1
            out.write(json.dumps(entry) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} (trusted={n_ok}, uncertain={n_uncertain})")

    print(f"\nDone: {n_ok} trusted labels, {n_uncertain} disagreements, "
          f"{n_fail} parse failures.")
    print("Next: python scripts/label_intents.py --report")
    if n_uncertain:
        print('Optional audit: grep \'"uncertain": true\' data/intent_labels.jsonl '
              "to inspect the disagreement slice.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
