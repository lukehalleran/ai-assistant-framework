"""Read-only replay of historic light-path context size; prints no personal text.

Run: python scripts/audit_context_budget.py
Counts are the application's local tokenizer estimate, not provider billing.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=repo / "data/corpus_v4.json")
    parser.add_argument("--before", default="2026-09-04T17:26:27")
    args = parser.parse_args()
    from models.tokenizer_manager import TokenizerManager
    from core.prompt.token_manager import TokenManager

    rows = json.loads(args.corpus.read_text())
    entries = sorted(
        [r for r in rows if str(r.get("timestamp", "")) <= args.before],
        key=lambda r: str(r.get("timestamp", "")), reverse=True,
    )[:3]
    if not entries:
        parser.error("No matching corpus entries")
    model = SimpleNamespace(is_api_model=lambda name: True, get_active_model_name=lambda: "kimi-3")
    tokenizer = TokenizerManager(model)
    manager = TokenManager(model, tokenizer, 10000)
    render = lambda items: "\n\n".join(manager._extract_text(item) for item in items)
    before = render(entries)
    tokenizer.count_tokens("warmup", "kimi-3")
    elapsed = []
    for _ in range(5):
        started = time.perf_counter()
        result = manager._manage_token_budget({"recent_conversations": entries})
        elapsed.append((time.perf_counter() - started) * 1000)
    after = render(result["recent_conversations"])
    print(json.dumps({
        "entries": len(entries), "before_chars": len(before),
        "before_estimated_tokens": tokenizer.count_tokens(before, "kimi-3"),
        "after_chars": len(after), "after_estimated_tokens": tokenizer.count_tokens(after, "kimi-3"),
        "cap_median_ms": round(statistics.median(elapsed), 2),
        "largest_query_before_chars": max(len(item.get("query", "")) for item in entries),
        "largest_query_after_estimated_tokens": max(
            (tokenizer.count_tokens(item.get("query", ""), "kimi-3")
             for item in result["recent_conversations"]), default=0,
        ),
        "scope": "old light path bypassed this cap; not a full turn or provider token count",
    }))


if __name__ == "__main__":
    main()
