#!/usr/bin/env python3
"""DM-30 — keyword-boundary corpus diff (read-only).

Every keyword list in this project is compiled through
``utils.trigger_match.compile_keyword_matcher``. A boundary rule change there
moves matching for ALL of them at once, in BOTH directions, and the cost of a
wrong rule is invisible in unit tests: on 2026-09-11 `'numb'` matched "number"
567 times and `'dead'` matched "deadline" 176 times in the owner's own corpus,
and the first cut of the 2026-09-12 fix let `'hate'` match "hat" (a gcc banner
reading "(Red Hat 14.3.1-4)" scored a HEAVY hit).

This probe answers the only question that matters before such a change ships:
against the real corpus, which word TOKENS does each keyword stop matching,
and which does it start matching? Review both columns — a lost token should be
an unrelated word, a gained token should be a true inflection.

Read-only: loads `data/corpus_v4.json`, writes nothing, calls no model, and
never touches a store. Safe to run while the daemon is live.

    python scripts/probe_keyword_boundary.py [--corpus PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _matchers() -> dict:
    """Every live matcher, by the name of the list it compiles."""
    from core.agentic import gate as g
    from utils import query_checker as qc
    from utils import tone_detector as td

    return {
        "HIGH_CRISIS": td._HIGH_MATCHER,
        "MEDIUM_CRISIS": td._MEDIUM_MATCHER,
        "CONCERN": td._CONCERN_MATCHER,
        "EVENT_DISTRESS": td._EVENT_MATCHER,
        "HEAVY": qc._HEAVY_MATCHER,
        "COMPUTATION": g._COMPUTATION_HIT,
        "WEB_SEARCH": g._WEB_SEARCH_HIT,
        "TOOL": g._TOOL_HIT,
        "MEMORY": g._MEMORY_HIT,
        "KNOWLEDGE": g._KNOWLEDGE_HIT,
        "FILE_ACCESS": g._FILE_ACCESS_KEYWORD_HIT,
        "RECALL": g._RECALL_PHRASE_HIT,
    }


def _corpus_tokens(corpus_path: Path) -> collections.Counter:
    rows = json.loads(corpus_path.read_text())
    text = "\n".join(
        ((row.get("query") or "") + "\n" + (row.get("response") or "")).lower()
        for row in rows
        if isinstance(row, dict)
    )
    return collections.Counter(re.findall(r"[a-z_]+", text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus_v4.json"))
    ap.add_argument("--limit", type=int, default=40,
                    help="rows to print per direction (default 40)")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        return 2
    tokens = _corpus_tokens(corpus_path)
    print(f"corpus {corpus_path}: {sum(tokens.values())} word tokens, "
          f"{len(tokens)} distinct")

    lost: collections.Counter = collections.Counter()
    gained: collections.Counter = collections.Counter()
    for list_name, matcher in _matchers().items():
        for keyword, pattern in matcher._word_pats:
            # The pre-2026-09-12 rule: left word boundary only.
            legacy = re.compile(rf"\b{re.escape(keyword)}")
            for token, count in tokens.items():
                was, now = bool(legacy.match(token)), bool(pattern.match(token))
                if was and not now:
                    lost[(list_name, keyword, token)] = count
                elif now and not was:
                    gained[(list_name, keyword, token)] = count

    print(f"\nGAINED — tokens a keyword now matches ({sum(gained.values())} hits). "
          "Every one must be a true inflection of the keyword:")
    for (list_name, keyword, token), count in gained.most_common(args.limit) or []:
        print(f"  {list_name:14s} {keyword:16s} + {token!r} ({count})")
    if not gained:
        print("  none")

    print(f"\nLOST — tokens a keyword no longer matches ({sum(lost.values())} hits). "
          "Every one must be an unrelated word:")
    for (list_name, keyword, token), count in lost.most_common(args.limit):
        print(f"  {list_name:14s} {keyword:16s} - {token!r} ({count})")
    if not lost:
        print("  none")
    if len(lost) > args.limit:
        print(f"  … {len(lost) - args.limit} more (raise --limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
