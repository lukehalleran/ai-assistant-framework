#!/usr/bin/env python3
"""
One-time summary backlog catch-up, offline, through THE deployed summary path.
(2026-10-15, class: BC-95, BC-27)

Shutdown summarizes at most `SUMMARY_MAX_BLOCKS_PER_SHUTDOWN` (default 1) block of N
conversations per exit because it shares a 60 s budget with fact extraction.  After the
10-14 coverage fix that is ~3 weeks of shutdowns for the existing backlog.  This script
drains the backlog in one offline run by calling the SAME method the shutdown calls,
`ShutdownProcessor.summarize_uncovered_blocks(cap)` - block selection (content-time
watermark), the extractive prompt (`MemoryConsolidator`) and storage (`_store_summary`:
corpus + Chroma with temporal anchors + claim registration) are never re-derived here.

Default is a DRY RUN: prints the content-time watermark, the conversations newer than it,
the planned blocks and the estimated LLM calls.  No model call, no write, no ModelManager.

--apply:
  - REFUSES while a live Daemon main.py is detected (fails closed).
  - PREFLIGHT (refuses, nothing written): every Chroma `conversations` doc flagged
    `curation_quarantined` and newer than the watermark must have its CORPUS copy flagged
    too (`memory.quarantine_sync.match_corpus_index`; no corpus copy = fine, ambiguous =
    refuse).  Otherwise the selector (which reads `CorpusManager.visible_entries()`) would
    summarize the quarantined, possibly false, content again (BC-98).
  - then `summarize_uncovered_blocks(cap=--limit)`; the corpus is saved by
    `CorpusManager.add_summary` and the claim index by `ClaimIndex.save()` exactly as the
    shutdown's persistence phase does.

Usage (Daemon DOWN, from the repo root):
    python scripts/backfill_summaries.py                    # dry run
    python scripts/backfill_summaries.py --limit 2 --apply  # canary
    python scripts/backfill_summaries.py --apply            # the rest

Exit codes: 0 ok, 1 refused.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.daemon_guard import daemon_running  # noqa: E402

QUARANTINE_KEY = "curation_quarantined"
ALL_BLOCKS = 10 ** 9


# ---------------------------------------------------------------------------
# Component builders (patch points for tests; the app's own constructors)
# ---------------------------------------------------------------------------

def _open_store():
    from config import app_config
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
    return MultiCollectionChromaStore(persist_directory=app_config.CHROMA_PATH)


def _open_corpus():
    from config import app_config
    from memory.corpus_manager import CorpusManager
    return CorpusManager(app_config.CORPUS_FILE)


def _summary_block_size() -> int:
    """Block size the coordinator hands the consolidator (config memory.summary_interval)."""
    from config import app_config
    try:
        return max(1, int(((app_config.config.get("memory") or {}).get("summary_interval") or 10)))
    except (ValueError, TypeError):
        return 10


def _open_model_manager():
    """Same construction as main.build_orchestrator: .env authority, then the config's active model."""
    from dotenv import load_dotenv
    from utils.bootstrap import resolve_env_path
    load_dotenv(resolve_env_path(), override=True)
    from config import app_config
    from models.model_manager import ModelManager
    mm = ModelManager()
    active = (app_config.config.get("models", {}) or {}).get("active") or "gpt-5"
    mm.switch_model(active)
    return mm


def _open_claim_index():
    """Same gate and path as MemoryCoordinator._load_claim_index (None when staleness is off)."""
    from config import app_config
    if not app_config.STALENESS_ENABLED:
        return None
    from memory import claim_tracker
    return claim_tracker.ClaimIndex(persist_path=app_config.STALENESS_INDEX_PATH)


def _open_entity_resolver():
    """Read-only use (claim extraction); never saved here. None when the graph is off/unavailable."""
    from config import app_config
    if not app_config.KNOWLEDGE_GRAPH_ENABLED:
        return None
    from memory.entity_resolver import EntityResolver
    from memory.graph_memory import GraphMemory
    gm = GraphMemory(persist_path=app_config.KNOWLEDGE_GRAPH_PERSIST_PATH)
    return EntityResolver(graph_memory=gm, aliases_path=app_config.KNOWLEDGE_GRAPH_ALIASES_PATH)


def make_processor(store, cm, *, consolidator=None, claim_index=None, entity_resolver=None):
    """ShutdownProcessor wired for the summary path only.

    Passed: corpus_manager, chroma_store, consolidator (the three the selection/consolidation/
    storage call graph reads), claim_index + a memory_coordinator shim exposing only
    `entity_resolver` (the only attribute `_store_summary` reads, for claim hashing).
    None: fact_extractor, model_manager (the consolidator already holds it), user_profile,
    storage - none is touched by select/generate/store (verified by reading the call graph).
    """
    from memory.shutdown_processor import ShutdownProcessor
    from datetime import datetime
    mc = SimpleNamespace(entity_resolver=entity_resolver) if entity_resolver is not None else None
    return ShutdownProcessor(
        corpus_manager=cm, chroma_store=store, consolidator=consolidator, fact_extractor=None,
        model_manager=None, user_profile=None, storage=None, session_start=datetime.now(),
        memory_coordinator=mc, claim_index=claim_index,
    )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(store, cm, watermark) -> list[str]:
    """Problems that must block --apply: quarantined Chroma conversations newer than the
    watermark whose corpus copy is unflagged or cannot be matched unambiguously."""
    from memory import summary_coverage as cov
    from memory.quarantine_sync import OUTCOME_AMBIGUOUS, OUTCOME_MATCH, match_corpus_index
    problems: list[str] = []
    for doc in store.list_all("conversations"):
        md = doc.get("metadata") or {}
        if not md.get(QUARANTINE_KEY):
            continue
        ts = cov._parse(md.get("timestamp"))
        if watermark is not None and ts is not None and ts <= watermark:
            continue
        idx, reason = match_corpus_index("conversations", doc, cm.corpus)
        if reason == OUTCOME_MATCH and not cm.corpus[idx].get(QUARANTINE_KEY):
            problems.append(f"{doc.get('id')}\tcorpus copy at index {idx} is NOT flagged")
        elif reason == OUTCOME_AMBIGUOUS:
            problems.append(f"{doc.get('id')}\tcorpus copy is ambiguous (several candidates)")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt(dt) -> str:
    return dt.isoformat(timespec="minutes") if dt else "-"


def print_plan(processor, cap: int) -> int:
    """Print the dry-run report; returns the number of planned blocks."""
    from memory import summary_coverage as cov
    sel = processor._select_uncovered(cap)
    if sel is None:
        print("Summary coverage is UNREADABLE - the deployed path would skip summarizing. Nothing planned.")
        return 0
    non_summ, n, watermark, _ = sel
    newer = [e for e in non_summ if watermark is None or cov.entry_time(e) > watermark]
    print(f"watermark (content time of newest summary anchor): {_fmt(watermark)}")
    print(f"block size N={n}; conversations newer than the watermark: {len(newer)} "
          f"(= {len(newer) // n} full blocks + {len(newer) % n} waiting)")
    plan = processor.plan_uncovered_blocks(cap)
    for p in plan:
        print(f"  block {p['index']}: {p['n_entries']} conversations, {_fmt(p['start'])} -> {_fmt(p['end'])}")
    print(f"planned blocks: {len(plan)}; estimated LLM calls: {len(plan)} (one consolidate_memories call each)")
    return len(plan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true", help="actually summarize (Daemon DOWN; spends LLM credits)")
    ap.add_argument("--limit", type=int, default=None, help="cap on blocks (default: all)")
    args = ap.parse_args(argv)
    cap = ALL_BLOCKS if args.limit is None else max(0, args.limit)

    if args.apply and daemon_running():
        print("REFUSED: a live Daemon main.py process is running (or could not be ruled out) - it holds "
              "ChromaDB and the corpus in memory and would re-save over this change. Shut it down first.")
        return 1

    store = _open_store()
    cm = _open_corpus()
    processor = make_processor(store, cm, consolidator=SimpleNamespace(
        consolidation_threshold=_summary_block_size()))

    if not args.apply:
        print_plan(processor, cap)
        print("\nDRY RUN - no model call, nothing was written. Re-run with --apply (Daemon DOWN).")
        return 0

    sel = processor._select_uncovered(cap)
    if sel is None:
        print("REFUSED: summary coverage is unreadable; summarizing blind is never done.")
        return 1
    watermark = sel[2]
    problems = preflight(store, cm, watermark)
    if problems:
        print(f"REFUSED: {len(problems)} quarantined conversation(s) newer than the watermark "
              f"would be summarized again (corpus copy not flagged). Nothing was written.")
        for p in problems[:20]:
            print("  " + p)
        print("Fix: build a 'conversations<TAB>id' file of those ids and run\n"
              "  python scripts/quarantine_memory.py --from-file <file> --apply\n"
              "(flags the corpus copy too), then re-run this script.")
        return 1

    from memory.memory_consolidator import MemoryConsolidator
    consolidator = MemoryConsolidator(_open_model_manager(), consolidation_threshold=_summary_block_size())
    claim_index = _open_claim_index()
    try:
        resolver = _open_entity_resolver()
    except Exception as e:  # degrades: claims hashed without alias resolution (same as MemoryCoordinator)
        print(f"note: entity resolver unavailable ({type(e).__name__}); claims registered without alias resolution")
        resolver = None
    processor = make_processor(store, cm, consolidator=consolidator,
                               claim_index=claim_index, entity_resolver=resolver)

    created = asyncio.run(processor.summarize_uncovered_blocks(cap))
    if claim_index is not None:
        claim_index.save()
    cm.invalidate_read_cache()

    from memory import summary_coverage as cov
    new_wm = cov.coverage_watermark(cov.covered_ranges(store))
    print(f"Created {created} summary block(s). New watermark: {_fmt(new_wm)}")
    remaining = processor._select_uncovered(ALL_BLOCKS)
    if remaining is not None:
        print(f"Full blocks still uncovered: {len(remaining[3])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
