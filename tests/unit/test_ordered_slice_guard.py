"""
Source-level guard for the newest-first-then-truncate bug class (five prior
incidents: agentic digest 2026-08-02, LLM extractor 2026-08-05,
_recent_distress_from_history 2026-08-22, summary/reflection floors
2026-08-27, insight evidence 2026-08-31/2026-09-04).

Scans core/, memory/, utils/, gui/ for the pattern
``<name>[-N:]`` / ``<name>[:N]`` / ``[::-1]`` where ``<name>`` matches
``(recent|conversations|summaries|reflections|memories|history|entries|
evidence|events|items)\\w*`` — a variable name that SOUNDS like a
retrieval list being truncated by raw position instead of by timestamp.
Every hit must be either:
  (a) inside utils/ordered_slice.py itself (the single source of truth),
  (b) preceded within 3 lines by a sort/helper call establishing the order
      the slice then trusts, or
  (c) explicitly allowlisted below with a one-line reason.

The allowlist is where the doctrine lives — each entry documents WHY that
specific slice is not an instance of the bug (already sorted immediately
above, not chronological data at all, dead/unused code, etc.), reviewed
2026-09-04. Keep it short: a new hit should be fixed, not reflexively
allowlisted.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "utils", "gui")

_NAME_RE = r"(recent|conversations|summaries|reflections|memories|history|entries|evidence|events|items)\w*"
_PATTERN = re.compile(rf"\b{_NAME_RE}\s*\[(?:-\s*\w*\s*:\s*\w*|\w*\s*:\s*\w*|::-1)\]")
_SORT_HINT_RE = re.compile(r"\.sort\(|sorted\(|newest_first\(|oldest_first\(|window_fair_sample\(|round_robin_merge\(|head\(")

# (relative_path, line_number, matched_text) -> one-line reason.
# A hit is allowlisted by (path, line, match) so a line-number drift from an
# unrelated edit re-surfaces it here for re-review rather than silently
# staying "fixed" against the wrong line.
ALLOWLIST = {
    # --- Append-only / externally-pre-ordered sources: no separate sort needed ---
    ("core/actions/audit.py", 83, "entries[-limit:]"):
        "Append-only JSONL audit log read in file order — chronological by "
        "construction (each _append() appends at the end); nothing to sort.",
    ("core/actions/google_calendar.py", 120, "events[:max_events]"):
        "Google Calendar API call sets orderBy='startTime' — events already "
        "arrive date-ordered; [:max_events] correctly keeps the soonest N.",

    # --- Dead / unused helpers: nothing to fix ---
    ("core/prompt/base.py", 73, "items[-limit:]"):
        "_truncate_list has no production call sites (grep-confirmed; only "
        "re-export shims + direct unit tests with plain ints) — generic "
        "List[Any] utility with no timestamp concept to key on.",
    ("core/prompt/formatter.py", 118, "items[-limit:]"):
        "Duplicate _truncate_list definition, same as core/prompt/base.py — "
        "no production call sites.",
    ("core/prompt/context_gatherer.py", 386, "items[:max_items]"):
        "_bounded has no call sites anywhere in the codebase (dead helper).",
    ("core/prompt/base.py", 166, "memories[:3]"):
        "_FallbackMemoryCoordinator is an explicitly-named degraded-mode/"
        "test stub (see class docstring); its synthesized memory dicts "
        "carry no timestamp field at all (query/response/metadata only).",

    # --- Freshly LLM-generated output (single generation call, no chronology) ---
    ("core/prompt/summarizer.py", 455, "items[:needed]"):
        "items are lines parsed from ONE fresh LLM generation call — no "
        "per-item timestamp exists; [:needed] is a defensive cap on parsed "
        "output count, not a recency truncation.",
    ("memory/thread_extractor.py", 239, "items[:5]"):
        "items = freshly parsed LLM JSON output from one generation call — "
        "same class as summarizer.py above, no chronology involved.",

    # --- Relevance/score-ranked, not chronological ---
    ("core/prompt/gatherer_memory.py", 410, "memories[:limit]"):
        "_apply_valence_cap's `memories` arrives relevance-scored from the "
        "retrieval gate (memory_scorer/memory_retriever) — top-`limit` "
        "means top-SCORED, not most-recent.",
    ("core/prompt/gatherer_memory.py", 684, "summaries[:limit]"):
        "_get_summaries_hybrid_filtered pass-through/fallback branch — the "
        "success branch returns gate-relevance-ranked results; not chronological.",
    ("core/prompt/gatherer_memory.py", 696, "summaries[:limit]"):
        "Same function, exception fallback branch — passes through whatever "
        "order the caller supplied (already handled at the caller).",
    ("core/prompt/gatherer_memory.py", 748, "reflections[:limit]"):
        "_get_reflections_hybrid_filtered — same relevance-ranked class as "
        "the summaries hybrid-filter pair above.",
    ("core/prompt/gatherer_memory.py", 760, "reflections[:limit]"):
        "Same function, exception fallback branch.",
    ("memory/memory_retriever.py", 629, "entries[:max_hits]"):
        "Keyword-anchor entries: relevance evidence is LEXICAL match order "
        "(corpus_manager.search_keyword hits), not chronological — see "
        "CLAUDE.md Keyword-anchor retrieval.",
    ("memory/memory_retriever.py", 1606, "memories[:RERANK_TOP_N]"):
        "memories arrives sorted by scorer-assigned final_score (multi-"
        "factor relevance) — 'Items beyond this cutoff keep their scorer-"
        "assigned rank' per the docstring, correct as designed.",
    ("memory/memory_retriever.py", 1607, "memories[RERANK_TOP_N:]"):
        "Same split as above — the tail keeps its scorer rank by design.",

    # NOTE: hits immediately preceded (within 3 lines) by a .sort(/sorted(/
    # newest_first(/etc. call are auto-excluded by _find_hits()'s rule (b)
    # and never need an entry here — memory/corpus_manager.py's
    # get_summaries/get_summaries_of_type/get_items_by_type,
    # memory/cross_deduplicator.py's entries[1:], and
    # memory/user_profile.py's recent_candidates[:recent_count] all fall
    # in this category (each sorts by timestamp on the line(s) directly
    # above its slice).

    # --- Contract-guaranteed ordering documented at the source function ---
    ("memory/user_profile.py", 823, "history[-_tl_max:]"):
        "get_fact_history()'s own docstring + implementation guarantee "
        "oldest-first (ascending sort) — the tail slice correctly takes "
        "the newest N; comment at the call site restates the contract.",
    ("memory/memory_retriever.py", 928, "recent[:recent_budget]"):
        "recent = self.get_reflections(...), which returns newest-first "
        "(get_items_by_type sorts descending upstream) — head correctly "
        "keeps the newest recent_budget.",
    ("memory/memory_retriever.py", 1064, "recent[:recent_budget]"):
        "Same get_reflections(...) newest-first contract as above.",
    ("memory/memory_retriever.py", 1084, "recent[recent_budget:]"):
        "Same contract — the tail correctly holds the OLDER remainder of "
        "an already newest-first list.",

    # --- Debug/report/telemetry display only (not user-facing evidence) ---
    ("core/insight/coordinator.py", 389, "events[:12]"):
        "Builds the debug/telemetry 'source_ids' sample field only — the "
        "deliberation's actual comparison stats are computed over the FULL "
        "events list elsewhere; sample order doesn't affect correctness.",
    ("memory/memory_scorer.py", 650, "memories[:5]"):
        "Debug-log-only top-5 print, immediately preceded by an explicit "
        "memories.sort(key=final_score, reverse=True) two lines above.",
    ("memory/dedup_models.py", 100, "entries[:5]"):
        "Markdown dry-run dedup report renderer (human preview of up to 5 "
        "sample entries) — not part of the actual keep/delete decision "
        "logic (c.keep_id is computed elsewhere in cross_deduplicator.py).",

    # --- Shutdown-time best-effort context; already relevance-capped upstream ---
    ("memory/shutdown_processor.py", 1363, "memories[:5]"):
        "_gather_proposal_context: memories = mc.get_memories(query, "
        "limit=5) a few lines above — already capped+relevance-ordered; "
        "this re-slice is a redundant no-op. Shutdown-time best-effort "
        "proposal context, not the live prompt path.",
    ("memory/shutdown_processor.py", 1374, "summaries[:3]"):
        "Same function: summaries = mc.get_summaries_hybrid(query, limit=3) "
        "above — redundant no-op re-cap.",
    ("memory/shutdown_processor.py", 1384, "reflections[:2]"):
        "Same function: reflections via get_reflections_hybrid(query, "
        "limit=2) above — redundant no-op re-cap.",
    ("memory/corpus_manager.py", 489, "recent[:num_recent]"):
        "create_summary_now: recent = self.get_recent_memories(num_recent) "
        "on the line above already caps at num_recent — redundant no-op "
        "re-slice, not a truncation-order bug.",
}


def _iter_py_files():
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if base.exists():
            yield from base.rglob("*.py")


def _find_hits():
    hits = []
    for path in _iter_py_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel == "utils/ordered_slice.py":
            continue  # (a) the single source of truth itself
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                continue
            for m in _PATTERN.finditer(line):
                lineno = i + 1
                context = lines[max(0, i - 3):i + 1]
                if any(_SORT_HINT_RE.search(c) for c in context):
                    continue  # (b) sorted/helper-derived within 3 lines
                hits.append((rel, lineno, m.group(0)))
    return hits


class TestOrderedSliceGuard:
    def test_every_hit_is_fixed_or_allowlisted(self):
        hits = _find_hits()
        unexplained = [h for h in hits if h not in ALLOWLIST]
        assert not unexplained, (
            "New/changed newest-first-then-truncate-shaped slice(s) found "
            "with no sort nearby and no allowlist entry — fix it (sort via "
            "utils.ordered_slice) or add a reviewed, reasoned allowlist "
            "entry in tests/unit/test_ordered_slice_guard.py:\n"
            + "\n".join(f"  {h}" for h in unexplained)
        )

    def test_allowlist_entries_still_exist(self):
        """An allowlist entry whose (path, line, text) no longer matches
        current source is STALE — either the line moved (re-review it) or
        it was fixed (remove the entry). Keeps the allowlist honest."""
        hits = set(_find_hits())
        # Only check entries that would otherwise be findable at all — i.e.
        # confirm allowlisted call sites still exist verbatim in source
        # (ignoring the 3-line sort-hint carve-out, since allowlisted lines
        # by definition have none).
        stale = [entry for entry in ALLOWLIST if entry not in hits]
        assert not stale, (
            "Allowlist entries no longer found verbatim in source (line "
            "drifted or code changed) — re-review and update:\n"
            + "\n".join(f"  {e}" for e in stale)
        )

    def test_allowlist_is_reasonably_short(self):
        """The allowlist is where the doctrine lives — it should stay a
        reviewed, deliberate list, not grow unboundedly. This is a soft
        tripwire: if it more than doubles, someone should look at why."""
        assert len(ALLOWLIST) <= 40, (
            f"Allowlist grew to {len(ALLOWLIST)} entries — review whether "
            "genuine fixes are being allowlisted instead of applied."
        )
