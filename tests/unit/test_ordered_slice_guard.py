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

Anchoring (2026-09-05): an entry is keyed by a CONTENT anchor —
``(path, enclosing function, stripped source line)`` — not by line number.
The first version pinned line numbers and went red on CI the first time
unrelated hunks higher in two files shifted the slices (push a690a91:
+5/+9 lines, slice code untouched). A content anchor survives pure drift;
editing the slice line itself, or moving it into another function, still
surfaces the entry as STALE for re-review. Identical lines inside one
function (a success branch and its exception-fallback twin) each need
their own entry: the list is matched as a multiset, so a THIRD identical
line would still show up as unexplained.
"""
import ast
import re
from collections import Counter, namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "utils", "gui")

_NAME_RE = r"(recent|conversations|summaries|reflections|memories|history|entries|evidence|events|items)\w*"
_PATTERN = re.compile(rf"\b{_NAME_RE}\s*\[(?:-\s*\w*\s*:\s*\w*|\w*\s*:\s*\w*|::-1)\]")
_SORT_HINT_RE = re.compile(r"\.sort\(|sorted\(|newest_first\(|oldest_first\(|window_fair_sample\(|round_robin_merge\(|head\(")

Hit = namedtuple("Hit", "rel lineno scope text match")

# ((relative_path, enclosing_function, stripped_source_line), one-line reason)
#
# enclosing_function is the dotted qualified name (Class.method,
# outer.inner) or "<module>". The stripped source line is the whole line,
# trailing comment included, with leading/trailing whitespace removed.
# A pure line-number drift keeps the entry valid; any edit to the line, or
# a move into a different function, makes the entry stale (re-review it).
ALLOWLIST = (
    # --- Append-only / externally-pre-ordered sources: no separate sort needed ---
    (("core/actions/audit.py", "ActionAuditLog.get_history",
      "return entries[-limit:]"),
     "Append-only JSONL audit log read in file order — chronological by "
     "construction (each _append() appends at the end); nothing to sort."),
    (("core/actions/google_calendar.py", "fetch_upcoming_events",
      "return events[:max_events]"),
     "Google Calendar API call sets orderBy='startTime' — events already "
     "arrive date-ordered; [:max_events] correctly keeps the soonest N."),

    # --- Dead / unused helpers: nothing to fix ---
    (("core/prompt/base.py", "_truncate_list",
      "return items[-limit:] if len(items) > limit else items"),
     "_truncate_list has no production call sites (grep-confirmed; only "
     "re-export shims + direct unit tests with plain ints) — generic "
     "List[Any] utility with no timestamp concept to key on."),
    (("core/prompt/formatter.py", "_truncate_list",
      "return items[-limit:] if len(items) > limit else items"),
     "Duplicate _truncate_list definition, same as core/prompt/base.py — "
     "no production call sites."),
    (("core/prompt/context_gatherer.py", "ContextGatherer._bounded",
      "return items[:max_items] if len(items) > max_items else items"),
     "_bounded has no call sites anywhere in the codebase (dead helper)."),
    (("core/prompt/base.py", "_FallbackMemoryCoordinator.retrieve_relevant_memories",
      '"recent_conversations": memories[:3],'),
     "_FallbackMemoryCoordinator is an explicitly-named degraded-mode/"
     "test stub (see class docstring); its synthesized memory dicts "
     "carry no timestamp field at all (query/response/metadata only)."),

    # --- Freshly LLM-generated output (single generation call, no chronology) ---
    (("core/prompt/summarizer.py", "LLMSummarizer._reflect_on_demand",
      "return items[:needed]"),
     "items are lines parsed from ONE fresh LLM generation call — no "
     "per-item timestamp exists; [:needed] is a defensive cap on parsed "
     "output count, not a recency truncation."),
    (("memory/thread_extractor.py", "ThreadExtractor.extract_new_threads",
      "for item in items[:5]:  # cap at 5"),
     "items = freshly parsed LLM JSON output from one generation call — "
     "same class as summarizer.py above, no chronology involved."),

    # --- Relevance/score-ranked, not chronological ---
    (("core/prompt/gatherer_memory.py", "MemoryRetrievalMixin._apply_valence_cap",
      "top = memories[:limit]"),
     "_apply_valence_cap's `memories` arrives relevance-scored from the "
     "retrieval gate (memory_scorer/memory_retriever) — top-`limit` "
     "means top-SCORED, not most-recent."),
    (("core/prompt/gatherer_memory.py", "MemoryRetrievalMixin._get_summaries_hybrid_filtered",
      "return summaries[:limit]"),
     "_get_summaries_hybrid_filtered pass-through/fallback branch — the "
     "success branch returns gate-relevance-ranked results; not chronological."),
    (("core/prompt/gatherer_memory.py", "MemoryRetrievalMixin._get_summaries_hybrid_filtered",
      "return summaries[:limit]"),
     "Same function, exception fallback branch (second identical line — "
     "multiset match, so both branches carry an entry) — passes through "
     "whatever order the caller supplied (already handled at the caller)."),
    (("core/prompt/gatherer_memory.py", "MemoryRetrievalMixin._get_reflections_hybrid_filtered",
      "return reflections[:limit]"),
     "_get_reflections_hybrid_filtered — same relevance-ranked class as "
     "the summaries hybrid-filter pair above."),
    (("core/prompt/gatherer_memory.py", "MemoryRetrievalMixin._get_reflections_hybrid_filtered",
      "return reflections[:limit]"),
     "Same function, exception fallback branch (second identical line)."),
    (("memory/memory_retriever.py", "MemoryRetriever._keyword_anchor_memories",
      'entries[:max_hits], id_prefix="kwanchor", base_relevance=0.7'),
     "Keyword-anchor entries: relevance evidence is LEXICAL match order "
     "(corpus_manager.search_keyword hits), not chronological — see "
     "CLAUDE.md Keyword-anchor retrieval."),
    (("memory/memory_retriever.py", "MemoryRetriever._maybe_cross_encoder_rerank",
      "to_rerank = memories[:RERANK_TOP_N]"),
     "memories arrives sorted by scorer-assigned final_score (multi-"
     "factor relevance) — 'Items beyond this cutoff keep their scorer-"
     "assigned rank' per the docstring, correct as designed."),
    (("memory/memory_retriever.py", "MemoryRetriever._maybe_cross_encoder_rerank",
      "tail = memories[RERANK_TOP_N:]"),
     "Same split as above — the tail keeps its scorer rank by design."),

    # NOTE: hits immediately preceded (within 3 lines) by a .sort(/sorted(/
    # newest_first(/etc. call are auto-excluded by _find_hits()'s rule (b)
    # and never need an entry here — memory/corpus_manager.py's
    # get_summaries/get_summaries_of_type/get_items_by_type,
    # memory/cross_deduplicator.py's entries[1:], and
    # memory/user_profile.py's recent_candidates[:recent_count] all fall
    # in this category (each sorts by timestamp on the line(s) directly
    # above its slice).

    # --- Contract-guaranteed ordering documented at the source function ---
    (("memory/user_profile.py", "UserProfile.get_context_injection",
      "recent = history[-_tl_max:] if _tl_max > 0 else history"),
     "get_fact_history()'s own docstring + implementation guarantee "
     "oldest-first (ascending sort) — the tail slice correctly takes "
     "the newest N; comment at the call site restates the contract."),
    (("memory/memory_retriever.py", "MemoryRetriever.get_reflections_hybrid",
      "for item in recent[:recent_budget]:"),
     "recent = self.get_reflections(...), which returns newest-first "
     "(get_items_by_type sorts descending upstream) — head correctly "
     "keeps the newest recent_budget."),
    (("memory/memory_retriever.py", "MemoryRetriever.get_summaries_hybrid",
      "for item in recent[:recent_budget]:"),
     "Summaries twin of get_reflections_hybrid — same newest-first "
     "get_items_by_type contract."),
    (("memory/memory_retriever.py", "MemoryRetriever.get_summaries_hybrid",
      "for item in recent[recent_budget:]:"),
     "Same contract — the tail correctly holds the OLDER remainder of "
     "an already newest-first list."),

    # --- Debug/report/telemetry display only (not user-facing evidence) ---
    (("core/insight/coordinator.py", "_phase_manifest",
      '"source_ids": [event.source_id for event in comparison.events[:12]],'),
     "Builds the debug/telemetry 'source_ids' sample field only — the "
     "deliberation's actual comparison stats are computed over the FULL "
     "events list elsewhere; sample order doesn't affect correctness."),
    (("memory/memory_scorer.py", "MemoryScorer.rank_memories",
      "for i, mm in enumerate(memories[:5], 1):"),
     "Debug-log-only top-5 print, immediately preceded by an explicit "
     "memories.sort(key=final_score, reverse=True) two lines above."),
    (("memory/dedup_models.py", "DedupPlan.to_markdown",
      "for entry in c.entries[:5]:"),
     "Markdown dry-run dedup report renderer (human preview of up to 5 "
     "sample entries) — not part of the actual keep/delete decision "
     "logic (c.keep_id is computed elsewhere in cross_deduplicator.py)."),

    # --- Shutdown-time best-effort context; already relevance-capped upstream ---
    (("memory/shutdown_processor.py", "ShutdownProcessor._gather_proposal_context",
      "for m in memories[:5]:"),
     "_gather_proposal_context: memories = mc.get_memories(query, "
     "limit=5) a few lines above — already capped+relevance-ordered; "
     "this re-slice is a redundant no-op. Shutdown-time best-effort "
     "proposal context, not the live prompt path."),
    (("memory/shutdown_processor.py", "ShutdownProcessor._gather_proposal_context",
      "for s in summaries[:3]"),
     "Same function: summaries = mc.get_summaries_hybrid(query, limit=3) "
     "above — redundant no-op re-cap."),
    (("memory/shutdown_processor.py", "ShutdownProcessor._gather_proposal_context",
      "for r in reflections[:2]"),
     "Same function: reflections via get_reflections_hybrid(query, "
     "limit=2) above — redundant no-op re-cap."),
    (("core/insight/synthesizer.py", "recent_conversation_context",
      "for entry in history[-8:]:"),
     "history is the chat UI transcript (Gradio messages / API session "
     "history), appended per turn so oldest-first by construction — [-8:] "
     "keeps the NEWEST four exchanges; the corpus fallback in the same "
     "function iterates reversed(get_recent_memories(4)) for the "
     "newest-first source."),
    (("memory/corpus_manager.py", "CorpusManager.create_summary_now",
      "for entry in recent[:num_recent]:"),
     "create_summary_now: recent = self.get_recent_memories(num_recent) "
     "on the line above already caps at num_recent — redundant no-op "
     "re-slice, not a truncation-order bug."),
)

_ALLOWED = Counter(key for key, _reason in ALLOWLIST)


def _function_spans(path: Path):
    """(start_line, end_line, dotted_name) for every def/async def in the
    file, qualified through enclosing classes and functions. Empty on a
    parse failure (the hit is then attributed to "<unparsed>")."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError):
        return None
    spans = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                spans.append((child.lineno, child.end_lineno or child.lineno, name))
                walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return spans


def _scope_for(spans, lineno: int) -> str:
    if spans is None:
        return "<unparsed>"
    innermost = None
    for start, end, name in spans:
        if start <= lineno <= end and (
            innermost is None or (end - start) < (innermost[1] - innermost[0])
        ):
            innermost = (start, end, name)
    return innermost[2] if innermost else "<module>"


def _key(hit: Hit):
    return (hit.rel, hit.scope, hit.text)


def _find_hits_in_file(path: Path, rel: str):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    hits = []
    spans = None  # parsed lazily: only files with a raw hit pay for ast
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            continue
        for m in _PATTERN.finditer(line):
            context = lines[max(0, i - 3):i + 1]
            if any(_SORT_HINT_RE.search(c) for c in context):
                continue  # (b) sorted/helper-derived within 3 lines
            if spans is None:
                spans = _function_spans(path)
            lineno = i + 1
            hits.append(Hit(rel, lineno, _scope_for(spans, lineno), line.strip(), m.group(0)))
    return hits


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
        hits.extend(_find_hits_in_file(path, rel))
    return hits


def _fmt(hit: Hit) -> str:
    return (f"  {hit.rel}:{hit.lineno} [{hit.scope}] {hit.text!r}\n"
            f"    allowlist key: ({hit.rel!r}, {hit.scope!r}, {hit.text!r})")


class TestOrderedSliceGuard:
    def test_every_hit_is_fixed_or_allowlisted(self):
        seen = Counter()
        unexplained = []
        for hit in _find_hits():
            key = _key(hit)
            seen[key] += 1
            if seen[key] > _ALLOWED.get(key, 0):
                unexplained.append(hit)
        assert not unexplained, (
            "New/changed newest-first-then-truncate-shaped slice(s) found "
            "with no sort nearby and no allowlist entry — fix it (sort via "
            "utils.ordered_slice) or add a reviewed, reasoned allowlist "
            "entry in tests/unit/test_ordered_slice_guard.py:\n"
            + "\n".join(_fmt(h) for h in unexplained)
        )

    def test_allowlist_entries_still_exist(self):
        """An allowlist entry whose content anchor no longer matches a
        current hit is STALE — the slice line was edited or moved into
        another function (re-review it), or it was fixed (remove the
        entry). Pure line-number drift does NOT trip this. Keeps the
        allowlist honest."""
        found = Counter(_key(h) for h in _find_hits())
        stale = list((_ALLOWED - found).elements())
        assert not stale, (
            "Allowlist entries no longer match any current hit (slice line "
            "edited, moved to another function, or fixed) — re-review and "
            "update:\n" + "\n".join(f"  {e}" for e in stale)
        )

    def test_allowlist_is_reasonably_short(self):
        """The allowlist is where the doctrine lives — it should stay a
        reviewed, deliberate list, not grow unboundedly. This is a soft
        tripwire: if it more than doubles, someone should look at why."""
        assert len(ALLOWLIST) <= 40, (
            f"Allowlist grew to {len(ALLOWLIST)} entries — review whether "
            "genuine fixes are being allowlisted instead of applied."
        )

    # --- properties of the anchor itself ---------------------------------

    def test_anchor_survives_pure_line_drift(self, tmp_path):
        src = "def f(memories):\n    x = 1\n    return memories[:5]\n"
        p = tmp_path / "m.py"
        p.write_text(src, encoding="utf-8")
        before = _find_hits_in_file(p, "m.py")
        assert len(before) == 1 and before[0].lineno == 3

        p.write_text("import os\n\n\n" + src, encoding="utf-8")  # 3 lines added above
        after = _find_hits_in_file(p, "m.py")
        assert after[0].lineno == 6
        assert _key(after[0]) == _key(before[0])

    def test_anchor_changes_when_slice_line_is_edited_or_moved(self, tmp_path):
        src = "def f(memories):\n    return memories[:5]\n"
        p = tmp_path / "m.py"
        p.write_text(src, encoding="utf-8")
        base = _key(_find_hits_in_file(p, "m.py")[0])

        p.write_text(src.replace("memories[:5]", "memories[:6]"), encoding="utf-8")
        assert _key(_find_hits_in_file(p, "m.py")[0]) != base

        p.write_text(src.replace("def f(", "def g("), encoding="utf-8")
        assert _key(_find_hits_in_file(p, "m.py")[0]) != base

    def test_scope_is_qualified_through_classes_and_nesting(self, tmp_path):
        src = (
            "recent[:3]\n"
            "class C:\n"
            "    def m(self, recent):\n"
            "        def inner():\n"
            "            return recent[:2]\n"
            "        return recent[:1]\n"
            "async def a(events):\n"
            "    return events[:4]\n"
        )
        p = tmp_path / "m.py"
        p.write_text(src, encoding="utf-8")
        scopes = {h.match: h.scope for h in _find_hits_in_file(p, "m.py")}
        assert scopes == {
            "recent[:3]": "<module>",
            "recent[:2]": "C.m.inner",
            "recent[:1]": "C.m",
            "events[:4]": "a",
        }

    def test_identical_lines_are_matched_as_a_multiset(self, tmp_path):
        src = (
            "def f(summaries):\n"
            "    try:\n"
            "        return summaries[:3]\n"
            "    except Exception:\n"
            "        return summaries[:3]\n"
        )
        p = tmp_path / "m.py"
        p.write_text(src, encoding="utf-8")
        keys = Counter(_key(h) for h in _find_hits_in_file(p, "m.py"))
        assert list(keys.values()) == [2]  # one key, two occurrences
        one_entry = Counter({next(iter(keys)): 1})
        assert list((keys - one_entry).elements())  # a single entry leaves one unexplained
