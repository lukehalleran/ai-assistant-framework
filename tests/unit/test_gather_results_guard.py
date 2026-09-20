"""
Repo-wide guard for the gather/CancelledError class (BC-87, DM-33).

``asyncio.gather(..., return_exceptions=True)`` puts a cancelled child's
``CancelledError`` — a ``BaseException``, NOT an ``Exception`` — into the
results list. Every ``isinstance(x, Exception)`` / ``x is not None`` filter
and every tuple-unpack that assumes a clean value admits it into typed code
(2026-09-16 typecheck triage: agentic tool dispatch, best-of/duel/ensemble,
Gmail fetch, synthesis articulation; 2026-09-19: six more sites plus a
URL-fetch round-summary sibling; the pre-existing
``multi_collection_chroma_store.py`` slot unpacked every result before its
own check).

Scans ``core/, memory/, knowledge/, utils/, gui/, api/, models/, processing/,
agent_branch/, eval/`` plus ``main.py`` (never ``git ls-files`` — see
``tests/unit/test_no_git_state_in_tests.py``) for a ``Call`` to ``gather``
(attribute or name form) carrying the literal keyword
``return_exceptions=True``. For the innermost enclosing function, the site is
SAFE if:

  (a) the awaited result is discarded — the ``await`` is a bare expression
      statement, or the gather call sits directly inside
      ``asyncio.wait_for(...)`` that is itself a bare expression statement, or
  (b) the SAME enclosing function calls ``classify_gather_results`` or
      ``partition_gather_results`` (``utils/async_results.py`` — the single
      shared decision point that turns every ``BaseException`` result,
      including ``CancelledError``, into an explicit per-position error).

Otherwise it is a VIOLATION unless its anchor is explicitly ALLOWLISTED below
with a reviewed, one-line reason.

Anchoring (matches ``tests/unit/test_ordered_slice_guard.py``, 2026-09-05/13):
an entry is keyed by a CONTENT anchor — ``(path, enclosing function, stripped
source line)`` — not by line number, so it survives unrelated hunks shifting
line numbers elsewhere in the file. Editing the gather call itself, or moving
it into another function, makes the entry stale (re-review it). The list is
matched as a multiset: two textually identical gather calls in one function
each need their own entry.

2026-09-19: the first run of this guard found three sites no earlier sweep
had listed — ``memory/shutdown_processor.py`` ``process_shutdown_memory``
(phase_a and phase_b) and ``knowledge/document_generator.py``
``_gather_sources``. All three sit in accepted-debt files; the referee read
each one: the gathered values are consumed only under a POSITIVE type check
or a log call, so a CancelledError is dropped silently rather than crashing.
They are allowlisted below with that reason and stay BC-87 open sites.
"""
from __future__ import annotations

import ast
from collections import Counter, namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "knowledge", "utils", "gui", "api", "models", "processing", "agent_branch", "eval")
EXTRA_FILES = ("main.py",)

_HELPER_NAMES = frozenset({"classify_gather_results", "partition_gather_results"})

Hit = namedtuple("Hit", "rel lineno scope text")

# ((relative_path, enclosing_function, stripped_source_line), one-line reason)
#
# enclosing_function is the dotted qualified name (Class.method, outer.inner)
# or "<module>". The stripped source line is the AST Call node's OWN start
# line (for a multi-line call this is just the opening line, e.g.
# "asyncio.gather("), with leading/trailing whitespace removed. A pure
# line-number drift keeps the entry valid; any edit to that line, or moving
# the call into a different function, makes the entry stale (re-review it).
ALLOWLIST = (
    (("memory/shutdown_processor.py", "ShutdownProcessor._process_open_threads",
      "resolutions, new_threads = await asyncio.gather("),
     "accepted-debt file (memory/shutdown_processor.py); BC-87 open site — "
     "resolutions/new_threads are read via isinstance(x, Exception), which "
     "misses CancelledError (elif resolutions: would then iterate the "
     "exception instance). Tracked, not fixed here; closes when the file "
     "moves off accepted-debt."),
    (("knowledge/web_search_manager.py", "WebSearchManager.multi_search",
      "asyncio.gather(*search_tasks, return_exceptions=True),"),
     "accepted-debt file (knowledge/web_search_manager.py); BC-87 open site — "
     "sub-query results are merged with isinstance(page, Exception) checks "
     "downstream, missing CancelledError. Tracked, not fixed here; closes "
     "when the file moves off accepted-debt."),
    (("gui/handlers.py", "_write_turn_telemetry",
      "_telemetry_task = asyncio.gather(*_real_tasks, return_exceptions=True)"),
     "Not awaited at this call site: the Future is stored as a task object "
     "for a later waiter (PostResponseHookContext.telemetry_task) — no "
     "per-item result is read here, so there is nothing to misclassify."),
    (("main.py", "_do_shutdown_async",
      "asyncio.gather("),
     "log-only: _refl/_proc are only used for an isinstance(x, Exception) "
     "completion log line (a CancelledError would be mislabelled as "
     "completed rather than crash anything). Flagged for the referee; "
     "main.py is out of scope for this batch's edits."),
    (("memory/shutdown_processor.py", "ShutdownProcessor.process_shutdown_memory",
      "phase_a = await asyncio.gather("),
     "accepted-debt file; BC-87 open site, log-only: phase_a items are read "
     "only by `if isinstance(r, Exception): _log_shutdown_llm_failure(...)` — "
     "a CancelledError is not logged as a failure, nothing crashes."),
    (("memory/shutdown_processor.py", "ShutdownProcessor.process_shutdown_memory",
      "phase_b = await asyncio.gather("),
     "accepted-debt file; BC-87 open site, log-only: same shape as phase_a."),
    (("knowledge/document_generator.py", "DocumentGenerator._gather_sources",
      "results = await asyncio.gather(*tasks.values(), return_exceptions=True)"),
     "accepted-debt file; BC-87 open site: values are used only under "
     "`isinstance(result, list)`, so a CancelledError is skipped without its "
     "warning line rather than reaching typed code."),
)

_ALLOWED = Counter(key for key, _reason in ALLOWLIST)


def _func_name(call: ast.AST):
    if not isinstance(call, ast.Call):
        return None
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _is_gather_call(node) -> bool:
    if not isinstance(node, ast.Call) or _func_name(node) != "gather":
        return False
    return any(
        isinstance(kw, ast.keyword) and kw.arg == "return_exceptions"
        and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in node.keywords
    )


def _function_spans(tree):
    """(start_line, end_line, dotted_name, node) for every def/async def."""
    spans = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                spans.append((child.lineno, child.end_lineno or child.lineno, name, child))
                walk(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return spans


def _scope_for(spans, lineno: int):
    innermost = None
    for start, end, name, node in spans:
        if start <= lineno <= end and (
            innermost is None or (end - start) < (innermost[1] - innermost[0])
        ):
            innermost = (start, end, name, node)
    return innermost


def _func_has_helper(func_node) -> bool:
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call) and _func_name(n) in _HELPER_NAMES:
            return True
    return False


def _parent_map(tree):
    parent_of = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_of[id(child)] = parent
    return parent_of


def _is_discarded(node, parent_of) -> bool:
    """The gather Call's result is thrown away: a bare ``await`` expression
    statement, or ``await asyncio.wait_for(<this gather call>, ...)`` that is
    itself a bare expression statement."""
    parent = parent_of.get(id(node))
    if isinstance(parent, ast.Await):
        grandparent = parent_of.get(id(parent))
        return isinstance(grandparent, ast.Expr)
    if isinstance(parent, ast.Call) and _func_name(parent) == "wait_for":
        gp = parent_of.get(id(parent))
        if isinstance(gp, ast.Await):
            ggp = parent_of.get(id(gp))
            return isinstance(ggp, ast.Expr)
    return False


def _key(hit: Hit):
    return (hit.rel, hit.scope, hit.text)


def _find_hits_in_source(source: str, rel: str):
    """(violations, all_gather_sites) for one file's source text.

    ``all_gather_sites`` includes every matching call regardless of verdict
    (used only by the self-tests below); ``violations`` is what the real
    guard cares about.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [Hit(rel, 0, "<unparsed>", "<file does not parse>")], []
    lines = source.splitlines()
    spans = _function_spans(tree)
    parent_of = _parent_map(tree)
    violations = []
    all_sites = []
    for node in ast.walk(tree):
        if not _is_gather_call(node):
            continue
        scope = _scope_for(spans, node.lineno)
        scope_name = scope[2] if scope else "<module>"
        text = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
        all_sites.append(Hit(rel, node.lineno, scope_name, text))
        if _is_discarded(node, parent_of):
            continue  # (a) discarded
        if scope is not None and _func_has_helper(scope[3]):
            continue  # (b) routed through the shared helper
        violations.append(Hit(rel, node.lineno, scope_name, text))
    return violations, all_sites


def _find_hits_in_file(path: Path, rel: str):
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return [], []
    return _find_hits_in_source(source, rel)


def _iter_py_files():
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if base.exists():
            yield from sorted(base.rglob("*.py"))
    for name in EXTRA_FILES:
        p = REPO_ROOT / name
        if p.exists():
            yield p


def _find_all_violations():
    violations = []
    for path in _iter_py_files():
        rel = str(path.relative_to(REPO_ROOT))
        file_violations, _all = _find_hits_in_file(path, rel)
        violations.extend(file_violations)
    return violations


def _unexplained(hits, allowed: Counter):
    seen = Counter()
    unexplained = []
    for hit in hits:
        key = _key(hit)
        seen[key] += 1
        if seen[key] > allowed.get(key, 0):
            unexplained.append(hit)
    return unexplained


def _fmt(hit: Hit) -> str:
    return (
        f"  {hit.rel}:{hit.lineno} [{hit.scope}] {hit.text!r}\n"
        f"    fix: route through utils.async_results "
        f"(classify_gather_results / partition_gather_results), or add a "
        f"reviewed allowlist entry with key "
        f"({hit.rel!r}, {hit.scope!r}, {hit.text!r})"
    )


class TestGatherResultsGuard:
    def test_no_unclassified_gather_results(self):
        unexplained = _unexplained(_find_all_violations(), _ALLOWED)
        assert not unexplained, (
            "gather(..., return_exceptions=True) result(s) reach typed code "
            "without discarding the result or routing through "
            "utils.async_results, and are not on the reviewed allowlist:\n"
            + "\n".join(_fmt(h) for h in unexplained)
        )

    def test_allowlist_has_no_stale_entries(self):
        """An allowlist entry whose content anchor no longer matches a
        current violation is STALE — the call was fixed, edited, or moved
        into another function (re-review it, or remove the entry)."""
        found = Counter(_key(h) for h in _find_all_violations())
        stale = list((_ALLOWED - found).elements())
        assert not stale, (
            "Allowlist entries no longer match any current violation (fixed, "
            "edited, or moved) — re-review and update:\n"
            + "\n".join(f"  {e}" for e in stale)
        )

    def test_every_allowlist_entry_has_a_reason(self):
        assert all(isinstance(reason, str) and len(reason) > 20 for _key_, reason in ALLOWLIST)


class TestDetectorSelfTests:
    """Proves the detector on synthetic source strings, independent of the
    real repo's current state."""

    def test_flags_the_old_isinstance_exception_loop_shape(self):
        source = (
            "async def f(tasks):\n"
            "    results = await asyncio.gather(*tasks, return_exceptions=True)\n"
            "    for r in results:\n"
            "        if isinstance(r, Exception):\n"
            "            log(r)\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert [(h.lineno, h.scope) for h in violations] == [(2, "f")]

    def test_flags_a_tuple_unpack_with_no_helper(self):
        source = (
            "async def f(t1, t2):\n"
            "    a, b = await asyncio.gather(t1, t2, return_exceptions=True)\n"
            "    return a, b\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert len(violations) == 1

    def test_passes_bare_await_discard_shape(self):
        source = (
            "async def f(tasks):\n"
            "    await asyncio.gather(*tasks, return_exceptions=True)\n"
        )
        violations, all_sites = _find_hits_in_source(source, "m.py")
        assert violations == []
        assert len(all_sites) == 1  # the call is still SEEN, just not a violation

    def test_passes_wait_for_bare_discard_shape(self):
        source = (
            "async def f(tasks, timeout):\n"
            "    await asyncio.wait_for(\n"
            "        asyncio.gather(*tasks, return_exceptions=True),\n"
            "        timeout=timeout,\n"
            "    )\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert violations == []

    def test_wait_for_result_that_is_assigned_is_still_a_violation(self):
        """Discard only applies when the wait_for(...) call itself is a bare
        expression statement — assigning its result is not a discard."""
        source = (
            "async def f(tasks, timeout):\n"
            "    results = await asyncio.wait_for(\n"
            "        asyncio.gather(*tasks, return_exceptions=True),\n"
            "        timeout=timeout,\n"
            "    )\n"
            "    for r in results:\n"
            "        if isinstance(r, Exception):\n"
            "            log(r)\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert len(violations) == 1

    def test_passes_when_the_function_calls_the_shared_helper(self):
        source = (
            "async def f(tasks):\n"
            "    results = await asyncio.gather(*tasks, return_exceptions=True)\n"
            "    for pos, value, err in classify_gather_results(results):\n"
            "        handle(pos, value, err)\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert violations == []

    def test_passes_when_the_function_calls_partition_helper(self):
        source = (
            "async def f(coros):\n"
            "    results = await asyncio.gather(*coros, return_exceptions=True)\n"
            "    values, errors = partition_gather_results(results)\n"
            "    return values\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert violations == []

    def test_helper_call_elsewhere_in_the_function_still_counts(self):
        """(b) says 'the same function' — not 'the next line' — so a helper
        call used only to interpret the SAME variable further down still
        clears the site."""
        source = (
            "async def f(tasks):\n"
            "    results = await asyncio.gather(*tasks, return_exceptions=True)\n"
            "    n = len(results)\n"
            "    if n:\n"
            "        outcomes = classify_gather_results(results)\n"
            "        return outcomes\n"
            "    return []\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert violations == []

    def test_a_different_function_using_the_helper_does_not_launder_this_one(self):
        source = (
            "async def helper_user(results):\n"
            "    return classify_gather_results(results)\n\n"
            "async def f(tasks):\n"
            "    results = await asyncio.gather(*tasks, return_exceptions=True)\n"
            "    for r in results:\n"
            "        if isinstance(r, Exception):\n"
            "            log(r)\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert [(h.lineno, h.scope) for h in violations] == [(5, "f")]

    def test_gather_without_return_exceptions_true_is_not_matched(self):
        source = (
            "async def f(tasks):\n"
            "    results = await asyncio.gather(*tasks)\n"
            "    for r in results:\n"
            "        if isinstance(r, Exception):\n"
            "            log(r)\n"
        )
        violations, all_sites = _find_hits_in_source(source, "m.py")
        assert violations == [] and all_sites == []

    def test_bare_name_gather_import_form_is_matched(self):
        source = (
            "from asyncio import gather\n"
            "async def f(tasks):\n"
            "    results = await gather(*tasks, return_exceptions=True)\n"
            "    for r in results:\n"
            "        if isinstance(r, Exception):\n"
            "            log(r)\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert len(violations) == 1

    def test_unparseable_file_fails_closed(self):
        violations, _all = _find_hits_in_source("def broken(:\n", "m.py")
        assert [h.scope for h in violations] == ["<unparsed>"]

    def test_scope_is_qualified_through_classes_and_nesting(self):
        source = (
            "class C:\n"
            "    async def m(self, tasks):\n"
            "        async def inner():\n"
            "            results = await asyncio.gather(*tasks, return_exceptions=True)\n"
            "            for r in results:\n"
            "                if isinstance(r, Exception):\n"
            "                    log(r)\n"
            "        return inner\n"
        )
        violations, _all = _find_hits_in_source(source, "m.py")
        assert [h.scope for h in violations] == ["C.m.inner"]
