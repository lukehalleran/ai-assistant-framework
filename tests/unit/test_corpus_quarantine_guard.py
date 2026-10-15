"""DM-46 (proposed, BC-26/BC-58): corpus CONTENT reads go through the visibility predicate.

A quarantine flag set on the Chroma copy of a record hides half the record: the same
conversation / summary / reflection also lives as a dict in the corpus JSON.  Every
corpus CONTENT read must therefore use ``CorpusManager.visible_entries()`` (which drops
``curation_quarantined`` entries); only maintenance code and COUNT-only reads may touch
the raw ``.corpus`` list.

Receiver names treated as a CorpusManager reference (derived from the repo, 2026-10-15):
``corpus_manager`` / ``_corpus_manager`` (``self.corpus_manager``, ``memory_system.corpus_manager``),
``cm``, ``corpus_mgr`` and the ``corpus`` parameter of ``gui/setup_core``.
eval/ has its own, unrelated ``CorpusManager`` class and is out of scope, as are tests/ and scripts/.

Ceiling 0 un-allowlisted.  Inside memory/corpus_manager.py the content read methods must not
reference ``self.corpus`` directly.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "gui", "api", "models", "utils", "knowledge", "memory", "processing", "config")
RECEIVERS = {"corpus_manager", "_corpus_manager", "cm", "corpus_mgr", "corpus"}

# (file, enclosing function) -> reason.  COUNT/STAT reads only; none of these read entry text.
ALLOWLIST = {
    ("gui/setup_core.py", "_established_profile"): "count-only (corpus entry count for first-run detection)",
    ("gui/launch.py", "check_first_run"): "count-only (startup corpus count print)",
    ("gui/launch.py", "get_summary_status"): "count-only (summary/block cadence status line)",
    ("memory/user_profile.py", "is_first_run"): "count-only",
    ("memory/memory_coordinator.py", "debug_memory_state"): "count-only (entry counts in the debug dict)",
}

CONTENT_READ_METHODS = {
    "_get_episodic_sorted", "search_keyword", "get_summaries", "get_summaries_of_type",
    "get_items_by_type", "get_recent_memories", "get_recent_within_hours", "_episodic_with_pending",
    "create_summary_now",
}

def _receiver_name(node: ast.AST):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

def find_raw_corpus_reads(source: str):
    """[(lineno, enclosing function)] for ``<corpus manager ref>.corpus`` attribute reads."""
    tree = ast.parse(source)
    hits = []

    def visit(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if isinstance(node, ast.Attribute) and node.attr == "corpus" \
                and _receiver_name(node.value) in RECEIVERS:
            hits.append((node.lineno, func))
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(tree, "<module>")
    return hits

def _production_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.is_dir():
            yield from (p for p in base.rglob("*.py") if "tests" not in p.parts)
    yield from ROOT.glob("*.py")

def test_no_unallowlisted_raw_corpus_reads_outside_corpus_manager():
    problems, used = [], set()
    for path in _production_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "memory/corpus_manager.py":
            continue
        for lineno, func in find_raw_corpus_reads(path.read_text(encoding="utf-8")):
            key = (rel, func)
            if key in ALLOWLIST:
                used.add(key)
            else:
                problems.append(f"{rel}:{lineno} in {func}()")
    assert not problems, (
        "Raw `.corpus` read of a CorpusManager outside corpus_manager.py. Content readers must use "
        "`visible_entries()` (quarantined entries must not surface); count-only reads need an ALLOWLIST "
        "entry with a reason:\n" + "\n".join(problems))
    stale = set(ALLOWLIST) - used
    assert not stale, f"stale ALLOWLIST entries (no matching raw read any more): {sorted(stale)}"

def test_corpus_manager_content_readers_do_not_touch_raw_corpus():
    tree = ast.parse((ROOT / "memory" / "corpus_manager.py").read_text(encoding="utf-8"))
    bad = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "CorpusManager"):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in CONTENT_READ_METHODS):
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr == "corpus" \
                        and isinstance(node.value, ast.Name) and node.value.id == "self":
                    bad.append(f"{fn.name}:{node.lineno}")
    assert not bad, f"content read methods must use self.visible_entries(): {bad}"

def test_guard_sensitivity_flags_the_original_shape():
    snippet = "def f(self):\n    for e in self.corpus_manager.corpus:\n        use(e['query'])\n"
    assert find_raw_corpus_reads(snippet) == [(2, "f")]
    assert find_raw_corpus_reads("def g(cm):\n    return cm.visible_entries()\n") == []
