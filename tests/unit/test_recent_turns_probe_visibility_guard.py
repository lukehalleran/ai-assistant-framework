"""DM-50 (BC-105, BC-58): recent turns reach a turn only through ONE accessor.

Class BC-105 "probe traffic visible through an unfiltered reader" (incident 2026-10-28: a
real "Hey" carried four ``[test]`` probe exchanges because several readers called
``corpus_manager.get_recent_memories`` raw). Every TURN-PATH read of the corpus's recent turns
(prompt, gate, STM/tone/thread/topic, pacing, action guard) must go through
``memory.recent_turns.recent_turns_visible_to``; this guard scans core/gui/api/memory/utils/
knowledge/models/processing for a raw ``<corpus manager ref>.get_recent_memories(`` call and
allows only the listed NOT-turn-path readers (reason required). Ceiling 0 un-allowlisted;
receiver names match DM-46 (tests/unit/test_corpus_quarantine_guard.py).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "gui", "api", "models", "utils", "knowledge", "memory", "processing")
RECEIVERS = {"corpus_manager", "_corpus_manager", "cm", "corpus_mgr", "corpus"}
ACCESSOR_FILE = "memory/recent_turns.py"
OWNER_FILE = "memory/corpus_manager.py"

# (file, enclosing function) -> reason this reader is NOT a turn-path reader.
ALLOWLIST = {
    ("core/orchestrator.py", "get_memories"): "offline in-memory fallback coordinator (_InMemoryCorpus), not the live corpus",
    ("core/orchestrator.py", "retrieve_relevant_memories"): "offline in-memory fallback coordinator (_InMemoryCorpus), not the live corpus",
    ("core/prompt/base.py", "get_memories"): "_FallbackCorpusManager in-memory fallback, not the live corpus",
    ("memory/memory_retriever.py", "_get_recent_conversations"): "retrieval channel: drops EVERY probe via is_test_turn (stricter than the accessor)",
    ("gui/launch.py", "get_summary_status"): "status display of the last message time, feeds no prompt or decision",
    ("gui/tabs/proposals.py", "_generate_proposals_now"): "developer proposals tab (owner-triggered UI), not a turn",
}


def _receiver_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def find_raw_recent_reads(source: str):
    """[(lineno, enclosing function)] for ``<corpus ref>.get_recent_memories(...)`` calls."""
    hits = []

    def visit(node, func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_recent_memories"
                and _receiver_name(node.func.value) in RECEIVERS):
            hits.append((node.lineno, func))
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(ast.parse(source), "<module>")
    return hits


def _production_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.is_dir():
            yield from (p for p in base.rglob("*.py") if "tests" not in p.parts)
    yield from ROOT.glob("*.py")


def test_no_raw_recent_turn_reads_outside_the_accessor():
    problems, used = [], set()
    for path in _production_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in (OWNER_FILE,):
            continue
        for lineno, func in find_raw_recent_reads(path.read_text(encoding="utf-8")):
            if rel == ACCESSOR_FILE and func == "recent_turns_visible_to":
                continue
            key = (rel, func)
            if key in ALLOWLIST:
                used.add(key)
            else:
                problems.append(f"{rel}:{lineno} in {func}()")
    assert not problems, (
        "Raw `get_recent_memories(` on a corpus manager: turn-path readers must use "
        "`memory.recent_turns.recent_turns_visible_to` (probe turns must not reach a real turn); "
        "a non-turn-path reader needs an ALLOWLIST entry with a reason:\n" + "\n".join(problems))
    stale = set(ALLOWLIST) - used
    assert not stale, f"stale ALLOWLIST entries (no matching raw read any more): {sorted(stale)}"


def test_guard_sensitivity_flags_a_planted_raw_read(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text("def f(self):\n    return self.corpus_manager.get_recent_memories(3)\n")
    assert find_raw_recent_reads(planted.read_text()) == [(2, "f")]
    ok = "def g(cm):\n    return recent_turns_visible_to(cm, 3, current_query='x')\n"
    assert find_raw_recent_reads(ok) == []
