"""DM-42 / BC-96 ratchet: a record's creation time is never consumed as content time.

A summary written today about September carries creation `timestamp` = today.
Sorting or labelling it by that stamp made September content read as "recent".
Summaries must be read with `time_keys=` (content anchor first).
"""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("core", "memory", "knowledge", "utils", "api", "gui", "models", "processing", "scripts")


def find_unkeyed_summary_reads(source):
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get_recent" and node.args \
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "summaries":
            if not any(k.arg == "time_keys" for k in node.keywords):
                hits.append(node.lineno)
    return hits


def test_no_summary_get_recent_without_time_keys():
    bad = []
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if "tests" in p.parts:
                continue
            for ln in find_unkeyed_summary_reads(p.read_text(encoding="utf-8")):
                bad.append(f"{p.relative_to(ROOT)}:{ln}")
    assert bad == [], bad


def test_sensitivity_unkeyed_read_is_flagged():
    assert find_unkeyed_summary_reads("store.get_recent('summaries', limit=3)") == [1]
    assert find_unkeyed_summary_reads("store.get_recent('summaries', 3, time_keys=('a',))") == []
    assert find_unkeyed_summary_reads("store.get_recent('facts', 3)") == []


class _Store:
    """Real get_recent semantics over canned rows."""
    def __init__(self, rows):
        self.rows = rows

    def list_all(self, name):
        return list(self.rows)


def test_deployed_get_summaries_sorts_by_content_not_creation():
    from memory.memory_retriever import MemoryRetriever
    from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore as S
    rows = [
        {"id": "a", "content": "newer content written EARLY. Long enough summary text here.",
         "metadata": {"timestamp": "2026-10-01T10:00:00", "temporal_anchor_start": "2026-09-30T10:00:00",
                      "temporal_anchor_end": "2026-09-30T12:00:00"}},
        {"id": "b", "content": "older content written LATE. Long enough summary text here too.",
         "metadata": {"timestamp": "2026-10-13T10:00:00", "temporal_anchor_start": "2026-09-17T10:00:00",
                      "temporal_anchor_end": "2026-09-19T12:00:00"}},
    ]
    store = _Store(rows)
    store.get_recent = lambda name, limit=8, time_keys=("timestamp",): S.get_recent(store, name, limit, time_keys)
    r = MemoryRetriever.__new__(MemoryRetriever)
    r.chroma_store = store
    out = r.get_summaries(limit=2)
    assert [o["content"][:5] for o in out] == ["newer", "older"]
    assert out[0]["covers"] == "2026-09-30" and out[1]["covers"] == "2026-09-17 – 2026-09-19"
