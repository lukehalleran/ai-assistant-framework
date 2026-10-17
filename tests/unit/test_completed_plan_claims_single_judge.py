"""BC-26 closure for the thread quick-resolver / narrative completed-plan pair:
both must call THE one per-clause judge, and the resolver must not regrow a
whole-message keyword-overlap resolution path.  class: BC-26, BC-58

Original batch label: oct14e (renamed 2026-10-16; old path in tests/RENAMES.tsv).
"""
import ast
from pathlib import Path

import utils.completed_plan_claims as cpc
from memory import thread_store
from memory.thread_store import OpenThread, check_quick_resolutions

ROOT = Path(__file__).resolve().parents[2]


def test_both_consumers_call_the_shared_judge(monkeypatch):
    calls = []

    def spy(clause, head, modifier, **kw):
        calls.append(clause)
        return True

    monkeypatch.setattr(cpc, "clause_reports_completion", spy)
    th = OpenThread(thread_id="x", topic="Submit homework 7", summary="")
    assert check_quick_resolutions("I submitted homework 7", [th]) == ["x"]
    n_thread = len(calls)
    assert n_thread >= 1
    out = cpc.completed_by_user(
        "He plans to submit homework 7.",
        [{"user_text": "I submitted homework 7", "timestamp": "2026-10-14T10:00:00"}],
    )
    assert out is not None
    assert len(calls) > n_thread


def test_resolver_has_no_keyword_overlap_resolution():
    tree = ast.parse((ROOT / "memory" / "thread_store.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "check_quick_resolutions")
    for node in ast.walk(fn):
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)), \
            "check_quick_resolutions must not intersect keyword sets"
    src = (ROOT / "memory" / "thread_store.py").read_text(encoding="utf-8")
    assert "_object_hit" not in src


def test_sensitivity_scan_detects_overlap_shape():
    bad = ast.parse("def check_quick_resolutions(a, b):\n    return a & b\n")
    assert any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd) for n in ast.walk(bad))
