"""scripts/quarantine_facts.py — reversible by-id quarantine (2026-09-02)."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("quarantine_facts", _ROOT / "scripts" / "quarantine_facts.py")
qf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qf)


def test_parse_id_file_accepts_plain_and_jsonl_with_comments():
    text = '# audit\nfact_a  # whoop\n\n{"id": "fact_b", "content": "x"}\nfact_a\nnot json {\n'
    assert qf.parse_id_file(text) == ["fact_a", "fact_b"]


def test_plan_splits_by_current_state_and_existence():
    docs = [
        {"id": "f1", "content": "user | uses | WHOOP", "metadata": {}},
        {"id": "f2", "content": "user | has_dog | Mochi", "metadata": {"curation_quarantined": True}},
    ]
    to_change, already, missing = qf.plan_quarantine(docs, ["f1", "f2", "f9"])
    assert [d["id"] for d in to_change] == ["f1"]
    assert [d["id"] for d in already] == ["f2"]
    assert missing == ["f9"]
    # Undo inverts the split.
    to_change, already, _ = qf.plan_quarantine(docs, ["f1", "f2"], undo=True)
    assert [d["id"] for d in to_change] == ["f2"]
    assert [d["id"] for d in already] == ["f1"]


class _FakeStore:
    def __init__(self):
        self.calls = []

    def update_metadata(self, collection, doc_id, updates):
        self.calls.append((collection, doc_id, updates))
        return True


def test_apply_uses_deployed_update_metadata_and_is_reversible():
    store = _FakeStore()
    docs = [{"id": "f1", "content": "user | uses | WHOOP", "metadata": {}}]
    n = qf.apply_plan(store, docs, undo=False, reason="provenance audit", ts="T")
    assert n == 1
    coll, fid, upd = store.calls[0]
    assert (coll, fid) == ("facts", "f1")
    assert upd["curation_quarantined"] is True
    assert upd["curation_quarantine_reason"] == "provenance audit"
    # The retrieval-side predicate is what makes quarantine effective.
    from memory.utils import is_quarantined
    assert is_quarantined(upd)
    qf.apply_plan(store, docs, undo=True, reason="", ts="T2")
    assert store.calls[1][2]["curation_quarantined"] is False
    assert not is_quarantined(store.calls[1][2])
