"""Curation Center quarantine flips the corpus copy too (2026-10-15, class BC-98 / BC-26).

Every conversation/summary/reflection exists as a Chroma doc AND a corpus-JSON
entry. The engine used to flip Chroma only, leaving the corpus copy visible to
[RECENT CONVERSATION], STM, summaries, the keyword anchor. These tests drive the
deployed engine.apply / engine.undo with a real CorpusManager on a tmp file.
"""
import json
from datetime import datetime

import pytest

from memory.corpus_manager import CorpusManager
from memory.curation.adapters import QUARANTINE_KEY
from memory.curation.engine import CurationEngine, StoreBundle, new_proposal_id
from memory.curation.journal import CurationJournal
from memory.curation.types import (
    Confidence, CurationProposal, Instrument, ItemChange, ProposalStatus,
)
from memory.quarantine_sync import OUTCOME_AMBIGUOUS, OUTCOME_MATCH
from tests.unit.test_curation_engine import FakeChromaStore, FakeCollection

TS = "2026-10-10T12:00:00"


def _corpus(tmp_path, entries):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return CorpusManager(str(path)), path


def _engine(tmp_path, chroma, cm):
    return CurationEngine(
        StoreBundle(chroma_store=chroma, corpus_manager=cm),
        queue_path=str(tmp_path / "queue.json"),
        journal=CurationJournal(str(tmp_path / "audit.jsonl")),
    )


def _proposal(store, doc_id, change_type="quarantine", after=None):
    return CurationProposal(
        proposal_id=new_proposal_id(), curator="test", instrument=Instrument.METADATA,
        confidence=Confidence.DETERMINISTIC, title="t", evidence="e",
        items=[ItemChange(store=store, doc_id=doc_id, change_type=change_type,
                          after=after or {})],
    )


def _queue(eng, p):
    eng._proposals[p.proposal_id] = p
    return p.proposal_id


def _conv_chroma(extra=None):
    return FakeChromaStore({"conversations": FakeCollection({
        "d1": {"document": "User: hello there\nAssistant: hi",
               "metadata": {"query": "hello there", "timestamp": TS, **(extra or {})}}})})


CONV = {"query": "hello there", "response": "hi", "timestamp": TS, "tags": []}


def test_conversation_quarantine_flips_both_copies_and_persists(tmp_path):
    cm, path = _corpus(tmp_path, [dict(CONV)])
    chroma = _conv_chroma()
    eng = _engine(tmp_path, chroma, cm)
    pid = _queue(eng, _proposal("chroma:conversations", "d1"))
    eng.apply(pid)
    assert chroma.collections["conversations"].docs["d1"]["metadata"][QUARANTINE_KEY] is True
    assert cm.corpus[0][QUARANTINE_KEY] is True
    assert cm.visible_entries() == []
    on_disk = json.loads(path.read_text())
    assert on_disk[0][QUARANTINE_KEY] is True
    item = eng.get(pid).items[0]
    assert item.corpus_outcome == OUTCOME_MATCH
    audit = (tmp_path / "audit.jsonl").read_text()
    assert '"corpus_outcome": "match"' in audit
    assert "hello there" not in audit.replace('"query"', "")  # no content text in audit


def test_undo_restores_both_and_is_idempotent(tmp_path):
    cm, path = _corpus(tmp_path, [dict(CONV)])
    chroma = _conv_chroma()
    eng = _engine(tmp_path, chroma, cm)
    pid = _queue(eng, _proposal("chroma:conversations", "d1"))
    eng.apply(pid)
    eng.undo(pid)
    assert chroma.collections["conversations"].docs["d1"]["metadata"][QUARANTINE_KEY] is False
    assert QUARANTINE_KEY not in cm.corpus[0]
    assert len(cm.visible_entries()) == 1
    assert QUARANTINE_KEY not in json.loads(path.read_text())[0]
    # second revert straight through the adapter: no error, no change
    from memory.curation.adapters import revert_change
    revert_change(eng.get(pid).items[0], chroma_store=chroma, corpus_manager=cm)
    assert QUARANTINE_KEY not in cm.corpus[0]
    assert eng.get(pid).status == ProposalStatus.UNDONE


def test_summary_and_reflection_variants(tmp_path):
    cm, _ = _corpus(tmp_path, [
        {"type": "summary", "content": "sum body", "timestamp": TS},
        {"type": "reflection", "content": "refl body", "timestamp": TS},
    ])
    chroma = FakeChromaStore({
        "summaries": FakeCollection({"s1": {"document": "sum body", "metadata": {}}}),
        "reflections": FakeCollection({"r1": {"document": "refl body", "metadata": {"timestamp": TS}}}),
    })
    eng = _engine(tmp_path, chroma, cm)
    for store, did in (("chroma:summaries", "s1"), ("chroma:reflections", "r1")):
        pid = _queue(eng, _proposal(store, did))
        eng.apply(pid)
    assert all(e.get(QUARANTINE_KEY) is True for e in cm.corpus)
    assert cm.visible_entries() == []


def test_ambiguous_corpus_is_chroma_only(tmp_path):
    cm, path = _corpus(tmp_path, [dict(CONV), dict(CONV)])
    before = path.read_text()
    chroma = _conv_chroma()
    eng = _engine(tmp_path, chroma, cm)
    pid = _queue(eng, _proposal("chroma:conversations", "d1"))
    eng.apply(pid)
    assert chroma.collections["conversations"].docs["d1"]["metadata"][QUARANTINE_KEY] is True
    assert all(QUARANTINE_KEY not in e for e in cm.corpus)
    assert path.read_text() == before
    assert eng.get(pid).items[0].corpus_outcome == OUTCOME_AMBIGUOUS
    assert '"corpus_outcome": "ambiguous"' in (tmp_path / "audit.jsonl").read_text()
    eng.undo(pid)  # Chroma-only undo works


def test_non_quarantine_metadata_change_does_no_corpus_work(tmp_path):
    cm, path = _corpus(tmp_path, [dict(CONV)])
    before = path.read_text()
    chroma = _conv_chroma({"note": "x"})
    eng = _engine(tmp_path, chroma, cm)
    pid = _queue(eng, _proposal("chroma:conversations", "d1", "set_metadata", {"note": "y"}))
    eng.apply(pid)
    assert chroma.collections["conversations"].docs["d1"]["metadata"]["note"] == "y"
    assert eng.get(pid).items[0].corpus_outcome is None
    assert path.read_text() == before


def test_legacy_journal_item_without_new_fields_reverts(tmp_path):
    cm, _ = _corpus(tmp_path, [dict(CONV)])
    chroma = _conv_chroma({QUARANTINE_KEY: True})
    legacy = ItemChange.model_validate({
        "store": "chroma:conversations", "doc_id": "d1", "change_type": "quarantine",
        "before": {QUARANTINE_KEY: None}, "missing_before": [QUARANTINE_KEY], "after": {}})
    assert legacy.corpus_outcome is None
    from memory.curation.adapters import revert_change
    revert_change(legacy, chroma_store=chroma, corpus_manager=cm)
    assert chroma.collections["conversations"].docs["d1"]["metadata"][QUARANTINE_KEY] is False


def test_corpus_rematch_failure_rolls_back_chroma(tmp_path):
    cm, _ = _corpus(tmp_path, [dict(CONV)])
    chroma = _conv_chroma()
    eng = _engine(tmp_path, chroma, cm)
    pid = _queue(eng, _proposal("chroma:conversations", "d1"))
    real_save = cm.save_corpus
    calls = {"n": 0}
    def boom():
        calls["n"] += 1
        if calls["n"] == 1:
            cm.corpus.append(dict(CONV))  # a duplicate appears mid-apply -> ambiguous on re-check
            raise RuntimeError("disk")
        real_save()
    cm.save_corpus = boom
    with pytest.raises(RuntimeError):
        eng.apply(pid)
    assert chroma.collections["conversations"].docs["d1"]["metadata"][QUARANTINE_KEY] is False
