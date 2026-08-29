"""Curation engine + Wave-1 curators (docs/AUTONOMOUS_CURATION_DESIGN.md).

Covers: disposition ceiling (auto locked to queue), sentinel abort, anomaly
halt, rate cap, apply/undo with pre-images, all-or-nothing rollback,
dismissal journaling, queue persistence, the four Wave-1 curators against
fake stores (including a reproduction of the live drop-deadline fact), and
the quarantine retrieval helper.
"""

import json
from datetime import datetime, timedelta

import pytest

from memory.curation.adapters import QUARANTINE_KEY
from memory.curation.curators import (
    ErrorSentinelCurator,
    JunkFactCurator,
    StreamArtifactCurator,
    TemporalStalenessCurator,
)
from memory.curation.curators.temporal_staleness import (
    extract_explicit_dates,
    fact_expired,
)
from memory.curation.engine import CurationEngine, StoreBundle, new_proposal_id
from memory.curation.journal import CurationJournal
from memory.curation.types import (
    Confidence,
    CurationProposal,
    CuratorMode,
    Instrument,
    ItemChange,
    ProposalStatus,
    SentinelResult,
)
from memory.utils import is_quarantined


# ===========================================================================
# Fakes
# ===========================================================================

class FakeCollection:
    def __init__(self, docs=None):
        # id -> {"document": str, "metadata": dict}
        self.docs = dict(docs or {})

    def get(self, ids=None, limit=None, offset=0, include=None):
        if ids is not None:
            found = [(i, self.docs[i]) for i in ids if i in self.docs]
        else:
            items = list(self.docs.items())[offset:offset + (limit or len(self.docs))]
            found = items
        return {
            "ids": [i for i, _ in found],
            "documents": [d["document"] for _, d in found],
            "metadatas": [dict(d["metadata"]) for _, d in found],
        }

    def update(self, ids, documents=None, metadatas=None):
        for idx, doc_id in enumerate(ids):
            if documents is not None:
                self.docs[doc_id]["document"] = documents[idx]
            if metadatas is not None:
                self.docs[doc_id]["metadata"] = dict(metadatas[idx])

    def count(self):
        return len(self.docs)


class FakeChromaStore:
    def __init__(self, collections):
        self.collections = collections

    def _get_collection(self, name):
        return self.collections.get(name)


class FakeProfile:
    def __init__(self, facts):
        self.profile = {"categories": {"career": list(facts)}}
        self.saves = 0

    def save(self):
        self.saves += 1


def make_engine(tmp_path, stores=None, **kw):
    return CurationEngine(
        stores or StoreBundle(),
        queue_path=str(tmp_path / "queue.json"),
        journal=CurationJournal(str(tmp_path / "audit.jsonl")),
        **kw,
    )


def simple_proposal(store="chroma:conversations", doc_id="d1",
                    instrument=Instrument.METADATA,
                    confidence=Confidence.DETERMINISTIC, batch=False):
    return CurationProposal(
        proposal_id=new_proposal_id(), curator="test", instrument=instrument,
        confidence=confidence, title="t", evidence="e", batch=batch,
        items=[ItemChange(store=store, doc_id=doc_id, change_type="quarantine",
                          after={})],
    )


class StubCurator:
    name = "test"

    def __init__(self, proposals=None, sentinel_pass=True):
        self._proposals = proposals or []
        self._sentinel_pass = sentinel_pass

    def sentinels(self, stores):
        return [SentinelResult(name="stub", passed=self._sentinel_pass)]

    def scan(self, stores):
        return list(self._proposals)


# ===========================================================================
# 1. Engine disposition
# ===========================================================================

class TestDisposition:

    def test_max_mode_ceiling_caps_auto(self, tmp_path):
        eng = make_engine(tmp_path, max_mode="queue",
                          curator_modes={"test": "auto"})
        assert eng.mode_for("test") == CuratorMode.QUEUE

    def test_queue_mode_persists_proposal(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {"d1": {"document": "x", "metadata": {}}})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store))
        eng.register(StubCurator([simple_proposal()]))
        report = eng.run_scan()
        assert report.proposals_queued == 1
        assert len(eng.pending()) == 1
        # Nothing was applied — queue ceiling.
        assert store.collections["conversations"].docs["d1"]["metadata"] == {}

    def test_shadow_mode_journals_only(self, tmp_path):
        eng = make_engine(tmp_path, curator_modes={"test": "shadow"})
        eng.register(StubCurator([simple_proposal()]))
        report = eng.run_scan()
        assert report.proposals_shadowed == 1
        assert eng.pending() == []

    def test_auto_mode_applies_deterministic_metadata(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {f"d{i}": {"document": "x", "metadata": {}} for i in range(100)})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store),
                          max_mode="auto", curator_modes={"test": "auto"})
        eng.register(StubCurator([simple_proposal(doc_id="d1")]))
        eng.run_scan()
        assert store.collections["conversations"].docs["d1"]["metadata"][QUARANTINE_KEY]

    def test_auto_never_applies_delete(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {f"d{i}": {"document": "x", "metadata": {}} for i in range(100)})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store),
                          max_mode="auto", curator_modes={"test": "auto"})
        eng.register(StubCurator([simple_proposal(
            doc_id="d1", instrument=Instrument.DELETE)]))
        eng.run_scan()
        # Queued, untouched.
        assert len(eng.pending()) == 1
        assert store.collections["conversations"].docs["d1"]["metadata"] == {}

    def test_auto_never_applies_single_llm(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {f"d{i}": {"document": "x", "metadata": {}} for i in range(100)})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store),
                          max_mode="auto", curator_modes={"test": "auto"})
        eng.register(StubCurator([simple_proposal(
            doc_id="d1", confidence=Confidence.SINGLE_LLM)]))
        eng.run_scan()
        assert store.collections["conversations"].docs["d1"]["metadata"] == {}

    def test_sentinel_failure_aborts_batch(self, tmp_path):
        eng = make_engine(tmp_path)
        eng.register(StubCurator([simple_proposal()], sentinel_pass=False))
        report = eng.run_scan()
        assert eng.pending() == []
        assert "test" in report.halted_curators

    def test_anomaly_halt_blocks_auto(self, tmp_path):
        # Proposal touches 10/100 docs = 10% > 5% anomaly fraction.
        store = FakeChromaStore({"conversations": FakeCollection(
            {f"d{i}": {"document": "x", "metadata": {}} for i in range(100)})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store),
                          max_mode="auto", curator_modes={"test": "auto"},
                          anomaly_fraction=0.05)
        p = CurationProposal(
            proposal_id=new_proposal_id(), curator="test",
            instrument=Instrument.METADATA, confidence=Confidence.DETERMINISTIC,
            title="t", evidence="e", batch=True,
            items=[ItemChange(store="chroma:conversations", doc_id=f"d{i}",
                              change_type="quarantine") for i in range(10)],
        )
        eng.register(StubCurator([p]))
        eng.run_scan()
        assert all(store.collections["conversations"].docs[f"d{i}"]["metadata"] == {}
                   for i in range(10))
        assert len(eng.pending()) == 1

    def test_rate_cap_limits_auto_applies(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {f"d{i}": {"document": "x", "metadata": {}} for i in range(200)})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store),
                          max_mode="auto", curator_modes={"test": "auto"},
                          auto_rate_cap=3)
        eng.register(StubCurator([simple_proposal(doc_id=f"d{i}")
                                  for i in range(6)]))
        eng.run_scan()
        applied = sum(1 for d in store.collections["conversations"].docs.values()
                      if d["metadata"].get(QUARANTINE_KEY))
        assert applied == 3
        assert len(eng.pending()) == 3

    def test_duplicate_proposals_not_requeued(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {"d1": {"document": "x", "metadata": {}}})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store))
        eng.register(StubCurator([simple_proposal(doc_id="d1")]))
        eng.run_scan()
        eng.register(StubCurator([simple_proposal(doc_id="d1")]))
        eng.run_scan()
        assert len(eng.pending()) == 1

    def test_queue_card_cap(self, tmp_path):
        eng = make_engine(tmp_path, max_queue_items_per_curator=5)
        eng.register(StubCurator([simple_proposal(doc_id=f"d{i}")
                                  for i in range(9)]))
        eng.run_scan()
        assert len(eng.pending()) == 5


# ===========================================================================
# 2. Apply / undo / rollback / persistence
# ===========================================================================

class TestApplyUndo:

    def _engine_with_doc(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {"d1": {"document": "hello", "metadata": {"topic": "x"}}})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store))
        return eng, store

    def test_apply_captures_preimage_and_undo_restores(self, tmp_path):
        eng, store = self._engine_with_doc(tmp_path)
        p = simple_proposal(doc_id="d1")
        eng.register(StubCurator([p]))
        eng.run_scan()
        eng.apply(p.proposal_id)
        meta = store.collections["conversations"].docs["d1"]["metadata"]
        assert meta[QUARANTINE_KEY] is True
        assert meta["topic"] == "x"  # untouched keys preserved
        eng.undo(p.proposal_id)
        meta = store.collections["conversations"].docs["d1"]["metadata"]
        assert QUARANTINE_KEY not in meta
        assert eng.get(p.proposal_id).status == ProposalStatus.UNDONE

    def test_content_repair_undo_restores_document(self, tmp_path):
        store = FakeChromaStore({"summaries": FakeCollection(
            {"s1": {"document": "<|sep|>Summary text.", "metadata": {}}})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store))
        p = CurationProposal(
            proposal_id=new_proposal_id(), curator="test",
            instrument=Instrument.METADATA, confidence=Confidence.DETERMINISTIC,
            title="t", evidence="e",
            items=[ItemChange(store="chroma:summaries", doc_id="s1",
                              change_type="replace_content",
                              after={"document": "Summary text."})],
        )
        eng.register(StubCurator([p]))
        eng.run_scan()
        eng.apply(p.proposal_id)
        assert store.collections["summaries"].docs["s1"]["document"] == "Summary text."
        eng.undo(p.proposal_id)
        assert store.collections["summaries"].docs["s1"]["document"] == "<|sep|>Summary text."

    def test_apply_is_all_or_nothing(self, tmp_path):
        store = FakeChromaStore({"conversations": FakeCollection(
            {"d1": {"document": "x", "metadata": {}}})})
        eng = make_engine(tmp_path, StoreBundle(chroma_store=store))
        p = CurationProposal(
            proposal_id=new_proposal_id(), curator="test",
            instrument=Instrument.METADATA, confidence=Confidence.DETERMINISTIC,
            title="t", evidence="e", batch=True,
            items=[
                ItemChange(store="chroma:conversations", doc_id="d1",
                           change_type="quarantine"),
                ItemChange(store="chroma:conversations", doc_id="MISSING",
                           change_type="quarantine"),
            ],
        )
        eng.register(StubCurator([p]))
        eng.run_scan()
        with pytest.raises(Exception):
            eng.apply(p.proposal_id)
        # First item rolled back.
        assert store.collections["conversations"].docs["d1"]["metadata"] == {}
        assert eng.get(p.proposal_id).status == ProposalStatus.FAILED

    def test_profile_supersede_and_undo(self, tmp_path):
        profile = FakeProfile([{
            "fact_id": "f1", "relation": "deadline",
            "value": "drop deadline is Fri 2026-08-28 at 3 PM Central",
            "is_current": True,
        }])
        eng = make_engine(tmp_path, StoreBundle(user_profile=profile))
        p = CurationProposal(
            proposal_id=new_proposal_id(), curator="test",
            instrument=Instrument.METADATA, confidence=Confidence.DETERMINISTIC,
            title="t", evidence="e",
            items=[ItemChange(store="profile", doc_id="f1",
                              change_type="supersede_profile_fact",
                              after={"reason": "date_passed:2026-08-28"})],
        )
        eng.register(StubCurator([p]))
        eng.run_scan()
        eng.apply(p.proposal_id)
        fact = profile.profile["categories"]["career"][0]
        assert fact["is_current"] is False
        assert fact["curation_stale_reason"] == "date_passed:2026-08-28"
        assert profile.saves >= 1
        eng.undo(p.proposal_id)
        assert fact["is_current"] is True
        assert "curation_stale_reason" not in fact

    def test_queue_persistence_roundtrip(self, tmp_path):
        eng, _store = self._engine_with_doc(tmp_path)
        p = simple_proposal(doc_id="d1")
        eng.register(StubCurator([p]))
        eng.run_scan()
        eng2 = make_engine(tmp_path)
        assert [x.proposal_id for x in eng2.pending()] == [p.proposal_id]

    def test_corrupt_queue_cold_starts(self, tmp_path):
        (tmp_path / "queue.json").write_text("{not json")
        eng = make_engine(tmp_path)
        assert eng.pending() == []

    def test_dismiss_journals_confidence(self, tmp_path):
        eng, _ = self._engine_with_doc(tmp_path)
        p = simple_proposal(doc_id="d1")
        eng.register(StubCurator([p]))
        eng.run_scan()
        eng.dismiss(p.proposal_id, reason="wrong")
        events = eng.journal.tail()
        dism = [e for e in events if e["event"] == "dismissed"]
        assert dism and dism[0]["confidence"] == "deterministic"


# ===========================================================================
# 3. Wave-1 curators
# ===========================================================================

class TestErrorSentinelCurator:

    def test_flags_junk_and_skips_quarantined(self):
        store = FakeChromaStore({
            "conversations": FakeCollection({
                "junk": {"document": "User: hi\nAssistant: [API unavailable] err",
                         "metadata": {}},
                "already": {"document": "User: hi\nAssistant: [API Error] x",
                            "metadata": {QUARANTINE_KEY: True}},
                "clean": {"document": "User: how was the gym\nAssistant: Solid "
                                      "session — you hit the squat PR.",
                          "metadata": {}},
            }),
            "summaries": FakeCollection({
                "jsum": {"document": "[API Error] 402 credits", "metadata": {}},
            }),
        })
        cur = ErrorSentinelCurator()
        assert all(s.passed for s in cur.sentinels(StoreBundle()))
        props = cur.scan(StoreBundle(chroma_store=store))
        assert len(props) == 1 and props[0].batch
        ids = {i.doc_id for i in props[0].items}
        assert ids == {"junk", "jsum"}

    def test_no_junk_no_proposal(self):
        store = FakeChromaStore({
            "conversations": FakeCollection({
                "clean": {"document": "User: hey\nAssistant: Morning — ready "
                                      "for the exam review?", "metadata": {}}}),
            "summaries": FakeCollection({}),
        })
        assert ErrorSentinelCurator().scan(StoreBundle(chroma_store=store)) == []


class TestStreamArtifactCurator:

    def test_repairs_sep_prefix(self):
        store = FakeChromaStore({
            "conversations": FakeCollection({
                "a": {"document": "<|sep|>That's a good question about R.",
                      "metadata": {}},
                "clean": {"document": "Nothing wrong here.", "metadata": {}},
            }),
            "summaries": FakeCollection({}),
        })
        cur = StreamArtifactCurator()
        assert all(s.passed for s in cur.sentinels(StoreBundle()))
        props = cur.scan(StoreBundle(chroma_store=store))
        assert len(props) == 1
        item = props[0].items[0]
        assert item.doc_id == "a"
        assert item.after["document"] == "That's a good question about R."


class TestJunkFactCurator:

    def test_flags_fragment_object(self):
        store = FakeChromaStore({"facts": FakeCollection({
            "f_junk": {"document": "user | dad_show_up | for a bit", "metadata": {}},
            "f_ok": {"document": "user | likes | pizza", "metadata": {}},
            "f_unparseable": {"document": "free text no pipes", "metadata": {}},
        })})
        cur = JunkFactCurator()
        assert all(s.passed for s in cur.sentinels(StoreBundle()))
        props = cur.scan(StoreBundle(chroma_store=store))
        assert len(props) == 1
        assert {i.doc_id for i in props[0].items} == {"f_junk"}


class TestTemporalStalenessCurator:

    def test_extract_explicit_dates(self):
        assert extract_explicit_dates("due 2026-08-28 3 PM") == [datetime(2026, 8, 28)]
        assert extract_explicit_dates("closes Sat Oct 31, 2026") == [datetime(2026, 10, 31)]
        assert extract_explicit_dates("by 31 Aug 2026 latest") == [datetime(2026, 8, 31)]
        assert extract_explicit_dates("due Aug 31") == []          # yearless
        assert extract_explicit_dates("meeting sometime") == []

    def test_fact_expired_semantics(self):
        now = datetime(2026, 9, 15)
        assert fact_expired("deadline", "drop deadline Fri 2026-08-28",
                            now=now) == datetime(2026, 8, 28)
        # Future date → not expired
        assert fact_expired("deadline", "W closes Oct 31, 2026", now=now) is None
        # Grace window: deadline day itself + grace never expires
        assert fact_expired("deadline", "due 2026-09-14",
                            now=datetime(2026, 9, 15)) is None
        # Multi-date value: latest date governs
        assert fact_expired("deadline", "opens 2026-08-01, closes 2026-12-01",
                            now=now) is None
        # Non-event relation never touched, whatever the value
        assert fact_expired("birthday", "born 1993-02-09", now=now) is None

    def test_live_drop_deadline_fact_reproduction(self, tmp_path):
        """The stale fact from the 2026-08-28 session, verbatim."""
        profile = FakeProfile([
            {"fact_id": "f1", "relation": "deadline",
             "value": "drop deadline is Fri 2026-08-28 at 3 PM Central",
             "is_current": True},
            {"fact_id": "f2", "relation": "deadline",
             "value": "self-service W deadline Sat Oct 31, 2026 11:59 PM ET",
             "is_current": True},
            {"fact_id": "f3", "relation": "program",
             "value": "OMSA at Georgia Tech since 2026-01-05",
             "is_current": True},
        ])
        cur = TemporalStalenessCurator()
        assert all(s.passed for s in cur.sentinels(StoreBundle()))
        props = [p for p in cur.scan(StoreBundle(user_profile=profile))
                 # scan uses real now — guard the test against far-future runs
                 if p.items[0].doc_id == "f1"]
        if datetime.now() > datetime(2026, 8, 31):
            assert len(props) == 1
            assert "2026-08-28" in props[0].evidence
        # f2 must never appear before Nov 2026; f3 (non-event relation) never.
        all_props = cur.scan(StoreBundle(user_profile=profile))
        ids = {p.items[0].doc_id for p in all_props}
        assert "f3" not in ids
        if datetime.now() < datetime(2026, 11, 1):
            assert "f2" not in ids

    def test_superseded_and_idless_facts_skipped(self):
        profile = FakeProfile([
            {"fact_id": "f1", "relation": "deadline",
             "value": "due 2020-01-01", "is_current": False},
            {"relation": "deadline", "value": "due 2020-01-01",
             "is_current": True},  # no fact_id → unaddressable, leave alone
        ])
        assert TemporalStalenessCurator().scan(
            StoreBundle(user_profile=profile)) == []


# ===========================================================================
# 4. Quarantine retrieval helper
# ===========================================================================

class TestQuarantineHelper:

    def test_is_quarantined(self):
        assert is_quarantined({QUARANTINE_KEY: True})
        assert not is_quarantined({QUARANTINE_KEY: False})
        assert not is_quarantined({})
        assert not is_quarantined(None)
