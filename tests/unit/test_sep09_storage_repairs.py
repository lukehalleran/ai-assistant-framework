"""
Regression tests for the 2026-09-09 audit repair batch B3 — Storage
(docs/PLAN_20260909_audit_repairs.md, docs/HANDOFF_20260909_independent_bug_audit.md
F02/F05/F08).

All tests drive the DEPLOYED functions against a real ephemeral ChromaDB
instance (chromadb.EphemeralClient) — never a re-derivation or a mock of the
storage contract. A tiny deterministic, offline embedding function stands in
for the real SentenceTransformer (no network, no model download); vectors
carry no semantic meaning and are never used for similarity assertions here.

F02 — reference_docs_manager.upload_document/upload_text used to delete the
      existing same-title document BEFORE inserting the replacement; a
      mid-upload failure (chunk/embed/insert) lost the prior version.
F05 — get_ids_by_timestamp_range() passed raw ISO strings to Chroma's
      $gte/$lte, which require numeric operands; every call silently
      returned [] and shutdown summaries never got source_doc_ids.
F08 — MemoryExpander applied hygiene (quarantine/junk/supersession) to
      NEIGHBORS but not the ANCHOR, and cached expansions indefinitely with
      no way to learn about a later chroma mutation (e.g. a curation
      apply_change).
"""

import uuid
from datetime import datetime, timedelta

import chromadb
import pytest
from chromadb.config import Settings

from knowledge.reference_docs_manager import ReferenceDocsManager
from memory.curation.adapters import apply_change
from memory.curation.types import ItemChange
from memory.memory_expander import EXPANSION_CACHE_TTL_S, MemoryExpander
from memory.storage.multi_collection_chroma_store import (
    MultiCollectionChromaStore,
    timestamp_to_epoch,
)


# ---------------------------------------------------------------------------
# Shared test fixtures — real Chroma, deterministic offline embeddings
# ---------------------------------------------------------------------------

class _FixedEmbeddingFunction:
    """Deterministic 2-d embedding function — no network, no model download.
    Vector content is irrelevant to every test in this file (none exercise
    semantic similarity); this exists only so `collection.add()`/`.update()`
    have something to embed with."""

    def __call__(self, input):
        return [[1.0, 0.0] for _ in input]

    def name(self):
        return "fixed-test-embedder"


def _fresh_collection(client, prefix):
    return client.create_collection(
        name=f"{prefix}_{uuid.uuid4().hex[:8]}", embedding_function=_FixedEmbeddingFunction()
    )


def _make_store(*collection_names):
    """A real MultiCollectionChromaStore wired to real ephemeral collections
    for exactly the requested names (the __new__ + manual .collections
    pattern — bypasses __init__ so no real SentenceTransformer is loaded)."""
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {name: _fresh_collection(client, name) for name in collection_names}
    return store


# ===========================================================================
# F02 — staged replacement
# ===========================================================================

class TestReplacementPreservesOriginal:
    """reference_docs_manager.upload_document/upload_text staged replacement."""

    def _make_manager(self):
        store = _make_store("reference_docs")
        coll = store.collections["reference_docs"]
        mgr = ReferenceDocsManager(chroma_store=store)
        return mgr, store, coll

    # -- (a) upload_text failure preserves the original ---------------------
    def test_upload_text_failure_preserves_original(self):
        mgr, store, coll = self._make_manager()
        seed = mgr.upload_text(
            content="Original synthetic worksheet content used for the F02 replacement test.",
            title="Synthetic worksheet",
            metadata_overrides={"type": "user_upload"},
        )
        assert seed.success is True
        original_ids = sorted(c["id"] for c in mgr._get_document_chunks("Synthetic worksheet"))
        assert original_ids

        def _raise(*_a, **_kw):
            raise RuntimeError("synthetic embedding failure")

        orig_method = store.add_batch_to_collection
        store.add_batch_to_collection = _raise
        try:
            result = mgr.upload_text(
                content="Replacement content that must never reach storage on this failing call.",
                title="Synthetic worksheet",
                metadata_overrides={"type": "user_upload"},
            )
        finally:
            store.add_batch_to_collection = orig_method

        assert result.success is False
        assert result.errors

        still_there = coll.get(ids=original_ids, include=["documents"])
        assert sorted(still_there["ids"]) == original_ids
        all_docs = coll.get(include=["documents"])["documents"]
        assert not any("must never reach storage" in (d or "") for d in all_docs)

    # -- (b) upload_document failure preserves the original ------------------
    def test_upload_document_failure_preserves_original(self, tmp_path):
        mgr, store, coll = self._make_manager()
        f = tmp_path / "worksheet.txt"
        f.write_text("Original file-based synthetic worksheet content for the F02 test.")
        seed = mgr.upload_document(str(f), title="Synthetic worksheet")
        assert seed.success is True
        original_ids = sorted(c["id"] for c in mgr._get_document_chunks("Synthetic worksheet"))
        assert original_ids

        f.write_text("Replacement file content that must never reach storage on this failing call.")

        def _raise(*_a, **_kw):
            raise RuntimeError("synthetic embedding failure")

        orig_method = store.add_batch_to_collection
        store.add_batch_to_collection = _raise
        try:
            result = mgr.upload_document(str(f), title="Synthetic worksheet")
        finally:
            store.add_batch_to_collection = orig_method

        assert result.success is False
        still_there = coll.get(ids=original_ids, include=["documents"])
        assert sorted(still_there["ids"]) == original_ids
        all_docs = coll.get(include=["documents"])["documents"]
        assert not any("must never reach storage" in (d or "") for d in all_docs)

    # -- (c) success path replaces: old gone, new present, one batch --------
    def test_success_replaces_old_version(self):
        mgr, store, coll = self._make_manager()
        seed = mgr.upload_text(
            content="Version one of the synthetic worksheet, long enough to chunk cleanly.",
            title="Synthetic worksheet",
            metadata_overrides={"type": "user_upload"},
        )
        assert seed.success is True
        old_ids = sorted(c["id"] for c in mgr._get_document_chunks("Synthetic worksheet"))

        result = mgr.upload_text(
            content="Version two of the synthetic worksheet, also long enough to chunk cleanly.",
            title="Synthetic worksheet",
            metadata_overrides={"type": "user_upload"},
        )
        assert result.success is True

        remaining = mgr._get_document_chunks("Synthetic worksheet")
        remaining_ids = {c["id"] for c in remaining}
        assert remaining_ids.isdisjoint(old_ids)
        batches = {c["metadata"].get("upload_batch") for c in remaining}
        assert len(batches) == 1
        assert all("Version two" in c["content"] for c in remaining)

    # -- (d) same-title, different type is never cross-deleted ---------------
    def test_replacement_preserves_different_type_same_title(self):
        mgr, store, coll = self._make_manager()
        # Seed a self-seeded reference_doc under the title first.
        doc_result = mgr.upload_text(
            content="Self-seeded documentation content sharing a title with a user upload.",
            title="Shared Title",
            metadata_overrides={"type": "reference_doc"},
        )
        assert doc_result.success is True
        doc_ids = {c["id"] for c in mgr._get_document_chunks("Shared Title")
                   if c["metadata"].get("type") == "reference_doc"}
        assert doc_ids

        # A user_upload with the SAME title must not delete the doc above,
        # and replacing the user_upload again must not touch it either.
        up1 = mgr.upload_text(
            content="First version of the user-uploaded content under the shared title.",
            title="Shared Title",
            metadata_overrides={"type": "user_upload"},
        )
        assert up1.success is True
        up2 = mgr.upload_text(
            content="Second version of the user-uploaded content under the shared title.",
            title="Shared Title",
            metadata_overrides={"type": "user_upload"},
        )
        assert up2.success is True

        all_chunks = mgr._get_document_chunks("Shared Title")
        by_type = {}
        for c in all_chunks:
            by_type.setdefault(c["metadata"].get("type"), []).append(c)
        assert doc_ids <= {c["id"] for c in by_type.get("reference_doc", [])}
        assert len(by_type.get("user_upload", [])) >= 1
        assert all("Second version" in c["content"] for c in by_type.get("user_upload", []))

    # -- (e) retry after failure with changed content succeeds, one version --
    def test_retry_after_failure_with_changed_content_succeeds(self):
        mgr, store, coll = self._make_manager()
        seed = mgr.upload_text(
            content="Original content before the failing retry attempt for this worksheet.",
            title="Synthetic worksheet",
            metadata_overrides={"type": "user_upload"},
        )
        assert seed.success is True

        def _raise(*_a, **_kw):
            raise RuntimeError("synthetic embedding failure")

        orig_method = store.add_batch_to_collection
        store.add_batch_to_collection = _raise
        try:
            failed = mgr.upload_text(
                content="First replacement attempt content, which must fail.",
                title="Synthetic worksheet",
                metadata_overrides={"type": "user_upload"},
            )
        finally:
            store.add_batch_to_collection = orig_method
        assert failed.success is False

        retried = mgr.upload_text(
            content="Second replacement attempt content, with different text, which must succeed.",
            title="Synthetic worksheet",
            metadata_overrides={"type": "user_upload"},
        )
        assert retried.success is True

        remaining = mgr._get_document_chunks("Synthetic worksheet")
        batches = {c["metadata"].get("upload_batch") for c in remaining}
        assert len(batches) == 1
        assert all("Second replacement attempt" in c["content"] for c in remaining)


# ===========================================================================
# F05 — timestamp_epoch + get_ids_by_timestamp_range
# ===========================================================================

class TestTimestampToEpochHelper:
    """timestamp_to_epoch() direct unit coverage."""

    def test_naive_timestamp_treated_as_local(self):
        naive = "2026-09-09T12:00:00"
        epoch = timestamp_to_epoch(naive)
        assert epoch is not None
        # datetime.fromtimestamp() on a naive epoch round-trips through LOCAL
        # time — matching the documented convention.
        assert datetime.fromtimestamp(epoch) == datetime.fromisoformat(naive)

    def test_offset_aware_timestamp_respects_its_offset(self):
        aware = "2026-09-09T12:00:00-05:00"
        epoch = timestamp_to_epoch(aware)
        assert epoch == pytest.approx(datetime.fromisoformat(aware).timestamp())

    def test_date_only_string_parses(self):
        assert timestamp_to_epoch("2026-09-09") is not None

    def test_malformed_inputs_return_none(self):
        assert timestamp_to_epoch("not-a-date") is None
        assert timestamp_to_epoch("") is None
        assert timestamp_to_epoch(None) is None
        assert timestamp_to_epoch(12345) is None


class TestTimestampRange:
    """get_ids_by_timestamp_range() through the real store + real Chroma."""

    def test_inclusive_boundaries_and_adjacent_row_excluded(self):
        store = _make_store_convo()
        base = datetime(2026, 9, 9, 12, 0, 0)
        ids = []
        for minutes in (0, 30, 60, 61):
            ts = (base + timedelta(minutes=minutes)).isoformat()
            doc_id = store.add_to_collection(
                "conversations", f"User: q{minutes}\nAssistant: a{minutes}", {"timestamp": ts}
            )
            ids.append(doc_id)

        result_ids = store.get_ids_by_timestamp_range(
            "conversations", base.isoformat(), (base + timedelta(minutes=60)).isoformat()
        )
        assert set(result_ids) == set(ids[:3])
        assert ids[3] not in result_ids

    def test_mixed_legacy_and_new_rows_both_returned(self):
        store = _make_store_convo()
        coll = store.collections["conversations"]
        base = datetime(2026, 9, 9, 12, 0, 0)

        new_id = store.add_to_collection(
            "conversations", "User: a\nAssistant: b", {"timestamp": base.isoformat()}
        )
        # Simulate a pre-fix row: no timestamp_epoch, inserted directly on
        # the collection (bypassing the store's _derive_epoch()).
        legacy_id = str(uuid.uuid4())
        coll.add(
            ids=[legacy_id],
            documents=["User: c\nAssistant: d"],
            metadatas=[{"timestamp": (base + timedelta(minutes=10)).isoformat()}],
        )

        result_ids = store.get_ids_by_timestamp_range(
            "conversations", base.isoformat(), (base + timedelta(minutes=20)).isoformat()
        )
        assert new_id in result_ids
        assert legacy_id in result_ids

    def test_offset_aware_bound_lands_correctly(self):
        store = _make_store_convo()
        doc_id = store.add_to_collection(
            "conversations", "User: x\nAssistant: y", {"timestamp": "2026-09-09T12:00:00-05:00"}
        )
        result_ids = store.get_ids_by_timestamp_range(
            "conversations", "2026-09-09T16:55:00+00:00", "2026-09-09T17:05:00+00:00"
        )
        assert doc_id in result_ids

    def test_malformed_row_is_skipped_and_counted(self):
        store = _make_store_convo()
        coll = store.collections["conversations"]
        base = datetime(2026, 9, 9, 12, 0, 0)
        good_id = store.add_to_collection(
            "conversations", "User: g\nAssistant: h", {"timestamp": base.isoformat()}
        )
        bad_id = str(uuid.uuid4())
        coll.add(
            ids=[bad_id],
            documents=["User: bad\nAssistant: bad"],
            metadatas=[{"timestamp": "not-a-real-date"}],
        )
        result_ids = store.get_ids_by_timestamp_range(
            "conversations", base.isoformat(), (base + timedelta(minutes=5)).isoformat()
        )
        assert good_id in result_ids
        assert bad_id not in result_ids

    def test_malformed_bound_returns_empty(self):
        store = _make_store_convo()
        assert store.get_ids_by_timestamp_range("conversations", "garbage", "also garbage") == []

    def test_legacy_scan_pages_never_a_single_unbounded_call(self):
        """Regression guard for the "never the whole collection in one call"
        requirement: seed more rows than the page size and assert the
        underlying collection.get() is always called with a bounded limit
        during the legacy scan."""
        store = _make_store_convo()
        coll = store.collections["conversations"]
        page_size = store._TIMESTAMP_RANGE_PAGE_SIZE
        base = datetime(2026, 9, 9, 0, 0, 0)
        n_rows = page_size + 5
        ids = [str(uuid.uuid4()) for _ in range(n_rows)]
        texts = [f"User: r{i}\nAssistant: a{i}" for i in range(n_rows)]
        metas = [{"timestamp": (base + timedelta(minutes=i)).isoformat()} for i in range(n_rows)]
        coll.add(ids=ids, documents=texts, metadatas=metas)

        orig_get = coll.get
        seen_limits = []

        def _spy_get(*args, **kwargs):
            if "limit" in kwargs:
                seen_limits.append(kwargs["limit"])
            return orig_get(*args, **kwargs)

        coll.get = _spy_get
        try:
            result_ids = store.get_ids_by_timestamp_range(
                "conversations", base.isoformat(), (base + timedelta(minutes=n_rows)).isoformat()
            )
        finally:
            coll.get = orig_get

        assert len(result_ids) == n_rows
        assert seen_limits, "legacy scan must page via limit="
        assert all(limit <= page_size for limit in seen_limits)
        assert len(seen_limits) >= 2  # more than one page was needed


def _make_store_convo():
    return _make_store("conversations")


class TestShutdownSummarySourceDocIds:
    """Exercises the deployed ShutdownProcessor._store_summary() — the
    smallest deployed method that owns the get_ids_by_timestamp_range() call
    site named in the audit (shutdown_processor.py, near the historical
    L488) — end to end, asserting the persisted summary carries
    source_doc_ids."""

    class _StubCorpus:
        def __init__(self):
            self.added = []

        def get_summaries(self, limit):
            return []

        def add_summary(self, entry):
            self.added.append(entry)

    def _make_processor(self, store):
        from memory.shutdown_processor import ShutdownProcessor

        return ShutdownProcessor(
            corpus_manager=self._StubCorpus(),
            chroma_store=store,
            consolidator=None,
            fact_extractor=None,
            model_manager=None,
            user_profile=None,
            storage=None,
            session_start=datetime(2026, 9, 9, 0, 0, 0),
            claim_index=None,
        )

    def test_store_summary_populates_source_doc_ids(self):
        store = _make_store("conversations", "summaries")
        base = datetime(2026, 9, 9, 12, 0, 0)
        convo_ids = []
        for minutes in (0, 10, 20):
            ts = (base + timedelta(minutes=minutes)).isoformat()
            convo_ids.append(
                store.add_to_collection(
                    "conversations", f"User: q{minutes}\nAssistant: a{minutes}", {"timestamp": ts}
                )
            )

        sp = self._make_processor(store)
        block = [{"timestamp": (base + timedelta(minutes=m)).isoformat()} for m in (0, 10, 20)]
        sp._store_summary(
            "A consolidated summary of the early conversation block used for this F05 regression test.",
            N=1,
            b=0,
            start=0,
            end=3,
            block=block,
        )

        summaries = store.collections["summaries"].get(include=["metadatas"])
        metas = summaries.get("metadatas") or []
        assert metas, "summary was not stored"
        source_doc_ids = (metas[0] or {}).get("source_doc_ids", "")
        assert source_doc_ids
        stored_ids = set(source_doc_ids.split(","))
        assert stored_ids == set(convo_ids)


# ===========================================================================
# F08 — expander anchor hygiene + fingerprinted/TTL cache
# ===========================================================================

class TestExpansionHygieneAndCache:
    def _make_store_and_expander(self):
        store = _make_store("conversations", "facts", "summaries", "reference_docs")
        expander = MemoryExpander(store)
        return store, expander

    def test_quarantined_anchor_is_suppressed(self):
        store, expander = self._make_store_and_expander()
        doc_id = store.add_to_collection(
            "conversations",
            "User: hello there\nAssistant: hi back",
            {"timestamp": datetime.now().isoformat(), "curation_quarantined": True},
        )
        result = expander.expand(doc_id, collection="conversations")
        assert result["turns"] == []
        assert result["error"] is not None
        assert "quarantined" in result["error"].lower()

    def test_superseded_fact_anchor_is_suppressed(self):
        store, expander = self._make_store_and_expander()
        doc_id = store.add_to_collection(
            "facts",
            "user likes decaf coffee in the mornings on weekdays",
            {"timestamp": datetime.now().isoformat(), "is_current": False},
        )
        result = expander.expand(doc_id, collection="facts")
        assert result["turns"] == []
        assert result["error"] is not None
        assert "superseded" in result["error"].lower()

    def test_expand_then_edit_anchor_no_manual_clear_cache_needed(self):
        store, expander = self._make_store_and_expander()
        ts = datetime.now().isoformat()
        doc_id = store.add_to_collection(
            "conversations", "User: original wording is here\nAssistant: ok", {"timestamp": ts}
        )
        r1 = expander.expand(doc_id, collection="conversations")
        assert any("original wording" in t["content"] for t in r1["turns"])

        coll = store.collections["conversations"]
        coll.update(ids=[doc_id], documents=["User: CHANGED wording now\nAssistant: ok"])

        r2 = expander.expand(doc_id, collection="conversations")  # no clear_cache() call
        assert any("CHANGED wording" in t["content"] for t in r2["turns"])
        assert r2 is not r1

    def test_expand_then_quarantine_anchor_via_curation_adapter_is_suppressed(self):
        store, expander = self._make_store_and_expander()
        ts = datetime.now().isoformat()
        doc_id = store.add_to_collection(
            "conversations", "User: quarantine me later please\nAssistant: ok", {"timestamp": ts}
        )
        r1 = expander.expand(doc_id, collection="conversations")
        assert r1["error"] is None

        change = ItemChange(store="chroma:conversations", doc_id=doc_id, change_type="quarantine", after={})
        apply_change(change, chroma_store=store)

        r2 = expander.expand(doc_id, collection="conversations")  # no manual clear_cache()
        assert r2["turns"] == []
        assert "quarantined" in (r2["error"] or "").lower()

    def test_curation_mutation_on_a_neighbor_invalidates_the_whole_cache(self):
        """The per-call anchor fingerprint only re-validates the ANCHOR; a
        curation mutation to a NEIGHBOR document inside an already-cached
        window is caught only via the adapter's notify_chroma_mutation()
        hook clearing the whole cache — this test fails if that wiring is
        removed even though the two anchor-only tests above would still
        pass."""
        store, expander = self._make_store_and_expander()
        base = datetime.now()
        anchor_id = store.add_to_collection(
            "conversations", "User: anchor turn text\nAssistant: ok",
            {"timestamp": base.isoformat()},
        )
        neighbor_id = store.add_to_collection(
            "conversations", "User: neighbor turn text\nAssistant: ok",
            {"timestamp": (base + timedelta(minutes=1)).isoformat()},
        )

        r1 = expander.expand(anchor_id, window=2, collection="conversations")
        assert any(t["id"] == neighbor_id for t in r1["turns"])

        change = ItemChange(store="chroma:conversations", doc_id=neighbor_id, change_type="quarantine", after={})
        apply_change(change, chroma_store=store)

        r2 = expander.expand(anchor_id, window=2, collection="conversations")
        assert r2 is not r1
        assert not any(t["id"] == neighbor_id for t in r2["turns"])

    def test_ttl_expiry_forces_recompute(self, monkeypatch):
        store, expander = self._make_store_and_expander()
        ts = datetime.now().isoformat()
        doc_id = store.add_to_collection(
            "conversations", "User: ttl bound test content\nAssistant: ok", {"timestamp": ts}
        )
        r1 = expander.expand(doc_id, collection="conversations")

        import memory.memory_expander as me_mod

        real_time = me_mod.time.time()
        monkeypatch.setattr(me_mod.time, "time", lambda: real_time + EXPANSION_CACHE_TTL_S + 1)

        r2 = expander.expand(doc_id, collection="conversations")
        assert r2 is not r1
