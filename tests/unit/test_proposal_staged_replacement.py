"""Tests for the CM-07 staged replacement in memory/proposal_store.py.

`update_status` must write the new version of a proposal BEFORE deleting the
old document, and delete the old document only after a confirmed write, so a
failed re-store never loses the existing record (failure_outcome_design.md,
"[verified] Delete-then-store loses the record"). `update_tracking_metadata`
must return the real bool from the store's `update_metadata()` merge instead
of always reporting success once a matching item was found.

A local, in-memory fake store/collection is used throughout (no real
Chroma), following tests/unit/test_thread_staged_replacement.py's shape as a
local copy (not imported) per F11b-2's brief. The fake can be told to raise
on the next `add_to_collection()` call and/or on a `delete()` call naming a
specific document id, so each staged-replacement branch (write fails, delete
fails and rolls back, success, missing collection handle) can be exercised
directly, and it exposes a settable `update_metadata()` result for the
tracking-metadata sibling.
"""
import logging
from types import SimpleNamespace

from memory.code_proposal import CodeProposal, ProposalStatus, ProposalType
from memory.proposal_store import ProposalStore, COLLECTION_NAME

MARKER = "SYNTHETIC_test_proposal_staged_replacement_marker"


# ---------------------------------------------------------------------------
# Fake ChromaDB infrastructure (records adds/deletes; raises on demand)
# ---------------------------------------------------------------------------

class FakeCollection:
    """Fake ChromaDB collection: count()/delete(), with injectable delete failures."""

    def __init__(self, items):
        self._items = items  # shared list reference with FakeChromaStore
        self._fail_delete_ids = set()
        self.delete_calls = []

    def count(self):
        return len(self._items)

    def fail_delete(self, doc_id):
        """The next delete() call naming doc_id raises instead of deleting."""
        self._fail_delete_ids.add(doc_id)

    def delete(self, ids=None):
        ids = list(ids or [])
        self.delete_calls.append(ids)
        if any(i in self._fail_delete_ids for i in ids):
            raise RuntimeError(f"{MARKER}_delete")
        matched = set(ids)
        self._items[:] = [item for item in self._items if item.get("id") not in matched]


class FakeChromaStore:
    """Fake of MultiCollectionChromaStore's subset used by ProposalStore."""

    def __init__(self, items=None):
        self._items = items if items is not None else []
        self._id_counter = 0
        self.collection = FakeCollection(self._items)
        self.collections = {COLLECTION_NAME: self.collection}
        self._add_outcomes = []  # queue of bools; True = next add() raises
        self._update_metadata_result = True  # what the next update_metadata() returns
        self.update_metadata_calls = []

    def queue_add_failure(self):
        self._add_outcomes.append(True)

    def add_to_collection(self, name, text, metadata):
        fail = self._add_outcomes.pop(0) if self._add_outcomes else False
        if fail:
            raise RuntimeError(f"{MARKER}_add")
        self._id_counter += 1
        doc_id = f"doc_{self._id_counter}"
        self._items.append({"id": doc_id, "content": text, "metadata": dict(metadata)})
        return doc_id

    def list_all(self, name):
        return list(self._items)

    def create_collection(self, name):
        return self.collection

    def update_metadata(self, name, doc_id, metadata_updates):
        self.update_metadata_calls.append((name, doc_id, dict(metadata_updates)))
        if not self._update_metadata_result:
            return False
        for item in self._items:
            if item.get("id") == doc_id:
                item["metadata"].update(metadata_updates)
                return True
        return False


def _make_proposal(title="Test proposal", proposal_id=None, status=ProposalStatus.PENDING,
                    reasoning="Because"):
    kwargs = dict(title=title, proposal_type=ProposalType.FEATURE, status=status,
                  reasoning=reasoning)
    if proposal_id is not None:
        kwargs["id"] = proposal_id
    return CodeProposal(**kwargs)


def _items_for(chroma, proposal_id):
    return [i for i in chroma.list_all(COLLECTION_NAME)
            if i["metadata"].get("proposal_id") == proposal_id]


def _make_detection(confidence=0.8, status="detected", evidence="Found in commit"):
    return SimpleNamespace(confidence=confidence, status=status, evidence=evidence)


# ===========================================================================
# update_status — staged replacement
# ===========================================================================

class TestUpdateStatusStagedReplacement:

    def test_store_fails_keeps_old_document(self):
        """A raising re-store leaves the old (still-pending) document untouched."""
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Fix bug", proposal_id="p1"))

        chroma.queue_add_failure()
        assert store.update_status("p1", ProposalStatus.APPROVED) is False

        items = _items_for(chroma, "p1")
        assert len(items) == 1
        assert items[0]["metadata"]["status"] == "pending"
        assert chroma.collection.delete_calls == []  # no delete was attempted

    def test_delete_of_old_document_fails_rolls_back_new_document(self):
        """A delete failure rolls the new document back; exactly the old one remains."""
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Review PR", proposal_id="p2"))
        old_id = _items_for(chroma, "p2")[0]["id"]
        chroma.collection.fail_delete(old_id)

        assert store.update_status("p2", ProposalStatus.APPROVED) is False

        items = _items_for(chroma, "p2")
        assert len(items) == 1
        assert items[0]["id"] == old_id  # the new (approved) doc was rolled back
        assert items[0]["metadata"]["status"] == "pending"

    def test_success_approved_stages_replacement(self):
        """A clean write+delete for APPROVED: exactly one document, a fresh id."""
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Deploy", proposal_id="p3"))
        old_id = _items_for(chroma, "p3")[0]["id"]

        assert store.update_status("p3", ProposalStatus.APPROVED) is True

        items = _items_for(chroma, "p3")
        assert len(items) == 1
        assert items[0]["id"] != old_id  # a fresh uuid-style id, not an in-place edit
        assert items[0]["metadata"]["status"] == "approved"

    def test_success_rejected_with_reason_stages_replacement(self):
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Skip this", proposal_id="p4"))
        old_id = _items_for(chroma, "p4")[0]["id"]

        assert store.update_status("p4", ProposalStatus.REJECTED, reason="Not needed") is True

        items = _items_for(chroma, "p4")
        assert len(items) == 1
        assert items[0]["id"] != old_id
        assert items[0]["metadata"]["status"] == "rejected"
        assert items[0]["metadata"]["rejection_reason"] == "Not needed"

    def test_success_completed_with_hash_stages_replacement(self):
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Ship it", proposal_id="p5"))
        old_id = _items_for(chroma, "p5")[0]["id"]

        assert store.update_status("p5", ProposalStatus.COMPLETED, commit_hash="abc123") is True

        items = _items_for(chroma, "p5")
        assert len(items) == 1
        assert items[0]["id"] != old_id
        assert items[0]["metadata"]["status"] == "completed"
        assert items[0]["metadata"]["commit_hash"] == "abc123"

    def test_missing_collection_handle_returns_false_and_writes_nothing(self):
        """Today this silently stores a duplicate; staged replacement must refuse."""
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="No collection handle", proposal_id="p6"))
        chroma.collections[COLLECTION_NAME] = None  # _ensure_collection() still passes

        assert store.update_status("p6", ProposalStatus.APPROVED) is False

        items = _items_for(chroma, "p6")
        assert len(items) == 1  # nothing new was stored
        assert items[0]["metadata"]["status"] == "pending"  # nothing was deleted either
        assert chroma.collection.delete_calls == []

    def test_not_found_returns_false(self):
        """Control: unmatched proposal_id behaves exactly as before."""
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Other", proposal_id="other"))

        assert store.update_status("nonexistent", ProposalStatus.APPROVED) is False


# ===========================================================================
# update_tracking_metadata — honest bool
# ===========================================================================

class TestUpdateTrackingMetadataHonestBool:

    def test_merge_failure_returns_false_and_skips_success_log(self, caplog):
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Track me", proposal_id="t1"))
        chroma._update_metadata_result = False

        with caplog.at_level(logging.INFO):
            result = store.update_tracking_metadata("t1", _make_detection())

        assert result is False
        assert "Updated tracking for t1" not in caplog.text

    def test_merge_success_returns_true_and_logs(self, caplog):
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Track me too", proposal_id="t2"))

        with caplog.at_level(logging.INFO):
            result = store.update_tracking_metadata(
                "t2", _make_detection(confidence=0.5, status="verified")
            )

        assert result is True
        assert "Updated tracking for t2" in caplog.text

    def test_not_found_returns_false(self):
        """Control: unmatched proposal_id behaves exactly as before."""
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Other", proposal_id="other"))

        assert store.update_tracking_metadata("nonexistent", _make_detection()) is False


# ===========================================================================
# Privacy: injected markers never reach the new log lines
# ===========================================================================

def _staged_replacement_lines(caplog):
    """Only the new _replace_stored_proposal()/update_status() log lines —
    not the pre-existing, out-of-scope store_proposal() outer-except line,
    which interpolates the raw exception and is untouched by this batch."""
    return [r.message for r in caplog.records if "Staged replacement" in r.message]


class TestStagedReplacementPrivacy:

    def test_store_failure_marker_not_in_logs(self, caplog):
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Fix bug", proposal_id="pv1"))

        chroma.queue_add_failure()
        with caplog.at_level(logging.ERROR):
            assert store.update_status("pv1", ProposalStatus.APPROVED) is False

        lines = _staged_replacement_lines(caplog)
        assert lines  # the new log line fired
        assert not any(MARKER in line for line in lines)
        assert any("StoreWriteError" in line for line in lines)

    def test_delete_failure_marker_not_in_logs(self, caplog):
        chroma = FakeChromaStore()
        store = ProposalStore(chroma_store=chroma)
        store.store_proposal(_make_proposal(title="Fix bug", proposal_id="pv2"))
        old_id = _items_for(chroma, "pv2")[0]["id"]
        chroma.collection.fail_delete(old_id)

        with caplog.at_level(logging.ERROR):
            assert store.update_status("pv2", ProposalStatus.APPROVED) is False

        lines = _staged_replacement_lines(caplog)
        assert lines
        assert not any(MARKER in line for line in lines)
