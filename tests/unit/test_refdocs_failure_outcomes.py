"""Regression tests for CGR-20260913-008 #103-#106 and the upload-refusal
defect (F3a, docs/execution/generalization/failure_outcome_design.md).

Root defects (verified against the deployed source before this batch):
`_get_document_chunks`/`_keyword_search`/`list_documents` collapse "no
matches" and a raised store error into a bare `[]` (dm18 #103/#105/#106);
`get_documents` does the same for its hybrid read (#104). `upload_document`/
`upload_text` snapshot the PRIOR version via `_get_document_chunks` before
inserting the new one (F02 staged replacement); a silently-empty failed
snapshot let the upload proceed, leaving both the stale old chunks and the
new ones retrievable under the same title (CM-07) — the same failure shape
as the 2026-08-02 cold-store incident regression-guarded by
tests/unit/test_refdocs_lazy_collection.py, triggered here by a genuine read
failure rather than an unopened collection.

Drives the DEPLOYED `ReferenceDocsManager` methods directly, never a
re-derivation. Isolation follows test_refdocs_lazy_collection.py's
`_store_with_unopened_collection` pattern: a `MagicMock` store whose
`_get_collection()` is wired directly, no real ChromaDB or embedder. File
uploads use `tmp_path`.
"""
from unittest.mock import MagicMock

import pytest

from knowledge.reference_docs_manager import ReferenceDocsManager
from utils.retrieval_outcome import OutcomeList

UPLOAD_REFUSAL_ERROR = (
    "Could not read the existing version of this document, so the upload "
    "was refused to avoid keeping two versions. Please try again."
)

# Distinctive markers so a privacy assertion proves absence, not luck.
PRIVATE_EXC_TEXT = "leaked exception detail EXCZQX9"
PRIVATE_TITLE = "Confidential Report TITLEZQX9"
PRIVATE_QUERY = "sensitive question about TITLEZQX9 health"


# Store/collection builders
def _mgr_with_collection(coll):
    store = MagicMock()
    store._get_collection = MagicMock(return_value=coll)
    return ReferenceDocsManager(chroma_store=store)


def _mock_collection_returning(payload):
    coll = MagicMock()
    coll.get = MagicMock(return_value=payload)
    return coll


def _mgr_with_raising_get(exc):
    coll = MagicMock()
    coll.get = MagicMock(side_effect=exc)
    return _mgr_with_collection(coll)


def _mgr_with_unavailable_collection():
    store = MagicMock()
    store._get_collection = MagicMock(return_value=None)
    return ReferenceDocsManager(chroma_store=store)


# One shared payload satisfying all three plain-read methods' "success"
# control at once; one shared "genuinely empty, read succeeded" payload.
MATCHING_PAYLOAD = {
    "ids": ["c1"],
    "documents": ["Notes about architecture patterns used in the system."],
    "metadatas": [{"title": "Doc Title", "section": "Overview",
                    "type": "reference_doc", "file_type": "markdown",
                    "timestamp": "2026-01-01T00:00:00"}],
}
EMPTY_PAYLOAD = {"ids": [], "documents": [], "metadatas": []}


def _call_get_document_chunks(mgr):
    return mgr._get_document_chunks("Doc Title")


def _call_keyword_search(mgr):
    return mgr._keyword_search("architecture", limit=10)


def _call_list_documents(mgr):
    return mgr.list_documents()


READ_METHODS = [
    ("_get_document_chunks", _call_get_document_chunks),
    ("_keyword_search", _call_keyword_search),
    ("list_documents", _call_list_documents),
]
READ_METHOD_IDS = [name for name, _ in READ_METHODS]


# #103, #105, #106 — the three plain reads

class TestPlainReadOutcomes:
    @pytest.mark.parametrize("name,call", READ_METHODS, ids=READ_METHOD_IDS)
    def test_raising_collection_get_is_failed_and_empty(self, name, call):
        mgr = _mgr_with_raising_get(RuntimeError("boom"))
        result = call(mgr)
        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result.reason == "RuntimeError"
        assert result == []

    @pytest.mark.parametrize("name,call", READ_METHODS, ids=READ_METHOD_IDS)
    def test_missing_collection_is_unavailable_and_empty(self, name, call):
        mgr = _mgr_with_unavailable_collection()
        result = call(mgr)
        assert isinstance(result, OutcomeList)
        assert result.status == "unavailable"
        assert result.reason == "collection_unavailable"
        assert result == []

    @pytest.mark.parametrize("name,call", READ_METHODS, ids=READ_METHOD_IDS)
    def test_control_matching_read_succeeds(self, name, call):
        mgr = _mgr_with_collection(_mock_collection_returning(MATCHING_PAYLOAD))
        result = call(mgr)
        assert isinstance(result, OutcomeList)
        assert result.status == "succeeded"
        assert result != []

    @pytest.mark.parametrize("name,call", READ_METHODS, ids=READ_METHOD_IDS)
    def test_control_genuinely_empty_is_no_results(self, name, call):
        mgr = _mgr_with_collection(_mock_collection_returning(EMPTY_PAYLOAD))
        result = call(mgr)
        assert isinstance(result, OutcomeList)
        assert result.status == "no_results"
        assert result == []


# #104 — get_documents hybrid keyword + semantic

def _semantic_item(title="Sem Title"):
    return {"content": "semantic hit", "metadata": {"title": title, "section": ""},
            "relevance_score": 0.5}


def _keyword_item(title="Key Title"):
    return {"content": "keyword hit", "metadata": {"title": title, "section": ""},
            "relevance_score": 1.0}


class TestGetDocumentsHybrid:
    @pytest.mark.asyncio
    async def test_keyword_failed_items_kept_reason_prefixed(self):
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=OutcomeList.failed("RuntimeError"))
        mgr.chroma_store.query_collection = MagicMock(return_value=[_semantic_item()])

        result = await mgr.get_documents("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result.reason.startswith("keyword:")
        assert len(result) == 1  # the semantic item is kept

    @pytest.mark.asyncio
    async def test_semantic_raises_is_failed_and_empty(self):
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=OutcomeList([_keyword_item()]))
        mgr.chroma_store.query_collection = MagicMock(side_effect=RuntimeError("boom"))

        result = await mgr.get_documents("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result == []

    @pytest.mark.asyncio
    async def test_control_healthy_hybrid_succeeds(self):
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=OutcomeList([_keyword_item()]))
        mgr.chroma_store.query_collection = MagicMock(return_value=[_semantic_item()])

        result = await mgr.get_documents("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "succeeded"
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_control_healthy_but_empty_is_no_results(self):
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=OutcomeList([]))
        mgr.chroma_store.query_collection = MagicMock(return_value=[])

        result = await mgr.get_documents("q", limit=9)

        assert isinstance(result, OutcomeList)
        assert result.status == "no_results"
        assert result == []


# Upload refusal (CM-07): a failed/unavailable snapshot read must refuse
# BEFORE chunking/embedding/inserting/replacing; a genuine no_results
# snapshot (first upload) proceeds exactly as today.

CONTENT = "Some upload content that is long enough to chunk without error."


class TestUploadRefusal:
    def test_upload_text_refuses_on_failed_snapshot(self):
        mgr = _mgr_with_raising_get(RuntimeError("boom"))
        mgr.chroma_store.add_batch_to_collection = MagicMock()
        mgr._replace_old_chunks = MagicMock()

        result = mgr.upload_text(content=CONTENT, title="Doc")

        assert result.success is False
        assert UPLOAD_REFUSAL_ERROR in result.errors
        mgr.chroma_store.add_batch_to_collection.assert_not_called()
        mgr._replace_old_chunks.assert_not_called()

    def test_upload_document_refuses_on_failed_snapshot(self, tmp_path):
        mgr = _mgr_with_raising_get(RuntimeError("boom"))
        mgr.chroma_store.add_batch_to_collection = MagicMock()
        mgr._replace_old_chunks = MagicMock()
        f = tmp_path / "doc.txt"
        f.write_text(CONTENT)

        result = mgr.upload_document(str(f), title="Doc")

        assert result.success is False
        assert UPLOAD_REFUSAL_ERROR in result.errors
        mgr.chroma_store.add_batch_to_collection.assert_not_called()
        mgr._replace_old_chunks.assert_not_called()

    def test_upload_text_refuses_on_unavailable_snapshot(self):
        mgr = _mgr_with_unavailable_collection()
        mgr.chroma_store.add_batch_to_collection = MagicMock()

        result = mgr.upload_text(content=CONTENT, title="Doc")

        assert result.success is False
        assert UPLOAD_REFUSAL_ERROR in result.errors
        mgr.chroma_store.add_batch_to_collection.assert_not_called()

    def test_control_first_upload_on_empty_collection_succeeds(self):
        mgr = _mgr_with_collection(_mock_collection_returning(EMPTY_PAYLOAD))
        mgr.chroma_store.add_batch_to_collection = MagicMock(return_value=["id1"])

        result = mgr.upload_text(content=CONTENT, title="Doc")

        assert result.success is True
        mgr.chroma_store.add_batch_to_collection.assert_called_once()

    def test_control_prior_version_is_replaced(self):
        mgr = _mgr_with_collection(_mock_collection_returning(MATCHING_PAYLOAD))
        mgr.chroma_store.add_batch_to_collection = MagicMock(return_value=["id1"])
        mgr._replace_old_chunks = MagicMock()

        # "Doc Title" matches MATCHING_PAYLOAD so the snapshot is non-empty.
        result = mgr.upload_text(content=CONTENT, title="Doc Title")

        assert result.success is True
        mgr._replace_old_chunks.assert_called_once()


class TestSyncFileNoInsertOnFailedSnapshot:
    """2026-08-02 duplicate class, one layer deeper: sync_file's hash-read
    failure falls through to upload_document, whose refusal must still fire."""

    def test_sync_file_reports_failed_and_never_inserts(self, tmp_path):
        mgr = _mgr_with_raising_get(RuntimeError("boom"))
        mgr.chroma_store.add_batch_to_collection = MagicMock()
        f = tmp_path / "doc.txt"
        f.write_text(CONTENT)

        status = mgr.sync_file(str(f), title="Doc")

        assert status == "failed"
        mgr.chroma_store.add_batch_to_collection.assert_not_called()


# list_document_titles propagates list_documents()'s status

class TestListDocumentTitles:
    def test_propagates_failed_status(self):
        mgr = _mgr_with_raising_get(RuntimeError("boom"))
        result = mgr.list_document_titles()
        assert isinstance(result, OutcomeList)
        assert result.status == "failed"
        assert result == []

    def test_control_succeeds_with_sorted_titles(self):
        mgr = _mgr_with_collection(_mock_collection_returning(MATCHING_PAYLOAD))
        result = mgr.list_document_titles()
        assert isinstance(result, OutcomeList)
        assert result.status == "succeeded"
        assert result == ["Doc Title"]


# Privacy: no reason/error string carries the title, query or exception text

class TestPrivacyNoLeakedText:
    def test_get_document_chunks_failure_reason_has_no_leaked_text(self):
        mgr = _mgr_with_raising_get(RuntimeError(PRIVATE_EXC_TEXT))
        result = mgr._get_document_chunks(PRIVATE_TITLE)
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_TITLE not in result.reason

    def test_keyword_search_failure_reason_has_no_leaked_text(self):
        mgr = _mgr_with_raising_get(RuntimeError(PRIVATE_EXC_TEXT))
        result = mgr._keyword_search(PRIVATE_QUERY)
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason

    @pytest.mark.asyncio
    async def test_get_documents_failure_reason_has_no_leaked_text(self):
        mgr = ReferenceDocsManager(chroma_store=MagicMock())
        mgr._keyword_search = MagicMock(return_value=OutcomeList.failed("RuntimeError"))
        mgr.chroma_store.query_collection = MagicMock(side_effect=RuntimeError(PRIVATE_EXC_TEXT))
        result = await mgr.get_documents(PRIVATE_QUERY)
        assert PRIVATE_EXC_TEXT not in result.reason
        assert PRIVATE_QUERY not in result.reason

    def test_upload_refusal_error_has_no_leaked_text(self):
        mgr = _mgr_with_raising_get(RuntimeError(PRIVATE_EXC_TEXT))
        result = mgr.upload_text(content=CONTENT, title=PRIVATE_TITLE)
        assert not any(PRIVATE_EXC_TEXT in e for e in result.errors)
        assert not any(PRIVATE_TITLE in e for e in result.errors)
        assert result.errors == [UPLOAD_REFUSAL_ERROR]
