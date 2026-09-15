"""Regression tests for CGR-20260913-010 anchors #127 and #140 (F10a).

``add_conversation_memory`` (annotated ``-> str``) and ``store_interaction``
used to swallow ANY exception into the same ``None`` a DELIBERATE skip
returns — a caller could not tell "chose not to persist" from "tried and
failed". These drive the DEPLOYED path: the real ``add_conversation_memory``
(fake-collection pattern, F9a's test_store_get_by_id_outcomes.py), the real
``store_interaction`` (a LOCAL copy of test_api_error_storage_guard.py's
``storage`` fixture — that file is unedited, per the FIXTURE RULE), the real
``MemoryCoordinator.store_interaction`` (test_provenance.py's
``MemoryCoordinator.__new__`` pattern), and the real
``gui.handlers._background_store_interaction`` (test_calendar_turn_round3.py
~609 pattern).

Contract (docs/execution/generalization/briefs/F10a.md): ``StoreWriteError``
is keyword-only source/reason, str "source: reason", a RuntimeError, NOT a
RetrievalError. ``add_conversation_memory`` raises it on a failed ``.add``.
``store_interaction``'s four deliberate skips still return None before any
write; a ``StoreWriteError`` from Chroma propagates unchanged; any other
body exception wraps as ``StoreWriteError(source="store_interaction", ...)``.
The coordinator syncs context/counter back even when storage raises
(try/finally), then re-raises; thread resolution runs only on success. The
handler logs a raised ``StoreWriteError`` and still records the transcript
with db_id=None. A privacy marker in an injected exception must never reach
a ``StoreWriteError``'s reason/str or the handler's log.

``StoreWriteError`` does not exist pre-edit, so every test needing it
imports it LOCALLY (inside the test body), not at module scope — so the
file still collects and the skip/success controls run and pass unedited.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.memory_coordinator import MemoryCoordinator
from memory.memory_storage import MemoryStorage
from memory.storage.multi_collection_chroma_store import MultiCollectionChromaStore
from utils.retrieval_outcome import RetrievalError

_MARKER = "SYNTH_MARKER_f10a_9d3e7b21"


# ---------------------------------------------------------------------------
# Leaf: StoreWriteError (utils/retrieval_outcome.py)
# ---------------------------------------------------------------------------


class TestStoreWriteErrorLeaf:
    def test_str_and_attributes_and_subclassing(self):
        from utils.retrieval_outcome import StoreWriteError
        err = StoreWriteError(source="chroma_conversations", reason="RuntimeError")
        assert err.source == "chroma_conversations"
        assert err.reason == "RuntimeError"
        assert str(err) == "chroma_conversations: RuntimeError"
        assert issubclass(StoreWriteError, RuntimeError)
        assert not issubclass(StoreWriteError, RetrievalError)
        assert not isinstance(err, RetrievalError)


# ---------------------------------------------------------------------------
# Chroma: MultiCollectionChromaStore.add_conversation_memory
# ---------------------------------------------------------------------------


class _FakeConversationsCollection:
    """Stand-in for the 'conversations' Chroma collection's ``.add(...)``."""

    def __init__(self, *, raises=None):
        self._raises = raises
        self.add_calls = []

    def add(self, ids=None, documents=None, metadatas=None):
        self.add_calls.append({"ids": ids, "documents": documents, "metadatas": metadatas})
        if self._raises is not None:
            raise self._raises


def _make_store(collection):
    """A MultiCollectionChromaStore wired to a fake 'conversations' collection,
    bypassing __init__ (test_store_get_by_id_outcomes.py / F9a pattern) so no
    real Chroma client or SentenceTransformer is constructed."""
    store = MultiCollectionChromaStore.__new__(MultiCollectionChromaStore)
    store.collections = {"conversations": collection}
    return store


class TestAddConversationMemoryOutcomes:
    def test_failed_add_raises_store_write_error(self):
        from utils.retrieval_outcome import StoreWriteError
        coll = _FakeConversationsCollection(raises=RuntimeError(f"backend exploded {_MARKER}"))
        store = _make_store(coll)

        with pytest.raises(StoreWriteError) as exc_info:
            store.add_conversation_memory("hi", "hello", {"a": 1})

        err = exc_info.value
        assert err.source == "chroma_conversations"
        assert err.reason == "RuntimeError"
        assert _MARKER not in err.reason
        assert _MARKER not in str(err)

    def test_success_returns_doc_id(self):
        coll = _FakeConversationsCollection()
        store = _make_store(coll)

        doc_id = store.add_conversation_memory("hi", "hello", {"a": 1})

        assert isinstance(doc_id, str) and doc_id
        assert coll.add_calls[0]["documents"] == ["User: hi\nAssistant: hello"]


# ---------------------------------------------------------------------------
# Storage: MemoryStorage.store_interaction
# (local copy of test_api_error_storage_guard.py's `storage` fixture — never
# shared/imported; that file's own fixture and tests are unedited here)
# ---------------------------------------------------------------------------


def _make_storage(*, chroma_store=None, corpus_manager=None):
    corpus_manager = corpus_manager if corpus_manager is not None else MagicMock()
    chroma_store = chroma_store if chroma_store is not None else MagicMock()
    ms = MemoryStorage(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        fact_extractor=MagicMock(),
    )
    return ms, corpus_manager, chroma_store


class TestStoreInteractionFailureModes:
    @pytest.mark.asyncio
    async def test_store_write_error_from_chroma_propagates_unchanged(self):
        from utils.retrieval_outcome import StoreWriteError
        swe = StoreWriteError(source="chroma_conversations", reason="RuntimeError")
        chroma_store = MagicMock()
        chroma_store.add_conversation_memory.side_effect = swe
        ms, _corpus, _chroma = _make_storage(chroma_store=chroma_store)

        with pytest.raises(StoreWriteError) as exc_info:
            await ms.store_interaction(query="hi", response="hello there")

        assert exc_info.value is swe  # same object — not re-wrapped

    @pytest.mark.asyncio
    async def test_corpus_add_entry_failure_wraps_as_store_write_error(self):
        from utils.retrieval_outcome import StoreWriteError
        corpus_manager = MagicMock()
        corpus_manager.add_entry.side_effect = RuntimeError(f"disk full {_MARKER}")
        ms, _corpus, chroma_store = _make_storage(corpus_manager=corpus_manager)

        with pytest.raises(StoreWriteError) as exc_info:
            await ms.store_interaction(query="hi", response="hello there")

        err = exc_info.value
        assert err.source == "store_interaction"
        assert err.reason == "RuntimeError"
        assert _MARKER not in err.reason
        assert _MARKER not in str(err)
        assert not chroma_store.add_conversation_memory.called  # never reached

    @pytest.mark.asyncio
    async def test_success_control_returns_id(self):
        """Positive control paired with the failures above and the skips below."""
        chroma_store = MagicMock()
        chroma_store.add_conversation_memory.return_value = "doc-999"
        ms, corpus_manager, _chroma = _make_storage(chroma_store=chroma_store)

        result = await ms.store_interaction(query="hi", response="hello there")

        assert result == "doc-999"
        assert corpus_manager.add_entry.called


@pytest.mark.asyncio
@pytest.mark.parametrize("label,response", [
    ("thinking_only", "<thinking>unfinished reasoning"),
    ("empty", "   "),
    ("file_error", "[error reading notes.docx] could not parse"),
    ("api_error", "[API Error] Error code: 402 - Insufficient credits"),
])
async def test_deliberate_skips_return_none_and_never_write(label, response):
    """Paired controls (BC-64): the four skip paths still return None BEFORE
    any write, and never raise — guards against a regression that would
    turn a deliberate skip into a StoreWriteError."""
    ms, corpus_manager, chroma_store = _make_storage()

    result = await ms.store_interaction(query="hi", response=response)

    assert result is None
    assert not corpus_manager.add_entry.called
    assert not chroma_store.add_conversation_memory.called


# ---------------------------------------------------------------------------
# Coordinator: MemoryCoordinator.store_interaction
# ---------------------------------------------------------------------------

_SYNCED_CONTEXT = object()


class _FakeStorage:
    """Lightest stand-in for MemoryStorage: the coordinator method only
    reads/writes `.current_topic`, `.conversation_context`,
    `.interactions_since_consolidation` and calls `.store_interaction(...)`."""

    def __init__(self, *, raises=None, memory_id="mem-1"):
        self._raises = raises
        self._memory_id = memory_id
        self.current_topic = None
        self.conversation_context = None
        self.interactions_since_consolidation = 0

    async def store_interaction(self, query, response, tags, session_id=None,
                                 provenance=None, user_text=None):
        # Simulate the real MemoryStorage mutating its own state (context
        # append, consolidation counter) before it raises or returns.
        self.conversation_context = _SYNCED_CONTEXT
        self.interactions_since_consolidation = 7
        if self._raises is not None:
            raise self._raises
        return self._memory_id


def _make_coordinator(storage, *, thread_store=None):
    coord = MemoryCoordinator.__new__(MemoryCoordinator)
    coord._storage = storage
    coord.current_topic = "general"
    coord.conversation_context = []
    coord.interactions_since_consolidation = 0
    coord.thread_store = thread_store
    return coord


class TestCoordinatorStoreInteraction:
    @pytest.mark.asyncio
    async def test_raise_still_syncs_context_back(self):
        from utils.retrieval_outcome import StoreWriteError
        swe = StoreWriteError(source="store_interaction", reason="RuntimeError")
        thread_store = MagicMock()
        coord = _make_coordinator(_FakeStorage(raises=swe), thread_store=thread_store)

        with pytest.raises(StoreWriteError) as exc_info:
            await coord.store_interaction("I completed the task", "ok")

        assert exc_info.value is swe
        assert coord.conversation_context is _SYNCED_CONTEXT
        assert coord.interactions_since_consolidation == 7
        # Quick thread resolution is unreached code after a raise.
        assert not thread_store.list_open_threads.called

    @pytest.mark.asyncio
    async def test_success_control_syncs_and_returns(self):
        thread_store = MagicMock()
        thread_store.list_open_threads.return_value = []
        coord = _make_coordinator(_FakeStorage(), thread_store=thread_store)

        result = await coord.store_interaction("I completed the task", "ok")

        assert result == "mem-1"
        assert coord.conversation_context is _SYNCED_CONTEXT
        assert coord.interactions_since_consolidation == 7
        assert thread_store.list_open_threads.called


# ---------------------------------------------------------------------------
# Handler: gui.handlers._background_store_interaction
# ---------------------------------------------------------------------------


class TestBackgroundStoreInteraction:
    @pytest.mark.asyncio
    async def test_store_write_error_logged_and_transcript_still_recorded(self, caplog):
        import gui.handlers as handlers
        from utils.retrieval_outcome import StoreWriteError

        # reason mirrors what the real chain produces from a marker-laden
        # exception: only the class name survives (leaf contract #1) — this
        # also checks the marker cannot reach the handler's error log.
        underlying = RuntimeError(f"disk full {_MARKER}")
        swe = StoreWriteError(source="store_interaction", reason=type(underlying).__name__)
        memory = SimpleNamespace(store_interaction=AsyncMock(side_effect=swe))
        orchestrator = SimpleNamespace(memory_system=memory, current_topic="general")
        conversation_logger = MagicMock()

        with caplog.at_level(logging.ERROR, logger="gradio_gui"):
            await handlers._background_store_interaction(
                orchestrator=orchestrator,
                merged_input="hi",
                response_to_store="Clean response",
                tags=["topic:general"],
                user_text="hi",
                final_output="<|sep|>Clean response",
                personality="default",
                file_names=[],
                conversation_logger=conversation_logger,
            )

        assert "Background storage failed" in caplog.text
        assert _MARKER not in caplog.text
        conversation_logger.log_interaction.assert_called_once()
        call = conversation_logger.log_interaction.call_args
        assert call.kwargs["metadata"]["db_id"] is None
        assert call.kwargs["assistant_response"] == "Clean response"

    @pytest.mark.asyncio
    async def test_success_control_unchanged(self):
        import gui.handlers as handlers

        memory = SimpleNamespace(store_interaction=AsyncMock(return_value="mem-42"))
        orchestrator = SimpleNamespace(memory_system=memory, current_topic="general")
        conversation_logger = MagicMock()

        await handlers._background_store_interaction(
            orchestrator=orchestrator,
            merged_input="hi",
            response_to_store="Clean response",
            tags=[],
            user_text="hi",
            final_output="hi",
            personality="default",
            file_names=[],
            conversation_logger=conversation_logger,
        )

        conversation_logger.log_interaction.assert_called_once()
        call = conversation_logger.log_interaction.call_args
        assert call.kwargs["metadata"]["db_id"] == "mem-42"
