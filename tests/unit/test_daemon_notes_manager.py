"""
Tests for knowledge/daemon_notes_manager.py.

Module Contract
- Purpose: Verify note creation, file writing, ChromaDB embedding,
  index management, path safety, and trigger detection.
- Inputs: Mock LLM, mock ChromaDB, tmp_path fixtures.
- Outputs: Pass/fail assertions.
- Dependencies: pytest, pytest-asyncio, unittest.mock, standard library.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from knowledge.daemon_notes_manager import (
    DaemonNote,
    DaemonNotesManager,
    detect_self_note_intent,
    VALID_CATEGORIES,
    VALID_CONFIDENCE,
    MAX_AUTONOMOUS_NOTES_PER_SESSION,
    DEDUP_SIMILARITY_THRESHOLD,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_manager():
    mm = AsyncMock()
    mm.generate_once = AsyncMock(return_value="This is a working note about the topic. It covers key decisions and rationale.")
    return mm


@pytest.fixture
def mock_chroma_store():
    store = MagicMock()
    store.add_to_collection = MagicMock(return_value="doc-123")
    return store


@pytest.fixture
def manager(mock_model_manager, mock_chroma_store, tmp_path):
    return DaemonNotesManager(
        model_manager=mock_model_manager,
        chroma_store=mock_chroma_store,
        output_dir=tmp_path / "daemon_notes",
        repo_root=tmp_path,
    )


# ############################################################################
#
#  Data model
#
# ############################################################################

class TestDaemonNote:

    def test_creation(self):
        note = DaemonNote(
            id="test-note-2026-05-23",
            title="Test Note",
            category="implementation",
            summary="A test summary.",
            confidence="medium",
        )
        assert note.id == "test-note-2026-05-23"
        assert note.status == "active"
        assert note.tags == []

    def test_to_dict(self):
        note = DaemonNote(
            id="test", title="T", category="research",
            summary="S", confidence="high", tags=["a", "b"],
        )
        d = note.to_dict()
        assert d["category"] == "research"
        assert d["tags"] == ["a", "b"]

    def test_chromadb_metadata(self):
        note = DaemonNote(
            id="test", title="T", category="architecture",
            summary="S", confidence="low",
            tags=["tag1"], related_files=["file.py"],
            created="2026-01-01T00:00:00Z",
        )
        meta = note.to_chromadb_metadata()
        assert meta["source_type"] == "daemon_self_note"
        assert meta["author"] == "daemon"
        assert meta["ground_truth"] is False
        assert meta["trust_level"] == "working_context"
        assert meta["category"] == "architecture"
        assert meta["confidence"] == "low"
        assert "tag1" in meta["tags"]
        assert "file.py" in meta["related_files"]


# ############################################################################
#
#  Note creation
#
# ############################################################################

class TestCreateNote:

    @pytest.mark.asyncio
    async def test_creates_markdown_file(self, manager, tmp_path):
        note = await manager.create_note(
            title="Test Implementation Note",
            category="implementation",
            summary="This is a test note about implementation.",
            confidence="medium",
        )
        assert Path(note.path).exists()
        content = Path(note.path).read_text()
        assert content.startswith("---")
        assert "type: daemon_self_note" in content
        assert "category: implementation" in content
        assert "confidence: medium" in content

    @pytest.mark.asyncio
    async def test_embeds_into_chromadb(self, manager, mock_chroma_store):
        note = await manager.create_note(
            title="ChromaDB Test",
            category="architecture",
            summary="Testing ChromaDB embedding with ground_truth=False.",
            confidence="high",
        )
        mock_chroma_store.add_to_collection.assert_called_once()
        call_args = mock_chroma_store.add_to_collection.call_args
        assert call_args[1]["metadata"]["ground_truth"] is False
        assert call_args[1]["metadata"]["source_type"] == "daemon_self_note"

    @pytest.mark.asyncio
    async def test_updates_index(self, manager, tmp_path):
        await manager.create_note(
            title="Index Test",
            category="research",
            summary="Testing index.json update.",
            confidence="low",
        )
        index_path = tmp_path / "daemon_notes" / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert len(data) == 1
        assert data[0]["category"] == "research"

    @pytest.mark.asyncio
    async def test_validates_title_too_short(self, manager):
        with pytest.raises(ValueError, match="Title"):
            await manager.create_note(title="ab", summary="Valid summary text here.", confidence="medium")

    @pytest.mark.asyncio
    async def test_validates_summary_too_short(self, manager):
        with pytest.raises(ValueError, match="Summary"):
            await manager.create_note(title="Valid Title", summary="Short", confidence="medium")

    @pytest.mark.asyncio
    async def test_invalid_category_defaults(self, manager):
        note = await manager.create_note(
            title="Category Test",
            category="invalid_cat",
            summary="Testing invalid category defaults.",
            confidence="medium",
        )
        assert note.category == "implementation"

    @pytest.mark.asyncio
    async def test_invalid_confidence_defaults(self, manager):
        note = await manager.create_note(
            title="Confidence Test",
            category="research",
            summary="Testing invalid confidence defaults.",
            confidence="extreme",
        )
        assert note.confidence == "medium"

    @pytest.mark.asyncio
    async def test_with_body(self, manager):
        note = await manager.create_note(
            title="Body Test",
            category="decisions",
            summary="Testing note with freeform body.",
            confidence="high",
            body="## Decisions\n- We decided to use direct invocation.\n\n## Rationale\nBecause the agentic loop doesn't work.",
        )
        content = Path(note.path).read_text()
        assert "## Decisions" in content
        assert "direct invocation" in content

    @pytest.mark.asyncio
    async def test_with_tags_and_files(self, manager):
        note = await manager.create_note(
            title="Tags Test",
            category="implementation",
            summary="Testing tags and related files.",
            confidence="medium",
            tags=["tag1", "tag2"],
            related_files=["file.py", "other.py"],
        )
        content = Path(note.path).read_text()
        assert "tag1" in content
        assert "file.py" in content

    @pytest.mark.asyncio
    async def test_collision_suffix(self, manager, tmp_path):
        n1 = await manager.create_note(title="Same Note", summary="First version of this note.", confidence="medium")
        n2 = await manager.create_note(title="Same Note", summary="Second version of this note.", confidence="medium")
        assert n1.path != n2.path
        assert Path(n1.path).exists()
        assert Path(n2.path).exists()


# ############################################################################
#
#  File writing / path safety
#
# ############################################################################

class TestFileWriting:

    def test_safe_slug(self):
        assert DaemonNotesManager._safe_slug("Hello World!") == "hello-world"
        assert DaemonNotesManager._safe_slug("résumé") == "r-sum"
        assert DaemonNotesManager._safe_slug("") == "note"
        assert DaemonNotesManager._safe_slug("a" * 100)[:60] == "a" * 60

    @pytest.mark.asyncio
    async def test_output_inside_daemon_notes(self, manager, tmp_path):
        note = await manager.create_note(
            title="Path Test", summary="Testing path safety.", confidence="medium",
        )
        assert str(Path(note.path).resolve()).startswith(
            str((tmp_path / "daemon_notes").resolve())
        )

    @pytest.mark.asyncio
    async def test_creates_directory(self, manager, tmp_path):
        note = await manager.create_note(
            title="Dir Test", summary="Testing directory creation.", confidence="medium",
        )
        assert (tmp_path / "daemon_notes").is_dir()


# ############################################################################
#
#  Index management
#
# ############################################################################

class TestIndex:

    @pytest.mark.asyncio
    async def test_creates_if_missing(self, manager, tmp_path):
        index_path = tmp_path / "daemon_notes" / "index.json"
        assert not index_path.exists()
        await manager.create_note(title="First", summary="First note ever.", confidence="medium")
        assert index_path.exists()

    @pytest.mark.asyncio
    async def test_preserves_existing(self, manager, tmp_path):
        index_path = tmp_path / "daemon_notes" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps([{"id": "old"}]))
        await manager.create_note(title="New", summary="New note added.", confidence="medium")
        data = json.loads(index_path.read_text())
        assert len(data) == 2
        assert data[0]["id"] == "old"

    @pytest.mark.asyncio
    async def test_handles_corrupt_index(self, manager, tmp_path):
        index_path = tmp_path / "daemon_notes" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("not json!")
        await manager.create_note(title="Recovery", summary="Testing index corruption recovery.", confidence="medium")
        data = json.loads(index_path.read_text())
        assert len(data) == 1


# ############################################################################
#
#  LLM summary generation
#
# ############################################################################

class TestSummaryGeneration:

    @pytest.mark.asyncio
    async def test_generates_summary(self, manager, mock_model_manager):
        result = await manager._generate_summary("memory system", mock_model_manager)
        assert len(result) > 10
        mock_model_manager.generate_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_no_model(self, tmp_path):
        mgr = DaemonNotesManager(model_manager=None, output_dir=tmp_path)
        result = await mgr._generate_summary("test", None)
        assert "Working notes on: test" in result


# ############################################################################
#
#  Trigger detection
#
# ############################################################################

class TestTriggerDetection:

    @pytest.mark.parametrize("query", [
        "save a note for yourself about the memory system",
        "write yourself a note about the document pipeline",
        "leave an implementation note about the scorer",
        "note that we decided to use direct invocation",
        "remember this for next time: always use direct invocation",
        "create a note for your future self about ChromaDB patterns",
        "save a note for later about the retrieval pipeline",
    ])
    def test_strong_triggers(self, query):
        result = detect_self_note_intent(query)
        assert result is not None, f"Should trigger on: {query}"
        assert result["topic"]

    @pytest.mark.parametrize("query", [
        "tell me about cats",
        "write a report about climate change",
        "save a report about something",
        "hello",
        "what's the weather",
        "note: this is just a colon",
        "research something for me",
    ])
    def test_non_triggers(self, query):
        result = detect_self_note_intent(query)
        assert result is None, f"Should NOT trigger on: {query}"

    def test_detects_architecture_category(self):
        result = detect_self_note_intent("save a note for yourself about the architecture of the scorer")
        assert result is not None
        assert result["category"] == "architecture"

    def test_detects_decisions_category(self):
        result = detect_self_note_intent("note that we decided to use ChromaDB")
        assert result is not None
        assert result["category"] == "decisions"

    def test_detects_research_category(self):
        result = detect_self_note_intent("leave a research note for yourself about embedding models")
        assert result is not None
        assert result["category"] == "research"

    def test_extracts_topic(self):
        result = detect_self_note_intent("save a note for yourself about the memory scoring algorithm")
        assert result is not None
        assert "memory scoring" in result["topic"].lower()


# ############################################################################
#
#  Autonomous note guardrails
#
# ############################################################################

class TestAutonomousGuardrails:

    @pytest.fixture
    def auto_manager(self, mock_model_manager, mock_chroma_store, tmp_path):
        """Manager for autonomous note tests."""
        # Make chroma query return empty (no duplicates by default)
        mock_chroma_store.query_collection = MagicMock(return_value=[])
        return DaemonNotesManager(
            model_manager=mock_model_manager,
            chroma_store=mock_chroma_store,
            output_dir=tmp_path / "daemon_notes",
            repo_root=tmp_path,
        )

    @pytest.mark.asyncio
    async def test_per_session_cap(self, auto_manager):
        """After MAX_AUTONOMOUS_NOTES_PER_SESSION, notes are rejected."""
        session = "test-session-123"
        created = []
        for i in range(MAX_AUTONOMOUS_NOTES_PER_SESSION + 2):
            result = await auto_manager.create_autonomous_note(
                title=f"Note {i}",
                category="implementation",
                summary=f"This is autonomous note number {i} with enough text.",
                session_id=session,
            )
            created.append(result)

        # First N should succeed
        assert all(n is not None for n in created[:MAX_AUTONOMOUS_NOTES_PER_SESSION])
        # Remaining should be None (skipped)
        assert all(n is None for n in created[MAX_AUTONOMOUS_NOTES_PER_SESSION:])

    @pytest.mark.asyncio
    async def test_session_reset(self, auto_manager):
        """New session_id resets the counter."""
        for i in range(MAX_AUTONOMOUS_NOTES_PER_SESSION):
            await auto_manager.create_autonomous_note(
                title=f"Session 1 Note {i}",
                category="research",
                summary=f"Note from session one, number {i} with detail.",
                session_id="session-1",
            )

        # Cap reached for session-1
        result = await auto_manager.create_autonomous_note(
            title="Should be rejected",
            category="research",
            summary="This should be rejected due to session cap.",
            session_id="session-1",
        )
        assert result is None

        # New session resets
        result = await auto_manager.create_autonomous_note(
            title="New session note",
            category="research",
            summary="This is from a new session and should succeed.",
            session_id="session-2",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_semantic_dedup_skips_duplicate(self, auto_manager, mock_chroma_store):
        """Near-duplicate notes are rejected."""
        # Mock ChromaDB returning a high-similarity match
        mock_chroma_store.query_collection = MagicMock(return_value=[
            {"content": "Very similar content", "relevance_score": 0.92, "metadata": {}}
        ])

        result = await auto_manager.create_autonomous_note(
            title="Duplicate Note",
            category="implementation",
            summary="This is very similar to an existing note.",
            session_id="test",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_dedup_allows_different(self, auto_manager, mock_chroma_store):
        """Sufficiently different notes pass dedup."""
        mock_chroma_store.query_collection = MagicMock(return_value=[
            {"content": "Different content", "relevance_score": 0.3, "metadata": {}}
        ])

        result = await auto_manager.create_autonomous_note(
            title="Unique Note",
            category="implementation",
            summary="This is sufficiently different from existing notes.",
            session_id="test",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_status_defaults_to_tentative(self, auto_manager):
        """Autonomous notes default to tentative status."""
        result = await auto_manager.create_autonomous_note(
            title="Tentative Note",
            category="decisions",
            summary="This should have tentative status by default.",
            session_id="test",
        )
        assert result is not None
        assert result.status == "tentative"

    @pytest.mark.asyncio
    async def test_failed_attempt_status(self, auto_manager):
        """Notes can be marked as failed_attempt."""
        result = await auto_manager.create_autonomous_note(
            title="Failed Approach",
            category="implementation",
            summary="Tried using the agentic loop for direct invocation. It did not work.",
            session_id="test",
            status="failed_attempt",
        )
        assert result is not None
        assert result.status == "failed_attempt"

    @pytest.mark.asyncio
    async def test_hypothesis_status(self, auto_manager):
        """Notes can be marked as hypothesis."""
        result = await auto_manager.create_autonomous_note(
            title="Embedding Hypothesis",
            category="research",
            summary="BGE-small may outperform MiniLM for this use case based on initial results.",
            session_id="test",
            status="hypothesis",
        )
        assert result is not None
        assert result.status == "hypothesis"

    @pytest.mark.asyncio
    async def test_chromadb_metadata_has_session_id(self, auto_manager, mock_chroma_store):
        """Autonomous notes include session_id in ChromaDB metadata."""
        result = await auto_manager.create_autonomous_note(
            title="Audit Trail Test",
            category="implementation",
            summary="This note should have session_id in its metadata.",
            session_id="session-audit-123",
        )
        assert result is not None
        call_args = mock_chroma_store.add_to_collection.call_args
        meta = call_args[1]["metadata"]
        assert meta["ground_truth"] is False
        assert meta["source_type"] == "daemon_self_note"

    @pytest.mark.asyncio
    async def test_note_does_not_enter_user_profile(self, auto_manager, mock_chroma_store):
        """Autonomous notes only go to daemon_self_notes, never to facts or user_profile."""
        await auto_manager.create_autonomous_note(
            title="Should Not Be A Fact",
            category="implementation",
            summary="This note must not appear in the facts collection.",
            session_id="test",
        )
        call_args = mock_chroma_store.add_to_collection.call_args
        collection_name = call_args[0][0] if call_args[0] else call_args[1].get("name", "")
        assert collection_name == "daemon_self_notes"


# ############################################################################
#
#  Persistence outcome flags (anti-confabulation support)
#
# ############################################################################

class TestPersistenceFlags:

    @pytest.mark.asyncio
    async def test_flags_all_true_on_clean_save(self, manager):
        note = await manager.create_note(
            title="Clean Save", category="implementation",
            summary="A note that persists fully across disk, embed, and index.",
        )
        assert note.disk_written is True
        assert note.embedded is True
        assert note.indexed is True
        assert note.fully_persisted is True

    @pytest.mark.asyncio
    async def test_embedded_vacuously_true_without_store(self, mock_model_manager, tmp_path):
        mgr = DaemonNotesManager(
            model_manager=mock_model_manager, chroma_store=None,
            output_dir=tmp_path / "daemon_notes", repo_root=tmp_path,
        )
        note = await mgr.create_note(
            title="No Store", category="implementation",
            summary="No chroma store means embedding is vacuously satisfied.",
        )
        assert note.embedded is True
        assert note.fully_persisted is True

    @pytest.mark.asyncio
    async def test_embed_failure_surfaces_in_flags(self, mock_model_manager, tmp_path):
        store = MagicMock()
        store.add_to_collection = MagicMock(side_effect=RuntimeError("chroma down"))
        mgr = DaemonNotesManager(
            model_manager=mock_model_manager, chroma_store=store,
            output_dir=tmp_path / "daemon_notes", repo_root=tmp_path,
        )
        note = await mgr.create_note(
            title="Embed Fails", category="implementation",
            summary="Disk write succeeds but the ChromaDB embed raises.",
        )
        # Disk file still written, but the partial save is observable.
        assert note.disk_written is True
        assert Path(note.path).exists()
        assert note.embedded is False
        assert note.fully_persisted is False
