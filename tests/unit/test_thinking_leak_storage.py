"""
Tests for the thinking-leak storage defense (2026-06-10).

Covers:
- ResponseParser.sanitize_for_storage() — the canonical pre-persistence strip
- memory_storage.store_interaction() — the boundary guard (no path can persist leaks)
- scripts/repair_thinking_leaks.scrub_thinking() — historical-data repair transform

Background: synthetic <thinking></thinking> stream markers (emitted by
response_generator around API-separated reasoning) and tagged thinking blocks
were persisted raw by the agentic storage path, polluting 752 conversation docs.
Replayed pollution teaches the model to emit literal tags itself.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.response_parser import ResponseParser


# ---------------------------------------------------------------------------
# sanitize_for_storage
# ---------------------------------------------------------------------------

class TestSanitizeForStorage:
    def test_empty_pair_prefix_stripped(self):
        assert ResponseParser.sanitize_for_storage(
            "<thinking></thinking>Good to hear the bug's sorted."
        ) == "Good to hear the bug's sorted."

    def test_empty_pair_with_whitespace_between(self):
        assert ResponseParser.sanitize_for_storage(
            "<thinking>  \n </thinking>Answer here."
        ) == "Answer here."

    def test_empty_pair_short_think_variant(self):
        assert ResponseParser.sanitize_for_storage(
            "<think></think>Answer."
        ) == "Answer."

    def test_leading_tagged_block_removed_answer_kept(self):
        text = "<thinking>\nLuke just mentioned a breakthrough.\n</thinking>\nThat's a real result."
        assert ResponseParser.sanitize_for_storage(text) == "That's a real result."

    def test_unclosed_leading_block_returns_empty(self):
        text = "<thinking>The user is asking about X and I should consider"
        assert ResponseParser.sanitize_for_storage(text) == ""

    def test_orphan_fragment_stripped(self):
        assert ResponseParser.sanitize_for_storage(
            "/think>Here's the answer."
        ) == "Here's the answer."

    def test_reflection_block_stripped(self):
        text = "Real answer.\n<reflect>quality notes</reflect>"
        assert ResponseParser.sanitize_for_storage(text) == "Real answer."

    def test_clean_text_unchanged(self):
        text = "A normal response with no artifacts.\n\nTwo paragraphs."
        assert ResponseParser.sanitize_for_storage(text) == text

    def test_empty_and_none_inputs(self):
        assert ResponseParser.sanitize_for_storage("") == ""
        assert ResponseParser.sanitize_for_storage(None) == ""

    def test_does_not_apply_untagged_heuristics(self):
        # Heuristic-looking prose with NO tags must survive storage sanitization —
        # heuristics can false-positive and storage must never eat real content.
        text = "I should mention that the user asked a good question. Let me explain the tradeoff in detail here."
        assert ResponseParser.sanitize_for_storage(text) == text


# ---------------------------------------------------------------------------
# store_interaction boundary guard
# ---------------------------------------------------------------------------

@pytest.fixture
def storage():
    from memory.memory_storage import MemoryStorage
    corpus_manager = MagicMock()
    chroma_store = MagicMock()
    chroma_store.add_conversation_memory.return_value = "doc-123"
    ms = MemoryStorage(
        corpus_manager=corpus_manager,
        chroma_store=chroma_store,
        fact_extractor=MagicMock(),
    )
    return ms, corpus_manager, chroma_store


class TestStoreInteractionGuard:
    @pytest.mark.asyncio
    async def test_empty_pair_stripped_before_persistence(self, storage):
        ms, corpus_manager, _ = storage
        await ms.store_interaction(
            query="how's it going",
            response="<thinking></thinking>All good over here.",
        )
        assert corpus_manager.add_entry.called
        stored_response = corpus_manager.add_entry.call_args[0][1]
        assert "<thinking>" not in stored_response
        assert stored_response.startswith("All good")

    @pytest.mark.asyncio
    async def test_all_thinking_response_skipped(self, storage):
        ms, corpus_manager, chroma_store = storage
        result = await ms.store_interaction(
            query="hey",
            response="<thinking>I should figure out what the user wants",
        )
        assert result is None
        assert not corpus_manager.add_entry.called
        assert not chroma_store.add_conversation_memory.called

    @pytest.mark.asyncio
    async def test_leading_block_stripped_answer_stored(self, storage):
        ms, corpus_manager, _ = storage
        await ms.store_interaction(
            query="q",
            response="<thinking>reasoning here</thinking>The actual answer survives.",
        )
        stored_response = corpus_manager.add_entry.call_args[0][1]
        assert stored_response == "The actual answer survives."

    @pytest.mark.asyncio
    async def test_clean_response_stored_unchanged(self, storage):
        ms, corpus_manager, _ = storage
        await ms.store_interaction(query="q", response="Plain answer.")
        assert corpus_manager.add_entry.call_args[0][1] == "Plain answer."


# ---------------------------------------------------------------------------
# repair script transform
# ---------------------------------------------------------------------------

class TestRepairScrub:
    @pytest.fixture(autouse=True)
    def _import_script(self):
        import importlib.util
        script = Path(__file__).resolve().parent.parent.parent / "scripts" / "repair_thinking_leaks.py"
        spec = importlib.util.spec_from_file_location("repair_thinking_leaks", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.scrub = mod.scrub_thinking

    def test_empty_pair(self):
        assert self.scrub("<thinking></thinking>Answer.") == "Answer."

    def test_full_block_mid_doc(self):
        doc = "User: hi\nAssistant: <thinking>\nreasoning\n</thinking>\nReal reply."
        out = self.scrub(doc)
        assert "<thinking>" not in out
        assert "reasoning" not in out
        assert "Real reply." in out

    def test_orphan_tag(self):
        assert self.scrub("</thinking>Reply text.") == "Reply text."

    def test_clean_text_untouched(self):
        text = "User: hello\nAssistant: A normal stored conversation."
        assert self.scrub(text) == text

    def test_collapses_leftover_blank_lines(self):
        doc = "Before.\n\n<thinking>x</thinking>\n\n\nAfter."
        out = self.scrub(doc)
        assert "\n\n\n" not in out
