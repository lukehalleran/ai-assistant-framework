"""Unit tests for core/agentic/gate.py — 4-tier agentic gate evaluation.

Tests cover each tier (keyword, entity, doc/note intent, LLM fallback),
casual skip filter, continuation override, and intent-based veto.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.agentic.gate import evaluate_agentic_gate, AgenticDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_trigger_decision():
    """WebSearchDecision-like mock that triggers nothing."""
    return MagicMock(
        should_search=False,
        search_terms=[],
        needs_memory_search=False,
        needs_knowledge_search=False,
        needs_document_generation=False,
    )


# ===========================================================================
# Tier 1: Keyword heuristics
# ===========================================================================

class TestTier1Keywords:

    @pytest.mark.asyncio
    async def test_computation_keyword(self):
        d = await evaluate_agentic_gate("calculate fibonacci 10")
        assert d.should_trigger
        assert "computation" in d.modes

    @pytest.mark.asyncio
    async def test_memory_keyword(self):
        d = await evaluate_agentic_gate("do you remember my brother's name?")
        assert d.should_trigger
        assert "memory" in d.modes

    @pytest.mark.asyncio
    async def test_knowledge_keyword_4plus_words(self):
        d = await evaluate_agentic_gate("explain in depth how photosynthesis works")
        assert d.should_trigger
        assert "knowledge" in d.modes

    @pytest.mark.asyncio
    async def test_knowledge_keyword_needs_4_words(self):
        """Knowledge keywords don't fire for short queries."""
        d = await evaluate_agentic_gate("wiki python")
        # Only 2 words, knowledge should not trigger via keyword
        assert "knowledge" not in d.modes

    @pytest.mark.asyncio
    async def test_web_search_keyword(self):
        d = await evaluate_agentic_gate("search the web for python tutorials please")
        assert d.should_trigger
        assert "web_search" in d.modes

    @pytest.mark.asyncio
    async def test_tool_name_keyword(self):
        d = await evaluate_agentic_gate("show me github open issues for this repo")
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_url_triggers_web_search(self):
        d = await evaluate_agentic_gate("check out https://example.com for details")
        assert d.should_trigger
        assert "web_search" in d.modes


class TestFileAccessKeywords:
    """File / saved-document RETRIEVAL must route to agentic so the
    file_read / file_list / get_full_document tools are offered. Regression:
    these fell through to enhanced mode and the model confabulated
    "I don't have file access tools right now".
    """

    @pytest.mark.asyncio
    async def test_pull_and_print_full_document(self):
        """The exact reported request."""
        d = await evaluate_agentic_gate(
            "can you pull and print the full document for me"
        )
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_read_the_file(self):
        d = await evaluate_agentic_gate("read the file config/config.yaml please")
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_document_you_wrote(self):
        d = await evaluate_agentic_gate(
            "print the document you wrote yesterday about the plan"
        )
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_file_access_survives_intent_veto(self):
        """A casual_social-classified file request must NOT be vetoed —
        file access counts as an explicit request.
        """
        intent = MagicMock()
        intent.intent_type = MagicMock(value="casual_social")
        intent.confidence = 0.95
        d = await evaluate_agentic_gate(
            "yeah pull the document for me",
            intent_info=intent,
        )
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_inflected_verbs_and_intervening_words(self):
        """Regression from transcript: 'pulling up and printing the document'.
        Inflected verbs + words between verb and object must still trigger.
        """
        d = await evaluate_agentic_gate(
            "Can we verify the fix by pulling up and printing the document "
            "we have been discussing"
        )
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_capability_assertion_you_have_the_tool(self):
        """Regression from transcript: 'No. I mean can you pull it. You have the tool'."""
        d = await evaluate_agentic_gate("No. I mean can you pull it. You have the tool")
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_use_the_file_read_tool(self):
        d = await evaluate_agentic_gate("use the file_read tool to grab it")
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_pronoun_retrieval_after_document_turn(self):
        """Terse 'pull it up' routes to tools when the previous turn was about
        a saved document (pronoun pattern is gated on prior file/doc context).
        """
        corpus = MagicMock()
        corpus.get_recent_memories.return_value = [
            {"query": "pull the implementation plan document",
             "response": "Here is the document I saved to disk..."},
        ]
        d = await evaluate_agentic_gate("can you pull it up", corpus_manager=corpus)
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_pronoun_retrieval_without_doc_context_does_not_trigger(self):
        """'pull it together' (motivational) must NOT trigger — no prior doc turn."""
        corpus = MagicMock()
        corpus.get_recent_memories.return_value = [
            {"query": "I'm feeling overwhelmed today", "response": "That's rough, hang in there."},
        ]
        d = await evaluate_agentic_gate(
            "lets pull it together",
            corpus_manager=corpus,
            model_manager=None,
        )
        assert "tools" not in d.modes

    @pytest.mark.asyncio
    async def test_affirmation_after_file_offer_triggers(self):
        """'yes please' right after the model OFFERED to pull a file routes to
        tools — this makes the enhanced-mode honesty offer get carried out.
        """
        corpus = MagicMock()
        corpus.get_recent_memories.return_value = [
            {"query": "verify the save",
             "response": "I can't read files this turn. Want me to pull that up next turn?"},
        ]
        d = await evaluate_agentic_gate("yes please", corpus_manager=corpus)
        assert d.should_trigger
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_affirmation_without_file_offer_does_not_trigger(self):
        """A bare 'yeah' after a turn that did NOT offer a file must NOT trigger."""
        corpus = MagicMock()
        corpus.get_recent_memories.return_value = [
            {"query": "how are you", "response": "Doing well, thanks for asking!"},
        ]
        d = await evaluate_agentic_gate("yeah", corpus_manager=corpus, model_manager=None)
        assert "tools" not in d.modes


# ===========================================================================
# Tier 2: Entity match
# ===========================================================================

class TestTier2Entity:

    @pytest.mark.asyncio
    async def test_entity_with_recall_signal(self):
        resolver = MagicMock()
        with patch(
            "memory.graph_utils.extract_graph_entities",
            return_value={"flapjack"},
        ):
            d = await evaluate_agentic_gate(
                "what do you know about Flapjack?",
                entity_resolver=resolver,
            )
        assert d.should_trigger
        assert "memory" in d.modes
        assert "flapjack" in d.matched_entities

    @pytest.mark.asyncio
    async def test_entity_without_recall_no_trigger(self):
        resolver = MagicMock()
        with patch(
            "memory.graph_utils.extract_graph_entities",
            return_value={"flapjack"},
        ):
            d = await evaluate_agentic_gate(
                "Flapjack is really cute today",
                entity_resolver=resolver,
            )
        # No '?' and no recall signal words → entity match alone is not enough
        assert "memory" not in d.modes


# ===========================================================================
# Casual skip filter
# ===========================================================================

class TestCasualSkip:

    @pytest.mark.asyncio
    async def test_short_no_signal_skips(self):
        d = await evaluate_agentic_gate("ok cool")
        assert not d.should_trigger
        assert "casual" in d.reason

    @pytest.mark.asyncio
    async def test_casual_starter_skips(self):
        d = await evaluate_agentic_gate("thanks for that info")
        assert not d.should_trigger

    @pytest.mark.asyncio
    async def test_all_filler_skips(self):
        d = await evaluate_agentic_gate("yes ok sure")
        assert not d.should_trigger


# ===========================================================================
# Continuation override
# ===========================================================================

class TestContinuationOverride:

    @pytest.mark.asyncio
    async def test_continuation_inherits_agentic(self):
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {"query": "search for python tutorials?", "response": "Let me search for that..."},
        ])
        # Continuation override prevents casual skip, then LLM fallback fires
        mm = MagicMock()
        llm_decision = MagicMock(
            should_search=True,
            search_terms=["python tutorials"],
            needs_memory_search=False,
            needs_knowledge_search=False,
            needs_document_generation=False,
        )
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock,
            return_value=llm_decision,
        ):
            d = await evaluate_agentic_gate(
                "yes please",
                corpus_manager=corpus,
                model_manager=mm,
            )
        # "yes please" is normally casual-skipped, but previous turn had
        # agentic signals → override skip → LLM fallback triggers
        assert d.should_trigger

    @pytest.mark.asyncio
    async def test_no_continuation_without_prev_signals(self):
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {"query": "hello there", "response": "Hi! How are you?"},
        ])
        d = await evaluate_agentic_gate(
            "yes please",
            corpus_manager=corpus,
        )
        # Previous turn had no agentic signals → stays skipped
        assert not d.should_trigger


# ===========================================================================
# Intent-based veto
# ===========================================================================

class TestIntentVeto:

    @pytest.mark.asyncio
    async def test_meta_conversational_vetoes(self):
        """High-confidence meta_conversational intent vetoes agentic trigger."""
        intent = {"intent_type": "meta_conversational", "confidence": 0.85}
        # Use a memory keyword to trigger, then veto via intent
        d = await evaluate_agentic_gate(
            "do you remember anything about our conversations?",
            intent_info=intent,
        )
        assert not d.should_trigger

    @pytest.mark.asyncio
    async def test_veto_suppressed_by_explicit_search(self):
        """Explicit search keyword prevents intent veto."""
        intent = {"intent_type": "meta_conversational", "confidence": 0.85}
        d = await evaluate_agentic_gate(
            "search your memory for our past conversations",
            intent_info=intent,
        )
        # "search" is in EXPLICIT_SEARCH_KEYWORDS, so veto is suppressed
        assert d.should_trigger


# ===========================================================================
# Tier 4: LLM fallback
# ===========================================================================

class TestTier4LLMFallback:

    @pytest.mark.asyncio
    async def test_llm_memory_search(self):
        mm = MagicMock()
        decision = MagicMock(
            should_search=True,
            search_terms=["test"],
            needs_memory_search=True,
            needs_knowledge_search=False,
            needs_document_generation=False,
        )
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock,
            return_value=decision,
        ):
            # Query that doesn't hit any keyword but is long enough
            d = await evaluate_agentic_gate(
                "Can you describe the color of my cat's fur exactly?",
                model_manager=mm,
            )
        assert d.should_trigger
        assert "memory" in d.modes

    @pytest.mark.asyncio
    async def test_llm_knowledge_search(self):
        mm = MagicMock()
        decision = MagicMock(
            should_search=False,
            search_terms=[],
            needs_memory_search=False,
            needs_knowledge_search=True,
            needs_document_generation=False,
        )
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock,
            return_value=decision,
        ):
            d = await evaluate_agentic_gate(
                "Can you describe the color of my cat's fur exactly?",
                model_manager=mm,
            )
        assert d.should_trigger
        assert "knowledge" in d.modes

    @pytest.mark.asyncio
    async def test_llm_web_search(self):
        mm = MagicMock()
        decision = MagicMock(
            should_search=True,
            search_terms=["python tutorials 2026"],
            needs_memory_search=False,
            needs_knowledge_search=False,
            needs_document_generation=False,
        )
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock,
            return_value=decision,
        ):
            d = await evaluate_agentic_gate(
                "Can you describe the color of my cat's fur exactly?",
                model_manager=mm,
            )
        assert d.should_trigger
        assert d.search_terms == ["python tutorials 2026"]


# ===========================================================================
# Special intents
# ===========================================================================

class TestSpecialIntents:

    @pytest.mark.asyncio
    async def test_doc_gen_intent(self):
        with patch(
            "knowledge.document_generator.detect_document_intent",
            return_value={"topic": "AI overview", "doc_type": "report", "focus": None},
        ):
            d = await evaluate_agentic_gate("write a report about recent AI developments")
        assert d.should_trigger
        assert d.doc_gen_intent is not None
        assert d.doc_gen_intent["topic"] == "AI overview"

    @pytest.mark.asyncio
    async def test_self_note_intent(self):
        with patch(
            "knowledge.daemon_notes_manager.detect_self_note_intent",
            return_value={"topic": "architecture decisions", "category": "architecture"},
        ):
            d = await evaluate_agentic_gate("save a note about the architecture decisions we made")
        assert d.should_trigger
        assert d.self_note_intent is not None
        assert d.self_note_intent["category"] == "architecture"


# ===========================================================================
# Combined triggers
# ===========================================================================

class TestCombined:

    @pytest.mark.asyncio
    async def test_knowledge_and_entity(self):
        """Both knowledge keyword and entity match can fire together."""
        resolver = MagicMock()
        with patch(
            "memory.graph_utils.extract_graph_entities",
            return_value={"flapjack"},
        ):
            d = await evaluate_agentic_gate(
                "tell me about the history of Flapjack and his background?",
                entity_resolver=resolver,
            )
        assert d.should_trigger
        # Knowledge should fire ("tell me about " keyword + 4+ words)
        assert "knowledge" in d.modes
        # Entity match + recall signal (?) should add memory
        assert "memory" in d.modes


# ===========================================================================
# skip_initial_search
# ===========================================================================

class TestSkipInitialSearch:

    @pytest.mark.asyncio
    async def test_computation_skips_initial(self):
        d = await evaluate_agentic_gate("compute the integral of x squared")
        assert d.skip_initial_search

    @pytest.mark.asyncio
    async def test_web_search_with_terms_does_not_skip(self):
        mm = MagicMock()
        decision = MagicMock(
            should_search=True,
            search_terms=["python 3.12 release"],
            needs_memory_search=False,
            needs_knowledge_search=False,
            needs_document_generation=False,
        )
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock,
            return_value=decision,
        ):
            d = await evaluate_agentic_gate(
                "Can you describe the color of my cat's fur exactly?",
                model_manager=mm,
            )
        # Web search WITH search terms should not skip initial search
        assert not d.skip_initial_search
