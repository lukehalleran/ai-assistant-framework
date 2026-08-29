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
    async def test_what_is_bigram_does_not_trigger_tools(self):
        """'what is' mid-sentence is not a tool signal (2026-07-16: 'I will
        see what is said on Reddit…' matched the bare 'what is ' keyword and
        fired irrelevant web searches)."""
        d = await evaluate_agentic_gate(
            "I will see what is said on Reddit I don't wanna hear that guy talk"
        )
        assert "tools" not in d.modes
        assert not d.should_trigger

    @pytest.mark.asyncio
    async def test_contact_lookup_by_possessive_still_triggers(self):
        """Removing 'what is ' must not lose 'what is <name>'s email' coverage."""
        d = await evaluate_agentic_gate("what is Meagan's email")
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
            return_value={"biscuit"},
        ):
            d = await evaluate_agentic_gate(
                "what do you know about Biscuit?",
                entity_resolver=resolver,
            )
        assert d.should_trigger
        assert "memory" in d.modes
        assert "biscuit" in d.matched_entities

    @pytest.mark.asyncio
    async def test_entity_without_recall_no_trigger(self):
        resolver = MagicMock()
        with patch(
            "memory.graph_utils.extract_graph_entities",
            return_value={"biscuit"},
        ):
            d = await evaluate_agentic_gate(
                "Biscuit is really cute today",
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

    @pytest.mark.asyncio
    async def test_long_casual_sentence_is_not_continuation(self):
        """Regression (2026-07-15): an 11-word vibe remark starting with 'yeah'
        matched 'yeah' as a substring, and 'issues' in the prior query ("sleep
        issues") looked like a GitHub signal — overriding the casual skip and
        burning a 60s agentic loop + web credits on a conversational statement.
        A continuation must be terse; this message stays skipped and the LLM
        fallback is never consulted.
        """
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {
                "query": ("I wish benzos were not horrifically addictive and "
                          "life destroying I feel like they totally fix my "
                          "sleep issues"),
                "response": "That tradeoff is real...",
            },
        ])
        mock_llm = AsyncMock(return_value=MagicMock(
            should_search=True,
            search_terms=["benzodiazepine addiction 2026"],
            needs_memory_search=False,
            needs_knowledge_search=False,
            needs_document_generation=False,
        ))
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new=mock_llm,
        ):
            d = await evaluate_agentic_gate(
                "Yeah they seem like the worst drug to get addicted to",
                corpus_manager=corpus,
                model_manager=MagicMock(),
            )
        assert not d.should_trigger
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_stored_agentic_mode_enables_continuation(self):
        """response_mode='agentic-search' on the previous corpus entry is the
        ground-truth continuation signal — no keyword inference needed."""
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {
                "query": "how did my sleep trend this month",
                "response": "Pulled the data...",
                "response_mode": "agentic-search",
            },
        ])
        llm_decision = MagicMock(
            should_search=True,
            search_terms=["sleep trends"],
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
                model_manager=MagicMock(),
            )
        assert d.should_trigger

    @pytest.mark.asyncio
    async def test_stored_enhanced_mode_blocks_keyword_inference(self):
        """When response_mode is present and NOT agentic, tool-flavored words
        in the previous query ('issues', 'search') must not resurrect the
        override — ground truth wins over inference."""
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {
                "query": "search results for my sleep issues on github were funny",
                "response": "Ha, yes.",
                "response_mode": "enhanced",
            },
        ])
        d = await evaluate_agentic_gate(
            "yes please",
            corpus_manager=corpus,
        )
        assert not d.should_trigger

    @pytest.mark.asyncio
    async def test_legacy_entry_ambiguous_word_does_not_override(self):
        """Legacy entries (no response_mode) fall back to word-boundary
        inference — 'issues' inside 'sleep issues' no longer counts."""
        corpus = MagicMock()
        corpus.get_recent_memories = MagicMock(return_value=[
            {
                "query": "they totally fix my sleep issues",
                "response": "They do work fast.",
            },
        ])
        d = await evaluate_agentic_gate(
            "yes please",
            corpus_manager=corpus,
        )
        assert not d.should_trigger


# ===========================================================================
# Post-hoc intent veto (gate run concurrently with the context pipeline)
# ===========================================================================

class TestApplyIntentVetoPostHoc:

    @pytest.mark.asyncio
    async def test_posthoc_veto_suppresses_trigger(self):
        """A decision made without intent_info can be vetoed afterwards."""
        from core.agentic.gate import apply_intent_veto
        d = await evaluate_agentic_gate(
            "do you remember anything about our conversations?"
        )
        assert d.should_trigger
        assert d.veto_exempt is False
        d2 = apply_intent_veto(
            d, {"intent_type": "meta_conversational", "confidence": 0.85}
        )
        assert not d2.should_trigger
        assert d2.reason.startswith("intent-veto")

    @pytest.mark.asyncio
    async def test_posthoc_veto_respects_exemption(self):
        """Explicit search requests carry veto_exempt and are never suppressed."""
        from core.agentic.gate import apply_intent_veto
        d = await evaluate_agentic_gate("search for python tutorials")
        assert d.should_trigger
        assert d.veto_exempt is True
        d2 = apply_intent_veto(
            d, {"intent_type": "meta_conversational", "confidence": 0.95}
        )
        assert d2.should_trigger

    def test_posthoc_veto_noop_without_intent(self):
        from core.agentic.gate import apply_intent_veto
        d = AgenticDecision(should_trigger=True)
        assert apply_intent_veto(d, None).should_trigger is True


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

    @pytest.mark.asyncio
    async def test_llm_knowledge_search_suppressed_for_task_followup(self):
        """Incident 2026-07-24: the LLM trigger set needs_knowledge_search=True on a
        collaborative-task follow-up ("...ats friendly format"), routing it into the
        agentic knowledge loop where a slow model hung the turn ~2 min. With no
        knowledge-QUESTION signal in the query, the gate must suppress it."""
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
                "Yeah and I will need both in my current format but also them "
                "in ats friendly format",
                model_manager=mm,
            )
        assert "knowledge" not in d.modes
        assert not d.should_trigger

    @pytest.mark.asyncio
    async def test_llm_knowledge_search_honored_with_question_signal(self):
        """The suppression must NOT block a genuine knowledge question — one with an
        explicit explain/how/what signal but no keyword (so it reaches Tier 4)."""
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
                "explain the practical tradeoffs between optimistic and pessimistic "
                "locking in a busy multi user database system",
                model_manager=mm,
            )
        assert d.should_trigger
        assert "knowledge" in d.modes


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
            return_value={"biscuit"},
        ):
            d = await evaluate_agentic_gate(
                "tell me about the history of Biscuit and his background?",
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


# ===========================================================================
# Real-IntentResult regression (2026-07-03)
# ===========================================================================
# The veto reads intent_info.intent_type. IntentResult's field is `.intent`;
# before the intent_type alias property existed, the real object silently
# yielded None here and the veto NEVER fired in production (mocks/dicts in
# the tests above have intent_type, so they kept passing). These tests pin
# the veto to the actual production object.

class TestIntentVetoWithRealIntentResult:

    @pytest.mark.asyncio
    async def test_real_intent_result_vetoes(self):
        from core.intent_classifier import IntentClassifier, IntentType
        result = IntentClassifier().classify("what do you know about my projects?")
        assert result.intent == IntentType.META_CONVERSATIONAL
        assert result.confidence >= 0.75
        d = await evaluate_agentic_gate(
            "do you remember anything about our conversations?",
            intent_info=result,
        )
        assert not d.should_trigger
        assert d.reason.startswith("intent-veto:")
        assert "meta_conversational" in d.reason

    @pytest.mark.asyncio
    async def test_intent_type_alias_matches_intent(self):
        from core.intent_classifier import IntentClassifier
        result = IntentClassifier().classify("hey")
        assert result.intent_type == result.intent
        # The exact extraction expression used by builder.py / gate.py:
        _it = getattr(result, "intent_type", None)
        assert getattr(_it, "value", None) == "casual_social"
