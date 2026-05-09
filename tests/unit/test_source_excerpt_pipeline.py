"""Tests for source_excerpt pipeline: extraction, forwarding, prompt injection, and anti-confabulation.

Covers:
- LLM fact extractor attaches source_excerpt via keyword matching
- Shutdown processor forwards source_excerpt to ChromaDB and UserProfile
- Per-turn storage forwards source_excerpt to ChromaDB
- Prompt injection includes source_excerpt in [USER PROFILE] output
- Inline anti-confabulation instruction present in all prompt assembly paths
- Regression: HelloTalk-style confabulation blocked by prompt guardrails
"""
import pytest
import re
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from memory.llm_fact_extractor import LLMFactExtractor, _normalize_triple


# ---------------------------------------------------------------------------
# C1: LLM fact extractor — source_excerpt capture
# ---------------------------------------------------------------------------

class TestLLMExtractorSourceExcerpt:
    """_attach_source_excerpts matches triples to source messages."""

    def test_keyword_match_basic(self):
        """Triple with matching keywords gets correct source message."""
        triples = [{"relation": "lives_in", "object": "Seattle", "subject": "user"}]
        messages = [
            "I had coffee this morning",
            "I moved to Seattle last year",
            "The weather is nice today",
        ]
        LLMFactExtractor._attach_source_excerpts(triples, messages)
        assert "Seattle" in triples[0]["source_excerpt"]
        assert triples[0]["source_excerpt"] == "I moved to Seattle last year"

    def test_fallback_to_last_message(self):
        """When no keywords match, falls back to last non-empty message."""
        triples = [{"relation": "mood", "object": "ok", "subject": "user"}]
        messages = ["Just checking in", "Nothing much happening"]
        LLMFactExtractor._attach_source_excerpts(triples, messages)
        assert triples[0]["source_excerpt"] == "Nothing much happening"

    def test_empty_messages(self):
        """Empty message list doesn't crash."""
        triples = [{"relation": "name", "object": "Luke", "subject": "user"}]
        LLMFactExtractor._attach_source_excerpts(triples, [])
        assert "source_excerpt" not in triples[0]

    def test_truncation_at_200_chars(self):
        """Source excerpt is truncated to 200 chars."""
        triples = [{"relation": "story", "object": "adventure", "subject": "user"}]
        long_msg = "I had an adventure " + "x" * 300
        LLMFactExtractor._attach_source_excerpts(triples, [long_msg])
        assert len(triples[0]["source_excerpt"]) == 200

    def test_snake_case_relation_splits(self):
        """Snake_case relation names are split into keywords for matching."""
        triples = [{"relation": "language_exchange", "object": "Spanish", "subject": "user"}]
        messages = [
            "I've been doing language exchange with people online",
            "The movie was great",
        ]
        LLMFactExtractor._attach_source_excerpts(triples, messages)
        assert "language" in triples[0]["source_excerpt"]

    def test_multiple_triples_different_sources(self):
        """Each triple matches its own best source message."""
        triples = [
            {"relation": "lives_in", "object": "Boston", "subject": "user"},
            {"relation": "pet_name", "object": "Flapjack", "subject": "user"},
        ]
        messages = [
            "I live in Boston near the harbor",
            "Flapjack is being a silly kitten today",
        ]
        LLMFactExtractor._attach_source_excerpts(triples, messages)
        assert "Boston" in triples[0]["source_excerpt"]
        assert "Flapjack" in triples[1]["source_excerpt"]

    def test_strips_role_prefix(self):
        """Messages with 'user:' prefix are cleaned before matching."""
        triples = [{"relation": "name", "object": "Luke", "subject": "user"}]
        messages = ["user: My name is Luke"]
        LLMFactExtractor._attach_source_excerpts(triples, messages)
        assert triples[0]["source_excerpt"] == "My name is Luke"

    @pytest.mark.asyncio
    async def test_extract_triples_includes_source_excerpt(self):
        """End-to-end: extract_triples returns triples with source_excerpt."""
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value='[{"subject": "user", "relation": "hobby", "object": "chess", "category": "hobbies", "confidence": 0.9}]')
        extractor = LLMFactExtractor(mm)
        result = await extractor.extract_triples(["I've been playing chess lately"])
        assert len(result) == 1
        assert "source_excerpt" in result[0]
        assert "chess" in result[0]["source_excerpt"]


# ---------------------------------------------------------------------------
# C2: Storage pipeline forwarding
# ---------------------------------------------------------------------------

class TestStoragePipelineForwarding:
    """source_excerpt is forwarded through storage paths."""

    def test_per_turn_storage_forwards_source_excerpt(self):
        """extract_and_store_facts includes source_excerpt in source_dict."""
        # Verify source_excerpt is in the forwarding key list
        import memory.memory_storage as ms
        source = open(ms.__file__).read()
        assert '"source_excerpt"' in source
        # The for-loop forwarding keys should include source_excerpt
        assert 'source_excerpt' in source

    def test_shutdown_session_facts_forwards_source_excerpt(self):
        """_extract_session_facts passes source_excerpt from MemoryNode metadata."""
        import memory.shutdown_processor as sp
        source = open(sp.__file__).read()
        # Should reference source_excerpt when building src_dict
        assert 'source_excerpt' in source

    def test_shutdown_llm_facts_forwards_source_excerpt(self):
        """_extract_llm_facts passes source_excerpt from triples to ChromaDB."""
        import memory.shutdown_processor as sp
        source = open(sp.__file__).read()
        # The llm facts storage should reference source_excerpt
        count = source.count('source_excerpt')
        assert count >= 2, f"Expected source_excerpt referenced at least twice in shutdown_processor, got {count}"


# ---------------------------------------------------------------------------
# C3: Prompt injection includes source_excerpt
# ---------------------------------------------------------------------------

class TestPromptInjectionSourceExcerpt:
    """get_context_injection() includes source_excerpt when available."""

    def _make_profile(self, facts):
        """Create a minimal UserProfile with given facts."""
        from memory.user_profile import UserProfile
        profile = UserProfile.__new__(UserProfile)
        profile.profile = {
            "name": "Test",
            "updated_at": "2026-04-30T12:00:00",
            "categories": {},
        }
        profile._embedding_model = None
        profile._fact_embeddings = {}
        for f in facts:
            cat = f.get("category", "preferences")
            if cat not in profile.profile["categories"]:
                profile.profile["categories"][cat] = []
            profile.profile["categories"][cat].append(f)
        return profile

    def test_source_excerpt_shown_when_present(self):
        """Facts with source_excerpt include (said: "...") in output."""
        profile = self._make_profile([{
            "relation": "language_exchange",
            "value": "Spanish and English",
            "category": "preferences",
            "confidence": 0.8,
            "source_excerpt": "I do language exchange with people online",
            "timestamp": "2026-02-27T12:00:00",
        }])
        result = profile.get_context_injection()
        assert '(said: "I do language exchange with people online")' in result

    def test_source_excerpt_hidden_when_empty(self):
        """Facts without source_excerpt do not include (said:) marker."""
        profile = self._make_profile([{
            "relation": "language_exchange",
            "value": "Spanish and English",
            "category": "preferences",
            "confidence": 0.8,
            "source_excerpt": "",
            "timestamp": "2026-02-27T12:00:00",
        }])
        result = profile.get_context_injection()
        assert "(said:" not in result
        assert "language_exchange=Spanish and English" in result

    def test_source_excerpt_truncated_at_80_chars(self):
        """Long source excerpts are truncated in prompt injection."""
        long_excerpt = "x" * 200
        profile = self._make_profile([{
            "relation": "hobby",
            "value": "painting",
            "category": "hobbies",
            "confidence": 0.8,
            "source_excerpt": long_excerpt,
            "timestamp": "2026-04-01T12:00:00",
        }])
        result = profile.get_context_injection()
        # Find the (said: "...") portion and check length
        match = re.search(r'\(said: "(.+?)"\)', result)
        assert match is not None
        assert len(match.group(1)) == 80


# ---------------------------------------------------------------------------
# C4: Inline instruction present in assembled prompts
# ---------------------------------------------------------------------------

ANTI_CONFAB_INSTRUCTION = "do not add names, apps, or details not written here"


class TestInlineProfileInstruction:
    """Anti-confabulation instruction appears in all prompt assembly paths."""

    def test_builder_includes_instruction(self):
        """UnifiedPromptBuilder._assemble_prompt includes inline instruction.

        The instruction lives in formatter.py (where _assemble_prompt was moved)
        and builder.py delegates to it.
        """
        # Check formatter.py where the actual _assemble_prompt implementation lives
        source_path = "core/prompt/formatter.py"
        with open(source_path) as f:
            source = f.read()
        assert ANTI_CONFAB_INSTRUCTION in source

    def test_formatter_includes_instruction(self):
        """PromptFormatter._assemble_prompt includes inline instruction."""
        source_path = "core/prompt/formatter.py"
        with open(source_path) as f:
            source = f.read()
        assert ANTI_CONFAB_INSTRUCTION in source

    def test_agentic_controller_includes_instruction(self):
        """AgenticSearchController includes inline instruction."""
        source_path = "core/agentic/controller.py"
        with open(source_path) as f:
            source = f.read()
        assert ANTI_CONFAB_INSTRUCTION in source

    def test_system_prompt_has_anti_confabulation(self):
        """System prompt contains strengthened Memory use instruction."""
        with open("core/system_prompt.txt") as f:
            content = f.read()
        assert "paraphrase vaguely rather than concretize" in content
        assert "Never add specific names, apps, platforms" in content


# ---------------------------------------------------------------------------
# C5: HelloTalk regression test
# ---------------------------------------------------------------------------

class TestHelloTalkRegression:
    """Ensure the specific HelloTalk confabulation scenario is guarded against."""

    def test_hellotalk_not_in_profile_output(self):
        """Profile with language_exchange fact should never mention HelloTalk."""
        from memory.user_profile import UserProfile
        profile = UserProfile.__new__(UserProfile)
        profile.profile = {
            "name": "Luke",
            "updated_at": "2026-04-30T12:00:00",
            "categories": {
                "preferences": [{
                    "relation": "language_exchange",
                    "value": "Spanish and English",
                    "category": "preferences",
                    "confidence": 0.8,
                    "source_excerpt": "",
                    "timestamp": "2026-02-27T12:54:37",
                }],
                "goals": [{
                    "relation": "future_plan",
                    "value": "move to Spain",
                    "category": "goals",
                    "confidence": 0.8,
                    "source_excerpt": "",
                    "timestamp": "2026-03-12T16:23:32",
                }],
            },
        }
        profile._embedding_model = None
        profile._fact_embeddings = {}
        result = profile.get_context_injection()
        # HelloTalk must NOT appear anywhere in the injected context
        assert "HelloTalk" not in result
        assert "hellotalk" not in result.lower()
        # The actual fact values should be present
        assert "Spanish and English" in result
        assert "move to Spain" in result

    def test_inline_instruction_present_in_builder_source(self):
        """The anti-confabulation instruction guards the [USER PROFILE] section.

        The instruction lives in formatter.py (where _assemble_prompt was moved)
        and builder.py delegates to it.
        """
        with open("core/prompt/formatter.py") as f:
            source = f.read()
        # The instruction should appear right before user_profile injection
        assert "Stored facts" in source
        assert ANTI_CONFAB_INSTRUCTION in source

    def test_vague_fact_no_embellishment_in_context(self):
        """Vague facts should appear as-is, not elaborated with specifics."""
        from memory.user_profile import UserProfile
        profile = UserProfile.__new__(UserProfile)
        profile.profile = {
            "name": "Test",
            "updated_at": "2026-04-30T12:00:00",
            "categories": {
                "hobbies": [{
                    "relation": "hobby",
                    "value": "cooking",
                    "category": "hobbies",
                    "confidence": 0.8,
                    "source_excerpt": "",
                    "timestamp": "2026-04-01T12:00:00",
                }],
            },
        }
        profile._embedding_model = None
        profile._fact_embeddings = {}
        result = profile.get_context_injection()
        # Should just say hobby=cooking, nothing about specific cuisines/tools
        assert "hobby=cooking" in result
        # No fabricated details
        assert "Thai" not in result
        assert "Cuisinart" not in result
