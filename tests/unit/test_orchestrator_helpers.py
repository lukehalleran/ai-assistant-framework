"""
Unit tests for core/orchestrator.py helper functions and classes

Tests pure utility components:
- _parse_thinking_block static method
- _InMemoryCorpus fallback storage
- _SimplePromptBuilder fallback builder
- _get_tone_instructions tone mapping
"""

import pytest
from datetime import datetime
from core.orchestrator import (
    DaemonOrchestrator,
    _InMemoryCorpus,
    _SimplePromptBuilder,
    _FallbackMemoryCoordinator,
)
from utils.tone_detector import CrisisLevel


# =============================================================================
# _parse_thinking_block Tests
# =============================================================================

def test_parse_thinking_block_with_thinking():
    """Extracts thinking content and final answer"""
    response = "<thinking>Let me analyze this...</thinking>The answer is 42"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert thinking == "Let me analyze this..."
    assert answer == "The answer is 42"


def test_parse_thinking_block_no_thinking():
    """Returns empty thinking when no block present"""
    response = "This is just a regular answer"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert thinking == ""
    assert answer == "This is just a regular answer"


def test_parse_thinking_block_multiple_lines():
    """Handles multiline thinking blocks"""
    response = """<thinking>
    First step: analyze
    Second step: conclude
    </thinking>
    Final answer here"""
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert "First step" in thinking
    assert "Second step" in thinking
    assert answer.strip() == "Final answer here"


def test_parse_thinking_block_empty_thinking():
    """Handles empty thinking block"""
    response = "<thinking></thinking>Just the answer"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert thinking == ""
    assert answer == "Just the answer"


def test_parse_thinking_block_whitespace():
    """Strips whitespace from both parts"""
    response = "<thinking>  thought  </thinking>  answer  "
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert thinking == "thought"
    assert answer == "answer"


def test_parse_thinking_block_none_input():
    """Handles None input gracefully"""
    thinking, answer = DaemonOrchestrator._parse_thinking_block(None)

    assert thinking == ""
    assert answer == ""


def test_parse_thinking_block_empty_string():
    """Handles empty string input"""
    thinking, answer = DaemonOrchestrator._parse_thinking_block("")

    assert thinking == ""
    assert answer == ""


def test_parse_thinking_block_only_opening_tag():
    """Handles malformed block with only opening tag"""
    response = "<thinking>Some thought but no closing tag"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    # No </thinking> delimiter found, so returns full response as answer
    assert thinking == ""
    assert answer == response


def test_parse_thinking_block_no_opening_tag():
    """Handles malformed block with only closing tag"""
    response = "No opening tag</thinking>Answer"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    # Finds </thinking> delimiter, extracts before it
    assert "</thinking>" not in answer
    assert answer == "Answer"


def test_parse_thinking_block_multiple_closing_tags():
    """Only splits on first closing tag"""
    response = "<thinking>thought</thinking>answer</thinking>more"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert thinking == "thought"
    assert answer == "answer</thinking>more"


# =============================================================================
# _InMemoryCorpus Tests
# =============================================================================

def test_in_memory_corpus_initialization():
    """Initializes with empty entries"""
    corpus = _InMemoryCorpus()

    assert corpus._entries == []


def test_in_memory_corpus_add_entry_basic():
    """Adds entry with timestamp"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("query", "response")

    assert len(corpus._entries) == 1
    entry = corpus._entries[0]
    assert entry["query"] == "query"
    assert entry["response"] == "response"
    assert isinstance(entry["timestamp"], datetime)
    assert entry["tags"] == []


def test_in_memory_corpus_add_entry_with_tags():
    """Adds entry with tags"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("query", "response", tags=["tag1", "tag2"])

    entry = corpus._entries[0]
    assert entry["tags"] == ["tag1", "tag2"]


def test_in_memory_corpus_get_recent_memories():
    """Retrieves recent memories in order"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("q1", "r1")
    corpus.add_entry("q2", "r2")
    corpus.add_entry("q3", "r3")

    recent = corpus.get_recent_memories(limit=2)

    assert len(recent) == 2
    assert recent[0]["query"] == "q2"
    assert recent[1]["query"] == "q3"


def test_in_memory_corpus_get_recent_zero_limit():
    """Returns empty list for zero limit"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("q1", "r1")

    recent = corpus.get_recent_memories(limit=0)

    assert recent == []


def test_in_memory_corpus_get_recent_negative_limit():
    """Returns empty list for negative limit"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("q1", "r1")

    recent = corpus.get_recent_memories(limit=-1)

    assert recent == []


def test_in_memory_corpus_get_recent_exceeds_size():
    """Returns all entries when limit exceeds size"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("q1", "r1")
    corpus.add_entry("q2", "r2")

    recent = corpus.get_recent_memories(limit=10)

    assert len(recent) == 2


def test_in_memory_corpus_get_summaries():
    """Always returns empty list (not implemented)"""
    corpus = _InMemoryCorpus()
    corpus.add_entry("q1", "r1")

    # get_summaries takes positional arg, not keyword
    summaries = corpus.get_summaries(5)

    assert summaries == []


# =============================================================================
# _SimplePromptBuilder Tests
# =============================================================================

@pytest.mark.asyncio
async def test_simple_prompt_builder_basic():
    """Returns user input as-is"""
    builder = _SimplePromptBuilder()
    result = await builder.build_prompt("test input")

    assert result == "test input"


@pytest.mark.asyncio
async def test_simple_prompt_builder_empty():
    """Handles empty string"""
    builder = _SimplePromptBuilder()
    result = await builder.build_prompt("")

    assert result == ""


@pytest.mark.asyncio
async def test_simple_prompt_builder_none():
    """Handles None input"""
    builder = _SimplePromptBuilder()
    result = await builder.build_prompt(None)

    assert result == ""


@pytest.mark.asyncio
async def test_simple_prompt_builder_ignores_kwargs():
    """Ignores all keyword arguments"""
    builder = _SimplePromptBuilder()
    result = await builder.build_prompt(
        "input",
        system_prompt="ignored",
        topic="ignored",
        extra_arg="also ignored"
    )

    assert result == "input"


# =============================================================================
# _FallbackMemoryCoordinator Tests
# =============================================================================

@pytest.mark.asyncio
async def test_fallback_memory_initialization():
    """Initializes with in-memory corpus"""
    coordinator = _FallbackMemoryCoordinator()

    assert isinstance(coordinator.corpus_manager, _InMemoryCorpus)
    assert coordinator.gate_system is None


@pytest.mark.asyncio
async def test_fallback_memory_store_interaction():
    """Stores interaction in corpus"""
    coordinator = _FallbackMemoryCoordinator()
    await coordinator.store_interaction("query", "response", tags=["test"])

    entries = coordinator.corpus_manager._entries
    assert len(entries) == 1
    assert entries[0]["query"] == "query"
    assert entries[0]["response"] == "response"
    assert entries[0]["tags"] == ["test"]


@pytest.mark.asyncio
async def test_fallback_memory_get_memories():
    """Retrieves memories in reverse order"""
    coordinator = _FallbackMemoryCoordinator()
    await coordinator.store_interaction("q1", "r1")
    await coordinator.store_interaction("q2", "r2")
    await coordinator.store_interaction("q3", "r3")

    memories = await coordinator.get_memories("test query", limit=2)

    assert len(memories) == 2
    # Should be in reverse order (most recent first)
    assert memories[0]["query"] == "q3"
    assert memories[1]["query"] == "q2"


@pytest.mark.asyncio
async def test_fallback_memory_get_memories_metadata():
    """Includes source and score metadata"""
    coordinator = _FallbackMemoryCoordinator()
    await coordinator.store_interaction("query", "response")

    memories = await coordinator.get_memories("test", limit=10)

    assert len(memories) == 1
    assert memories[0]["metadata"]["source"] == "recent"
    assert memories[0]["metadata"]["final_score"] == 1.0


@pytest.mark.asyncio
async def test_fallback_memory_retrieve_relevant_memories():
    """Retrieves memories with counts"""
    coordinator = _FallbackMemoryCoordinator()
    await coordinator.store_interaction("q1", "r1")
    await coordinator.store_interaction("q2", "r2")

    result = await coordinator.retrieve_relevant_memories("test", config={"recent_count": 1})

    assert "memories" in result
    assert "counts" in result
    assert len(result["memories"]) == 1
    assert result["counts"]["recent"] == 1
    assert result["counts"]["semantic"] == 0
    assert result["counts"]["hierarchical"] == 0


@pytest.mark.asyncio
async def test_fallback_memory_retrieve_uses_config():
    """Uses config to determine limit"""
    coordinator = _FallbackMemoryCoordinator()
    for i in range(10):
        await coordinator.store_interaction(f"q{i}", f"r{i}")

    result = await coordinator.retrieve_relevant_memories("test", config={"recent_count": 3})

    assert len(result["memories"]) == 3


@pytest.mark.asyncio
async def test_fallback_memory_retrieve_default_limit():
    """Uses default limit of 5 when no config"""
    coordinator = _FallbackMemoryCoordinator()
    for i in range(10):
        await coordinator.store_interaction(f"q{i}", f"r{i}")

    result = await coordinator.retrieve_relevant_memories("test")

    assert len(result["memories"]) == 5


# =============================================================================
# _get_tone_instructions Tests (instance method)
# =============================================================================

class MockOrchestrator(DaemonOrchestrator):
    """Mock orchestrator with minimal dependencies for testing"""
    def __init__(self):
        self.logger = None


def test_get_tone_instructions_high():
    """Returns crisis support instructions"""
    orch = MockOrchestrator()
    instructions = orch._get_tone_instructions(CrisisLevel.HIGH)

    assert "CRISIS SUPPORT" in instructions
    assert "severe crisis" in instructions
    assert "therapeutic presence" in instructions
    assert "crisis lines" in instructions


def test_get_tone_instructions_medium():
    """Returns elevated support instructions"""
    orch = MockOrchestrator()
    instructions = orch._get_tone_instructions(CrisisLevel.MEDIUM)

    assert "ELEVATED SUPPORT" in instructions
    assert "acute distress" in instructions
    assert "2-3 paragraphs" in instructions
    assert "don't over-therapize" in instructions


def test_get_tone_instructions_concern():
    """Returns light support instructions"""
    orch = MockOrchestrator()
    instructions = orch._get_tone_instructions(CrisisLevel.CONCERN)

    assert "LIGHT SUPPORT" in instructions
    assert "concern, anxiety" in instructions
    assert "2-4 sentences" in instructions
    assert "That sucks" in instructions


def test_get_tone_instructions_conversational():
    """Returns conversational mode instructions"""
    orch = MockOrchestrator()
    instructions = orch._get_tone_instructions(CrisisLevel.CONVERSATIONAL)

    assert "CONVERSATIONAL" in instructions
    assert "casual conversation" in instructions
    assert "1-3 sentences" in instructions
    assert "No therapeutic language" in instructions


def test_get_tone_instructions_all_levels_formatted():
    """All tone levels return properly formatted strings"""
    orch = MockOrchestrator()

    for level in CrisisLevel:
        instructions = orch._get_tone_instructions(level)

        # Should start with newlines and header
        assert instructions.startswith("\n\n## RESPONSE MODE:")
        # Should contain "RESPONSE MODE:" header
        assert "RESPONSE MODE:" in instructions
        # Should be non-empty and contain mode guidance
        assert len(instructions) > 50


# =============================================================================
# Integration Tests
# =============================================================================

def test_thinking_block_with_tone_context():
    """Parse thinking block from response with tone-aware content"""
    response = "<thinking>User seems distressed, using ELEVATED_SUPPORT mode</thinking>I understand this is difficult."
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)

    assert "ELEVATED_SUPPORT" in thinking
    assert "difficult" in answer


@pytest.mark.asyncio
async def test_fallback_memory_full_workflow():
    """Complete workflow: store → retrieve → verify structure"""
    coordinator = _FallbackMemoryCoordinator()

    # Store multiple interactions
    await coordinator.store_interaction("What's 2+2?", "4", tags=["math"])
    await coordinator.store_interaction("Who won?", "Alice", tags=["game"])

    # Retrieve via get_memories
    memories = await coordinator.get_memories("test", limit=10)
    assert len(memories) == 2

    # Retrieve via retrieve_relevant_memories
    result = await coordinator.retrieve_relevant_memories("test")
    assert result["counts"]["recent"] == 2


def test_in_memory_corpus_ordering():
    """Verify LIFO ordering for recent memories"""
    corpus = _InMemoryCorpus()

    for i in range(5):
        corpus.add_entry(f"q{i}", f"r{i}")

    # Get last 3
    recent = corpus.get_recent_memories(limit=3)

    assert recent[0]["query"] == "q2"
    assert recent[1]["query"] == "q3"
    assert recent[2]["query"] == "q4"
