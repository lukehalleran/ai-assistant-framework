"""
Integration tests for ContextPipeline with Orchestrator.

Tests the integration between the new ContextPipeline and the existing
orchestrator infrastructure.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from core.context_pipeline import ContextPipeline, ContextResult, ToneLevel


# --- Fixtures ---

@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = Mock()
    manager.generate_once = AsyncMock(return_value="rewritten query")
    return manager


@pytest.fixture
def mock_topic_manager():
    """Create a mock topic manager."""
    manager = Mock()
    manager.update_from_user_input = Mock()
    manager.get_primary_topic = Mock(return_value="python")
    manager.get_entities = Mock(return_value=["debugging"])
    return manager


@pytest.fixture
def mock_memory_coordinator():
    """Create a mock memory coordinator."""
    coordinator = Mock()
    corpus_manager = Mock()
    corpus_manager.get_recent_memories = Mock(return_value=[
        {"role": "user", "content": "Previous question", "timestamp": "2024-01-01T12:00:00"},
        {"role": "assistant", "content": "Previous answer", "timestamp": "2024-01-01T12:01:00"},
    ])
    coordinator.corpus_manager = corpus_manager
    coordinator.get_thread_context = Mock(return_value={
        "thread_id": "thread-123",
        "thread_depth": 2,
    })
    coordinator.get_memories = AsyncMock(return_value=[])
    return coordinator


@pytest.fixture
def mock_prompt_builder():
    """Create a mock prompt builder."""
    builder = Mock()
    builder.build_prompt = AsyncMock(return_value={
        "recent_conversations": [],
        "memories": [],
        "user_profile": "",
        "summaries": [],
        "reflections": [],
        "dreams": [],
        "semantic_chunks": [],
        "wiki": [],
        "memory_id_map": {},
    })
    builder._assemble_prompt = Mock(return_value="Assembled prompt text")
    return builder


@pytest.fixture
def mock_response_generator():
    """Create a mock response generator."""
    generator = Mock()
    generator.time_manager = Mock()
    return generator


@pytest.fixture
def mock_file_processor():
    """Create a mock file processor."""
    processor = Mock()
    processor.process_files = AsyncMock(return_value="user input with file content")
    return processor


# --- Integration Tests ---

class TestContextPipelineOrchestratorIntegration:
    """Test integration between ContextPipeline and Orchestrator."""

    @pytest.mark.asyncio
    async def test_build_context_method_available(
        self,
        mock_model_manager,
        mock_topic_manager,
        mock_memory_coordinator,
        mock_prompt_builder,
        mock_response_generator,
        mock_file_processor,
    ):
        """Test that build_context method is available on orchestrator."""
        from core.orchestrator import DaemonOrchestrator

        orchestrator = DaemonOrchestrator(
            model_manager=mock_model_manager,
            response_generator=mock_response_generator,
            file_processor=mock_file_processor,
            prompt_builder=mock_prompt_builder,
            memory_system=mock_memory_coordinator,
            topic_manager=mock_topic_manager,
        )

        # Verify build_context method exists
        assert hasattr(orchestrator, 'build_context')
        assert callable(orchestrator.build_context)

    @pytest.mark.asyncio
    async def test_build_context_returns_context_result(
        self,
        mock_model_manager,
        mock_topic_manager,
        mock_memory_coordinator,
        mock_prompt_builder,
        mock_response_generator,
        mock_file_processor,
    ):
        """Test that build_context returns a ContextResult."""
        from core.orchestrator import DaemonOrchestrator

        orchestrator = DaemonOrchestrator(
            model_manager=mock_model_manager,
            response_generator=mock_response_generator,
            file_processor=mock_file_processor,
            prompt_builder=mock_prompt_builder,
            memory_system=mock_memory_coordinator,
            topic_manager=mock_topic_manager,
        )

        # Patch tone detection to avoid external calls
        with patch('utils.emotional_context.analyze_emotional_context', new_callable=AsyncMock) as mock_tone:
            mock_tone.return_value = None

            context = await orchestrator.build_context("How do I debug Python?")

        assert isinstance(context, ContextResult)
        assert context.original_query == "How do I debug Python?"

    @pytest.mark.asyncio
    async def test_context_pipeline_syncs_state_with_orchestrator(
        self,
        mock_model_manager,
        mock_topic_manager,
        mock_memory_coordinator,
        mock_prompt_builder,
        mock_response_generator,
        mock_file_processor,
    ):
        """Test that ContextPipeline syncs state back to orchestrator."""
        from core.orchestrator import DaemonOrchestrator

        orchestrator = DaemonOrchestrator(
            model_manager=mock_model_manager,
            response_generator=mock_response_generator,
            file_processor=mock_file_processor,
            prompt_builder=mock_prompt_builder,
            memory_system=mock_memory_coordinator,
            topic_manager=mock_topic_manager,
        )

        # Set up topic manager to return a specific topic
        mock_topic_manager.get_primary_topic.return_value = "async_programming"

        with patch('utils.emotional_context.analyze_emotional_context', new_callable=AsyncMock) as mock_tone:
            mock_tone.return_value = None

            await orchestrator.build_context("Tell me about async programming")

        # Verify orchestrator state was updated
        assert orchestrator.current_topic == "async_programming"


class TestContextPipelinePromptBuilderIntegration:
    """Test integration between ContextPipeline and PromptBuilder."""

    @pytest.mark.asyncio
    async def test_build_prompt_from_context_method_exists(
        self,
        mock_model_manager,
        mock_memory_coordinator,
    ):
        """Test that build_prompt_from_context method exists on prompt builder."""
        from core.prompt.builder import UnifiedPromptBuilder

        builder = UnifiedPromptBuilder(
            memory_coordinator=mock_memory_coordinator,
            model_manager=mock_model_manager,
        )

        assert hasattr(builder, 'build_prompt_from_context')
        assert callable(builder.build_prompt_from_context)

    @pytest.mark.asyncio
    async def test_build_prompt_from_context_accepts_context_result(
        self,
        mock_model_manager,
        mock_memory_coordinator,
    ):
        """Test that build_prompt_from_context accepts ContextResult."""
        from core.prompt.builder import UnifiedPromptBuilder

        builder = UnifiedPromptBuilder(
            memory_coordinator=mock_memory_coordinator,
            model_manager=mock_model_manager,
        )

        # Create a ContextResult
        context = ContextResult(
            processed_query="How do I fix async bugs?",
            original_query="How do I fix async bugs?",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="Be helpful",
            topics=["python", "async"],
            primary_topic="python",
        )

        # Patch the underlying build_prompt to avoid external calls
        with patch.object(builder, 'build_prompt', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {"memories": [], "memory_id_map": {}}

            result = await builder.build_prompt_from_context(context)

        assert isinstance(result, dict)
        mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_prompt_from_context_passes_correct_params(
        self,
        mock_model_manager,
        mock_memory_coordinator,
    ):
        """Test that build_prompt_from_context passes correct parameters."""
        from core.prompt.builder import UnifiedPromptBuilder

        builder = UnifiedPromptBuilder(
            memory_coordinator=mock_memory_coordinator,
            model_manager=mock_model_manager,
        )

        # Create a ContextResult with specific values
        context = ContextResult(
            processed_query="rewritten query for search",
            original_query="original query",
            tone_level=ToneLevel.ELEVATED,
            tone_instructions="Be empathetic",
            topics=["mental_health"],
            primary_topic="mental_health",
            stm_summary={"topic": "mental health", "tone": "concerned"},
            is_heavy_topic=True,
            extracted_facts=[{"content": "User mentioned anxiety"}],
        )

        with patch.object(builder, 'build_prompt', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {"memories": []}

            await builder.build_prompt_from_context(context)

        # Verify parameters were passed correctly
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs['user_input'] == "rewritten query for search"
        assert call_kwargs['crisis_level'] == "MEDIUM"
        assert call_kwargs['current_topic'] == "mental_health"
        assert call_kwargs['stm_summary'] == {"topic": "mental health", "tone": "concerned"}
        assert call_kwargs['fresh_facts'] == [{"content": "User mentioned anxiety"}]


class TestContextResultProperties:
    """Test ContextResult dataclass properties."""

    def test_crisis_level_str_returns_enum_value(self):
        """Test that crisis_level_str returns the enum value."""
        context = ContextResult(
            processed_query="test",
            original_query="test",
            tone_level=ToneLevel.CRISIS,
            tone_instructions="",
        )
        assert context.crisis_level_str == "HIGH"

        context = ContextResult(
            processed_query="test",
            original_query="test",
            tone_level=ToneLevel.ELEVATED,
            tone_instructions="",
        )
        assert context.crisis_level_str == "MEDIUM"

    def test_has_files_with_content(self):
        """Test has_files property with file content."""
        context = ContextResult(
            processed_query="test",
            original_query="test",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            file_context="Some file content here",
        )
        assert context.has_files is True

    def test_has_thread_with_valid_thread_id(self):
        """Test has_thread property with valid thread context."""
        context = ContextResult(
            processed_query="test",
            original_query="test",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            thread_context={"thread_id": "abc123", "thread_depth": 3},
        )
        assert context.has_thread is True

    def test_has_stm_with_summary(self):
        """Test has_stm property with STM summary."""
        context = ContextResult(
            processed_query="test",
            original_query="test",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            stm_summary={"topic": "python", "intent": "learning"},
        )
        assert context.has_stm is True


class TestToneLevelMapping:
    """Test ToneLevel enum mapping."""

    def test_from_string_all_values(self):
        """Test ToneLevel.from_string for all valid values."""
        assert ToneLevel.from_string("HIGH") == ToneLevel.CRISIS
        assert ToneLevel.from_string("MEDIUM") == ToneLevel.ELEVATED
        assert ToneLevel.from_string("CONCERN") == ToneLevel.CONCERN
        assert ToneLevel.from_string("CONVERSATIONAL") == ToneLevel.CONVERSATIONAL

    def test_from_string_invalid_defaults_to_conversational(self):
        """Test that invalid strings default to CONVERSATIONAL."""
        assert ToneLevel.from_string("INVALID") == ToneLevel.CONVERSATIONAL
        assert ToneLevel.from_string("") == ToneLevel.CONVERSATIONAL
        assert ToneLevel.from_string("low") == ToneLevel.CONVERSATIONAL
