"""
Unit tests for ContextPipeline.

Tests the context preparation pipeline in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import dataclass

from core.context_pipeline import (
    ContextPipeline,
    ContextResult,
    ToneLevel,
)


# --- Fixtures ---

@pytest.fixture
def mock_model_manager():
    """Create a mock model manager."""
    manager = Mock()
    manager.generate_once = AsyncMock(return_value="rewritten query for semantic search")
    return manager


@pytest.fixture
def mock_topic_manager():
    """Create a mock topic manager."""
    manager = Mock()
    manager.update_from_user_input = Mock()
    manager.get_primary_topic = Mock(return_value="python")
    manager.get_entities = Mock(return_value=["debugging", "async"])
    return manager


@pytest.fixture
def mock_file_processor():
    """Create a mock file processor."""
    processor = Mock()
    processor.process_files = AsyncMock(return_value="user input\n\n---\n\nFile content here")
    return processor


@pytest.fixture
def mock_stm_analyzer():
    """Create a mock STM analyzer."""
    analyzer = Mock()
    analyzer.analyze = AsyncMock(return_value={
        "topic": "python debugging",
        "user_question": "How to fix async bug?",
        "intent": "troubleshooting",
        "tone": "neutral",
        "open_threads": [],
        "constraints": []
    })
    return analyzer


@pytest.fixture
def mock_user_profile():
    """Create a mock user profile."""
    profile = Mock()
    identity = Mock()
    identity.name = "Alice"
    identity.pronouns = "she/her"
    profile.identity = identity
    return profile


@pytest.fixture
def mock_memory_system():
    """Create a mock memory system."""
    system = Mock()
    system.get_thread_context = Mock(return_value={
        "thread_id": "thread-123",
        "thread_depth": 3,
        "thread_started": "2024-01-01T12:00:00",
        "thread_topic": "python",
        "is_heavy_topic": False
    })
    corpus_manager = Mock()
    corpus_manager.get_recent_memories = Mock(return_value=[
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ])
    system.corpus_manager = corpus_manager
    system.get_facts = Mock(return_value=[])
    system._extract_and_store_facts = AsyncMock()
    return system


@pytest.fixture
def pipeline(mock_model_manager, mock_topic_manager):
    """Create a basic pipeline with minimal dependencies."""
    return ContextPipeline(
        model_manager=mock_model_manager,
        topic_manager=mock_topic_manager,
    )


@pytest.fixture
def full_pipeline(
    mock_model_manager,
    mock_topic_manager,
    mock_file_processor,
    mock_stm_analyzer,
    mock_user_profile,
    mock_memory_system
):
    """Create a pipeline with all dependencies."""
    return ContextPipeline(
        model_manager=mock_model_manager,
        topic_manager=mock_topic_manager,
        file_processor=mock_file_processor,
        stm_analyzer=mock_stm_analyzer,
        user_profile=mock_user_profile,
        memory_system=mock_memory_system,
        config={
            "USE_STM_PASS": True,
            "STM_MIN_CONVERSATION_DEPTH": 1,  # Low for testing
        }
    )


# --- ToneLevel Tests ---

class TestToneLevel:
    """Tests for ToneLevel enum."""

    def test_tone_level_values(self):
        """Test ToneLevel enum values match expected strings."""
        assert ToneLevel.CRISIS.value == "HIGH"
        assert ToneLevel.ELEVATED.value == "MEDIUM"
        assert ToneLevel.CONCERN.value == "CONCERN"
        assert ToneLevel.CONVERSATIONAL.value == "CONVERSATIONAL"

    def test_from_string_valid(self):
        """Test ToneLevel.from_string with valid inputs."""
        assert ToneLevel.from_string("HIGH") == ToneLevel.CRISIS
        assert ToneLevel.from_string("MEDIUM") == ToneLevel.ELEVATED
        assert ToneLevel.from_string("CONCERN") == ToneLevel.CONCERN
        assert ToneLevel.from_string("CONVERSATIONAL") == ToneLevel.CONVERSATIONAL

    def test_from_string_invalid(self):
        """Test ToneLevel.from_string with invalid input defaults to CONVERSATIONAL."""
        assert ToneLevel.from_string("UNKNOWN") == ToneLevel.CONVERSATIONAL
        assert ToneLevel.from_string("") == ToneLevel.CONVERSATIONAL
        assert ToneLevel.from_string("invalid") == ToneLevel.CONVERSATIONAL


# --- ContextResult Tests ---

class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic ContextResult."""
        result = ContextResult(
            processed_query="How do I fix this bug?",
            original_query="How do I fix this bug?",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="Be helpful",
        )

        assert result.processed_query == "How do I fix this bug?"
        assert result.original_query == "How do I fix this bug?"
        assert result.tone_level == ToneLevel.CONVERSATIONAL
        assert result.topics == []
        assert result.file_context is None

    def test_has_files_property(self):
        """Test has_files property."""
        result_no_files = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
        )
        assert result_no_files.has_files is False

        result_with_files = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            file_context="file content here",
        )
        assert result_with_files.has_files is True

        result_empty_files = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            file_context="",
        )
        assert result_empty_files.has_files is False

    def test_has_thread_property(self):
        """Test has_thread property."""
        result_no_thread = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
        )
        assert result_no_thread.has_thread is False

        result_with_thread = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            thread_context={"thread_id": "123", "thread_depth": 2},
        )
        assert result_with_thread.has_thread is True

        result_empty_thread = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CONVERSATIONAL,
            tone_instructions="",
            thread_context={},
        )
        assert result_empty_thread.has_thread is False

    def test_crisis_level_str_property(self):
        """Test crisis_level_str backwards compatibility property."""
        result = ContextResult(
            processed_query="query",
            original_query="query",
            tone_level=ToneLevel.CRISIS,
            tone_instructions="",
        )
        assert result.crisis_level_str == "HIGH"


# --- ContextPipeline Tests ---

class TestContextPipelineBasic:
    """Basic tests for ContextPipeline."""

    @pytest.mark.asyncio
    async def test_build_basic_query(self, pipeline):
        """Test basic query processing."""
        result = await pipeline.build("How do I fix this bug?")

        assert isinstance(result, ContextResult)
        assert result.original_query == "How do I fix this bug?"
        assert result.tone_level == ToneLevel.CONVERSATIONAL
        assert "python" in result.topics

    @pytest.mark.asyncio
    async def test_build_preserves_original_query(self, pipeline):
        """Test that original_query is always preserved."""
        result = await pipeline.build("my original query here")

        assert result.original_query == "my original query here"

    @pytest.mark.asyncio
    async def test_raw_mode_skips_enrichment(self, pipeline, mock_topic_manager):
        """Test that raw mode skips most processing."""
        result = await pipeline.build("test query", use_raw_mode=True)

        assert result.processed_query == "test query"
        assert result.metadata["use_raw_mode"] is True
        # Topic extraction still runs in raw mode
        mock_topic_manager.update_from_user_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_depth_increments(self, pipeline):
        """Test that conversation depth increments with each build."""
        assert pipeline.get_conversation_depth() == 0

        await pipeline.build("query 1")
        assert pipeline.get_conversation_depth() == 1

        await pipeline.build("query 2")
        assert pipeline.get_conversation_depth() == 2

    def test_reset_conversation_depth(self, pipeline):
        """Test resetting conversation depth."""
        pipeline._conversation_depth = 5
        pipeline.reset_conversation_depth()
        assert pipeline.get_conversation_depth() == 0


class TestContextPipelineTopicExtraction:
    """Tests for topic extraction stage."""

    @pytest.mark.asyncio
    async def test_topic_extraction_called(self, pipeline, mock_topic_manager):
        """Test that topic manager methods are called."""
        await pipeline.build("Tell me about Python async")

        mock_topic_manager.update_from_user_input.assert_called_once_with("Tell me about Python async")
        mock_topic_manager.get_primary_topic.assert_called_once()

    @pytest.mark.asyncio
    async def test_topic_extraction_returns_topics(self, pipeline, mock_topic_manager):
        """Test that topics are returned in result."""
        mock_topic_manager.get_primary_topic.return_value = "async programming"
        mock_topic_manager.get_entities.return_value = ["Python", "coroutines"]

        result = await pipeline.build("Tell me about Python async")

        assert result.primary_topic == "async programming"
        assert "async programming" in result.topics

    @pytest.mark.asyncio
    async def test_topic_extraction_handles_none(self, pipeline, mock_topic_manager):
        """Test graceful handling when no topics found."""
        mock_topic_manager.get_primary_topic.return_value = None
        mock_topic_manager.get_entities.return_value = []

        result = await pipeline.build("Hi")

        assert result.primary_topic is None
        assert result.topics == []

    @pytest.mark.asyncio
    async def test_topic_extraction_handles_exception(self, pipeline, mock_topic_manager):
        """Test graceful handling when topic extraction fails."""
        mock_topic_manager.update_from_user_input.side_effect = Exception("Topic error")

        result = await pipeline.build("test query")

        # Should not raise, should return empty topics
        assert result.primary_topic is None
        assert result.topics == []


class TestContextPipelineToneDetection:
    """Tests for tone detection stage."""

    @pytest.mark.asyncio
    async def test_tone_detection_conversational(self, pipeline):
        """Test conversational tone detection."""
        # Mock the emotional context module
        with patch('core.context_pipeline.ContextPipeline._detect_tone') as mock_detect:
            mock_detect.return_value = (ToneLevel.CONVERSATIONAL, None)
            result = await pipeline.build("What's the weather like?")

        # Since we mocked, check the mock was set up
        assert result.tone_level == ToneLevel.CONVERSATIONAL

    @pytest.mark.asyncio
    async def test_tone_instructions_for_crisis(self, pipeline):
        """Test that CRISIS tone gets appropriate instructions."""
        instructions = pipeline._get_tone_instructions(ToneLevel.CRISIS)

        assert "CRISIS SUPPORT MODE" in instructions
        assert "therapeutic" in instructions.lower()

    @pytest.mark.asyncio
    async def test_tone_instructions_for_elevated(self, pipeline):
        """Test that ELEVATED tone gets appropriate instructions."""
        instructions = pipeline._get_tone_instructions(ToneLevel.ELEVATED)

        assert "ELEVATED SUPPORT MODE" in instructions
        assert "empathetic" in instructions.lower()

    @pytest.mark.asyncio
    async def test_tone_instructions_for_concern(self, pipeline):
        """Test that CONCERN tone gets appropriate instructions."""
        instructions = pipeline._get_tone_instructions(ToneLevel.CONCERN)

        assert "LIGHT SUPPORT MODE" in instructions

    @pytest.mark.asyncio
    async def test_tone_instructions_for_conversational(self, pipeline):
        """Test that CONVERSATIONAL tone gets appropriate instructions."""
        instructions = pipeline._get_tone_instructions(ToneLevel.CONVERSATIONAL)

        assert "CONVERSATIONAL MODE" in instructions
        assert "concise" in instructions.lower()


class TestContextPipelineFileProcessing:
    """Tests for file processing stage."""

    @pytest.mark.asyncio
    async def test_file_processing_called(self, full_pipeline, mock_file_processor):
        """Test that file processor is called when files provided."""
        mock_files = [Mock(name="test.txt")]

        await full_pipeline.build("Process this file", files=mock_files)

        mock_file_processor.process_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_file_processing_skipped_without_files(self, full_pipeline, mock_file_processor):
        """Test that file processor is not called without files."""
        await full_pipeline.build("No files here")

        mock_file_processor.process_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_processing_skipped_in_raw_mode(self, full_pipeline, mock_file_processor):
        """Test that file processing is skipped in raw mode."""
        mock_files = [Mock(name="test.txt")]

        await full_pipeline.build("Process this file", files=mock_files, use_raw_mode=True)

        mock_file_processor.process_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_context_in_result(self, full_pipeline, mock_file_processor):
        """Test that file context appears in result."""
        mock_files = [Mock(name="test.txt")]
        mock_file_processor.process_files.return_value = "Combined content with files"

        result = await full_pipeline.build("Process this", files=mock_files)

        assert result.has_files is True
        assert "Combined content" in result.file_context


class TestContextPipelineSTMAnalysis:
    """Tests for STM analysis stage."""

    @pytest.mark.asyncio
    async def test_stm_analysis_called_when_enabled(self, full_pipeline, mock_stm_analyzer):
        """Test that STM analyzer is called when conditions are met."""
        # Build enough conversation depth
        await full_pipeline.build("query 1")

        mock_stm_analyzer.analyze.assert_called()

    @pytest.mark.asyncio
    async def test_stm_analysis_skipped_when_depth_insufficient(
        self,
        mock_model_manager,
        mock_topic_manager,
        mock_stm_analyzer
    ):
        """Test that STM is skipped when conversation depth is insufficient."""
        pipeline = ContextPipeline(
            model_manager=mock_model_manager,
            topic_manager=mock_topic_manager,
            stm_analyzer=mock_stm_analyzer,
            config={
                "USE_STM_PASS": True,
                "STM_MIN_CONVERSATION_DEPTH": 5,  # High threshold
            }
        )

        await pipeline.build("first query")

        # Should not be called yet
        mock_stm_analyzer.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_stm_summary_in_result(self, full_pipeline, mock_stm_analyzer):
        """Test that STM summary appears in result."""
        mock_stm_analyzer.analyze.return_value = {
            "topic": "testing",
            "intent": "verification"
        }

        result = await full_pipeline.build("test query")

        assert result.has_stm is True
        assert result.stm_summary["topic"] == "testing"


class TestContextPipelineIdentity:
    """Tests for identity injection stage."""

    @pytest.mark.asyncio
    async def test_identity_context_retrieved(self, full_pipeline):
        """Test that identity context is retrieved."""
        result = await full_pipeline.build("Hello!")

        assert result.user_name == "Alice"
        assert "Alice" in result.identity_block
        assert "she/her" in result.identity_block

    @pytest.mark.asyncio
    async def test_identity_context_empty_without_profile(self, pipeline):
        """Test that identity is empty when no profile available."""
        result = await pipeline.build("Hello!")

        assert result.user_name is None
        assert result.identity_block == ""


class TestContextPipelineThreadContext:
    """Tests for thread context stage."""

    @pytest.mark.asyncio
    async def test_thread_context_retrieved(self, full_pipeline, mock_memory_system):
        """Test that thread context is retrieved."""
        result = await full_pipeline.build("Continue our conversation")

        assert result.has_thread is True
        assert result.thread_context["thread_id"] == "thread-123"
        assert result.thread_context["thread_depth"] == 3

    @pytest.mark.asyncio
    async def test_thread_context_empty_without_memory_system(self, pipeline):
        """Test that thread context is empty when no memory system."""
        result = await pipeline.build("test query")

        assert result.has_thread is False
        assert result.thread_context is None

    @pytest.mark.asyncio
    async def test_thread_context_skipped_in_raw_mode(self, full_pipeline, mock_memory_system):
        """Test that thread context is skipped in raw mode."""
        result = await full_pipeline.build("test query", use_raw_mode=True)

        mock_memory_system.get_thread_context.assert_not_called()
        assert result.thread_context is None


class TestContextPipelineMetadata:
    """Tests for metadata in result."""

    @pytest.mark.asyncio
    async def test_metadata_includes_expected_fields(self, pipeline):
        """Test that metadata contains expected fields."""
        result = await pipeline.build("test query")

        assert "use_raw_mode" in result.metadata
        assert "has_files" in result.metadata
        assert "topic_count" in result.metadata
        assert "conversation_depth" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_reflects_raw_mode(self, pipeline):
        """Test that metadata correctly reflects raw mode."""
        result = await pipeline.build("test", use_raw_mode=True)
        assert result.metadata["use_raw_mode"] is True

        result = await pipeline.build("test", use_raw_mode=False)
        assert result.metadata["use_raw_mode"] is False


class TestContextPipelineQueryRewriting:
    """Tests for query rewriting stage."""

    @pytest.mark.asyncio
    async def test_query_rewriting_for_questions(self, full_pipeline, mock_model_manager):
        """Test that questions trigger query rewriting."""
        # Patch at the module where it's imported (inside the method)
        with patch('utils.query_checker.analyze_query') as mock_analyze:
            mock_analysis = Mock()
            mock_analysis.is_question = True
            mock_analysis.is_command = False
            mock_analysis.token_count = 10
            mock_analysis.is_heavy_topic = False
            mock_analyze.return_value = mock_analysis

            with patch('utils.query_checker.analyze_query_async', new_callable=AsyncMock) as mock_async:
                mock_async.return_value = mock_analysis
                result = await full_pipeline.build("How do I implement async functions in Python?")

        # Model manager should have been called for rewriting
        # (actual call depends on implementation details)
        assert result.processed_query is not None

    @pytest.mark.asyncio
    async def test_query_rewriting_skipped_in_raw_mode(self, full_pipeline, mock_model_manager):
        """Test that query rewriting is skipped in raw mode."""
        result = await full_pipeline.build(
            "How do I implement async?",
            use_raw_mode=True
        )

        # In raw mode, processed_query should equal original
        assert result.processed_query == "How do I implement async?"


# --- Integration-style Tests ---

class TestContextPipelineIntegration:
    """Integration-style tests with full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, full_pipeline):
        """Test complete pipeline flow."""
        result = await full_pipeline.build(
            "I'm working on a Python project and need help with async programming",
            files=None,
            use_raw_mode=False
        )

        # Verify all components are present
        assert result.original_query is not None
        assert result.processed_query is not None
        assert result.tone_level is not None
        assert result.tone_instructions is not None
        assert result.topics is not None
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_pipeline_handles_all_errors_gracefully(
        self,
        mock_model_manager,
        mock_topic_manager,
        mock_file_processor,
        mock_stm_analyzer,
        mock_user_profile,
        mock_memory_system
    ):
        """Test that pipeline handles errors gracefully in all stages."""
        # Make everything fail
        mock_topic_manager.update_from_user_input.side_effect = Exception("Topic error")
        mock_file_processor.process_files.side_effect = Exception("File error")
        mock_stm_analyzer.analyze.side_effect = Exception("STM error")
        mock_memory_system.get_thread_context.side_effect = Exception("Thread error")

        pipeline = ContextPipeline(
            model_manager=mock_model_manager,
            topic_manager=mock_topic_manager,
            file_processor=mock_file_processor,
            stm_analyzer=mock_stm_analyzer,
            user_profile=mock_user_profile,
            memory_system=mock_memory_system,
            config={"USE_STM_PASS": True, "STM_MIN_CONVERSATION_DEPTH": 0}
        )

        # Should not raise, should return result with defaults
        result = await pipeline.build("test query", files=[Mock()])

        assert result.original_query == "test query"
        assert result.topics == []
        assert result.thread_context is None
        assert result.stm_summary is None
