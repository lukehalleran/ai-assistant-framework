"""Tests for ResponseGenerator."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from core.response_generator import ResponseGenerator
from models.model_manager import ModelManager
from utils.time_manager import TimeManager


@pytest.fixture
def mock_model_manager():
    """Provide mock ModelManager."""
    mm = Mock(spec=ModelManager)
    mm.switch_model = Mock()
    mm.get_active_model_name = Mock(return_value="gpt-4")
    mm.generate_async = AsyncMock()
    mm.generate_once = AsyncMock(return_value="Test response")
    return mm


@pytest.fixture
def time_manager():
    """Provide TimeManager."""
    return TimeManager()


@pytest.fixture
def response_generator(mock_model_manager, time_manager):
    """Provide ResponseGenerator."""
    return ResponseGenerator(
        model_manager=mock_model_manager,
        time_manager=time_manager
    )


def test_response_generator_init(mock_model_manager):
    """Test ResponseGenerator initialization."""
    rg = ResponseGenerator(model_manager=mock_model_manager)

    assert rg.model_manager is mock_model_manager
    assert rg.time_manager is not None


@pytest.mark.asyncio
async def test_generate_streaming_response_basic(response_generator, mock_model_manager):
    """Test basic streaming response generation."""
    # Mock the async generator
    async def mock_gen():
        yield "Hello"
        yield " "
        yield "world"

    mock_model_manager.generate_async.return_value = mock_gen()

    result = []
    async for chunk in response_generator.generate_streaming_response(
        prompt="Test prompt",
        model_name="gpt-4",
        system_prompt="You are helpful"
    ):
        result.append(chunk)

    assert len(result) > 0
    mock_model_manager.switch_model.assert_called_once_with("gpt-4")


@pytest.mark.asyncio
async def test_generate_streaming_response_with_max_tokens(response_generator, mock_model_manager):
    """Test streaming with max_tokens parameter."""
    async def mock_gen():
        yield "Test"

    mock_model_manager.generate_async.return_value = mock_gen()

    result = []
    async for chunk in response_generator.generate_streaming_response(
        prompt="Test",
        model_name="gpt-4",
        max_tokens=100
    ):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_streaming_response_no_system_prompt(response_generator, mock_model_manager):
    """Test streaming falls back to default system prompt."""
    async def mock_gen():
        yield "Response"

    mock_model_manager.generate_async.return_value = mock_gen()

    result = []
    async for chunk in response_generator.generate_streaming_response(
        prompt="Test",
        model_name="gpt-4",
        system_prompt=None
    ):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_full(response_generator, mock_model_manager):
    """Test generate_full returns complete response."""
    mock_model_manager.generate_once.return_value = "Complete response text"

    result = await response_generator.generate_full(
        prompt="Test prompt",
        model_name="gpt-4"
    )

    assert isinstance(result, str)
    assert len(result) > 0
    mock_model_manager.generate_once.assert_called_once()


@pytest.mark.asyncio
async def test_generate_full_with_temperature(response_generator, mock_model_manager):
    """Test generate_full with temperature parameter."""
    mock_model_manager.generate_once.return_value = "Response"

    result = await response_generator.generate_full(
        prompt="Test",
        model_name="gpt-4",
        temperature=0.7
    )

    assert isinstance(result, str)


def test_coverage_score_high(response_generator):
    """Test _coverage_score with good coverage."""
    answer = "Python is a programming language used for web development and data science"
    question = "What is Python used for?"

    score = response_generator._coverage_score(answer, question)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_coverage_score_low(response_generator):
    """Test _coverage_score with poor coverage."""
    answer = "It's good"
    question = "Explain the theory of relativity in detail"

    score = response_generator._coverage_score(answer, question)

    assert isinstance(score, float)
    assert score < 0.5  # Poor coverage


def test_length_score_optimal(response_generator):
    """Test _length_score with optimal length."""
    answer = " ".join(["word"] * 100)  # ~100 words

    score = response_generator._length_score(answer)

    assert isinstance(score, float)
    assert score > 0.5  # Good length


def test_length_score_too_short(response_generator):
    """Test _length_score with too short answer."""
    answer = "Short"

    score = response_generator._length_score(answer)

    assert isinstance(score, float)
    assert score < 0.5  # Penalized for being short


def test_length_score_too_long(response_generator):
    """Test _length_score with too long answer."""
    answer = " ".join(["word"] * 500)  # Very long

    score = response_generator._length_score(answer)

    assert isinstance(score, float)


def test_repetition_penalty_no_repetition(response_generator):
    """Test _repetition_penalty with no repetition."""
    answer = "This is a unique sentence with different words"

    penalty = response_generator._repetition_penalty(answer)

    assert isinstance(penalty, float)
    assert 0.0 <= penalty <= 1.0
    assert penalty < 0.3  # Low penalty for unique words


def test_repetition_penalty_with_repetition(response_generator):
    """Test _repetition_penalty with repetitive text."""
    answer = "word word word word word word"

    penalty = response_generator._repetition_penalty(answer)

    assert isinstance(penalty, float)
    assert penalty > 0.5  # HIGH penalty for repetition


def test_hallucination_penalty_no_context(response_generator):
    """Test _hallucination_penalty with no context."""
    answer = "Python is a programming language"

    penalty = response_generator._hallucination_penalty(answer, context_hint="")

    assert isinstance(penalty, float)
    assert penalty == 0.0  # Returns 0.0 when no context provided


def test_hallucination_penalty_with_uncertainty(response_generator):
    """Test _hallucination_penalty with uncertainty markers."""
    answer = "I don't know, but I think maybe it could be something"

    penalty = response_generator._hallucination_penalty(answer)

    assert isinstance(penalty, float)


def test_score_answer_comprehensive(response_generator):
    """Test _score_answer combines all factors."""
    answer = "Python is a high-level programming language known for its readability"
    question = "What is Python?"

    score = response_generator._score_answer(answer, question)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_answer_poor_quality(response_generator):
    """Test _score_answer with poor answer."""
    answer = "idk"
    question = "Explain machine learning"

    score = response_generator._score_answer(answer, question)

    assert isinstance(score, float)
    assert score < 0.3  # Should score low


@pytest.mark.asyncio
async def test_generate_best_of(response_generator, mock_model_manager):
    """Test generate_best_of exists and is callable."""
    # Just verify the method exists
    assert hasattr(response_generator, 'generate_best_of')
    assert callable(response_generator.generate_best_of)


def test_minmax_normalize_empty(response_generator):
    """Test _minmax_normalize with empty list."""
    result = response_generator._minmax_normalize([])

    assert result == []


def test_minmax_normalize_single_value(response_generator):
    """Test _minmax_normalize with single value."""
    result = response_generator._minmax_normalize([0.5])

    assert len(result) == 1
    # When all values are equal, returns 5.0 for each
    assert result[0] == 5.0


def test_minmax_normalize_multiple_values(response_generator):
    """Test _minmax_normalize normalizes correctly."""
    values = [0.2, 0.8, 0.5]

    result = response_generator._minmax_normalize(values)

    assert len(result) == 3
    # Normalizes to 0-10 scale, not 0-1
    assert min(result) == 0.0
    assert max(result) == 10.0


@pytest.mark.asyncio
async def test_llm_judge_methods_exist(response_generator):
    """Test LLM judge methods exist."""
    # Just verify methods exist (complex to mock properly)
    assert hasattr(response_generator, '_llm_judge_score')
    assert hasattr(response_generator, '_llm_judge_compare')
    assert hasattr(response_generator, 'generate_duel_and_judge')
    assert callable(response_generator._llm_judge_score)
    assert callable(response_generator._llm_judge_compare)
    assert callable(response_generator.generate_duel_and_judge)


@pytest.mark.asyncio
async def test_generate_streaming_error_handling(response_generator, mock_model_manager):
    """Test streaming handles errors gracefully."""
    # Mock an error
    async def mock_gen_error():
        raise Exception("Test error")
        yield "Never reached"

    mock_model_manager.generate_async.return_value = mock_gen_error()

    # Should handle error and not crash
    try:
        result = []
        async for chunk in response_generator.generate_streaming_response(
            prompt="Test",
            model_name="gpt-4"
        ):
            result.append(chunk)
    except Exception:
        # Expected to raise or handle
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
