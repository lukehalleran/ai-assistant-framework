"""Deterministic contracts for the deployed ResponseGenerator; no providers load."""
import pytest
from core.response_generator import ResponseGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from utils.time_manager import TimeManager


@pytest.fixture
def model_manager():
    """Only the model/provider boundary is replaced."""
    manager = MagicMock()
    manager.get_active_model_name.return_value = "synthetic-model"
    manager.generate_once = AsyncMock(return_value="  Synthetic answer.  ")
    return manager


@pytest.fixture
def time_manager():
    """Provide TimeManager."""
    return TimeManager()


@pytest.fixture
def response_generator(model_manager, time_manager):
    """Provide ResponseGenerator."""
    return ResponseGenerator(model_manager=model_manager, time_manager=time_manager)


def test_coverage_score_basic():
    """Test _coverage_score static method."""
    score = ResponseGenerator._coverage_score(
        answer="Python is a programming language",
        question="What is Python?"
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_coverage_score_empty_answer():
    """Test _coverage_score with empty answer."""
    score = ResponseGenerator._coverage_score(
        answer="",
        question="What is Python?"
    )
    assert isinstance(score, float)


def test_coverage_score_with_context():
    """Test _coverage_score with context hint."""
    score = ResponseGenerator._coverage_score(
        answer="Python uses dynamic typing",
        question="How does Python handle types?",
        context_hint="Python programming"
    )
    assert isinstance(score, float)


def test_length_score_normal():
    """Test _length_score with normal length."""
    text = "This is a reasonable length answer " * 10
    score = ResponseGenerator._length_score(text)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_length_score_too_short():
    """Test _length_score with very short text."""
    score = ResponseGenerator._length_score("Short")
    assert isinstance(score, float)


def test_length_score_too_long():
    """Test _length_score with very long text."""
    long_text = "word " * 500
    score = ResponseGenerator._length_score(long_text)
    assert isinstance(score, float)


def test_length_score_custom_bounds():
    """Test _length_score with custom min/max."""
    text = "Some text here"
    score = ResponseGenerator._length_score(text, min_w=5, max_w=10)
    assert isinstance(score, float)


def test_repetition_penalty_no_repetition():
    """Test _repetition_penalty with unique words."""
    text = "Every word in this sentence is completely different and unique"
    penalty = ResponseGenerator._repetition_penalty(text)
    assert isinstance(penalty, float)
    assert 0.0 <= penalty <= 1.0


def test_repetition_penalty_high_repetition():
    """Test _repetition_penalty with repeated words."""
    text = "same same same same same same"
    penalty = ResponseGenerator._repetition_penalty(text)
    assert isinstance(penalty, float)


def test_repetition_penalty_empty():
    """Test _repetition_penalty with empty string."""
    penalty = ResponseGenerator._repetition_penalty("")
    assert isinstance(penalty, float)


def test_hallucination_penalty_normal():
    """Test _hallucination_penalty with normal text."""
    answer = "Python is a programming language created in 1991"
    penalty = ResponseGenerator._hallucination_penalty(answer)
    assert isinstance(penalty, float)


def test_hallucination_penalty_with_context():
    """Test _hallucination_penalty with context hint."""
    answer = "The answer involves Python programming"
    penalty = ResponseGenerator._hallucination_penalty(
        answer=answer,
        context_hint="Python"
    )
    assert isinstance(penalty, float)


def test_hallucination_penalty_suspicious():
    """Test _hallucination_penalty with suspicious patterns."""
    answer = "I don't have information about that"
    penalty = ResponseGenerator._hallucination_penalty(answer)
    assert isinstance(penalty, float)


def test_score_answer(response_generator):
    """Test _score_answer method."""
    score = response_generator._score_answer(
        answer="Python is a versatile programming language",
        question="What is Python?"
    )
    assert isinstance(score, float)


def test_score_answer_with_context(response_generator):
    """Test _score_answer with context hint."""
    score = response_generator._score_answer(
        answer="Python uses dynamic typing and has a large ecosystem",
        question="Tell me about Python",
        context_hint="Python programming language"
    )
    assert isinstance(score, float)


def test_score_answer_poor_quality(response_generator):
    """Test _score_answer with poor quality answer."""
    score = response_generator._score_answer(
        answer="yes",
        question="Explain Python in detail"
    )
    assert isinstance(score, float)


def test_minmax_normalize_normal():
    """Test _minmax_normalize static method (scales to 0-10 range)."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = ResponseGenerator._minmax_normalize(values)
    assert isinstance(normalized, list)
    assert len(normalized) == len(values)
    assert max(normalized) <= 10.0
    assert min(normalized) >= 0.0


def test_minmax_normalize_single_value():
    """Test _minmax_normalize with single value."""
    values = [5.0]
    normalized = ResponseGenerator._minmax_normalize(values)
    assert isinstance(normalized, list)
    assert len(normalized) == 1


def test_minmax_normalize_all_same():
    """Test _minmax_normalize with all same values."""
    values = [3.0, 3.0, 3.0, 3.0]
    normalized = ResponseGenerator._minmax_normalize(values)
    assert isinstance(normalized, list)


def test_minmax_normalize_empty():
    """Test _minmax_normalize with empty list."""
    values = []
    normalized = ResponseGenerator._minmax_normalize(values)
    assert isinstance(normalized, list)
    assert len(normalized) == 0


GOOD = "Python is a programming language with readable syntax, functions and dynamic typing."
OTHER = "An unrelated statement about clouds."


@pytest.mark.asyncio
async def test_generate_full_basic(response_generator):
    result = await response_generator.generate_full("What is 2+2?", "synthetic-model")
    assert result == "Synthetic answer."
    response_generator.model_manager.generate_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_full_with_system_prompt(response_generator):
    result = await response_generator.generate_full("Hello", "synthetic-model", system_prompt="Be precise")
    assert result == "Synthetic answer."
    assert response_generator.model_manager.generate_once.await_args.kwargs["system_prompt"] == "Be precise"


@pytest.mark.asyncio
async def test_generate_full_with_temperature(response_generator):
    result = await response_generator.generate_full("Write a haiku", "synthetic-model", temperature=0.9)
    assert result == "Synthetic answer."
    assert response_generator.model_manager.generate_once.await_args.kwargs["temperature"] == 0.9


@pytest.mark.asyncio
async def test_generate_best_of_basic(response_generator):
    response_generator.model_manager.generate_once.side_effect = [OTHER, GOOD]
    result = await response_generator.generate_best_of(
        "What is Python?", "synthetic-model", "Be precise", "What is Python?", n=2)
    assert result == GOOD
    assert response_generator.model_manager.generate_once.await_count == 2


@pytest.mark.asyncio
async def test_generate_best_of_with_context(response_generator):
    response_generator.model_manager.generate_once.side_effect = [OTHER, GOOD, "No."]
    result = await response_generator.generate_best_of(
        "Explain Python", "synthetic-model", "Be precise", "What is Python?",
        context_hint="Python programming language", n=3, temps=(0.2, 0.5, 0.8))
    assert result == GOOD
    assert response_generator.model_manager.generate_once.await_count == 3


@pytest.mark.asyncio
async def test_llm_judge_score(response_generator):
    response_generator.model_manager.generate_once.return_value = '{"score": 8.5}'
    score = await response_generator._llm_judge_score("synthetic-judge", "What is Python?", GOOD)
    assert score == 8.5
    sent = response_generator.model_manager.generate_once.await_args.kwargs
    assert GOOD in sent["prompt"] and sent["model_name"] == "synthetic-judge"


@pytest.mark.asyncio
async def test_llm_judge_compare(response_generator):
    response_generator.model_manager.generate_once.return_value = (
        '{"winner":"B","score_A":2,"score_B":9,"reason":"B answers the question"}')
    result = await response_generator._llm_judge_compare(
        "synthetic-judge", "What is Python?", OTHER, GOOD)
    assert result == {"winner": "B", "score_A": 2.0, "score_B": 9.0, "reason": "B answers the question"}


@pytest.mark.asyncio
async def test_generate_duel_and_judge(response_generator):
    response_generator.model_manager.generate_once.side_effect = [
        "<thinking>First reasoning.</thinking>" + OTHER,
        "<thinking>Second reasoning.</thinking>" + GOOD,
        '{"winner":"B","score_A":2,"score_B":9,"reason":"B is relevant"}',
    ]
    result = await response_generator.generate_duel_and_judge(
        "What is Python?", "synthetic-a", "synthetic-b", "synthetic-judge",
        "Be precise", "What is Python?")
    assert isinstance(result, dict)
    assert result["answer"] == GOOD and result["winner"] == "B"
    assert result["thinking_a"] == "First reasoning." and result["thinking_b"] == "Second reasoning."
    assert result["scores"] == {"A": 2.0, "B": 9.0}
    judged = response_generator.model_manager.generate_once.await_args.kwargs["prompt"]
    assert GOOD in judged and OTHER in judged and "thinking>" not in judged


@pytest.mark.asyncio
async def test_generate_best_of_ensemble(response_generator):
    response_generator.model_manager.generate_once.side_effect = [OTHER, GOOD, '{"score":1}', '{"score":9}']
    result = await response_generator.generate_best_of_ensemble(
        "What is Python?", ["synthetic-a", "synthetic-b"], "Be precise", "What is Python?",
        n_total=2, selector_models=["synthetic-judge"], weight_heuristic=0, weight_llm=1)
    assert result == GOOD
    assert response_generator.model_manager.generate_once.await_count == 4


def test_response_generator_initialization(model_manager):
    generator = ResponseGenerator(model_manager=model_manager)
    assert generator.model_manager is model_manager


def test_response_generator_with_time_manager(model_manager, time_manager):
    generator = ResponseGenerator(model_manager=model_manager, time_manager=time_manager)
    assert generator.time_manager is time_manager


def stream_provider(response_generator, chunks, error=None):
    """Mimic the deployed `generate_async` contract: an ``async def`` method
    that, once awaited, returns an object supporting ``__aiter__`` (see
    ``ModelManager.generate_async`` -> ``response_generator = await
    self.model_manager.generate_async(...)`` then ``hasattr(response_generator,
    "__aiter__")`` in core/response_generator.py). An async-generator *function*
    used directly as an AsyncMock side_effect is itself an async generator
    when called -- not awaitable -- which raised
    ``TypeError: object async_generator can't be used in 'await' expression``
    and was silently swallowed by the outer except-yield-error path. The fix
    is a plain (synchronous) factory that RETURNS the async-generator object;
    AsyncMock then awaits its own call and hands back that object untouched.
    """
    state = SimpleNamespace(closed=False)

    async def _stream_chunks():
        try:
            for chunk in chunks:
                yield chunk
            if error:
                raise error
        finally:
            state.closed = True

    def make_stream(*args, **kwargs):
        return _stream_chunks()

    response_generator.model_manager.generate_async = AsyncMock(side_effect=make_stream)
    return state


@pytest.mark.asyncio
async def test_generate_streaming_response_basic(response_generator):
    state = stream_provider(response_generator, ["Synthetic ", "complete ", "answer."])
    chunks = [chunk async for chunk in response_generator.generate_streaming_response("Hello", "synthetic-model")]
    assert chunks and all(isinstance(chunk, str) for chunk in chunks)
    # The deployed generator yields one word per chunk with the delimiting
    # space stripped out by its internal `buffer.split(" ")` logic (see
    # core/response_generator.py) -- real callers reassemble the answer with
    # their own join convention (core/orchestrator.py's standard streaming
    # path does `full_response += (chunk + " ")` then `.strip()`, which for
    # plain word chunks is equivalent to a single space join). A bare
    # `"".join(chunks)` does not reflect how any deployed caller actually
    # reconstructs the text and previously masked a real defect (calling the
    # async-generator-function side_effect directly, which is not awaitable)
    # by comparing against the wrong joiner.
    assert " ".join(chunks) == "Synthetic complete answer."
    assert state.closed


@pytest.mark.asyncio
async def test_generate_streaming_response_with_system(response_generator):
    state = stream_provider(response_generator, ["One. ", "Two. ", "Three. ", "Four."])
    chunks = [chunk async for chunk in response_generator.generate_streaming_response(
        "Hi", "synthetic-model", system_prompt="Be brief")]
    assert " ".join(chunks) == "One. Two. Three. Four." and state.closed
    assert response_generator.model_manager.generate_async.call_args.kwargs["system_prompt"] == "Be brief"


@pytest.mark.asyncio
async def test_generate_full_provider_failure_is_explicit(response_generator):
    response_generator.model_manager.generate_once.side_effect = RuntimeError("synthetic provider failure")
    result = await response_generator.generate_full("Hello", "synthetic-model")
    assert result == "[Generation error] synthetic provider failure"


@pytest.mark.asyncio
async def test_best_of_preserves_successful_candidate_when_peer_fails(response_generator):
    response_generator.model_manager.generate_once.side_effect = [RuntimeError("synthetic provider failure"), GOOD]
    result = await response_generator.generate_best_of("Python", "synthetic-model", "Be precise", "Python", n=2)
    assert result == GOOD


@pytest.mark.asyncio
async def test_judge_provider_failure_uses_documented_zero_score(response_generator):
    response_generator.model_manager.generate_once.side_effect = RuntimeError("synthetic provider failure")
    assert await response_generator._llm_judge_score("synthetic-judge", "Python?", GOOD) == 0.0


@pytest.mark.asyncio
async def test_streaming_provider_failure_surfaces_and_closes(response_generator):
    state = stream_provider(response_generator, [], RuntimeError("synthetic provider failure"))
    chunks = [chunk async for chunk in response_generator.generate_streaming_response("Hello", "synthetic-model")]
    assert state.closed and chunks
    assert "synthetic provider failure" in "".join(chunks)
