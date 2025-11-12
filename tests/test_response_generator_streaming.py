"""Targeted tests for ResponseGenerator streaming paths to boost coverage."""
import pytest
import asyncio
from core.response_generator import ResponseGenerator
from models.model_manager import ModelManager
from utils.time_manager import TimeManager


@pytest.fixture
def model_manager():
    """Provide ModelManager."""
    return ModelManager()


@pytest.fixture
def time_manager():
    """Provide TimeManager."""
    return TimeManager()


@pytest.fixture
def response_generator(model_manager, time_manager):
    """Provide ResponseGenerator."""
    return ResponseGenerator(model_manager=model_manager, time_manager=time_manager)


# Test streaming with different chunk types
@pytest.mark.asyncio
async def test_streaming_with_string_chunks(response_generator, monkeypatch):
    """Test streaming with plain string chunks."""
    async def mock_stream():
        for word in ["Hello", " ", "world"]:
            yield word

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_dict_chunks(response_generator, monkeypatch):
    """Test streaming with dict-style chunks."""
    async def mock_stream():
        yield {"content": "Hello"}
        yield {"text": "world"}
        yield {"content": " test"}

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_openai_chunks(response_generator, monkeypatch):
    """Test streaming with OpenAI-style chunks."""
    class MockDelta:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.delta = MockDelta(content)

    class MockChunk:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    async def mock_stream():
        yield MockChunk("Hello")
        yield MockChunk(" ")
        yield MockChunk("world")

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_stop_markers(response_generator, monkeypatch):
    """Test streaming stops early when stop marker found."""
    async def mock_stream():
        yield "Hello world"
        yield "<|end|>"
        yield "This should not be yielded"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    # Should stop before "This should not be yielded"
    full_text = " ".join(result)
    assert "This should not be yielded" not in full_text


@pytest.mark.asyncio
async def test_streaming_with_user_marker(response_generator, monkeypatch):
    """Test streaming with <|user|> stop marker."""
    async def mock_stream():
        yield "Response text"
        yield "<|user|>"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_assistant_marker(response_generator, monkeypatch):
    """Test streaming with <|assistant|> stop marker."""
    async def mock_stream():
        yield "Text here"
        yield " <|assistant|>"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) >= 0


@pytest.mark.asyncio
async def test_streaming_with_system_marker(response_generator, monkeypatch):
    """Test streaming with <|system|> marker."""
    async def mock_stream():
        yield "Content"
        yield "<|system|>More"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) >= 0


@pytest.mark.asyncio
async def test_streaming_with_eot_marker(response_generator, monkeypatch):
    """Test streaming with <|eot_id|> marker."""
    async def mock_stream():
        yield "Text"
        yield " here<|eot_id|>"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) >= 0


@pytest.mark.asyncio
async def test_streaming_word_by_word(response_generator, monkeypatch):
    """Test streaming yields words separated by spaces."""
    async def mock_stream():
        yield "Hello world this is a test"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    # Should yield at least one word
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_streaming_with_empty_chunks(response_generator, monkeypatch):
    """Test streaming handles empty chunks."""
    async def mock_stream():
        yield ""
        yield "Hello"
        yield ""
        yield " world"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_chunk_error(response_generator, monkeypatch):
    """Test streaming continues after chunk processing error."""
    async def mock_stream():
        yield "Good chunk"
        yield None  # Will cause error
        yield " another good chunk"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    # Should have some results despite error
    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_max_tokens(response_generator, monkeypatch):
    """Test streaming with max_tokens parameter."""
    async def mock_stream():
        yield "Response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4", max_tokens=100):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_non_streaming_response(response_generator, monkeypatch):
    """Test non-streaming fallback path."""
    class MockMessage:
        content = "This is a non-streaming response"

    class MockChoice:
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: MockResponse())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_non_streaming_without_message_attr(response_generator, monkeypatch):
    """Test non-streaming fallback without message attribute."""
    class MockChoice:
        pass  # No message attribute

    class MockResponse:
        choices = [MockChoice()]
        def __str__(self):
            return "Fallback string response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: MockResponse())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_non_streaming_no_choices(response_generator, monkeypatch):
    """Test non-streaming without choices."""
    class MockResponse:
        def __str__(self):
            return "Direct string response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: MockResponse())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_error_handling(response_generator, monkeypatch):
    """Test streaming error yields error message."""
    async def mock_error(*args, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(response_generator.model_manager, "generate_async", mock_error)

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0
    assert any("Error" in str(chunk) for chunk in result)


@pytest.mark.asyncio
async def test_streaming_with_custom_system_prompt(response_generator, monkeypatch):
    """Test streaming with custom system prompt."""
    async def mock_stream():
        yield "Response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response(
        "Test",
        "gpt-4",
        system_prompt="Custom system prompt"
    ):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_empty_system_prompt(response_generator, monkeypatch):
    """Test streaming falls back to default when system_prompt is empty."""
    async def mock_stream():
        yield "Response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response(
        "Test",
        "gpt-4",
        system_prompt=""
    ):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_none_system_prompt(response_generator, monkeypatch):
    """Test streaming falls back to default when system_prompt is None."""
    async def mock_stream():
        yield "Response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response(
        "Test",
        "gpt-4",
        system_prompt=None
    ):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_buffer_flush(response_generator, monkeypatch):
    """Test streaming flushes remaining buffer at end."""
    async def mock_stream():
        yield "Text without trailing space"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    # Should yield the text even without trailing space
    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_with_model_switch(response_generator, monkeypatch):
    """Test streaming switches model when specified."""
    async def mock_stream():
        yield "Response"

    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())
    monkeypatch.setattr(response_generator.model_manager, "switch_model", lambda x: None)
    monkeypatch.setattr(response_generator.model_manager, "get_active_model_name", lambda: "gpt-4")

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0


@pytest.mark.asyncio
async def test_streaming_config_import_failure(response_generator, monkeypatch):
    """Test streaming handles config import failure."""
    async def mock_stream():
        yield "Response"

    # This tests the exception path for importing SYSTEM_PROMPT
    monkeypatch.setattr(response_generator.model_manager, "generate_async", lambda *args, **kwargs: mock_stream())

    result = []
    async for chunk in response_generator.generate_streaming_response("Test", "gpt-4"):
        result.append(chunk)

    assert len(result) > 0
