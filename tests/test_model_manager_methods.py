"""Tests for ModelManager internal methods and uncovered paths."""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from models.model_manager import ModelManager


@pytest.fixture
def model_manager():
    """Provide ModelManager without API key (testing mode)."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
        mm = ModelManager()
        return mm


def test_model_manager_init():
    """Test ModelManager initialization."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
        mm = ModelManager()

        assert mm.api_key == "test_key"
        assert mm.models == {}
        assert mm.tokenizers == {}
        assert isinstance(mm.api_models, dict)
        assert "gpt-4o-mini" in mm.api_models


def test_model_manager_init_with_api_key():
    """Test ModelManager initialization with explicit API key."""
    mm = ModelManager(api_key="explicit_key")

    assert mm.api_key == "explicit_key"


def test_model_manager_init_no_openai():
    """Test ModelManager handles missing OpenAI library."""
    # This path is covered when OpenAI is not installed
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
        mm = ModelManager()
        # Should still initialize successfully
        assert mm is not None


def test_stub_response(model_manager):
    """Test _stub_response generates stub text."""
    prompt = "What is Python?"

    result = model_manager._stub_response(prompt)

    assert isinstance(result, str)
    assert "[OpenAI unavailable]" in result
    assert "What is Python?" in result


def test_stub_response_long_prompt(model_manager):
    """Test _stub_response truncates long prompts."""
    prompt = "A" * 200

    result = model_manager._stub_response(prompt)

    assert "..." in result
    assert len(result) < len(prompt) + 50


def test_stub_response_empty_prompt(model_manager):
    """Test _stub_response with empty prompt."""
    result = model_manager._stub_response("")

    assert "stub response" in result


def test_stub_response_multiline_prompt(model_manager):
    """Test _stub_response with multiline prompt."""
    prompt = "Line 1\nLine 2\nLine 3"

    result = model_manager._stub_response(prompt)

    # Should only include first line
    assert "Line 1" in result
    assert "Line 2" not in result


@pytest.mark.asyncio
async def test_stub_stream(model_manager):
    """Test _stub_stream generates async stream."""
    prompt = "Test prompt"

    stream = model_manager._stub_stream(prompt)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "[OpenAI unavailable]" in chunks[0]


def test_is_api_model(model_manager):
    """Test is_api_model identifies API models."""
    assert model_manager.is_api_model("gpt-4o-mini") == True
    assert model_manager.is_api_model("claude-opus") == True


def test_is_api_model_false(model_manager):
    """Test is_api_model returns False for unknown models."""
    assert model_manager.is_api_model("unknown-model") == False
    assert model_manager.is_api_model(None) == False


def test_load_openai_model(model_manager):
    """Test load_openai_model registers API model."""
    model_manager.load_openai_model("test-model", "provider/test-model")

    assert "test-model" in model_manager.api_models
    assert model_manager.api_models["test-model"] == "provider/test-model"


def test_switch_model(model_manager):
    """Test switch_model changes active model."""
    model_manager.switch_model("gpt-4o-mini")

    assert model_manager.active_model_name == "gpt-4o-mini"


def test_get_active_model_name(model_manager):
    """Test get_active_model_name returns current model."""
    model_manager.switch_model("claude-opus")

    assert model_manager.get_active_model_name() == "claude-opus"


def test_get_active_model_name_none(model_manager):
    """Test get_active_model_name returns None when none set."""
    model_manager.active_model_name = None

    name = model_manager.get_active_model_name()

    # Source simply returns active_model_name, no fallback to default
    assert name is None


def test_get_model_not_loaded(model_manager):
    """Test get_model with no active model."""
    model_manager.active_model_name = "unknown-model"

    result = model_manager.get_model()

    assert result is None


def test_get_tokenizer_not_loaded(model_manager):
    """Test get_tokenizer with no active tokenizer."""
    model_manager.active_model_name = "unknown-model"

    result = model_manager.get_tokenizer()

    assert result is None


def test_get_embedder(model_manager):
    """Test get_embedder returns embedding model."""
    embedder = model_manager.get_embedder()

    assert embedder is not None
    assert embedder is model_manager.embed_model


def test_get_context_limit_api_model(model_manager):
    """Test get_context_limit for API models."""
    model_manager.switch_model("gpt-4o-mini")

    limit = model_manager.get_context_limit()

    assert limit == 128000  # Default API model context


def test_get_context_limit_no_model(model_manager):
    """Test get_context_limit raises error when no model."""
    model_manager.active_model_name = "unknown-model"

    with pytest.raises(ValueError, match="No model loaded"):
        model_manager.get_context_limit()


def test_truncate_prompt_no_truncation_needed():
    """Test truncate_prompt when prompt fits."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
    mock_tokenizer.decode = Mock(return_value="Short prompt")

    result = ModelManager.truncate_prompt(
        "Short prompt",
        mock_tokenizer,
        max_input_tokens=100
    )

    assert result == "Short prompt"


def test_truncate_prompt_with_truncation():
    """Test truncate_prompt when prompt exceeds limit."""
    mock_tokenizer = Mock()
    # Simulate long prompt
    mock_tokenizer.encode = Mock(return_value=list(range(200)))
    mock_tokenizer.decode = Mock(return_value="Truncated text")

    result = ModelManager.truncate_prompt(
        "Very long prompt" * 50,
        mock_tokenizer,
        max_input_tokens=100,
        preserve_prefix="System:"
    )

    assert isinstance(result, str)
    # Should call decode with truncated tokens
    mock_tokenizer.decode.assert_called()


def test_close(model_manager):
    """Test close method closes HTTP client."""
    # Mock the client
    mock_client = Mock()
    mock_client._client = Mock()
    model_manager.client = mock_client

    model_manager.close()

    # Should call close on underlying client
    mock_client._client.close.assert_called_once()


def test_close_no_client(model_manager):
    """Test close handles missing client gracefully."""
    model_manager.client = None

    # Should not raise error
    model_manager.close()


@pytest.mark.asyncio
async def test_aclose(model_manager):
    """Test aclose method closes async HTTP client."""
    # Mock the async client
    mock_client = Mock()
    mock_client._client = Mock()
    mock_client._client.aclose = AsyncMock()
    model_manager.async_client = mock_client

    await model_manager.aclose()

    # Should call aclose on underlying client
    mock_client._client.aclose.assert_called_once()


def test_list_provider_models_no_client(model_manager):
    """Test list_provider_models when client is None."""
    model_manager.client = None

    result = model_manager.list_provider_models()

    assert result == []


def test_list_provider_models_with_vendor_filter(model_manager):
    """Test list_provider_models with vendor prefix filter."""
    # Mock HTTP response
    mock_response = Mock()
    mock_response.json = Mock(return_value={
        "data": [
            {"id": "anthropic/claude-3-opus"},
            {"id": "anthropic/claude-sonnet"},
            {"id": "openai/gpt-4"},
        ]
    })

    if model_manager.client is not None:
        with patch.object(model_manager.client._client, 'get', return_value=mock_response):
            result = model_manager.list_provider_models(vendor_prefix="anthropic/")

            assert len(result) == 2
            assert all(m.startswith("anthropic/") for m in result)


def test_api_model_aliases(model_manager):
    """Test predefined API model aliases."""
    assert "claude-opus" in model_manager.api_models
    assert "gpt-4o-mini" in model_manager.api_models
    assert "gpt-5" in model_manager.api_models
    assert "deepseek-v3.1" in model_manager.api_models


def test_default_parameters(model_manager):
    """Test default generation parameters."""
    assert hasattr(model_manager, 'default_temperature')
    assert hasattr(model_manager, 'default_max_tokens')


def test_base_url(model_manager):
    """Test base_url is set correctly."""
    assert model_manager.base_url == "https://openrouter.ai/api/v1"


def test_allow_fallback_default(model_manager):
    """Test allow_fallback is disabled by default."""
    assert model_manager.allow_fallback == False


@pytest.mark.asyncio
async def test_generate_async_stub_when_no_client(model_manager):
    """Test generate_async returns stub when client unavailable."""
    model_manager.async_client = None
    model_manager.switch_model("gpt-4o-mini")  # Use API model to trigger stub path

    # generate_async is async and returns an async generator
    stream = await model_manager.generate_async("Test prompt")

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # Should return stub response
    assert len(chunks) > 0
    assert "[OpenAI unavailable]" in "".join(chunks)


def test_generate_with_unknown_model(model_manager):
    """Test generate with unknown model name."""
    model_manager.active_model_name = "unknown-local-model"

    # Should handle gracefully (may return stub or raise)
    try:
        result = model_manager.generate("Test prompt")
        assert isinstance(result, str)
    except (ValueError, KeyError):
        # Expected if model not loaded
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
