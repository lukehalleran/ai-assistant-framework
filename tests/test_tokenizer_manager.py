"""Test suite for TokenizerManager."""
import pytest
from models.tokenizer_manager import TokenizerManager
from models.model_manager import ModelManager


@pytest.fixture
def model_manager():
    """Fixture to provide a ModelManager instance."""
    return ModelManager()


@pytest.fixture
def tokenizer_manager(model_manager):
    """Fixture to provide a TokenizerManager instance."""
    return TokenizerManager(model_manager)


class TestTokenizerManager:
    """Tests for TokenizerManager class."""

    def test_initialization(self, model_manager):
        """Test TokenizerManager can be initialized."""
        tm = TokenizerManager(model_manager)
        assert tm is not None

    def test_count_tokens_basic(self, tokenizer_manager):
        """Test basic token counting."""
        tm = tokenizer_manager

        # Simple text
        text = "Hello world"
        count = tm.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self, tokenizer_manager):
        """Test token counting with empty string."""
        count = tokenizer_manager.count_tokens("")
        assert count == 0

    def test_count_tokens_long_text(self, tokenizer_manager):
        """Test token counting with longer text."""
        long_text = "This is a longer piece of text. " * 100
        count = tokenizer_manager.count_tokens(long_text)
        assert count > 100  # Should have many tokens

    def test_count_tokens_different_models(self, tokenizer_manager):
        """Test token counting for different model types."""
        text = "Hello, how are you?"

        # Test with different model identifiers if supported
        count1 = tokenizer_manager.count_tokens(text, model_name="gpt-4")
        count2 = tokenizer_manager.count_tokens(text, model_name="gpt-3.5-turbo")

        # Counts should be similar for similar models
        assert abs(count1 - count2) < 5

    def test_count_tokens_special_characters(self, tokenizer_manager):
        """Test token counting with special characters."""
        text = "Hello! How are you? 😊"
        count = tokenizer_manager.count_tokens(text)
        assert count > 0

    def test_count_tokens_code(self, tokenizer_manager):
        """Test token counting with code snippets."""
        code = """
def hello():
    print("Hello world")
    return 42
"""
        count = tokenizer_manager.count_tokens(code)
        assert count > 10

    def test_count_tokens_json(self, tokenizer_manager):
        """Test token counting with JSON data."""
        json_text = '{"name": "test", "value": 123, "nested": {"key": "value"}}'
        count = tokenizer_manager.count_tokens(json_text)
        assert count > 0

    def test_count_tokens_unicode(self, tokenizer_manager):
        """Test token counting with Unicode characters."""
        unicode_text = "Hello 世界 🌍"
        count = tokenizer_manager.count_tokens(unicode_text)
        assert count > 0

    def test_count_tokens_very_long(self, tokenizer_manager):
        """Test token counting with very long text."""
        # Create text that exceeds typical context windows
        very_long = "word " * 10000
        count = tokenizer_manager.count_tokens(very_long)
        assert count > 5000

    def test_count_tokens_none(self, tokenizer_manager):
        """Test token counting with None input."""
        count = tokenizer_manager.count_tokens(None)
        assert count == 0 or count is None

    def test_count_tokens_numbers(self, tokenizer_manager):
        """Test token counting with numbers."""
        numbers = "1234567890 " * 10
        count = tokenizer_manager.count_tokens(numbers)
        assert count > 0

    def test_truncate_to_token_limit(self, tokenizer_manager):
        """Test text truncation to token limit."""
        text = "This is a test sentence. " * 100
        limit = 50

        if hasattr(tokenizer_manager, 'truncate_to_token_limit'):
            truncated = tokenizer_manager.truncate_to_token_limit(text, limit)
            count = tokenizer_manager.count_tokens(truncated)
            assert count <= limit

    def test_estimate_tokens_per_message(self, tokenizer_manager):
        """Test token estimation for messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        if hasattr(tokenizer_manager, 'count_messages_tokens'):
            count = tokenizer_manager.count_messages_tokens(messages)
            assert count > 0

    def test_tokenizer_singleton_behavior(self, model_manager):
        """Test that tokenizer instances behave consistently."""
        tm1 = TokenizerManager(model_manager)
        tm2 = TokenizerManager(model_manager)

        text = "Test text for consistency"
        count1 = tm1.count_tokens(text)
        count2 = tm2.count_tokens(text)

        assert count1 == count2

    def test_count_tokens_markdown(self, tokenizer_manager):
        """Test token counting with Markdown."""
        markdown = """
# Header
## Subheader
- List item 1
- List item 2

**Bold text** and *italic text*
"""
        count = tokenizer_manager.count_tokens(markdown)
        assert count > 10

    def test_count_tokens_whitespace(self, tokenizer_manager):
        """Test token counting with various whitespace."""
        text = "word\n\n\nword\t\tword    word"
        count = tokenizer_manager.count_tokens(text)
        assert count > 0


class TestTokenizerManagerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_initialization_without_model_manager(self):
        """Test that TokenizerManager requires model_manager."""
        with pytest.raises(ValueError, match="model_manager is required"):
            TokenizerManager(None)

    def test_whitespace_tokenizer(self, model_manager):
        """Test whitespace tokenizer fallback."""
        from models.tokenizer_manager import _WhitespaceTokenizer
        tok = _WhitespaceTokenizer()
        tokens = tok.encode("hello world test")
        assert len(tokens) == 3
        assert tokens == ["hello", "world", "test"]

    def test_whitespace_tokenizer_empty(self):
        """Test whitespace tokenizer with empty string."""
        from models.tokenizer_manager import _WhitespaceTokenizer
        tok = _WhitespaceTokenizer()
        tokens = tok.encode("")
        # Empty string split returns empty list
        assert tokens == []

    def test_tiktoken_wrapper(self):
        """Test TiktokenWrapper if tiktoken is available."""
        try:
            import tiktoken
            from models.tokenizer_manager import _TiktokenWrapper
            enc = tiktoken.get_encoding("cl100k_base")
            wrapper = _TiktokenWrapper(enc)
            tokens = wrapper.encode("hello world")
            assert len(tokens) > 0
        except ImportError:
            pytest.skip("tiktoken not available")

    def test_choose_tiktoken_encoding_o_series(self):
        """Test encoding selection for o-series models."""
        from models.tokenizer_manager import _choose_tiktoken_encoding
        assert _choose_tiktoken_encoding("gpt-4o") == "o200k_base"
        assert _choose_tiktoken_encoding("gpt-4o-mini") == "o200k_base"
        assert _choose_tiktoken_encoding("o3-mini") == "o200k_base"

    def test_choose_tiktoken_encoding_standard(self):
        """Test encoding selection for standard models."""
        from models.tokenizer_manager import _choose_tiktoken_encoding
        assert _choose_tiktoken_encoding("gpt-4") == "cl100k_base"
        assert _choose_tiktoken_encoding("gpt-3.5-turbo") == "cl100k_base"
        assert _choose_tiktoken_encoding("claude-3") == "cl100k_base"

    def test_looks_like_api_model(self):
        """Test API model detection heuristic."""
        from models.tokenizer_manager import _looks_like_api_model
        assert _looks_like_api_model("gpt-4")
        assert _looks_like_api_model("gpt-3.5-turbo")
        assert _looks_like_api_model("claude-3-opus")
        assert _looks_like_api_model("gemini-pro")
        assert not _looks_like_api_model("mistralai/Mistral-7B")
        assert not _looks_like_api_model("meta-llama/Llama-2-7b")

    def test_is_api_model_with_exception(self, model_manager):
        """Test _is_api_model handles exceptions gracefully."""
        tm = TokenizerManager(model_manager)
        # Should not crash even if model_manager.is_api_model raises
        result = tm._is_api_model("gpt-4")
        assert isinstance(result, bool)

    def test_count_tokens_fallback(self, tokenizer_manager):
        """Test token counting fallback for unknown models."""
        # This should use character-based fallback
        count = tokenizer_manager.count_tokens("test text", "unknown-model-xyz")
        assert count > 0

    def test_get_tokenizer_caching(self, tokenizer_manager):
        """Test that tokenizers are cached."""
        tok1 = tokenizer_manager.get_tokenizer("gpt-4")
        tok2 = tokenizer_manager.get_tokenizer("gpt-4")
        # Should return the same cached instance
        assert tok1 is tok2


    def test_count_tokens_with_none_model(self, tokenizer_manager):
        """Test count_tokens when model_name is None."""
        count = tokenizer_manager.count_tokens("test", None)
        assert count > 0
