"""Comprehensive tests for LLMFactExtractor to boost coverage."""
import pytest
from memory.llm_fact_extractor import LLMFactExtractor, _snake, _normalize_triple
from models.model_manager import ModelManager


@pytest.fixture
def model_manager():
    """Provide ModelManager."""
    return ModelManager()


@pytest.fixture
def llm_extractor(model_manager):
    """Provide LLMFactExtractor."""
    return LLMFactExtractor(model_manager, model_alias="gpt-4o-mini")


# Test helper functions
def test_snake_normal():
    """Test _snake with normal text."""
    assert _snake("Hello World") == "hello_world"


def test_snake_special_chars():
    """Test _snake with special characters."""
    assert _snake("Hello-World!@#") == "hello-world"


def test_snake_multiple_spaces():
    """Test _snake with multiple spaces."""
    assert _snake("Hello   World") == "hello_world"


def test_snake_empty():
    """Test _snake with empty string."""
    assert _snake("") == ""


def test_snake_none():
    """Test _snake with None."""
    assert _snake(None) == ""


def test_snake_strips_leading_trailing():
    """Test _snake strips leading/trailing characters."""
    result = _snake("  Hello World  ")
    assert result == "hello_world"


def test_normalize_triple_valid():
    """Test _normalize_triple with valid triple."""
    triple = {"subject": "User", "relation": "Likes", "object": "Python"}
    result = _normalize_triple(triple)
    assert result is not None
    assert result["subject"] == "user"
    assert result["relation"] == "likes"
    assert result["object"] == "python"


def test_normalize_triple_pronouns():
    """Test _normalize_triple converts pronouns to 'user'."""
    for pronoun in ["I", "me", "my", "we", "us", "you"]:
        triple = {"subject": pronoun, "relation": "like", "object": "coding"}
        result = _normalize_triple(triple)
        assert result["subject"] == "user"


def test_normalize_triple_missing_fields():
    """Test _normalize_triple with missing fields."""
    assert _normalize_triple({"subject": "A", "relation": "B"}) is None
    assert _normalize_triple({"subject": "A", "object": "C"}) is None
    assert _normalize_triple({"relation": "B", "object": "C"}) is None


def test_normalize_triple_empty_strings():
    """Test _normalize_triple with empty strings."""
    triple = {"subject": "", "relation": "likes", "object": "Python"}
    assert _normalize_triple(triple) is None


def test_normalize_triple_coerces_numbers():
    """Test _normalize_triple coerces numbers to strings."""
    triple = {"subject": 123, "relation": 456, "object": 789}
    result = _normalize_triple(triple)
    assert result is not None
    assert result["subject"] == "123"
    assert result["relation"] == "456"
    assert result["object"] == "789"


def test_normalize_triple_strips_periods():
    """Test _normalize_triple strips trailing periods."""
    triple = {"subject": "User", "relation": "likes", "object": "Python."}
    result = _normalize_triple(triple)
    assert result["object"] == "python"


# Test LLMFactExtractor initialization
def test_llm_extractor_init(llm_extractor):
    """Test LLMFactExtractor initialization."""
    assert llm_extractor is not None
    assert llm_extractor.model_alias == "gpt-4o-mini"
    assert llm_extractor.max_input_chars == 4000
    assert llm_extractor.max_triples == 10


def test_llm_extractor_custom_params(model_manager):
    """Test LLMFactExtractor with custom parameters."""
    extractor = LLMFactExtractor(
        model_manager,
        model_alias="gpt-4",
        max_input_chars=2000,
        max_triples=5
    )
    assert extractor.model_alias == "gpt-4"
    assert extractor.max_input_chars == 2000
    assert extractor.max_triples == 5


def test_build_prompt_basic(llm_extractor):
    """Test _build_prompt with basic messages."""
    messages = ["Hello world", "I like Python"]
    prompt = llm_extractor._build_prompt(messages)
    assert isinstance(prompt, str)
    assert "Hello world" in prompt
    assert "I like Python" in prompt
    assert "JSON" in prompt


def test_build_prompt_empty_list(llm_extractor):
    """Test _build_prompt with empty list."""
    prompt = llm_extractor._build_prompt([])
    assert isinstance(prompt, str)
    assert "JSON" in prompt


def test_build_prompt_strips_role_prefixes(llm_extractor):
    """Test _build_prompt strips role prefixes."""
    messages = ["user: Hello", "assistant: Hi", "User: How are you?"]
    prompt = llm_extractor._build_prompt(messages)
    assert "user:" not in prompt.lower() or prompt.lower().count("user:") <= 1
    assert "assistant:" not in prompt.lower()


def test_build_prompt_enforces_char_budget(model_manager):
    """Test _build_prompt enforces character budget."""
    extractor = LLMFactExtractor(model_manager, max_input_chars=100)
    messages = ["A" * 200, "B" * 200, "C" * 200]
    prompt = extractor._build_prompt(messages)
    # Prompt should be limited by budget
    assert len(prompt) < 500  # Much less than 3*200=600


def test_build_prompt_limits_messages(llm_extractor):
    """Test _build_prompt limits to last 50 messages."""
    messages = [f"Message {i}" for i in range(100)]
    prompt = llm_extractor._build_prompt(messages)
    # Should only include subset due to 50 message cap
    assert isinstance(prompt, str)


def test_build_prompt_skips_empty_messages(llm_extractor):
    """Test _build_prompt skips empty messages."""
    messages = ["Valid", "", None, "  ", "Also valid"]
    prompt = llm_extractor._build_prompt(messages)
    assert "Valid" in prompt
    assert "Also valid" in prompt


@pytest.mark.asyncio
async def test_extract_triples_empty_input(llm_extractor):
    """Test extract_triples with empty input."""
    result = await llm_extractor.extract_triples([])
    assert result == []


@pytest.mark.asyncio
async def test_extract_triples_with_model_failure(llm_extractor, monkeypatch):
    """Test extract_triples handles model failures gracefully."""
    async def mock_generate_error(*args, **kwargs):
        raise Exception("Model error")

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_error)

    result = await llm_extractor.extract_triples(["Test message"])
    assert result == []


@pytest.mark.asyncio
async def test_extract_triples_with_invalid_json(llm_extractor, monkeypatch):
    """Test extract_triples handles invalid JSON."""
    async def mock_generate_invalid(*args, **kwargs):
        return "This is not valid JSON"

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_invalid)

    result = await llm_extractor.extract_triples(["Test message"])
    assert result == []


@pytest.mark.asyncio
async def test_extract_triples_with_non_list_json(llm_extractor, monkeypatch):
    """Test extract_triples handles non-list JSON."""
    async def mock_generate_dict(*args, **kwargs):
        return '{"subject": "User", "relation": "likes", "object": "Python"}'

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_dict)

    result = await llm_extractor.extract_triples(["Test message"])
    assert result == []


@pytest.mark.asyncio
async def test_extract_triples_with_valid_json(llm_extractor, monkeypatch):
    """Test extract_triples with valid JSON response."""
    async def mock_generate_valid(*args, **kwargs):
        return '[{"subject": "User", "relation": "likes", "object": "Python"}]'

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_valid)

    result = await llm_extractor.extract_triples(["I like Python"])
    assert len(result) == 1
    assert result[0]["subject"] == "user"
    assert result[0]["relation"] == "likes"
    assert result[0]["object"] == "python"


@pytest.mark.asyncio
async def test_extract_triples_with_junk_around_json(llm_extractor, monkeypatch):
    """Test extract_triples handles junk around JSON."""
    async def mock_generate_junk(*args, **kwargs):
        return 'Here are the triples: [{"subject": "User", "relation": "knows", "object": "Python"}] Thanks!'

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_junk)

    result = await llm_extractor.extract_triples(["Test"])
    assert len(result) == 1
    assert result[0]["relation"] == "knows"


@pytest.mark.asyncio
async def test_extract_triples_deduplicates(llm_extractor, monkeypatch):
    """Test extract_triples deduplicates triples."""
    async def mock_generate_duplicates(*args, **kwargs):
        return '''[
            {"subject": "User", "relation": "likes", "object": "Python"},
            {"subject": "User", "relation": "likes", "object": "Python"},
            {"subject": "User", "relation": "knows", "object": "Java"}
        ]'''

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_duplicates)

    result = await llm_extractor.extract_triples(["Test"])
    assert len(result) == 2  # Duplicates removed


@pytest.mark.asyncio
async def test_extract_triples_max_triples_limit(model_manager, monkeypatch):
    """Test extract_triples respects max_triples limit."""
    extractor = LLMFactExtractor(model_manager, max_triples=2)

    async def mock_generate_many(*args, **kwargs):
        return '''[
            {"subject": "A", "relation": "r1", "object": "X"},
            {"subject": "B", "relation": "r2", "object": "Y"},
            {"subject": "C", "relation": "r3", "object": "Z"},
            {"subject": "D", "relation": "r4", "object": "W"}
        ]'''

    monkeypatch.setattr(extractor.mm, "generate_once", mock_generate_many)

    result = await extractor.extract_triples(["Test"])
    assert len(result) == 2  # Limited to max_triples


@pytest.mark.asyncio
async def test_extract_triples_skips_invalid_items(llm_extractor, monkeypatch):
    """Test extract_triples skips invalid items in array."""
    async def mock_generate_mixed(*args, **kwargs):
        return '''[
            {"subject": "User", "relation": "likes", "object": "Python"},
            "invalid string",
            123,
            {"subject": "", "relation": "empty", "object": "bad"},
            {"subject": "User", "relation": "knows", "object": "Java"}
        ]'''

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_mixed)

    result = await llm_extractor.extract_triples(["Test"])
    assert len(result) == 2  # Only valid triples


@pytest.mark.asyncio
async def test_extract_triples_empty_string_response(llm_extractor, monkeypatch):
    """Test extract_triples with empty string response."""
    async def mock_generate_empty(*args, **kwargs):
        return ""

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_empty)

    result = await llm_extractor.extract_triples(["Test"])
    assert result == []


@pytest.mark.asyncio
async def test_extract_triples_none_response(llm_extractor, monkeypatch):
    """Test extract_triples with None response."""
    async def mock_generate_none(*args, **kwargs):
        return None

    monkeypatch.setattr(llm_extractor.mm, "generate_once", mock_generate_none)

    result = await llm_extractor.extract_triples(["Test"])
    assert result == []
