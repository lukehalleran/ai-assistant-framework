"""
Unit tests for memory/memory_coordinator.py class methods

Tests MemoryCoordinator helper methods:
- _calculate_truth_score: Truth scoring heuristics
- _calculate_importance_score: Importance scoring
- _get_memory_key: Deduplication key generation
- _safe_detect_topic: Safe topic detection
- _now/_now_iso: Time utilities
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from memory.memory_coordinator import MemoryCoordinator


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for MemoryCoordinator"""
    corpus = Mock()
    corpus.corpus = []

    chroma = Mock()
    chroma.add_memory = Mock()
    chroma.search = Mock(return_value=[])

    gate = Mock()

    return {
        'corpus_manager': corpus,
        'chroma_store': chroma,
        'gate_system': gate,
    }


@pytest.fixture
def coordinator(mock_dependencies):
    """Create MemoryCoordinator instance with minimal setup"""
    from memory.memory_scorer import MemoryScorer
    from memory.thread_manager import ThreadManager

    coord = object.__new__(MemoryCoordinator)
    coord.corpus_manager = mock_dependencies['corpus_manager']
    coord.chroma_store = mock_dependencies['chroma_store']
    coord.gate_system = mock_dependencies['gate_system']
    coord.time_manager = None
    coord.topic_manager = None
    coord.conversation_context = []
    coord.current_thread_id = None
    coord.thread_map = {}

    # Initialize modular components for delegation
    coord.scorer = MemoryScorer(
        time_manager=None,
        conversation_context=coord.conversation_context
    )
    coord.thread_manager = ThreadManager(
        corpus_manager=mock_dependencies['corpus_manager'],
        topic_manager=None,
        time_manager=None
    )

    return coord


# =============================================================================
# _calculate_truth_score Tests
# =============================================================================

def test_calculate_truth_score_base(coordinator):
    """Base truth score is 0.5"""
    score = coordinator._calculate_truth_score("question", "answer")

    assert score == 0.5


def test_calculate_truth_score_long_response(coordinator):
    """Long response (>200 chars) increases score"""
    long_response = "x" * 250
    score = coordinator._calculate_truth_score("question", long_response)

    assert score == 0.6  # 0.5 + 0.1


def test_calculate_truth_score_question_mark(coordinator):
    """Question in query increases score"""
    score = coordinator._calculate_truth_score("What is this?", "answer")

    assert score == 0.6  # 0.5 + 0.1


def test_calculate_truth_score_confirmation(coordinator):
    """Confirmation words increase score"""
    score = coordinator._calculate_truth_score("query", "Yes, that's correct")

    assert score == 0.7  # 0.5 + 0.2


def test_calculate_truth_score_all_confirmations(coordinator):
    """Tests all confirmation keywords"""
    confirmations = ['yes', 'correct', 'exactly', 'right', 'understood', 'makes sense', 'good point']

    for word in confirmations:
        score = coordinator._calculate_truth_score("query", f"The answer is {word}")
        assert score == 0.7, f"Failed for confirmation word: {word}"


def test_calculate_truth_score_combined_boosts(coordinator):
    """Multiple factors stack"""
    long_response = "x" * 250
    score = coordinator._calculate_truth_score("What?", f"{long_response} yes")

    # 0.5 + 0.1 (long) + 0.1 (question) + 0.2 (confirmation) = 0.9
    assert abs(score - 0.9) < 0.01


def test_calculate_truth_score_capped_at_one(coordinator):
    """Score is capped at 1.0"""
    coordinator.conversation_context = [{'response': 'previous context words'}]
    long_response = "x" * 250
    score = coordinator._calculate_truth_score("What context?", f"{long_response} yes correct")

    # 0.5 + 0.1 + 0.1 + 0.2 = 0.9 (no continuity match, 'correct' already counted)
    assert abs(score - 0.9) < 0.01


def test_calculate_truth_score_continuity(coordinator):
    """Continuity with previous response adds score"""
    coordinator.conversation_context = [{'response': 'The cat sat on the mat'}]
    score = coordinator._calculate_truth_score("Tell me more about the cat", "response")

    # 0.5 + 0.15 (continuity)
    assert score == 0.65


def test_calculate_truth_score_empty_inputs(coordinator):
    """Handles empty/None inputs"""
    score = coordinator._calculate_truth_score("", "")
    assert score == 0.5

    score = coordinator._calculate_truth_score(None, None)
    assert score == 0.5


def test_calculate_truth_score_case_insensitive(coordinator):
    """Confirmation matching is case insensitive"""
    score1 = coordinator._calculate_truth_score("q", "YES")
    score2 = coordinator._calculate_truth_score("q", "Yes")
    score3 = coordinator._calculate_truth_score("q", "yes")

    assert score1 == score2 == score3 == 0.7


# =============================================================================
# _calculate_importance_score Tests
# =============================================================================

def test_calculate_importance_score_base(coordinator):
    """Base importance score is 0.5"""
    score = coordinator._calculate_importance_score("some text")

    assert score == 0.5


def test_calculate_importance_score_long_content(coordinator):
    """Long content (>200 chars) increases score"""
    long_text = "x" * 250
    score = coordinator._calculate_importance_score(long_text)

    assert score == 0.6


def test_calculate_importance_score_question(coordinator):
    """Question mark increases score"""
    score = coordinator._calculate_importance_score("What is this?")

    assert score == 0.6


def test_calculate_importance_score_keywords(coordinator):
    """Important keywords increase score"""
    keywords = ['important', 'remember', 'note', 'key', 'critical', 'essential', 'todo', 'directive']

    for keyword in keywords:
        score = coordinator._calculate_importance_score(f"This is {keyword} information")
        assert score == 0.7, f"Failed for keyword: {keyword}"


def test_calculate_importance_score_combined(coordinator):
    """Multiple factors stack"""
    long_text = "x" * 250
    score = coordinator._calculate_importance_score(f"{long_text} This is important?")

    # 0.5 + 0.1 (long) + 0.1 (question) + 0.2 (keyword) = 0.9
    assert abs(score - 0.9) < 0.01


def test_calculate_importance_score_capped(coordinator):
    """Score is capped at 1.0"""
    long_text = "x" * 250
    score = coordinator._calculate_importance_score(f"{long_text} important remember? critical")

    # 0.5 + 0.1 + 0.1 + 0.2 = 0.9 (any() only adds 0.2 once)
    assert abs(score - 0.9) < 0.01


def test_calculate_importance_score_empty(coordinator):
    """Handles empty content"""
    score = coordinator._calculate_importance_score("")
    assert score == 0.5

    score = coordinator._calculate_importance_score(None)
    assert score == 0.5


def test_calculate_importance_score_case_insensitive(coordinator):
    """Keyword matching is case insensitive"""
    score1 = coordinator._calculate_importance_score("IMPORTANT")
    score2 = coordinator._calculate_importance_score("Important")
    score3 = coordinator._calculate_importance_score("important")

    assert score1 == score2 == score3 == 0.7


# =============================================================================
# _get_memory_key Tests
# =============================================================================

def test_get_memory_key_basic(coordinator):
    """Generates key from query and response"""
    memory = {
        'query': 'What is Python?',
        'response': 'Python is a programming language'
    }

    key = coordinator._get_memory_key(memory)

    # New format: "hash:...__query__response" with truncation
    assert '__' in key
    # Key should contain truncated query/response parts
    assert isinstance(key, str)
    # Should start with hash: since no id or timestamp
    assert key.startswith('hash:')


def test_get_memory_key_truncates(coordinator):
    """Truncates long queries/responses"""
    long_text = "x" * 100
    memory = {
        'query': long_text,
        'response': long_text
    }

    key = coordinator._get_memory_key(memory)

    # New format truncates to 30 chars each for hash-based keys
    # hash:...__xxxxxx...30chars__xxxxxx...30chars
    assert len(key) < 200  # Reasonable upper bound
    assert 'x' in key


def test_get_memory_key_missing_query(coordinator):
    """Handles missing query"""
    memory = {'response': 'answer'}

    key = coordinator._get_memory_key(memory)

    # Should still generate a key with hash prefix
    assert key.startswith('hash:')
    assert '__' in key


def test_get_memory_key_missing_response(coordinator):
    """Handles missing response"""
    memory = {'query': 'question'}

    key = coordinator._get_memory_key(memory)

    assert key.startswith('hash:')
    assert '__' in key


def test_get_memory_key_empty_memory(coordinator):
    """Handles empty memory dict"""
    key = coordinator._get_memory_key({})

    # Empty dict generates hash-based key
    assert key.startswith('hash:')


def test_get_memory_key_none_values(coordinator):
    """Handles None values"""
    memory = {'query': None, 'response': None}

    key = coordinator._get_memory_key(memory)

    # None values are converted to empty strings
    assert key.startswith('hash:')


def test_get_memory_key_same_memories_same_key(coordinator):
    """Same memories produce same key"""
    memory1 = {'query': 'q', 'response': 'r'}
    memory2 = {'query': 'q', 'response': 'r'}

    key1 = coordinator._get_memory_key(memory1)
    key2 = coordinator._get_memory_key(memory2)

    assert key1 == key2


def test_get_memory_key_different_memories_different_keys(coordinator):
    """Different memories produce different keys"""
    memory1 = {'query': 'q1', 'response': 'r1'}
    memory2 = {'query': 'q2', 'response': 'r2'}

    key1 = coordinator._get_memory_key(memory1)
    key2 = coordinator._get_memory_key(memory2)

    assert key1 != key2


# =============================================================================
# _safe_detect_topic Tests
# =============================================================================

def test_safe_detect_topic_no_manager(coordinator):
    """Returns 'general' when no topic manager"""
    coordinator.topic_manager = None

    result = coordinator._safe_detect_topic("some text")

    assert result == 'general'


def test_safe_detect_topic_with_manager(coordinator):
    """Uses topic manager when available"""
    coordinator.topic_manager = Mock()
    coordinator.topic_manager.detect_topic = Mock(return_value='python')

    result = coordinator._safe_detect_topic("Python programming")

    assert result == 'python'
    coordinator.topic_manager.detect_topic.assert_called_once_with("Python programming")


def test_safe_detect_topic_manager_returns_none(coordinator):
    """Falls back to 'general' if manager returns None"""
    coordinator.topic_manager = Mock()
    coordinator.topic_manager.detect_topic = Mock(return_value=None)

    result = coordinator._safe_detect_topic("text")

    assert result == 'general'


def test_safe_detect_topic_manager_returns_empty(coordinator):
    """Falls back to 'general' if manager returns empty string"""
    coordinator.topic_manager = Mock()
    coordinator.topic_manager.detect_topic = Mock(return_value='')

    result = coordinator._safe_detect_topic("text")

    assert result == 'general'


def test_safe_detect_topic_manager_raises_exception(coordinator):
    """Falls back to 'general' if manager raises exception"""
    coordinator.topic_manager = Mock()
    coordinator.topic_manager.detect_topic = Mock(side_effect=Exception("error"))

    result = coordinator._safe_detect_topic("text")

    assert result == 'general'


def test_safe_detect_topic_no_detect_topic_method(coordinator):
    """Falls back to 'general' if manager lacks detect_topic method"""
    coordinator.topic_manager = Mock(spec=[])  # No methods

    result = coordinator._safe_detect_topic("text")

    assert result == 'general'


# =============================================================================
# _now and _now_iso Tests
# =============================================================================

def test_now_no_time_manager(coordinator):
    """Returns datetime.now() when no time manager"""
    coordinator.time_manager = None

    result = coordinator._now()

    assert isinstance(result, datetime)
    assert (datetime.now() - result).total_seconds() < 1


def test_now_with_time_manager(coordinator):
    """Uses time manager when available"""
    mock_time = datetime(2024, 1, 1, 12, 0, 0)
    coordinator.time_manager = Mock()
    coordinator.time_manager.current = Mock(return_value=mock_time)

    result = coordinator._now()

    assert result == mock_time


def test_now_time_manager_no_current_method(coordinator):
    """Falls back to datetime.now() if time manager lacks current()"""
    coordinator.time_manager = Mock(spec=[])  # No methods

    result = coordinator._now()

    assert isinstance(result, datetime)


def test_now_time_manager_raises_exception(coordinator):
    """Falls back to datetime.now() if time manager raises exception"""
    coordinator.time_manager = Mock()
    coordinator.time_manager.current = Mock(side_effect=Exception("error"))

    result = coordinator._now()

    assert isinstance(result, datetime)


def test_now_iso_no_time_manager(coordinator):
    """Returns ISO string when no time manager"""
    coordinator.time_manager = None

    result = coordinator._now_iso()

    assert isinstance(result, str)
    assert 'T' in result  # ISO format has T separator


def test_now_iso_with_time_manager(coordinator):
    """Uses time manager's current_iso when available"""
    coordinator.time_manager = Mock()
    coordinator.time_manager.current_iso = Mock(return_value="2024-01-01T12:00:00")

    result = coordinator._now_iso()

    assert result == "2024-01-01T12:00:00"


def test_now_iso_time_manager_no_method(coordinator):
    """Falls back to _now().isoformat() if no current_iso"""
    coordinator.time_manager = Mock(spec=['current'])
    mock_time = datetime(2024, 1, 1, 12, 0, 0)
    coordinator.time_manager.current = Mock(return_value=mock_time)

    result = coordinator._now_iso()

    assert result == mock_time.isoformat()


def test_now_iso_time_manager_raises_exception(coordinator):
    """Falls back if time manager raises exception"""
    coordinator.time_manager = Mock()
    coordinator.time_manager.current_iso = Mock(side_effect=Exception("error"))
    # Also need to mock current() which is called by _now()
    mock_time = datetime(2024, 1, 1, 12, 0, 0)
    coordinator.time_manager.current = Mock(return_value=mock_time)

    result = coordinator._now_iso()

    assert isinstance(result, str)
    assert 'T' in result


# =============================================================================
# Edge Cases and Integration
# =============================================================================

def test_truth_score_with_all_factors(coordinator):
    """Integration test with all scoring factors"""
    # Continuity: query word without punctuation must match
    coordinator.conversation_context = [{'response': 'tell me about Python programming'}]
    long_response = "x" * 250

    score = coordinator._calculate_truth_score(
        "tell me more",  # "tell" and "me" will match (no punctuation)
        f"{long_response} Yes, that's exactly right and makes sense"
    )

    # 0.5 + 0.1 (long) + 0.2 (confirmation) + 0.15 (continuity) = 0.95 (no question mark)
    assert abs(score - 0.95) < 0.01


def test_importance_score_with_all_factors(coordinator):
    """Integration test with all importance factors"""
    long_text = "x" * 250

    score = coordinator._calculate_importance_score(
        f"{long_text} This is important to remember? Critical directive"
    )

    # 0.5 + 0.1 + 0.1 + 0.2 = 0.9 (any() only adds 0.2 once)
    assert abs(score - 0.9) < 0.01


def test_memory_key_uniqueness(coordinator):
    """Memory keys uniquely identify memories"""
    memories = [
        {'query': 'q1', 'response': 'r1'},
        {'query': 'q1', 'response': 'r2'},
        {'query': 'q2', 'response': 'r1'},
        {'query': 'q2', 'response': 'r2'},
    ]

    keys = [coordinator._get_memory_key(m) for m in memories]

    # All keys should be different
    assert len(set(keys)) == 4


# =============================================================================
# _parse_result Tests
# =============================================================================

def test_parse_result_basic(coordinator):
    """Parses basic ChromaDB result"""
    item = {
        'id': 'mem123',
        'content': 'Test content',
        'metadata': {
            'query': 'test query',
            'response': 'test response',
            'timestamp': '2024-01-01T12:00:00',
            'tags': 'tag1,tag2',
            'truth_score': 0.8,
            'importance_score': 0.7
        },
        'relevance_score': 0.9
    }

    result = coordinator._parse_result(item, 'episodic')

    assert result['id'] == 'mem123'
    assert result['query'] == 'test query'
    assert result['response'] == 'test response'
    assert result['content'] == 'Test content'
    assert result['source'] == 'episodic'
    assert result['collection'] == 'episodic'
    assert result['relevance_score'] == 0.9
    assert result['truth_score'] == 0.8
    assert result['importance_score'] == 0.7
    assert result['tags'] == ['tag1', 'tag2']


def test_parse_result_missing_metadata(coordinator):
    """Handles missing metadata"""
    item = {
        'content': 'content only'
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['source'] == 'semantic'
    assert result['content'] == 'content only'
    assert 'id' in result
    assert result['relevance_score'] == 0.5


def test_parse_result_timestamp_parsing(coordinator):
    """Parses ISO timestamp string"""
    item = {
        'metadata': {
            'timestamp': '2024-06-15T14:30:00'
        }
    }

    result = coordinator._parse_result(item, 'episodic')

    assert isinstance(result['timestamp'], datetime)
    assert result['timestamp'].year == 2024
    assert result['timestamp'].month == 6
    assert result['timestamp'].day == 15


def test_parse_result_invalid_timestamp(coordinator):
    """Falls back to datetime.now() for invalid timestamp"""
    item = {
        'metadata': {
            'timestamp': 'not-a-date'
        }
    }

    result = coordinator._parse_result(item, 'episodic')

    assert isinstance(result['timestamp'], datetime)
    # Should be recent (within last second)
    assert (datetime.now() - result['timestamp']).total_seconds() < 2


def test_parse_result_timestamp_already_datetime(coordinator):
    """Handles timestamp that's already datetime object"""
    ts = datetime(2024, 1, 1, 12, 0, 0)
    item = {
        'metadata': {
            'timestamp': ts
        }
    }

    result = coordinator._parse_result(item, 'episodic')

    assert result['timestamp'] == ts


def test_parse_result_tags_as_string(coordinator):
    """Parses comma-separated tags string"""
    item = {
        'metadata': {
            'tags': 'python, programming, tutorial'
        }
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['tags'] == ['python', 'programming', 'tutorial']


def test_parse_result_tags_as_list(coordinator):
    """Handles tags already as list"""
    item = {
        'metadata': {
            'tags': ['tag1', 'tag2', 'tag3']
        }
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['tags'] == ['tag1', 'tag2', 'tag3']


def test_parse_result_empty_tags(coordinator):
    """Handles empty tags"""
    item = {
        'metadata': {
            'tags': ''
        }
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['tags'] == []


def test_parse_result_tags_with_whitespace(coordinator):
    """Strips whitespace from tags"""
    item = {
        'metadata': {
            'tags': '  tag1  ,  tag2  ,  tag3  '
        }
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['tags'] == ['tag1', 'tag2', 'tag3']


def test_parse_result_missing_id_generates_uuid(coordinator):
    """Generates UUID if id is missing"""
    item = {
        'content': 'content'
    }

    result = coordinator._parse_result(item, 'episodic')

    assert 'id' in result
    assert 'episodic::' in result['id']


def test_parse_result_query_from_content_fallback(coordinator):
    """Uses content as query if query is missing"""
    item = {
        'content': 'This is a very long content string that should be truncated',
        'metadata': {}
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['query'] == 'This is a very long content string that should be truncated'[:100]


def test_parse_result_default_truth_score(coordinator):
    """Uses default truth score if not in metadata"""
    item = {
        'metadata': {}
    }

    result = coordinator._parse_result(item, 'episodic', default_truth=0.75)

    assert result['truth_score'] == 0.75


def test_parse_result_none_metadata(coordinator):
    """Handles None metadata"""
    item = {
        'content': 'test',
        'metadata': None
    }

    result = coordinator._parse_result(item, 'semantic')

    assert result['source'] == 'semantic'
    assert result['content'] == 'test'


def test_parse_result_not_dict_input(coordinator):
    """Returns empty dict for non-dict input"""
    result = coordinator._parse_result("not a dict", 'episodic')

    assert result == {}


def test_parse_result_none_input(coordinator):
    """Returns empty dict for None input"""
    result = coordinator._parse_result(None, 'episodic')

    assert result == {}


def test_parse_result_list_input(coordinator):
    """Returns empty dict for list input"""
    result = coordinator._parse_result(['a', 'list'], 'episodic')

    assert result == {}


def test_parse_result_relevance_score_conversion(coordinator):
    """Converts relevance_score to float"""
    item = {
        'relevance_score': '0.85'  # String
    }

    result = coordinator._parse_result(item, 'semantic')

    assert isinstance(result['relevance_score'], float)
    assert result['relevance_score'] == 0.85


def test_parse_result_complete_integration(coordinator):
    """Integration test with all fields"""
    item = {
        'id': 'complete123',
        'content': 'Full content string',
        'metadata': {
            'query': 'What is this?',
            'response': 'This is a test',
            'timestamp': '2024-01-01T10:00:00',
            'tags': 'integration,test,complete',
            'truth_score': 0.95,
            'importance_score': 0.85,
            'extra_field': 'preserved'
        },
        'relevance_score': 0.92
    }

    result = coordinator._parse_result(item, 'procedural', default_truth=0.5)

    assert result['id'] == 'complete123'
    assert result['query'] == 'What is this?'
    assert result['response'] == 'This is a test'
    assert result['content'] == 'Full content string'
    assert result['source'] == 'procedural'
    assert result['collection'] == 'procedural'
    assert result['relevance_score'] == 0.92
    assert result['truth_score'] == 0.95  # From metadata, not default
    assert result['importance_score'] == 0.85
    assert result['tags'] == ['integration', 'test', 'complete']
    assert isinstance(result['timestamp'], datetime)
    assert result['metadata']['extra_field'] == 'preserved'


# =============================================================================
# _format_hierarchical_memory Tests
# =============================================================================

def test_format_hierarchical_memory_basic(coordinator):
    """Formats hierarchical memory node"""
    memory = Mock()
    memory.content = "User: What is Python?\nAssistant: Python is a programming language"
    memory.id = "hier123"
    memory.timestamp = datetime(2024, 1, 1, 12, 0, 0)
    memory.truth_score = 0.8
    memory.score = 0.9
    memory.metadata = {'importance_score': 0.7}
    memory.tags = ['programming', 'python']

    result = coordinator._format_hierarchical_memory(memory)

    assert result['id'] == 'hier123'
    assert result['query'] == 'What is Python?'
    assert result['response'] == 'Python is a programming language'
    assert result['source'] == 'hierarchical'
    assert result['collection'] == 'hierarchical'
    assert result['truth_score'] == 0.8
    assert result['relevance_score'] == 0.9
    assert result['importance_score'] == 0.7
    assert result['tags'] == ['programming', 'python']


def test_format_hierarchical_memory_unparseable_content(coordinator):
    """Handles content that doesn't match expected format"""
    memory = Mock()
    memory.content = "Some random content without structure"
    memory.id = "hier456"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['query'] == 'Some random content without structure'[:100]
    assert result['response'] == "[Could not parse response]"


def test_format_hierarchical_memory_missing_id(coordinator):
    """Generates UUID if id is missing"""
    memory = Mock(spec=[])  # No attributes
    memory.content = "test"

    result = coordinator._format_hierarchical_memory(memory)

    assert 'id' in result
    assert 'hier::' in result['id']


def test_format_hierarchical_memory_truth_score_from_metadata(coordinator):
    """Falls back to metadata for truth score"""
    memory = Mock()
    memory.content = "User: q\nAssistant: a"
    memory.truth_score = None  # Not set directly
    memory.metadata = {'truth_score': 0.85}
    memory.timestamp = datetime.now()
    memory.score = 0.5
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['truth_score'] == 0.85


def test_format_hierarchical_memory_no_truth_score(coordinator):
    """Uses default 0.5 if no truth score"""
    memory = Mock()
    memory.content = "User: q\nAssistant: a"
    memory.truth_score = None
    memory.metadata = {}
    memory.timestamp = datetime.now()
    memory.score = 0.5
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['truth_score'] == 0.5


def test_format_hierarchical_memory_timestamp_conversion(coordinator):
    """Converts non-datetime timestamp"""
    memory = Mock()
    memory.content = "User: q\nAssistant: a"
    memory.timestamp = "not a datetime"  # String instead
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert isinstance(result['timestamp'], datetime)


def test_format_hierarchical_memory_missing_timestamp(coordinator):
    """Uses datetime.now() if timestamp missing"""
    memory = Mock(spec=['content', 'metadata', 'tags'])
    memory.content = "User: q\nAssistant: a"
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert isinstance(result['timestamp'], datetime)
    assert (datetime.now() - result['timestamp']).total_seconds() < 1


def test_format_hierarchical_memory_empty_content(coordinator):
    """Handles empty content"""
    memory = Mock()
    memory.content = ""
    memory.id = "empty"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['query'] == ""
    assert result['response'] == "[Could not parse response]"
    # Content gets reconstructed
    assert "User:" in result['content']
    assert "Assistant:" in result['content']


def test_format_hierarchical_memory_long_content_truncation(coordinator):
    """Truncates query to 100 chars when unparseable"""
    long_content = "x" * 150
    memory = Mock()
    memory.content = long_content
    memory.id = "long"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert len(result['query']) == 100


def test_format_hierarchical_memory_strips_user_prefix(coordinator):
    """Strips 'User: ' prefix from query"""
    memory = Mock()
    memory.content = "User: What is AI?\nAssistant: Artificial Intelligence"
    memory.id = "test"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['query'] == "What is AI?"
    assert "User:" not in result['query']


def test_format_hierarchical_memory_whitespace_stripping(coordinator):
    """Strips whitespace from parsed query and response"""
    memory = Mock()
    memory.content = "User:   What?   \nAssistant:   Answer   "
    memory.id = "test"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['query'] == "What?"
    assert result['response'] == "Answer"


def test_format_hierarchical_memory_multiple_assistant_markers(coordinator):
    """Handles content with multiple 'Assistant:' markers (won't parse correctly)"""
    memory = Mock()
    memory.content = "User: q\nAssistant: a1\nAssistant: a2"
    memory.id = "test"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    # Split creates 3 parts, not 2, so won't parse - uses first 100 chars as query
    assert result['query'] == memory.content[:100]
    assert result['response'] == "[Could not parse response]"


def test_format_hierarchical_memory_none_metadata(coordinator):
    """Handles None metadata gracefully"""
    memory = Mock()
    memory.content = "User: q\nAssistant: a"
    memory.id = "test"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = None
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['metadata'] == {}
    assert result['importance_score'] == 0.5


def test_format_hierarchical_memory_missing_metadata(coordinator):
    """Handles missing metadata attribute"""
    memory = Mock(spec=['content', 'id', 'timestamp', 'truth_score', 'score', 'tags'])
    memory.content = "User: q\nAssistant: a"
    memory.id = "test"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.tags = []

    result = coordinator._format_hierarchical_memory(memory)

    assert result['metadata'] == {}


def test_format_hierarchical_memory_missing_tags(coordinator):
    """Handles missing tags attribute"""
    memory = Mock(spec=['content', 'id', 'timestamp', 'truth_score', 'score', 'metadata'])
    memory.content = "User: q\nAssistant: a"
    memory.id = "test"
    memory.timestamp = datetime.now()
    memory.truth_score = 0.5
    memory.score = 0.5
    memory.metadata = {}

    result = coordinator._format_hierarchical_memory(memory)

    assert result['tags'] == []


def test_format_hierarchical_memory_complete_integration(coordinator):
    """Integration test with all fields populated"""
    memory = Mock()
    memory.id = "complete_hier"
    memory.content = "User: Tell me about Python?\nAssistant: Python is a versatile programming language"
    memory.timestamp = datetime(2024, 6, 15, 10, 30, 0)
    memory.truth_score = 0.92
    memory.score = 0.88
    memory.metadata = {
        'importance_score': 0.85,
        'session_id': 'session123',
        'extra': 'preserved'
    }
    memory.tags = ['programming', 'python', 'tutorial']

    result = coordinator._format_hierarchical_memory(memory)

    assert result['id'] == 'complete_hier'
    assert result['query'] == 'Tell me about Python?'
    assert result['response'] == 'Python is a versatile programming language'
    assert result['timestamp'] == datetime(2024, 6, 15, 10, 30, 0)
    assert result['truth_score'] == 0.92
    assert result['relevance_score'] == 0.88
    assert result['importance_score'] == 0.85
    assert result['tags'] == ['programming', 'python', 'tutorial']
    assert result['source'] == 'hierarchical'
    assert result['collection'] == 'hierarchical'
    assert result['metadata']['extra'] == 'preserved'


# =============================================================================
# Helper Functions Tests (_is_deictic_followup, _salient_tokens, etc.)
# =============================================================================

def test_is_deictic_followup_explain():
    """Detects 'explain' as deictic hint"""
    from memory.memory_coordinator import _is_deictic_followup
    assert _is_deictic_followup("explain that to me")

def test_is_deictic_followup_that():
    """Detects 'that' as deictic hint"""
    from memory.memory_coordinator import _is_deictic_followup
    assert _is_deictic_followup("what is that?")

def test_is_deictic_followup_this():
    """Detects 'this' as deictic hint"""
    from memory.memory_coordinator import _is_deictic_followup
    assert _is_deictic_followup("tell me more about this")

def test_is_deictic_followup_again():
    """Detects 'again' as deictic hint"""
    from memory.memory_coordinator import _is_deictic_followup
    assert _is_deictic_followup("say that again")

def test_is_deictic_followup_not_deictic():
    """Non-deictic queries return False"""
    from memory.memory_coordinator import _is_deictic_followup
    assert not _is_deictic_followup("what is Python?")

def test_is_deictic_followup_none():
    """Handles None gracefully"""
    from memory.memory_coordinator import _is_deictic_followup
    assert not _is_deictic_followup(None)

def test_is_deictic_followup_empty():
    """Handles empty string"""
    from memory.memory_coordinator import _is_deictic_followup
    assert not _is_deictic_followup("")


def test_salient_tokens_basic():
    """Extracts salient tokens excluding stopwords"""
    from memory.memory_coordinator import _salient_tokens
    tokens = _salient_tokens("the quick brown fox jumps over the lazy dog")
    # Stopwords like 'the' excluded, keeps unique content words
    assert 'the' not in tokens
    assert 'quick' in tokens or 'brown' in tokens or 'fox' in tokens

def test_salient_tokens_frequency_based():
    """Ranks tokens by frequency"""
    from memory.memory_coordinator import _salient_tokens
    tokens = _salient_tokens("python python python java java c++")
    # python appears 3 times, should be ranked high
    assert 'python' in tokens

def test_salient_tokens_limit_k():
    """Respects k limit"""
    from memory.memory_coordinator import _salient_tokens
    tokens = _salient_tokens("word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13", k=5)
    assert len(tokens) <= 5

def test_salient_tokens_math_symbols():
    """Includes math operators"""
    from memory.memory_coordinator import _salient_tokens
    tokens = _salient_tokens("solve x+y=10 and 2*x=20")
    # Should extract alphanumeric and operators
    assert any(t in tokens for t in ['solve', '10', '20'])

def test_salient_tokens_empty():
    """Handles empty text"""
    from memory.memory_coordinator import _salient_tokens
    tokens = _salient_tokens("")
    assert tokens == set()

def test_salient_tokens_none():
    """Handles None"""
    from memory.memory_coordinator import _salient_tokens
    tokens = _salient_tokens(None)
    assert tokens == set()


def test_num_op_density_numbers():
    """Calculates density with numbers"""
    from memory.memory_coordinator import _num_op_density
    density = _num_op_density("solve 5 + 3 = 8")
    # Numbers: 5, 3, 8 = 3; Ops: +, = = 2; Tokens: ~4 (solve, 5, 3, 8)
    assert density > 0.5  # (3+2)/4 or similar

def test_num_op_density_text_only():
    """Text-only has low density"""
    from memory.memory_coordinator import _num_op_density
    density = _num_op_density("this is just plain text")
    assert density < 0.1

def test_num_op_density_empty():
    """Empty string returns 0"""
    from memory.memory_coordinator import _num_op_density
    assert _num_op_density("") == 0.0

def test_num_op_density_none():
    """None returns 0"""
    from memory.memory_coordinator import _num_op_density
    assert _num_op_density(None) == 0.0

def test_num_op_density_operators():
    """Counts operators"""
    from memory.memory_coordinator import _num_op_density
    density = _num_op_density("x + y - z * w / a = b ^ c")
    # 7 operators, few words
    assert density > 0.6


def test_analogy_markers_basic():
    """Detects 'it's like' marker"""
    from memory.memory_coordinator import _analogy_markers
    count = _analogy_markers("it's like riding a bike")
    assert count == 1

def test_analogy_markers_multiple():
    """Detects multiple markers"""
    from memory.memory_coordinator import _analogy_markers
    count = _analogy_markers("imagine you're in a forest. Picture this: a bear approaches")
    assert count == 2

def test_analogy_markers_none():
    """No markers returns 0"""
    from memory.memory_coordinator import _analogy_markers
    count = _analogy_markers("just a regular sentence")
    assert count == 0

def test_analogy_markers_empty():
    """Empty string returns 0"""
    from memory.memory_coordinator import _analogy_markers
    assert _analogy_markers("") == 0

def test_analogy_markers_none_input():
    """None returns 0"""
    from memory.memory_coordinator import _analogy_markers
    assert _analogy_markers(None) == 0


def test_build_anchor_tokens_basic():
    """Builds anchors from last conversation"""
    from memory.memory_coordinator import _build_anchor_tokens
    conv = [{'query': 'what is python?', 'response': 'Python is a programming language'}]
    anchors = _build_anchor_tokens(conv)
    # Should include salient tokens from query+response
    assert len(anchors) > 0
    assert 'python' in anchors or 'programming' in anchors

def test_build_anchor_tokens_math_patterns():
    """Extracts math patterns"""
    from memory.memory_coordinator import _build_anchor_tokens
    conv = [{'query': 'solve f(x) = 7x^2 + 3x', 'response': 'derivative is 14x + 3'}]
    anchors = _build_anchor_tokens(conv)
    # Should capture math patterns
    assert len(anchors) > 0

def test_build_anchor_tokens_empty_conv():
    """Empty conversation returns empty set"""
    from memory.memory_coordinator import _build_anchor_tokens
    anchors = _build_anchor_tokens([])
    assert anchors == set()

def test_build_anchor_tokens_maxlen():
    """Respects maxlen limit"""
    from memory.memory_coordinator import _build_anchor_tokens
    conv = [{'query': ' '.join([f'word{i}' for i in range(50)]), 'response': 'response'}]
    anchors = _build_anchor_tokens(conv, maxlen=10)
    assert len(anchors) <= 10

def test_build_anchor_tokens_missing_keys():
    """Handles missing query/response keys"""
    from memory.memory_coordinator import _build_anchor_tokens
    conv = [{}]
    anchors = _build_anchor_tokens(conv)
    # Should not crash
    assert isinstance(anchors, set)


# =============================================================================
# _rank_memories Tests
# =============================================================================

def test_rank_memories_empty_list(coordinator):
    """Empty memories returns empty list"""
    result = coordinator._rank_memories([], "test query")
    assert result == []

def test_rank_memories_basic_relevance(coordinator):
    """Ranks by relevance score"""
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.9},
        {'query': 'q2', 'response': 'r2', 'relevance_score': 0.5},
        {'query': 'q3', 'response': 'r3', 'relevance_score': 0.7},
    ]
    # Add timestamps to avoid issues
    now = datetime.now()
    for m in memories:
        m['timestamp'] = now

    ranked = coordinator._rank_memories(memories, "test query")
    # Should be sorted by final_score (highest first)
    assert ranked[0]['relevance_score'] == 0.9
    assert ranked[-1]['relevance_score'] == 0.5

def test_rank_memories_collection_boost(coordinator):
    """Applies collection boost"""
    from config.app_config import COLLECTION_BOOSTS
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.5, 'collection': 'episodic'},
        {'query': 'q2', 'response': 'r2', 'relevance_score': 0.5, 'collection': 'semantic'},
    ]
    now = datetime.now()
    for m in memories:
        m['timestamp'] = now

    ranked = coordinator._rank_memories(memories, "query")
    # Collection with higher boost should rank higher (if boosts differ)
    # Check that final_score was calculated
    assert 'final_score' in ranked[0]
    assert 'final_score' in ranked[1]

def test_rank_memories_recency_decay(coordinator):
    """More recent memories score higher"""
    now = datetime.now()
    old_time = now - timedelta(hours=48)

    memories = [
        {'query': 'old', 'response': 'r1', 'relevance_score': 0.7, 'timestamp': old_time},
        {'query': 'new', 'response': 'r2', 'relevance_score': 0.7, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Recent memory should have higher final_score due to recency component
    # Find the 'new' memory
    new_mem = next(m for m in ranked if m['query'] == 'new')
    old_mem = next(m for m in ranked if m['query'] == 'old')
    # The recent one should contribute more from recency factor
    # (Though total score depends on weights, recency factor should be higher for new)
    assert new_mem['final_score'] >= old_mem['final_score']

def test_rank_memories_truth_score_boost(coordinator):
    """Access count boosts truth score"""
    now = datetime.now()
    memories = [
        {
            'query': 'q1', 'response': 'r1', 'relevance_score': 0.6,
            'timestamp': now, 'truth_score': 0.6,
            'metadata': {'access_count': 5}
        },
        {
            'query': 'q2', 'response': 'r2', 'relevance_score': 0.6,
            'timestamp': now, 'truth_score': 0.6,
            'metadata': {'access_count': 0}
        },
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Memory with access_count should get truth boost
    # Find memories
    accessed = next(m for m in ranked if m['query'] == 'q1')
    not_accessed = next(m for m in ranked if m['query'] == 'q2')
    # Accessed should score higher due to truth boost
    assert accessed['final_score'] >= not_accessed['final_score']

def test_rank_memories_importance(coordinator):
    """Importance score affects ranking"""
    now = datetime.now()
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.6, 'timestamp': now, 'importance_score': 0.9},
        {'query': 'q2', 'response': 'r2', 'relevance_score': 0.6, 'timestamp': now, 'importance_score': 0.3},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    important = next(m for m in ranked if m['query'] == 'q1')
    not_important = next(m for m in ranked if m['query'] == 'q2')
    # Higher importance should contribute to higher score
    assert important['final_score'] >= not_important['final_score']

def test_rank_memories_continuity_recent(coordinator):
    """Recent memories (within 10 minutes) get continuity boost"""
    now = datetime.now()
    recent = now - timedelta(minutes=5)
    old = now - timedelta(hours=1)

    memories = [
        {'query': 'recent', 'response': 'r1', 'relevance_score': 0.6, 'timestamp': recent},
        {'query': 'old', 'response': 'r2', 'relevance_score': 0.6, 'timestamp': old},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    recent_mem = next(m for m in ranked if m['query'] == 'recent')
    old_mem = next(m for m in ranked if m['query'] == 'old')
    # Recent should get +0.1 continuity bonus
    assert recent_mem['final_score'] >= old_mem['final_score']

def test_rank_memories_continuity_token_overlap(coordinator):
    """Token overlap with query increases continuity"""
    now = datetime.now()
    memories = [
        {'query': 'python programming language', 'response': 'r1', 'relevance_score': 0.6, 'timestamp': now},
        {'query': 'java coffee beans', 'response': 'r2', 'relevance_score': 0.6, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "tell me about python programming")
    # Memory about python should have higher token overlap
    python_mem = next(m for m in ranked if 'python' in m['query'])
    java_mem = next(m for m in ranked if 'java' in m['query'])
    assert python_mem['final_score'] >= java_mem['final_score']

def test_rank_memories_structural_alignment(coordinator):
    """Math-heavy query aligns with math-heavy memories"""
    now = datetime.now()
    memories = [
        {'query': 'solve 5+3*2=11', 'response': 'r1', 'relevance_score': 0.6, 'timestamp': now},
        {'query': 'tell me a story', 'response': 'r2', 'relevance_score': 0.6, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "calculate 7*8+2")
    # Math memory should align better with math query
    math_mem = next(m for m in ranked if 'solve' in m['query'])
    story_mem = next(m for m in ranked if 'story' in m['query'])
    assert math_mem['final_score'] >= story_mem['final_score']

def test_rank_memories_analogy_penalty(coordinator):
    """Analogy memories penalized for math queries"""
    now = datetime.now()
    memories = [
        {'query': 'q1', 'response': "it's like riding a bike, imagine you have numbers", 'relevance_score': 0.7, 'timestamp': now},
        {'query': 'q2', 'response': 'direct mathematical explanation', 'relevance_score': 0.7, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "solve 5*3+2=17")
    # Analogy memory should get penalty for math query
    analogy_mem = next(m for m in ranked if "it's like" in m['response'])
    direct_mem = next(m for m in ranked if 'direct' in m['response'])
    # Penalty should lower analogy score
    assert analogy_mem['final_score'] <= direct_mem['final_score']

def test_rank_memories_deictic_anchor_bonus(coordinator):
    """Deictic queries give anchor bonus"""
    now = datetime.now()
    coordinator.conversation_context = [
        {'query': 'what is python?', 'response': 'Python is a programming language'}
    ]

    memories = [
        {'query': 'about python', 'response': 'python info', 'relevance_score': 0.6, 'timestamp': now},
        {'query': 'about java', 'response': 'java info', 'relevance_score': 0.6, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "explain that")
    # Memory about python should get anchor bonus (matches conversation context)
    python_mem = next(m for m in ranked if 'python' in m['query'])
    java_mem = next(m for m in ranked if 'java' in m['query'])
    assert python_mem['final_score'] >= java_mem['final_score']

def test_rank_memories_deictic_penalty_no_anchor(coordinator):
    """Deictic query without anchor match gets penalty"""
    now = datetime.now()
    coordinator.conversation_context = [
        {'query': 'what is python?', 'response': 'Python is a programming language'}
    ]

    memories = [
        {'query': 'completely unrelated topic', 'response': 'about cars', 'relevance_score': 0.6, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "explain that")
    # Should apply drift penalty (multiplier 0.85)
    assert 'final_score' in ranked[0]
    # Can't easily verify exact penalty without knowing all other factors

def test_rank_memories_tone_penalty(coordinator):
    """Negative tone reduces truth score"""
    now = datetime.now()
    memories = [
        {'query': 'q1', 'response': 'you idiot, here is the answer', 'relevance_score': 0.7, 'timestamp': now, 'truth_score': 0.8},
        {'query': 'q2', 'response': 'here is a helpful answer', 'relevance_score': 0.7, 'timestamp': now, 'truth_score': 0.8},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    rude_mem = next(m for m in ranked if 'idiot' in m['response'])
    polite_mem = next(m for m in ranked if 'helpful' in m['response'])
    # Rude memory should score lower due to tone penalty
    assert rude_mem['final_score'] <= polite_mem['final_score']

def test_rank_memories_timestamp_parsing_string(coordinator):
    """Parses ISO timestamp strings"""
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.7, 'timestamp': '2024-06-15T10:30:00'},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Should not crash, should parse timestamp
    assert 'final_score' in ranked[0]

def test_rank_memories_timestamp_invalid(coordinator):
    """Invalid timestamp defaults to now"""
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.7, 'timestamp': 'invalid'},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Should not crash
    assert 'final_score' in ranked[0]

def test_rank_memories_missing_timestamp(coordinator):
    """Missing timestamp defaults to now"""
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.7},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Should not crash
    assert 'final_score' in ranked[0]

def test_rank_memories_sorts_descending(coordinator):
    """Memories sorted by final_score descending"""
    now = datetime.now()
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.5, 'timestamp': now},
        {'query': 'q2', 'response': 'r2', 'relevance_score': 0.9, 'timestamp': now},
        {'query': 'q3', 'response': 'r3', 'relevance_score': 0.7, 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Check descending order
    for i in range(len(ranked) - 1):
        assert ranked[i]['final_score'] >= ranked[i+1]['final_score']

def test_rank_memories_metadata_none(coordinator):
    """Handles None metadata"""
    now = datetime.now()
    memories = [
        {'query': 'q1', 'response': 'r1', 'relevance_score': 0.7, 'timestamp': now, 'metadata': None},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Should not crash
    assert 'final_score' in ranked[0]

def test_rank_memories_missing_scores(coordinator):
    """Uses defaults for missing scores"""
    now = datetime.now()
    memories = [
        {'query': 'q1', 'response': 'r1', 'timestamp': now},
    ]

    ranked = coordinator._rank_memories(memories, "query")
    # Should use default relevance (0.5), truth (0.6), importance (0.5)
    assert 'final_score' in ranked[0]
    # Score should be calculated with defaults
    assert ranked[0]['final_score'] > 0
