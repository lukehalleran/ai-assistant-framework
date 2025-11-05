"""
Unit tests for utils/query_checker.py

Tests all pure functions that don't require external dependencies:
- Text normalization
- Query type detection (question, command, deictic, meta-conversational)
- Temporal window extraction
- Keyword extraction
- Thread continuity helpers
- Heavy topic classification (heuristic only)
"""

import pytest
from utils.query_checker import (
    _normalize,
    is_deictic,
    is_deictic_followup,
    is_question,
    is_command,
    is_meta_conversational,
    extract_temporal_window,
    keyword_tokens,
    analyze_query,
    extract_thread_keywords,
    has_thread_break_marker,
    calculate_thread_continuity_score,
    belongs_to_thread,
    _is_heavy_topic_heuristic,
    _parse_heavy_topic_response,
)


# =============================================================================
# Text Normalization Tests
# =============================================================================

def test_normalize_basic():
    assert _normalize("Hello World") == "hello world"


def test_normalize_strips_whitespace():
    assert _normalize("  Hello  ") == "hello"


def test_normalize_empty_string():
    assert _normalize("") == ""


def test_normalize_none():
    assert _normalize(None) == ""


# =============================================================================
# is_question Tests
# =============================================================================

def test_is_question_with_question_mark():
    assert is_question("What is Python?") == True


def test_is_question_with_how():
    assert is_question("how do I test?") == True


def test_is_question_with_what():
    assert is_question("What about this?") == True


def test_is_question_with_who():
    assert is_question("who wrote this?") == True


def test_is_question_with_when():
    assert is_question("when did it happen?") == True


def test_is_question_with_where():
    assert is_question("where is the file?") == True


def test_is_question_with_why():
    assert is_question("why does this work?") == True


def test_is_question_with_which():
    assert is_question("which option is best?") == True


def test_is_question_negative():
    assert is_question("This is a statement") == False


def test_is_question_case_insensitive():
    assert is_question("WHAT IS THIS") == True


# =============================================================================
# is_command Tests
# =============================================================================

def test_is_command_with_slash():
    assert is_command("/help") == True


def test_is_command_with_please():
    assert is_command("please do this") == True


def test_is_command_with_do():
    assert is_command("do this for me") == True


def test_is_command_with_create():
    assert is_command("create a file") == True


def test_is_command_with_generate():
    assert is_command("generate a report") == True


def test_is_command_with_write():
    assert is_command("write some code") == True


def test_is_command_with_summarize():
    assert is_command("summarize this article") == True


def test_is_command_negative():
    assert is_command("This is just a statement") == False


def test_is_command_case_insensitive():
    assert is_command("PLEASE HELP") == True


# =============================================================================
# is_deictic Tests
# =============================================================================

def test_is_deictic_short_with_that():
    assert is_deictic("what about that") == True


def test_is_deictic_short_with_explain():
    assert is_deictic("explain it") == True


def test_is_deictic_starts_with_that():
    assert is_deictic("that makes sense") == True


def test_is_deictic_starts_with_this():
    assert is_deictic("this is interesting") == True


def test_is_deictic_starts_with_it():
    assert is_deictic("it works") == True


def test_is_deictic_negative_long_sentence():
    # Note: starts with "This" so is_deictic returns True (as designed)
    # Using a different sentence that doesn't start with deictic markers
    assert is_deictic("Machine learning is a completely new topic") == False


def test_is_deictic_negative_no_markers():
    assert is_deictic("What is machine learning?") == False


# =============================================================================
# is_deictic_followup Tests
# =============================================================================

def test_is_deictic_followup_with_explain():
    assert is_deictic_followup("explain that") == True


def test_is_deictic_followup_with_elaborate():
    assert is_deictic_followup("can you elaborate?") == True


def test_is_deictic_followup_with_more():
    assert is_deictic_followup("tell me more") == True


def test_is_deictic_followup_negative():
    assert is_deictic_followup("What is Python?") == False


# =============================================================================
# is_meta_conversational Tests
# =============================================================================

def test_is_meta_conversational_do_you_recall():
    assert is_meta_conversational("do you recall what we discussed?") == True


def test_is_meta_conversational_do_you_remember():
    assert is_meta_conversational("do you remember yesterday?") == True


def test_is_meta_conversational_did_we():
    assert is_meta_conversational("didn't we talk about this?") == True


def test_is_meta_conversational_we_discussed():
    assert is_meta_conversational("we discussed this last week") == True


def test_is_meta_conversational_you_said():
    assert is_meta_conversational("you said something about that") == True


def test_is_meta_conversational_you_mentioned():
    assert is_meta_conversational("you mentioned a solution earlier") == True


def test_is_meta_conversational_last_time():
    assert is_meta_conversational("last time we talked about Python") == True


def test_is_meta_conversational_negative():
    assert is_meta_conversational("What is Python?") == False


# =============================================================================
# extract_temporal_window Tests
# =============================================================================

def test_extract_temporal_window_yesterday():
    assert extract_temporal_window("what did we discuss yesterday") == 1


def test_extract_temporal_window_last_night():
    assert extract_temporal_window("last night we talked about") == 1


def test_extract_temporal_window_days_ago():
    assert extract_temporal_window("a few days ago") == 3


def test_extract_temporal_window_last_week():
    assert extract_temporal_window("remember last week") == 7


def test_extract_temporal_window_last_month():
    assert extract_temporal_window("last month we discussed") == 30


def test_extract_temporal_window_specific_day():
    assert extract_temporal_window("on monday we talked") == 7


def test_extract_temporal_window_no_markers():
    assert extract_temporal_window("how do I code") == 0


def test_extract_temporal_window_empty():
    assert extract_temporal_window("") == 0


def test_extract_temporal_window_explicit_number():
    assert extract_temporal_window("5 days ago we talked") == 5


def test_extract_temporal_window_explicit_date():
    assert extract_temporal_window("on November 1st we met") == 30


def test_extract_temporal_window_multiple_markers():
    # Should return the largest window
    assert extract_temporal_window("yesterday and last week") == 7


# =============================================================================
# keyword_tokens Tests
# =============================================================================

def test_keyword_tokens_basic():
    result = keyword_tokens("hello world python")
    assert "hello" in result
    assert "world" in result
    assert "python" in result


def test_keyword_tokens_filters_short():
    result = keyword_tokens("hi to me is")
    assert "hi" not in result  # length 2
    assert "to" not in result  # length 2
    assert "me" not in result  # length 2
    assert "is" not in result  # length 2


def test_keyword_tokens_custom_min_length():
    result = keyword_tokens("hi hello world", min_len=5)
    assert "hi" not in result
    assert "hello" in result
    assert "world" in result


def test_keyword_tokens_case_insensitive():
    result = keyword_tokens("HELLO World PyThOn")
    assert "hello" in result
    assert "world" in result
    assert "python" in result


def test_keyword_tokens_empty():
    assert keyword_tokens("") == []


# =============================================================================
# extract_thread_keywords Tests
# =============================================================================

def test_extract_thread_keywords_basic():
    result = extract_thread_keywords("machine learning algorithms")
    assert "machine" in result
    assert "learning" in result
    assert "algorithms" in result


def test_extract_thread_keywords_filters_stopwords():
    result = extract_thread_keywords("the quick brown fox")
    assert "the" not in result  # stopword
    assert "quick" in result
    assert "brown" in result


def test_extract_thread_keywords_filters_short():
    result = extract_thread_keywords("machine learning is fun")
    assert "is" not in result  # too short
    assert "machine" in result
    assert "learning" in result
    assert "fun" in result


def test_extract_thread_keywords_punctuation():
    result = extract_thread_keywords("machine-learning, deep-learning!")
    # Should extract words, not punctuation
    assert "machine" in result
    assert "learning" in result
    assert "deep" in result


# =============================================================================
# has_thread_break_marker Tests
# =============================================================================

def test_has_thread_break_marker_changing_topics():
    assert has_thread_break_marker("changing topics, let's discuss") == True


def test_has_thread_break_marker_different_topic():
    assert has_thread_break_marker("on a different topic") == True


def test_has_thread_break_marker_switching_gears():
    assert has_thread_break_marker("switching gears now") == True


def test_has_thread_break_marker_anyway():
    assert has_thread_break_marker("anyway, what about Python?") == True


def test_has_thread_break_marker_moving_on():
    assert has_thread_break_marker("moving on to the next thing") == True


def test_has_thread_break_marker_negative():
    assert has_thread_break_marker("continuing the discussion") == False


# =============================================================================
# calculate_thread_continuity_score Tests
# =============================================================================

def test_thread_continuity_hard_cutoff():
    # More than 2 hours should return 0
    score = calculate_thread_continuity_score(
        current_query="about machine learning",
        last_query="machine learning is great",
        time_diff_seconds=7300,  # > 2 hours
    )
    assert score == 0.0


def test_thread_continuity_explicit_break():
    score = calculate_thread_continuity_score(
        current_query="changing topics, what about databases?",
        last_query="machine learning is great",
        time_diff_seconds=100,
    )
    assert score == 0.0


def test_thread_continuity_keyword_overlap():
    score = calculate_thread_continuity_score(
        current_query="tell me more about neural networks",
        last_query="neural networks are powerful",
        time_diff_seconds=60,  # Recent
    )
    # Should have positive score due to keyword overlap + time bonus
    assert score > 0.0


def test_thread_continuity_heavy_topics_both():
    score = calculate_thread_continuity_score(
        current_query="the raid was terrifying",
        last_query="people were arrested in the raid",
        time_diff_seconds=100,
        both_heavy=True,
    )
    # Should boost score for heavy topic continuity
    assert score > 0.0


def test_thread_continuity_same_topic_bonus():
    score = calculate_thread_continuity_score(
        current_query="more about databases",
        last_query="databases are useful",
        time_diff_seconds=100,
        same_topic=True,
        current_topic="databases"
    )
    # Should have topic bonus
    assert score > 0.0


# =============================================================================
# _is_heavy_topic_heuristic Tests
# =============================================================================

def test_heavy_topic_long_text():
    # Text over 2500 chars should be considered heavy
    long_text = "a" * 2600
    assert _is_heavy_topic_heuristic(long_text) == True


def test_heavy_topic_multiple_keywords():
    text = "The police raid resulted in multiple arrests and deportation"
    assert _is_heavy_topic_heuristic(text) == True


def test_heavy_topic_mental_health():
    text = "I'm feeling really depressed and having suicidal thoughts"
    assert _is_heavy_topic_heuristic(text) == True


def test_heavy_topic_crisis():
    text = "There was a shooting and people were wounded in the violence"
    assert _is_heavy_topic_heuristic(text) == True


def test_heavy_topic_normal_text():
    text = "How do I write a Python function?"
    assert _is_heavy_topic_heuristic(text) == False


def test_heavy_topic_empty():
    assert _is_heavy_topic_heuristic("") == False


def test_heavy_topic_none():
    assert _is_heavy_topic_heuristic(None) == False


# =============================================================================
# _parse_heavy_topic_response Tests
# =============================================================================

def test_parse_heavy_topic_response_heavy():
    assert _parse_heavy_topic_response("HEAVY") == True


def test_parse_heavy_topic_response_normal():
    assert _parse_heavy_topic_response("NORMAL") == False


def test_parse_heavy_topic_response_heavy_lowercase():
    assert _parse_heavy_topic_response("heavy") == True


def test_parse_heavy_topic_response_normal_lowercase():
    assert _parse_heavy_topic_response("normal") == False


def test_parse_heavy_topic_response_with_extra_text():
    assert _parse_heavy_topic_response("The answer is HEAVY because...") == True


def test_parse_heavy_topic_response_crisis():
    assert _parse_heavy_topic_response("CRISIS") == True


def test_parse_heavy_topic_response_empty():
    assert _parse_heavy_topic_response("") == False


def test_parse_heavy_topic_response_none():
    assert _parse_heavy_topic_response(None) == False


# =============================================================================
# analyze_query Tests (Integration of multiple functions)
# =============================================================================

def test_analyze_query_question():
    result = analyze_query("What is machine learning?")
    assert result.is_question == True
    assert "question" in result.intents
    assert result.is_command == False
    assert result.char_count > 0
    assert result.token_count > 0


def test_analyze_query_command():
    result = analyze_query("please create a report")
    assert result.is_command == True
    assert "command" in result.intents


def test_analyze_query_deictic():
    result = analyze_query("explain that")
    assert result.is_deictic == True
    assert result.is_followup == True


def test_analyze_query_meta_conversational():
    result = analyze_query("do you remember what we discussed?")
    assert result.is_meta_conversational == True
    assert "meta_conversational" in result.intents


def test_analyze_query_statement():
    result = analyze_query("The sky is blue")
    assert "statement" in result.intents


def test_analyze_query_heavy_topic():
    result = analyze_query("I'm feeling suicidal and depressed")
    assert result.is_heavy_topic == True


def test_analyze_query_normal_topic():
    result = analyze_query("How do I install Python?")
    assert result.is_heavy_topic == False


# =============================================================================
# belongs_to_thread Tests (Integration test)
# =============================================================================

def test_belongs_to_thread_recent_same_keywords():
    from datetime import datetime, timedelta

    last_conv = {
        "query": "tell me about neural networks",
        "response": "Neural networks are computational models",
        "timestamp": datetime.now() - timedelta(seconds=60),
        "is_heavy_topic": False,
        "topic": "machine_learning"
    }

    result = belongs_to_thread(
        current_query="how do neural networks learn?",
        last_conversation=last_conv,
        current_topic="machine_learning"
    )
    assert result == True


def test_belongs_to_thread_too_old():
    from datetime import datetime, timedelta

    last_conv = {
        "query": "tell me about neural networks",
        "response": "Neural networks are models",
        "timestamp": datetime.now() - timedelta(hours=3),  # Too old
        "is_heavy_topic": False,
        "topic": "machine_learning"
    }

    result = belongs_to_thread(
        current_query="how do neural networks learn?",
        last_conversation=last_conv,
        current_topic="machine_learning"
    )
    assert result == False


def test_belongs_to_thread_explicit_break():
    from datetime import datetime, timedelta

    last_conv = {
        "query": "tell me about neural networks",
        "response": "Neural networks are models",
        "timestamp": datetime.now() - timedelta(seconds=60),
        "is_heavy_topic": False,
        "topic": "machine_learning"
    }

    result = belongs_to_thread(
        current_query="changing topics, what about databases?",
        last_conversation=last_conv,
        current_topic="databases"
    )
    assert result == False


def test_belongs_to_thread_different_topics():
    from datetime import datetime, timedelta

    last_conv = {
        "query": "tell me about databases",
        "response": "Databases store data",
        "timestamp": datetime.now() - timedelta(seconds=60),
        "is_heavy_topic": False,
        "topic": "databases"
    }

    result = belongs_to_thread(
        current_query="what is machine learning?",
        last_conversation=last_conv,
        current_topic="machine_learning"
    )
    # Different topics with no keyword overlap should not continue thread
    assert result == False
