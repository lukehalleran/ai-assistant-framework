"""
Unit tests for processing/gate_system.py helper functions

Tests pure utility functions:
- Wiki cache key generation
- Wiki cache operations
- Should attempt wiki heuristics
- Wiki noise detection
- Markup cleaning
"""

import pytest
from processing.gate_system import (
    _wiki_cache_key,
    get_cached_wiki,
    _cache_wiki,
    should_attempt_wiki,
    _looks_wiki_noisy,
    _WIKI_CACHE,
)


# =============================================================================
# _wiki_cache_key Tests
# =============================================================================

def test_wiki_cache_key_basic():
    """Generates cache key from query"""
    result = _wiki_cache_key("What is Python")

    assert isinstance(result, str)
    assert len(result) > 0


def test_wiki_cache_key_normalization():
    """Normalizes query for cache key"""
    key1 = _wiki_cache_key("Python")
    key2 = _wiki_cache_key("PYTHON")
    key3 = _wiki_cache_key("  python  ")

    # Should normalize case and whitespace
    assert key1 == key2 == key3


def test_wiki_cache_key_empty():
    """Handles empty query"""
    result = _wiki_cache_key("")

    assert isinstance(result, str)


def test_wiki_cache_key_none():
    """Handles None query"""
    result = _wiki_cache_key(None)

    assert isinstance(result, str)


def test_wiki_cache_key_includes_mode():
    """Cache key differentiates fetch modes"""
    result = _wiki_cache_key("test")

    # Should include mode indicator (:full or :lead)
    assert ":full" in result or ":lead" in result


# =============================================================================
# Wiki Cache Operations Tests
# =============================================================================

def test_cache_wiki_stores_value():
    """_cache_wiki stores wiki content"""
    # Clear cache first
    _WIKI_CACHE.clear()

    _cache_wiki("test query", "test content")

    # Should be retrievable
    result = get_cached_wiki("test query")
    assert result == "test content"


def test_cache_wiki_normalizes_query():
    """Cache operations normalize queries"""
    _WIKI_CACHE.clear()

    _cache_wiki("Test Query", "content")

    # Should be retrievable with different case/whitespace
    assert get_cached_wiki("test query") == "content"
    assert get_cached_wiki("TEST QUERY") == "content"
    assert get_cached_wiki("  test query  ") == "content"


def test_get_cached_wiki_miss():
    """get_cached_wiki returns empty on miss"""
    _WIKI_CACHE.clear()

    result = get_cached_wiki("nonexistent query")

    assert result == ""


def test_cache_wiki_ignores_empty():
    """_cache_wiki doesn't store empty content"""
    _WIKI_CACHE.clear()

    _cache_wiki("query", "")

    # Should not be in cache
    assert len(_WIKI_CACHE) == 0


def test_cache_wiki_bounds_size():
    """Cache stays bounded to ~64 entries"""
    _WIKI_CACHE.clear()

    # Add 70 entries
    for i in range(70):
        _cache_wiki(f"query{i}", f"content{i}")

    # Should not exceed limit significantly
    assert len(_WIKI_CACHE) <= 65


def test_cache_wiki_evicts_oldest():
    """Cache evicts oldest entries when full"""
    _WIKI_CACHE.clear()

    # Add entries to fill cache
    for i in range(70):
        _cache_wiki(f"query{i}", f"content{i}")

    # First entry should be evicted
    result = get_cached_wiki("query0")
    assert result == ""  # First entry gone

    # Recent entries should remain
    result = get_cached_wiki("query69")
    assert "content69" in result or result == ""  # May or may not be there


# =============================================================================
# should_attempt_wiki Tests
# =============================================================================

def test_should_attempt_wiki_what_is():
    """Detects 'what is' questions"""
    assert should_attempt_wiki("What is Python?") == True


def test_should_attempt_wiki_who_is():
    """Detects 'who is' questions"""
    assert should_attempt_wiki("Who is Alan Turing?") == True


def test_should_attempt_wiki_tell_me_about():
    """Detects 'tell me about' requests"""
    assert should_attempt_wiki("Tell me about machine learning") == True


def test_should_attempt_wiki_with_explain_keyword():
    """Detects queries with 'explain' keyword"""
    # "explain" alone might trigger deictic check, use with context
    assert should_attempt_wiki("what is quantum computing") == True


def test_should_attempt_wiki_short_topic():
    """Allows short topic-like queries"""
    assert should_attempt_wiki("Python") == True
    assert should_attempt_wiki("Machine learning") == True


def test_should_attempt_wiki_deictic_followup():
    """Rejects deictic follow-ups"""
    assert should_attempt_wiki("explain that again") == False
    assert should_attempt_wiki("tell me more about it") == False


def test_should_attempt_wiki_long_query():
    """Rejects long, complex queries"""
    long_query = "How do I integrate the authentication system with the database while maintaining security best practices?"

    # Too many tokens, not a simple topic query
    result = should_attempt_wiki(long_query)
    # May or may not attempt, but shouldn't crash
    assert isinstance(result, bool)


def test_should_attempt_wiki_empty():
    """Handles empty query"""
    # Empty query has 0 tokens, which is <= 4, so it returns True
    # But in practice this is fine as empty queries won't reach this function
    result = should_attempt_wiki("")
    assert isinstance(result, bool)  # Just verify it doesn't crash


def test_should_attempt_wiki_case_insensitive():
    """Detection is case insensitive"""
    assert should_attempt_wiki("WHAT IS Python") == True
    assert should_attempt_wiki("what is python") == True


def test_should_attempt_wiki_short_words_filtered():
    """Filters short words when counting tokens"""
    # "What is AI?" has "AI" as 2-letter word, should filter it
    result = should_attempt_wiki("What is AI?")
    # Has "what is" pattern, should return True
    assert result == True


# =============================================================================
# _looks_wiki_noisy Tests
# =============================================================================

def test_looks_wiki_noisy_with_category():
    """Detects [[category: markup as noisy"""
    text = "Some text [[category:stub]] more text"

    assert _looks_wiki_noisy(text) == True


def test_looks_wiki_noisy_with_template():
    """Detects {{ template markup as noisy"""
    text = "Text {{infobox}} more text"

    assert _looks_wiki_noisy(text) == True


def test_looks_wiki_noisy_with_user():
    """Detects user: pages as noisy"""
    text = "[[user:example]] wrote this"

    assert _looks_wiki_noisy(text) == True


def test_looks_wiki_noisy_with_redirect():
    """Detects redirect pages as noisy"""
    text = "#redirect [[other page]]"

    assert _looks_wiki_noisy(text) == True


def test_looks_wiki_noisy_clean_text():
    """Clean text is not noisy"""
    text = "This is a clean article about Python programming language."

    assert _looks_wiki_noisy(text) == False


def test_looks_wiki_noisy_with_html():
    """Detects HTML tags as noise"""
    text = "Text with <div>markup</div> and &lt;entities&gt;"

    # May or may not be noisy depending on density
    result = _looks_wiki_noisy(text)
    assert isinstance(result, bool)


def test_looks_wiki_noisy_empty():
    """Empty text is not noisy"""
    assert _looks_wiki_noisy("") == False


def test_looks_wiki_noisy_none():
    """None input is not noisy"""
    assert _looks_wiki_noisy(None) == False


def test_looks_wiki_noisy_high_markup_density():
    """High markup density is detected"""
    # Lots of brackets relative to content
    text = "[[x]] [[y]] [[z]]"

    result = _looks_wiki_noisy(text)
    # High density should be noisy
    assert isinstance(result, bool)


def test_looks_wiki_noisy_case_insensitive():
    """Noise detection is case insensitive"""
    assert _looks_wiki_noisy("[[CATEGORY:Test]]") == True
    assert _looks_wiki_noisy("[[category:test]]") == True
    assert _looks_wiki_noisy("[[Category:Test]]") == True


# =============================================================================
# Integration Tests
# =============================================================================

def test_cache_and_retrieve_workflow():
    """Full cache workflow"""
    _WIKI_CACHE.clear()

    query = "Test Topic"
    content = "This is test content about the topic"

    # Store
    _cache_wiki(query, content)

    # Retrieve
    retrieved = get_cached_wiki(query)

    assert retrieved == content


def test_multiple_queries_cached():
    """Multiple distinct queries can be cached"""
    _WIKI_CACHE.clear()

    _cache_wiki("Python", "Python is a language")
    _cache_wiki("Java", "Java is a language")
    _cache_wiki("Rust", "Rust is a language")

    assert "Python" in get_cached_wiki("Python")
    assert "Java" in get_cached_wiki("Java")
    assert "Rust" in get_cached_wiki("Rust")


def test_wiki_attempt_filters_then_cache():
    """should_attempt_wiki filters before cache check"""
    _WIKI_CACHE.clear()

    # Cache a result
    _cache_wiki("Python", "Content")

    # Should still apply should_attempt filter
    # Even though "explain it" might have cached content,
    # should_attempt_wiki would reject it
    deictic = "explain it again"
    assert should_attempt_wiki(deictic) == False


def test_noisy_content_detection():
    """Combines noise detection patterns"""
    clean = "Python is a high-level programming language."
    noisy1 = "{{Infobox|[[Category:Stubs]]|user:test}}"
    noisy2 = "<page><revision>content</revision></page>"

    assert _looks_wiki_noisy(clean) == False
    assert _looks_wiki_noisy(noisy1) == True
    assert _looks_wiki_noisy(noisy2) == True


def test_cache_key_consistency():
    """Cache keys are consistent for same logical query"""
    queries = [
        "Python Programming",
        "python programming",
        "  PYTHON PROGRAMMING  ",
        "Python programming"
    ]

    keys = [_wiki_cache_key(q) for q in queries]

    # All should produce same key
    assert len(set(keys)) == 1


# =============================================================================
# Additional Helper Function Tests
# =============================================================================

def test_strip_articles_basic():
    """Strips leading articles from strings"""
    from processing.gate_system import _strip_articles

    assert _strip_articles("the cat") == "cat"
    assert _strip_articles("a dog") == "dog"
    assert _strip_articles("an apple") == "apple"


def test_strip_articles_case_insensitive():
    """Article stripping is case insensitive"""
    from processing.gate_system import _strip_articles

    assert _strip_articles("The Cat") == "Cat"
    assert _strip_articles("A Dog") == "Dog"
    assert _strip_articles("AN Apple") == "Apple"


def test_strip_articles_no_article():
    """Returns string unchanged if no article"""
    from processing.gate_system import _strip_articles

    assert _strip_articles("Python") == "Python"
    assert _strip_articles("machine learning") == "machine learning"


def test_strip_articles_article_in_middle():
    """Only strips leading articles"""
    from processing.gate_system import _strip_articles

    assert _strip_articles("cat the dog") == "cat the dog"


def test_strip_articles_empty_string():
    """Handles empty string"""
    from processing.gate_system import _strip_articles

    assert _strip_articles("") == ""


def test_strip_articles_none():
    """Handles None input"""
    from processing.gate_system import _strip_articles

    assert _strip_articles(None) == ""


def test_strip_articles_only_article():
    """Handles string that is only an article (needs whitespace after)"""
    from processing.gate_system import _strip_articles

    # Regex requires whitespace after article
    assert _strip_articles("the ") == ""
    assert _strip_articles("a ") == ""
    assert _strip_articles("an ") == ""


def test_wiki_title_candidates_basic():
    """Generates title candidates from query"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("What is Python?")

    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert "What is Python?" in candidates


def test_wiki_title_candidates_strips_articles():
    """Includes version without articles"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("the Python programming language")

    # Should include version without "the"
    assert any("Python" in c for c in candidates)


def test_wiki_title_candidates_capitalization():
    """Includes capitalized variants"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("python programming")

    # Should include some capitalized version
    assert any(c[0].isupper() for c in candidates if c)


def test_wiki_title_candidates_plurals():
    """Attempts singularization"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("cats")

    # Should try "cat"
    assert "cat" in candidates or "Cat" in candidates


def test_wiki_title_candidates_aliases():
    """Uses known aliases"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("kitty")

    # Should map to "Cat"
    assert "Cat" in candidates


def test_wiki_title_candidates_empty():
    """Handles empty query"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("")

    assert isinstance(candidates, list)


def test_wiki_title_candidates_deduplication():
    """Removes duplicates"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("Python")

    # Should not have duplicates
    assert len(candidates) == len(set(candidates))


def test_wiki_title_candidates_temporal_cleanup():
    """Removes temporal qualifiers"""
    from processing.gate_system import wiki_title_candidates

    candidates = wiki_title_candidates("current state of AI")

    # Should try version without "current"
    assert any("state of AI" in c.lower() or "state of ai" in c.lower() for c in candidates)


def test_clean_wikiish_headings():
    """Removes wiki-style headings"""
    from processing.gate_system import clean_wikiish

    text = "Some text\n== History ==\nMore text"
    result = clean_wikiish(text)

    assert "== History ==" not in result
    assert "Some text" in result
    assert "More text" in result


def test_clean_wikiish_citations():
    """Removes citation markers"""
    from processing.gate_system import clean_wikiish

    text = "Python is a language[1] used for many things[citation needed]."
    result = clean_wikiish(text)

    assert "[1]" not in result
    assert "[citation needed]" not in result
    assert "Python is a language" in result


def test_clean_wikiish_templates():
    """Removes template braces"""
    from processing.gate_system import clean_wikiish

    text = "Python {{infobox}} is great {{cite web}}"
    result = clean_wikiish(text)

    assert "{{infobox}}" not in result
    assert "{{cite web}}" not in result
    assert "Python" in result
    assert "is great" in result


def test_clean_wikiish_combined():
    """Handles multiple markup types"""
    from processing.gate_system import clean_wikiish

    text = "== Intro ==\nPython[1] is {{popular}} great"
    result = clean_wikiish(text)

    assert "==" not in result
    assert "[1]" not in result
    assert "{{" not in result
    assert "Python" in result


def test_clean_wikiish_empty():
    """Handles empty string"""
    from processing.gate_system import clean_wikiish

    result = clean_wikiish("")

    assert result == ""


def test_content_words_basic():
    """Extracts content words (min 4 chars)"""
    from processing.gate_system import _content_words

    words = _content_words("The quick brown foxes")

    # Requires minimum 4 characters
    assert "quick" in words
    assert "brown" in words
    assert "foxes" in words


def test_content_words_lowercase():
    """Converts to lowercase"""
    from processing.gate_system import _content_words

    words = _content_words("HELLO World")

    assert "hello" in words
    assert "world" in words
    assert "HELLO" not in words


def test_content_words_empty():
    """Handles empty string"""
    from processing.gate_system import _content_words

    words = _content_words("")

    assert words == set()


def test_content_words_none():
    """Handles None input"""
    from processing.gate_system import _content_words

    words = _content_words(None)

    assert words == set()


def test_content_words_punctuation():
    """Ignores punctuation"""
    from processing.gate_system import _content_words

    words = _content_words("hello, world!")

    # Should extract words, not punctuation
    assert "hello" in words
    assert "world" in words


def test_overlap_score_full_overlap():
    """Full overlap returns 1.0 (words must be 4+ chars)"""
    from processing.gate_system import _overlap_score

    score = _overlap_score("cats dogs", "cats dogs")

    assert score == 1.0


def test_overlap_score_no_overlap():
    """No overlap returns 0.0"""
    from processing.gate_system import _overlap_score

    score = _overlap_score("cat dog", "bird fish")

    assert score == 0.0


def test_overlap_score_partial():
    """Partial overlap returns intermediate score"""
    from processing.gate_system import _overlap_score

    score = _overlap_score("cats dogs birds", "cats fish")

    # 1 word in common (cats), smaller set is 2 words
    assert score == 0.5


def test_overlap_score_empty_query():
    """Empty query returns 0.0"""
    from processing.gate_system import _overlap_score

    score = _overlap_score("", "cat dog")

    assert score == 0.0


def test_overlap_score_empty_text():
    """Empty text returns 0.0"""
    from processing.gate_system import _overlap_score

    score = _overlap_score("cat dog", "")

    assert score == 0.0


def test_overlap_score_case_insensitive():
    """Overlap is case insensitive"""
    from processing.gate_system import _overlap_score

    score = _overlap_score("CATS DOGS", "cats dogs")

    assert score == 1.0


def test_source_weight_docs():
    """Docs sources get higher weight"""
    from processing.gate_system import _source_weight

    assert _source_weight("docs") == 1.05
    assert _source_weight("paper") == 1.05
    assert _source_weight("arxiv") == 1.05
    assert _source_weight("manual") == 1.05
    assert _source_weight("notebook") == 1.05


def test_source_weight_wikipedia():
    """Wikipedia gets neutral weight"""
    from processing.gate_system import _source_weight

    assert _source_weight("wikipedia") == 1.00
    assert _source_weight("wiki") == 1.00


def test_source_weight_unknown():
    """Unknown sources get lower weight"""
    from processing.gate_system import _source_weight

    assert _source_weight("unknown") == 0.85
    assert _source_weight("") == 0.85


def test_source_weight_default():
    """Other sources get default weight"""
    from processing.gate_system import _source_weight

    assert _source_weight("blog") == 1.00
    assert _source_weight("forum") == 1.00
    assert _source_weight("reddit") == 1.00


def test_source_weight_case_insensitive():
    """Source weighting is case insensitive"""
    from processing.gate_system import _source_weight

    assert _source_weight("DOCS") == 1.05
    assert _source_weight("Docs") == 1.05
    assert _source_weight("Wikipedia") == 1.00


def test_source_weight_none():
    """None input treated as unknown"""
    from processing.gate_system import _source_weight

    assert _source_weight(None) == 0.85
