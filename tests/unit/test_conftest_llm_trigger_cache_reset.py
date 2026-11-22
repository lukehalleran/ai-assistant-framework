"""tests/conftest.py resets the LLM trigger decision cache between tests.

utils.web_search_trigger caches each LLM routing decision by query text for the
whole process. Without a reset, a test that asks the same question as an earlier
test is answered from that test's fake model and never reaches its own fake.
The two tests run in file order: the first fills the cache, the second checks it
starts empty.
"""
import utils.web_search_trigger as wst


def test_a_fills_the_trigger_cache():
    wst._llm_trigger_cache[hash("conftest reset probe")] = (0.0, object())
    assert wst._llm_trigger_cache


def test_b_trigger_cache_starts_empty():
    assert wst._llm_trigger_cache == {}
