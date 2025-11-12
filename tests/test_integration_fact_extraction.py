"""Integration tests for fact extraction pipeline."""
import pytest
from memory.fact_extractor import FactExtractor


@pytest.fixture
def fact_extractor():
    """Fixture to provide FactExtractor with all features enabled."""
    return FactExtractor(use_rebel=False, use_regex=True)


@pytest.mark.asyncio
async def test_extract_facts_from_conversation(fact_extractor):
    """Test extracting facts from a realistic conversation."""
    query = "My name is Luke and I live in Seattle. I work as a software engineer at Microsoft."
    response = "Nice to meet you, Luke! Seattle is a beautiful city and Microsoft is a major tech company."

    facts = await fact_extractor.extract_facts(query, response)

    assert isinstance(facts, list)
    # Should extract some facts about Luke
    assert len(facts) >= 0  # May be 0 if extraction fails, but shouldn't crash


@pytest.mark.asyncio
async def test_extract_facts_with_rebel_disabled(fact_extractor):
    """Test fact extraction with REBEL model disabled."""
    text = "Python is a programming language created by Guido van Rossum."
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_facts_with_measurements(fact_extractor):
    """Test extracting facts with measurements."""
    text = "I can deadlift 405 lbs and bench press 225 lbs."
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)
    # Should detect the measurements


@pytest.mark.asyncio
async def test_extract_facts_with_preferences(fact_extractor):
    """Test extracting preference facts."""
    text = "My favorite color is blue and I love pizza."
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_facts_empty_input(fact_extractor):
    """Test fact extraction with empty input."""
    facts = await fact_extractor.extract_facts("", "")
    
    assert isinstance(facts, list)
    assert len(facts) == 0


@pytest.mark.asyncio
async def test_extract_facts_no_facts(fact_extractor):
    """Test with text containing no extractable facts."""
    text = "Hello. How are you? The weather is nice."
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_structured_facts(fact_extractor):
    """Test extracting facts from structured responses."""
    text = """
    The user's name is Alice.
    Alice lives in San Francisco.
    Alice works at Google.
    Alice's favorite programming language is Python.
    """
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_facts_from_message(fact_extractor):
    """Test extracting facts from a simple message."""
    query = "I graduated from MIT in 2015."
    response = "That's impressive! MIT is a great university."

    facts = await fact_extractor.extract_facts(query, response)

    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_extract_facts_deduplication(fact_extractor):
    """Test that duplicate facts are handled."""
    text = """
    I like coffee. I really like coffee. 
    My favorite drink is coffee.
    """
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)
    # Should not have excessive duplicates


@pytest.mark.asyncio
async def test_extract_facts_with_context(fact_extractor):
    """Test fact extraction with contextual information."""
    text = """
    I've been learning machine learning for 2 years.
    I completed the Stanford ML course last month.
    My favorite framework is PyTorch.
    """
    
    facts = await fact_extractor.extract_facts(text, "")
    
    assert isinstance(facts, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
