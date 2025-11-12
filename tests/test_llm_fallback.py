"""Test LLM fallback for tone detection."""
import pytest
from utils.tone_detector import detect_crisis_level, CrisisLevel


class MockModelManager:
    """Mock model manager for testing LLM fallback."""

    async def generate_async(self, prompt, max_tokens=None, temperature=None):
        """Mock LLM response based on prompt content."""
        # Check what's being asked
        if "Everything would be better if I just wasn't here" in prompt:
            return "HIGH"
        elif "Everything feels pointless" in prompt:
            return "HIGH"
        elif "I'm a bit tired" in prompt:
            return "CONVERSATIONAL"
        else:
            return "CONVERSATIONAL"


@pytest.mark.asyncio
async def test_llm_fallback_triggers_on_borderline():
    """Test that LLM fallback is called for borderline cases."""
    mock_mm = MockModelManager()

    # This message scores ~0.42 for HIGH (below 0.58 threshold)
    # Should trigger LLM fallback since 0.42 > 0.43-0.15
    message = "Everything would be better if I just wasn't here"

    result = await detect_crisis_level(message, model_manager=mock_mm)

    # With LLM fallback, should now detect as HIGH
    assert result.trigger == "llm_fallback", f"Expected llm_fallback trigger, got {result.trigger}"
    assert result.level == CrisisLevel.HIGH, f"Expected HIGH, got {result.level}"
    print(f"✅ LLM fallback successfully detected: {message}")


@pytest.mark.asyncio
async def test_llm_fallback_prevents_false_positive():
    """Test that LLM fallback prevents false positives."""
    mock_mm = MockModelManager()

    # "I'm a bit tired" might score borderline - LLM should keep it CONVERSATIONAL
    message = "I'm a bit tired today"

    result = await detect_crisis_level(message, model_manager=mock_mm)

    # Should be CONVERSATIONAL (either semantic or LLM)
    assert result.level == CrisisLevel.CONVERSATIONAL
    print(f"✅ Correctly identified as conversational: {message}")


@pytest.mark.asyncio
async def test_no_llm_without_model_manager():
    """Test that system works without model_manager (no LLM fallback)."""
    # No model_manager - should use semantic only
    message = "I want to die"  # Clear keyword match

    result = await detect_crisis_level(message, model_manager=None)

    # Should still detect via keywords
    assert result.level == CrisisLevel.HIGH
    assert result.trigger == "keyword: want to die"
    print(f"✅ Keyword detection works without LLM: {message}")


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        await test_llm_fallback_triggers_on_borderline()
        await test_llm_fallback_prevents_false_positive()
        await test_no_llm_without_model_manager()

    asyncio.run(run_tests())
    print("\n✅ All LLM fallback tests passed!")
