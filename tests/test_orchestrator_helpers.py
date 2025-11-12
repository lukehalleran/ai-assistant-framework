"""Targeted tests for DaemonOrchestrator helper methods to boost coverage."""
import pytest
from core.orchestrator import DaemonOrchestrator


# Test static helper methods
def test_parse_thinking_block_with_both_tags():
    """Test extracting thinking block with both tags present."""
    response = "<thinking>My reasoning here</thinking>Final answer text"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert thinking == "My reasoning here"
    assert answer == "Final answer text"


def test_parse_thinking_block_closing_only():
    """Test extracting thinking block with only closing tag."""
    response = "Some reasoning</thinking>Final answer"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert thinking == "Some reasoning"
    assert answer == "Final answer"


def test_parse_thinking_block_no_tags():
    """Test extracting thinking block when no tags present."""
    response = "Just a regular response"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert thinking == ""
    assert answer == "Just a regular response"


def test_parse_thinking_block_empty_response():
    """Test extracting thinking block from empty response."""
    thinking, answer = DaemonOrchestrator._parse_thinking_block("")
    assert thinking == ""
    assert answer == ""


def test_parse_thinking_block_nested_tags():
    """Test extracting thinking block with nested content."""
    response = "<thinking>Step 1: analyze\nStep 2: decide</thinking>My final answer"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert "Step 1" in thinking
    assert "Step 2" in thinking
    assert answer == "My final answer"


def test_strip_xml_wrappers_result_tag():
    """Test stripping <result> wrapper."""
    text = "<result>Content here</result>"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "Content here"


def test_strip_xml_wrappers_answer_tag():
    """Test stripping <answer> wrapper."""
    text = "<answer>The answer is 42</answer>"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "The answer is 42"


def test_strip_xml_wrappers_final_tag():
    """Test stripping <final> wrapper."""
    text = "<final>Final response</final>"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "Final response"


def test_strip_xml_wrappers_with_attributes():
    """Test stripping tags with attributes."""
    text = '<result type="answer">Content</result>'
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "Content"


def test_strip_xml_wrappers_no_tags():
    """Test stripping when no wrapper tags present."""
    text = "Plain text without tags"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "Plain text without tags"


def test_strip_xml_wrappers_empty():
    """Test stripping with empty text."""
    result = DaemonOrchestrator._strip_xml_wrappers("")
    assert result == ""


def test_strip_xml_wrappers_none():
    """Test stripping with None."""
    result = DaemonOrchestrator._strip_xml_wrappers(None)
    assert result is None


def test_strip_xml_wrappers_partial_tag():
    """Test stripping with partial/unclosed tag."""
    text = "<result>Content without closing"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert "<result>" in result  # Should not strip incomplete tags


def test_strip_xml_wrappers_multiple_lines():
    """Test stripping wrapper with multiline content."""
    text = "<result>\nLine 1\nLine 2\nLine 3\n</result>"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert "Line 1" in result
    assert "Line 2" in result
    assert "<result>" not in result


def test_strip_prompt_artifacts_time_context():
    """Test stripping [TIME CONTEXT] header."""
    text = "[TIME CONTEXT]\nSome time info\n\nActual response here"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[TIME CONTEXT]" not in result
    assert "Actual response here" in result


def test_strip_prompt_artifacts_recent_conversation():
    """Test stripping [RECENT CONVERSATION] header."""
    text = "[RECENT CONVERSATION]\nQ: Hello\nA: Hi\n\nMy answer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[RECENT CONVERSATION]" not in result
    assert "My answer" in result


def test_strip_prompt_artifacts_relevant_info():
    """Test stripping [RELEVANT INFORMATION] header."""
    text = "[RELEVANT INFORMATION]\nSome context\n\nResponse text"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[RELEVANT INFORMATION]" not in result
    assert "Response text" in result


def test_strip_prompt_artifacts_facts():
    """Test stripping [FACTS] header."""
    text = "[FACTS]\nFact 1\nFact 2\n\nAnswer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[FACTS]" not in result
    assert "Answer" in result


def test_strip_prompt_artifacts_directives():
    """Test stripping [DIRECTIVES] header."""
    text = "[DIRECTIVES]\nBe helpful\n\nMy response"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[DIRECTIVES]" not in result
    assert "My response" in result


def test_strip_prompt_artifacts_user_input():
    """Test stripping [USER INPUT] header."""
    text = "[USER INPUT]\nUser query here\n\nAssistant response"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[USER INPUT]" not in result
    assert "Assistant response" in result


def test_strip_prompt_artifacts_background_knowledge():
    """Test stripping [BACKGROUND KNOWLEDGE] header."""
    text = "[BACKGROUND KNOWLEDGE]\nWiki info\n\nResponse"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[BACKGROUND KNOWLEDGE]" not in result
    assert "Response" in result


def test_strip_prompt_artifacts_summaries():
    """Test stripping [CONVERSATION SUMMARIES] header."""
    text = "[CONVERSATION SUMMARIES]\nSummary text\n\nAnswer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[CONVERSATION SUMMARIES]" not in result
    assert "Answer" in result


def test_strip_prompt_artifacts_reflections():
    """Test stripping [RECENT REFLECTIONS] header."""
    text = "[RECENT REFLECTIONS]\nReflection content\n\nResponse"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[RECENT REFLECTIONS]" not in result
    assert "Response" in result


def test_strip_prompt_artifacts_multiple_headers():
    """Test stripping multiple headers in sequence."""
    text = "[TIME CONTEXT]\nTime info\n\n[FACTS]\nFact 1\n\nFinal answer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[TIME CONTEXT]" not in result
    assert "[FACTS]" not in result
    assert "Final answer" in result


def test_strip_prompt_artifacts_no_headers():
    """Test stripping when no headers present."""
    text = "Just a regular response without headers"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert result == text


def test_strip_prompt_artifacts_empty():
    """Test stripping with empty text."""
    result = DaemonOrchestrator._strip_prompt_artifacts("")
    assert result == ""


def test_strip_prompt_artifacts_none():
    """Test stripping with None."""
    result = DaemonOrchestrator._strip_prompt_artifacts(None)
    assert result is None


def test_strip_prompt_artifacts_case_insensitive():
    """Test stripping works case-insensitively."""
    text = "[time context]\nInfo\n\n[Facts]\nData\n\nAnswer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[time context]" not in result.lower()
    assert "Answer" in result


def test_strip_prompt_artifacts_preserves_content():
    """Test stripping preserves legitimate content after headers."""
    text = "[FACTS]\nFact block\n\nThis is the real response.\nWith multiple lines.\nAll important."
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "This is the real response" in result
    assert "With multiple lines" in result
    assert "All important" in result


def test_strip_prompt_artifacts_header_with_brackets():
    """Test stripping [RECENT CONVERSATION History] style headers."""
    text = "[RECENT CONVERSATION History]\nOld chat\n\nNew response"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    # Should strip since pattern matches
    assert "New response" in result


def test_strip_prompt_artifacts_current_facts():
    """Test stripping [CURRENT MESSAGE FACTS] header."""
    text = "[CURRENT MESSAGE FACTS]\nExtracted facts\n\nResponse"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[CURRENT MESSAGE FACTS]" not in result
    assert "Response" in result


def test_strip_prompt_artifacts_session_reflections():
    """Test stripping [SESSION REFLECTIONS] header."""
    text = "[SESSION REFLECTIONS]\nSession notes\n\nAnswer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[SESSION REFLECTIONS]" not in result
    assert "Answer" in result


def test_extract_thinking_with_whitespace():
    """Test extracting thinking block with extra whitespace."""
    response = "  <thinking>  Reasoning  </thinking>  Answer  "
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert thinking.strip() == "Reasoning"
    assert answer.strip() == "Answer"


def test_strip_xml_case_insensitive():
    """Test XML stripping is case insensitive."""
    text = "<RESULT>Content</RESULT>"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "Content"


def test_strip_xml_with_spaces():
    """Test XML stripping handles spaces in tags."""
    text = "< result >Content</ result >"
    result = DaemonOrchestrator._strip_xml_wrappers(text)
    assert result == "Content"


def test_extract_thinking_only_closing_tag_at_end():
    """Test thinking extraction when closing tag is at very end."""
    response = "Thinking content goes here</thinking>"
    thinking, answer = DaemonOrchestrator._parse_thinking_block(response)
    assert thinking == "Thinking content goes here"
    assert answer == ""


def test_strip_prompt_artifacts_relevant_memories():
    """Test stripping [RELEVANT MEMORIES] header."""
    text = "[RELEVANT MEMORIES]\nMemory 1\nMemory 2\n\nResponse here"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[RELEVANT MEMORIES]" not in result
    assert "Response here" in result


def test_strip_prompt_artifacts_recent_facts():
    """Test stripping [RECENT FACTS] header."""
    text = "[RECENT FACTS]\nFact A\nFact B\n\nMy answer"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[RECENT FACTS]" not in result
    assert "My answer" in result


def test_strip_prompt_artifacts_current_query():
    """Test stripping [CURRENT USER QUERY] header."""
    text = "[CURRENT USER QUERY]\nWhat is Python?\n\nPython is a language"
    result = DaemonOrchestrator._strip_prompt_artifacts(text)
    assert "[CURRENT USER QUERY]" not in result
    assert "Python is a language" in result
