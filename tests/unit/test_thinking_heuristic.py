"""
Tests for likely_untagged_thinking() streaming heuristic and the
streaming-loop integration behavior it drives in handlers.py.

Covers:
- True positives: multi-pattern thinking dumps are detected
- True negatives: normal responses, short text, single-pattern text
- Bail-out: responses >300 chars are never suppressed (false-positive guard)
- Edge cases: thinking at boundary, patterns only in later lines
- _detect_untagged_thinking consistency: if the full parser splits,
  the heuristic should also have fired
"""

import pytest
from core.response_parser import ResponseParser


# ── likely_untagged_thinking: true positives ──

class TestLikelyUntaggedThinkingPositives:
    """Cases that SHOULD be detected as untagged thinking."""

    def test_classic_two_pattern_thinking(self):
        text = (
            "I should think about what the user is asking.\n"
            "The user wants to know about Python.\n"
            "Let me consider the best way to explain this."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_meta_reasoning_plus_user_reference(self):
        text = (
            "Let me check what I know about this topic.\n"
            "The user is asking about machine learning.\n"
            "I need to be careful here."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_planning_plus_meta(self):
        text = (
            "How should I approach this question?\n"
            "I should mention the key differences.\n"
            "This is a technical question."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_user_reference_plus_strategy(self):
        text = (
            "The user asked about their project timeline.\n"
            "What would actually be useful here is a breakdown.\n"
            "I could mention the deadline they set."
        )
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_bullet_reasoning_plus_meta(self):
        text = (
            "I need to consider several things:\n"
            "- Explicitly the user wants dates\n"
            "- Temporal context matters here\n"
        )
        assert ResponseParser.likely_untagged_thinking(text) is True


# ── likely_untagged_thinking: true negatives ──

class TestLikelyUntaggedThinkingNegatives:
    """Cases that should NOT be detected as thinking."""

    def test_normal_response(self):
        text = (
            "Python is a great programming language for beginners.\n"
            "It has clean syntax and a large standard library.\n"
            "You can install it from python.org."
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_short_text(self):
        assert ResponseParser.likely_untagged_thinking("I should check") is False

    def test_empty(self):
        assert ResponseParser.likely_untagged_thinking("") is False

    def test_none(self):
        assert ResponseParser.likely_untagged_thinking(None) is False

    def test_single_line(self):
        text = "I should think about this and the user wants an answer."
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_single_pattern_match(self):
        """One pattern hit is not enough."""
        text = (
            "I should mention that Python supports async/await.\n"
            "Here is how you use it:\n"
            "```python\nasync def main(): pass\n```"
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_conversational_response_with_first_person(self):
        """Normal first-person language should not trigger."""
        text = (
            "I think that's a great idea!\n"
            "You could try using pandas for this.\n"
            "Here's an example of how to do it."
        )
        assert ResponseParser.likely_untagged_thinking(text) is False

    def test_response_with_let_me(self):
        """'Let me' in a normal response context."""
        text = (
            "Great question! Let me explain how this works.\n"
            "The HTTP protocol uses request-response pairs.\n"
            "Each request has a method like GET or POST."
        )
        # "Let me" alone matches pattern 0, but needs a second distinct pattern
        assert ResponseParser.likely_untagged_thinking(text) is False


# ── Bail-out behavior (streaming integration) ──

class TestStreamingBailout:
    """The 300-char bail-out prevents false-positive suppression."""

    def test_short_thinking_suppressed(self):
        """Under 300 chars with patterns: should suppress."""
        text = (
            "I should think about what the user is asking.\n"
            "The user wants to know about Python."
        )
        assert len(text) < 300
        assert ResponseParser.likely_untagged_thinking(text) is True

    def test_long_text_with_patterns_not_suppressed(self):
        """Over 300 chars: the streaming loop bails out regardless of heuristic.

        The heuristic itself doesn't enforce the 300-char limit (it's a pure
        pattern check), but handlers.py only applies it when len < 300.
        We test the condition handlers.py would check.
        """
        text = (
            "I should think about what the user is asking.\n"
            "The user wants to know about Python.\n"
            + "Here is a very long explanation. " * 20
        )
        assert len(text) > 300
        # Heuristic still fires (it doesn't know about length)
        assert ResponseParser.likely_untagged_thinking(text) is True
        # But the streaming guard would NOT suppress (len >= 300)
        would_suppress = len(text) < 300 and ResponseParser.likely_untagged_thinking(text)
        assert would_suppress is False

    def test_normal_long_response_never_suppressed(self):
        """Normal response that's long: never suppressed at any length."""
        text = (
            "Python is a great programming language.\n"
            "It was created by Guido van Rossum.\n"
            + "It supports many paradigms. " * 20
        )
        assert ResponseParser.likely_untagged_thinking(text) is False
        would_suppress = len(text) < 300 and ResponseParser.likely_untagged_thinking(text)
        assert would_suppress is False


# ── Consistency with _detect_untagged_thinking ──

class TestHeuristicConsistency:
    """If _detect_untagged_thinking splits text, likely_untagged_thinking
    should also have detected patterns (the heuristic is a superset check)."""

    def test_splittable_text_detected_by_both(self):
        """Text with a clean split should be caught by both methods."""
        text = (
            "I should think about this carefully.\n"
            "The user wants to know about their dog.\n"
            "Let me check my memory.\n"
            "\n"
            "Your dog Flapjack is a golden retriever! "
            "You've mentioned him several times before."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        if thinking and answer:
            # If the full parser found a split, the fast check must also fire
            assert ResponseParser.likely_untagged_thinking(text) is True

    def test_no_split_no_detection(self):
        """Text with no thinking patterns: neither method fires."""
        text = (
            "Here is your answer about Python.\n"
            "It supports async/await since version 3.5.\n"
            "You can use asyncio.gather for parallelism."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking == ""
        assert ResponseParser.likely_untagged_thinking(text) is False


# ── parse_thinking_block integration ──

class TestParseThinkingBlockIntegration:
    """Ensure parse_thinking_block still works correctly for all modes."""

    def test_tagged_thinking(self):
        text = "<thinking>Let me analyze.</thinking>The answer is 42."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert thinking == "Let me analyze."
        assert answer == "The answer is 42."

    def test_think_tag_variant(self):
        text = "<think>Planning my response.</think>Here you go."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert thinking == "Planning my response."
        assert answer == "Here you go."

    def test_no_thinking(self):
        text = "Just a normal response with no thinking."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert thinking == ""
        assert answer == text

    def test_output_wrapper(self):
        text = "Some reasoning here\n<output>The real answer.</output>"
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert answer == "The real answer."

    def test_has_incomplete_thinking_block(self):
        assert ResponseParser.has_incomplete_thinking_block("<thinking>partial") is True
        assert ResponseParser.has_incomplete_thinking_block("<thinking>done</thinking>answer") is False
        assert ResponseParser.has_incomplete_thinking_block("no tags here") is False


# ── Real-world thinking leak regression tests ──

class TestRealWorldLeaks:
    """Regression tests from actual observed thinking leaks in production."""

    def test_agentic_sandbox_failure_reasoning(self):
        """DeepSeek v4 leaked sandbox failure reasoning into visible response."""
        text = (
            "The user wants a simple pandas DataFrame with 5 rows of random "
            "name/age/score data. The sandbox had two failed attempts due to "
            "a dateutil.parser import error — that's an environment issue "
            "(likely a version mismatch in the sandbox's python-dateutil "
            "package), not a code problem. I'll try running it anyway since "
            "it might have been transient or fixed, but if it fails again "
            "I'll just give them the working code directly rather than "
            "burning more rounds debugging the sandbox environment.\n\n"
            "Looks like the sandbox has a dateutil version conflict. "
            "Here is the code:\n\n"
            "```python\nimport pandas as pd\ndf = pd.DataFrame({'a': [1]})\n```"
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking, "Should detect sandbox reasoning as thinking"
        assert "```python" in answer, "Code block should be in the answer"
        assert "The user wants" not in answer, "User-reference should be stripped"

    def test_agentic_tool_dispatch_planning(self):
        """LLM planning which tools to use."""
        text = (
            "The user is asking about recent ML papers. I should use "
            "search_arxiv for academic content and maybe search_hackernews "
            "for community discussion. Let me try both in parallel.\n\n"
            "Here are some recent papers on retrieval-augmented generation:\n"
            "1. RAG-Token by Facebook (2024)..."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking, "Tool dispatch planning should be caught"
        assert "recent papers" in answer

    def test_agentic_retry_reasoning(self):
        """LLM reasoning about retrying a failed search."""
        text = (
            "The previous search didn't return relevant results. "
            "I need to adjust my query to be more specific. "
            "This time I should include the exact framework name.\n\n"
            "Based on my research, ChromaDB handles persistence via..."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking, "Retry reasoning should be caught"

    def test_environment_issue_diagnosis(self):
        """LLM diagnosing environment issues shouldn't leak."""
        text = (
            "That's an environment issue with the sandbox. Not a code problem. "
            "The dateutil package is incompatible. I'll just give them the "
            "code to run locally.\n\n"
            "Here's the working code:\n```python\nprint('hello')\n```"
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking, "Environment diagnosis should be caught"
        assert "```python" in answer

    def test_credit_aware_meta_reasoning(self):
        """LLM reasoning about API costs."""
        text = (
            "I've already used 3 search rounds. Rather than burning more "
            "rounds on this, I have enough information to answer.\n\n"
            "The answer to your question is..."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking, "Cost-aware reasoning should be caught"

    def test_user_wants_mid_paragraph(self):
        """'The user wants' in a single flowing paragraph."""
        text = (
            "The user wants to understand recursion. I should use a simple "
            "example like factorial since they mentioned they're a beginner. "
            "Let me keep it concise.\n\n"
            "Recursion is when a function calls itself!"
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking, "Single-paragraph reasoning should be caught"

    def test_mixed_thinking_and_code(self):
        """Thinking followed by code blocks — code should survive."""
        text = (
            "I need to figure out the right approach here. The user seems "
            "to want a sorting algorithm. Let me check what language they're "
            "using.\n\n"
            "Here's a quicksort implementation in Python:\n\n"
            "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n"
            "        return arr\n    pivot = arr[0]\n    return quicksort("
            "[x for x in arr[1:] if x < pivot]) + [pivot] + quicksort("
            "[x for x in arr[1:] if x >= pivot])\n```"
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert thinking
        assert "quicksort" in answer
        assert "I need to figure" not in answer


# ── False positive prevention ──

class TestFalsePositivePrevention:
    """Ensure normal responses aren't misdetected as thinking."""

    def test_educational_let_me_explain(self):
        """'Let me explain' in a teaching context is NOT thinking."""
        text = (
            "Great question! Let me explain how HTTP works.\n\n"
            "HTTP is a request-response protocol. The client sends a "
            "request with a method (GET, POST, etc.) and the server "
            "returns a response with a status code."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert not thinking, "'Let me explain' in context should not trigger"

    def test_i_think_as_opinion(self):
        """'I think' expressing an opinion, not reasoning."""
        text = (
            "I think Python is the best language for beginners. "
            "It has clean syntax and great documentation. "
            "You should try starting with the official tutorial."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert not thinking, "Opinion 'I think' should not trigger"

    def test_discussing_users_in_third_person(self):
        """Talking ABOUT users generically, not reasoning about THE user."""
        text = (
            "When users first start learning SQL, they often struggle "
            "with JOIN operations. The key is to visualize the tables. "
            "Here's a diagram that helps:"
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert not thinking, "Generic 'users' reference should not trigger"

    def test_sandbox_in_user_facing_explanation(self):
        """Mentioning 'sandbox' in a user-facing explanation."""
        text = (
            "The code sandbox lets you run Python safely. "
            "It has pandas, numpy, and matplotlib pre-installed. "
            "Just paste your code and it'll execute in an isolated environment."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert not thinking, "User-facing sandbox description should not trigger"

    def test_first_person_narrative_response(self):
        """Response about the user's life should not trigger."""
        text = (
            "Based on what you've told me, you've been working out "
            "about twice a week and focusing on compound lifts. "
            "Your deadlift has improved from 225 to 275 since March. "
            "That's solid progress!"
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert not thinking, "Narrative about user should not trigger"

    def test_long_technical_response_with_i_should(self):
        """Technical response that happens to use 'I should mention'."""
        text = (
            "Here's how to set up a Python virtual environment.\n\n"
            "First, install venv:\n```bash\npython -m venv myenv\n```\n\n"
            "Then activate it:\n```bash\nsource myenv/bin/activate\n```\n\n"
            "I should mention that on Windows, the activation command "
            "is different: `myenv\\Scripts\\activate`."
        )
        thinking, answer = ResponseParser._detect_untagged_thinking(text)
        assert not thinking, "Mid-response 'I should mention' is conversational"


# ── Tag-based edge cases ──

class TestTagEdgeCases:
    """Edge cases for tag-based thinking block parsing."""

    def test_nested_thinking_tags(self):
        """Only outermost thinking tags should be parsed."""
        text = "<thinking>Outer <thinking>inner</thinking> still thinking</thinking>Answer."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert "Answer" in answer

    def test_think_tag_with_whitespace(self):
        text = "  <think>  spaces  </think>  The answer.  "
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert "spaces" in thinking
        assert "answer" in answer

    def test_multiple_thinking_blocks(self):
        """First thinking block should be extracted."""
        text = "<thinking>First block</thinking>Middle text<thinking>Second</thinking>End."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert "First block" in thinking

    def test_thinking_with_code_inside(self):
        """Code blocks inside thinking tags should be captured."""
        text = "<thinking>```python\nx = 1\n```</thinking>The answer is 1."
        thinking, answer = ResponseParser.parse_thinking_block(text)
        assert "x = 1" in thinking
        assert answer == "The answer is 1."

    def test_leaked_closing_tag(self):
        """Partial closing tag should be stripped."""
        text = "Some answer text </think> and more."
        cleaned = ResponseParser.strip_thinking_tag_leaks(text)
        assert "</think>" not in cleaned

    def test_leaked_opening_tag(self):
        text = "Normal text <thinking> leaked."
        # If no closing tag, this is an incomplete block
        assert ResponseParser.has_incomplete_thinking_block(text) is True
