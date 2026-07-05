"""
Tests for explicit truncation markers in agentic prompt formatting.

Regression for the 2026-07-03 "outag" confabulation: recent-conversation
previews were silently cut mid-word ([:500]), the model quoted the cut
preview as the full message and invented a "stored version is truncated"
claim. Truncation must now always be visible, and explicit memory search
on the conversations collection must return the full stored document
(up to 2000 chars).
"""

from core.agentic.formatters import AgenticFormatter, clip_text, TRUNCATION_MARKER


class TestClipText:
    def test_short_text_unchanged(self):
        assert clip_text("hello", 10) == "hello"

    def test_exact_limit_unchanged(self):
        assert clip_text("a" * 10, 10) == "a" * 10

    def test_over_limit_gets_marker(self):
        result = clip_text("a" * 11, 10)
        assert result == "a" * 10 + TRUNCATION_MARKER

    def test_none_and_empty_safe(self):
        assert clip_text(None, 10) == ""
        assert clip_text("", 10) == ""


class TestRecentConversationTruncation:
    def test_long_response_marked_truncated(self):
        fmt = AgenticFormatter()
        long_response = "x" * 600
        result = fmt.format_recent_conversations(
            [{'timestamp': 't', 'query': 'q', 'response': long_response}]
        )
        assert TRUNCATION_MARKER in result
        assert "x" * 500 + TRUNCATION_MARKER in result

    def test_short_response_no_marker(self):
        fmt = AgenticFormatter()
        result = fmt.format_recent_conversations(
            [{'timestamp': 't', 'query': 'q', 'response': 'short answer'}]
        )
        assert TRUNCATION_MARKER not in result
        assert 'short answer' in result

    def test_long_query_marked_truncated(self):
        fmt = AgenticFormatter()
        result = fmt.format_recent_conversations(
            [{'timestamp': 't', 'query': 'y' * 600, 'response': 'r'}]
        )
        assert "y" * 500 + TRUNCATION_MARKER in result


class TestMemoryResultsTruncation:
    def test_conversations_full_document_up_to_2000(self):
        """Explicit memory search on conversations returns the full stored doc."""
        fmt = AgenticFormatter()
        content = "z" * 1500  # would have been cut at 500 before
        result = fmt.format_memory_results(
            [{'content': content, 'relevance_score': 0.9, 'metadata': {}, 'id': 'abc'}],
            'conversations',
        )
        assert content in result
        assert 'truncated' not in result

    def test_conversations_over_2000_points_to_expand_memory(self):
        fmt = AgenticFormatter()
        result = fmt.format_memory_results(
            [{'content': 'z' * 2500, 'relevance_score': 0.9, 'metadata': {}, 'id': 'doc42'}],
            'conversations',
        )
        assert 'truncated' in result
        # must name the REGISTERED tool — 'memory_expand' was a dropped call
        assert 'expand_memory' in result
        assert 'doc42' in result

    def test_other_collections_keep_500_with_marker(self):
        fmt = AgenticFormatter()
        result = fmt.format_memory_results(
            [{'content': 'w' * 800, 'relevance_score': 0.5, 'metadata': {}, 'id': 'd1'}],
            'facts',
        )
        assert 'w' * 500 in result
        assert 'w' * 501 not in result
        assert 'truncated' in result

    def test_no_id_falls_back_to_plain_marker(self):
        fmt = AgenticFormatter()
        result = fmt.format_memory_results(
            [{'content': 'w' * 800, 'relevance_score': 0.5, 'metadata': {}}],
            'facts',
        )
        assert TRUNCATION_MARKER in result
        assert 'expand_memory' not in result


class TestGateRecentContextTruncation:
    def test_gate_digest_marks_cut_messages(self):
        from core.agentic.gate import _build_recent_context

        class FakeCorpus:
            def get_recent_memories(self, n):
                return [{'query': 'q' * 300, 'response': 'r' * 400}]

        ctx = _build_recent_context(FakeCorpus())
        assert ctx is not None
        assert f"User: {'q' * 200}{TRUNCATION_MARKER}" in ctx
        assert f"Assistant: {'r' * 300}{TRUNCATION_MARKER}" in ctx

    def test_gate_digest_short_messages_unmarked(self):
        from core.agentic.gate import _build_recent_context

        class FakeCorpus:
            def get_recent_memories(self, n):
                return [{'query': 'short q', 'response': 'short r'}]

        ctx = _build_recent_context(FakeCorpus())
        assert TRUNCATION_MARKER not in ctx


class TestTriggerPromptContextTruncation:
    def test_context_block_over_1200_marked(self):
        from utils.web_search_trigger import _build_llm_trigger_prompt
        prompt = _build_llm_trigger_prompt(
            query="test",
            current_date="2026-07-03",
            conversation_context="c" * 1500,
        )
        assert "c" * 1200 + " [...truncated]" in prompt

    def test_context_block_short_unmarked(self):
        from utils.web_search_trigger import _build_llm_trigger_prompt
        prompt = _build_llm_trigger_prompt(
            query="test",
            current_date="2026-07-03",
            conversation_context="c" * 100,
        )
        assert "[...truncated]" not in prompt
