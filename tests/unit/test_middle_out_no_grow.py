"""
Regression tests for two middle-out compression bugs (2026-07-25):

1. Growth: the ~4-chars/token heuristic overshoots on token-dense text, so
   "compression" could RETURN MORE TOKENS than it received (live log:
   "Compressed git_commits[1]: 829 → 900 tokens"). _middle_out must never
   return a result that isn't smaller than its input.

2. Literal "\\n": the snip marker was built with escaped backslash-n, so
   prompts carried literal `\\n… [middle-out snipped …] …\\n` character
   sequences instead of newlines (visible in live prompt dumps).
"""

from unittest.mock import MagicMock

from core.prompt.token_manager import TokenManager


def _tm(counts_per_word=1.0):
    tm = TokenManager.__new__(TokenManager)
    tm.model_manager = MagicMock()
    tm.model_manager.get_active_model_name.return_value = "test-model"
    # Deterministic token counting: words * factor.
    tm.get_token_count = lambda text, model_name=None: int(
        len((text or "").split()) * counts_per_word
    )
    tm.token_budget = 10000
    return tm


class TestMiddleOutNoGrow:
    def test_token_dense_text_returns_original(self):
        tm = _tm()
        # Short "words" → ~2 chars/token. 4-chars/token heuristic would keep
        # max_tokens*4 chars ≈ 2x max_tokens tokens: a growth, not a squeeze.
        text = "a " * 900  # 900 tokens, 1800 chars
        out = tm._middle_out(text, max_tokens=800, force=True)
        assert out == text  # never grow — return original unchanged

    def test_compressible_text_shrinks(self):
        tm = _tm()
        # Long words → ~9 chars/token: heuristic genuinely compresses.
        text = ("abcdefgh " * 2000).strip()  # 2000 tokens, ~18000 chars
        out = tm._middle_out(text, max_tokens=500, force=True)
        assert len(out) < len(text)
        assert tm.get_token_count(out) < tm.get_token_count(text)
        assert "[middle-out snipped" in out

    def test_snip_marker_uses_real_newlines(self):
        tm = _tm()
        text = ("abcdefgh " * 2000).strip()
        out = tm._middle_out(text, max_tokens=500, force=True)
        assert "\\n" not in out, "snip marker leaked literal backslash-n"
        assert "\n… [middle-out snipped" in out

    def test_under_limit_unchanged(self):
        tm = _tm()
        text = "short text that fits"
        assert tm._middle_out(text, max_tokens=100, force=True) == text
