"""
Regression tests for the 2026-08-21 mid-stream API-error fail-fast.

Incident (2026-08-18): OpenRouter 402 "requires more credits" errors fell
through _classify_api_error to the generic [API Error] fallback carrying the
FULL raw error payload (~3.2K of JSON incl. previous_errors), and the display
loops streamed it chunk-by-chunk into the chat bubble — three times, all
mid-distress. The friendly-error conversion only ran after the stream ended.

Fixes under test:
  1. _classify_api_error recognizes 402 / "requires more credits" /
     "can only afford" → short [CREDITS EXHAUSTED] message, no raw payload.
  2. gui.handlers._friendly_api_error — single shared display converter.
  3. Both display loops (enhanced + agentic) break at the stream head when the
     accumulated output starts with a classified prefix (source-level guard).
"""

import inspect

import pytest

from models.model_manager import API_ERROR_PREFIXES, _classify_api_error
from gui.handlers import _API_ERROR_DISPLAY, _friendly_api_error


class _FakeApiError(Exception):
    def __init__(self, msg, status_code=0, code=""):
        super().__init__(msg)
        self.status_code = status_code
        self.code = code


RAW_402 = (
    "Error code: 402 - {'error': {'message': 'This request requires more "
    "credits, or fewer max_tokens. You requested up to 9984 tokens, but can "
    "only afford 1985. To increase, visit https://openrouter.ai/settings/"
    "credits and upgrade to a paid account', 'code': 402, 'metadata': "
    "{'provider_name': None}}, 'user_id': 'user_abc'}"
)


class TestClassify402:
    def test_status_code_402_is_credits_exhausted(self):
        out = _classify_api_error(_FakeApiError("payment required", status_code=402))
        assert out.startswith("[CREDITS EXHAUSTED]")

    def test_requires_more_credits_text(self):
        out = _classify_api_error(_FakeApiError(RAW_402))
        assert out.startswith("[CREDITS EXHAUSTED]")

    def test_no_raw_payload_in_message(self):
        """The whole point: the classified message must NOT carry the raw
        error JSON that was streamed into the bubble on 08-18."""
        out = _classify_api_error(_FakeApiError(RAW_402))
        assert "user_id" not in out
        assert "{'error'" not in out
        assert len(out) < 400

    def test_generic_errors_still_fall_through(self):
        out = _classify_api_error(_FakeApiError("something odd happened"))
        assert out.startswith("[API Error]")


class TestFriendlyApiError:
    def test_all_registered_prefixes_convert(self):
        for prefix in API_ERROR_PREFIXES:
            sample = f"{prefix} something went wrong"
            friendly = _friendly_api_error(sample)
            assert friendly is not None, f"prefix {prefix!r} not converted"
            assert prefix not in friendly  # raw tag never shown to the user

    def test_display_map_covers_all_model_prefixes(self):
        """Every prefix the model layer can emit must have a display entry —
        an uncovered prefix would stream raw again."""
        for prefix in API_ERROR_PREFIXES:
            assert any(prefix.startswith(k) or k.startswith(prefix)
                       for k in _API_ERROR_DISPLAY), f"no display entry covers {prefix!r}"

    def test_normal_text_returns_none(self):
        assert _friendly_api_error("A perfectly normal answer.") is None
        assert _friendly_api_error("") is None
        assert _friendly_api_error(None) is None

    def test_streaming_error_shapes(self):
        for sample in ("[Streaming Error: peer closed connection]",
                       "[Streaming Error] peer closed connection"):
            friendly = _friendly_api_error(sample)
            assert friendly is not None
            assert "peer closed connection" in friendly

    def test_partial_answer_with_trailing_marker_not_converted(self):
        """Head-match only: a real answer with an appended marker keeps its
        content (the storage boundary strips the marker)."""
        assert _friendly_api_error(
            "Here is most of the answer. [Streaming Error: dropped]"
        ) is None


class TestStreamHeadFailFast:
    """Source-level guards: both display loops abort at the stream head."""

    def _handlers_source(self):
        import gui.handlers as handlers
        return inspect.getsource(handlers)

    def test_enhanced_loop_breaks_on_error_head(self):
        src = self._handlers_source()
        assert src.count("API_ERROR_PREFIXES as _api_err_prefixes") >= 2, (
            "expected the stream-head fail-fast in BOTH the enhanced and "
            "agentic display loops"
        )

    def test_no_inline_prefix_map_left(self):
        """The display map must stay single-sourced in _API_ERROR_DISPLAY."""
        src = self._handlers_source()
        assert src.count("Out of API Credits") == 1
