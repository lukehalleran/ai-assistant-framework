"""
Strict JSON contract tests for core.response_planner (S02, BC-21 sibling of
F03/A04's core.grounding_check._parse_verdict and S01's
utils.web_search_trigger.LLMSearchTriggerResponse.parse).

Covers:
- ResponsePlanner._parse_review(): rejections for every taught field, a
  non-object top level (today an uncaught AttributeError), malformed JSON;
  controls for valid/minimal/boundary payloads.
- ResponsePlanner.review_answer(): the deployed caller — a malformed/
  missing-`passes` payload returns None; a valid payload returns a real
  ReviewResult (paired control). No network; the fake model_manager only.
- The deployed review gate in gui/handlers.py (log-only since 2026-08-28,
  handlers.py:4628-4680): a malformed or missing-`passes` review is never
  RECORDED into turn telemetry, and never ACTED ON, as a pass.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.response_planner import ResponsePlan, ResponsePlanner, ReviewResult

from tests.unit.test_handle_submit import (
    _final_content, _make_orchestrator, _no_trigger_decision, _run_submit,
)
import gui.handlers as _gui_handlers


def _p(**overrides) -> str:
    """A fully-typed, valid review payload as JSON, with overrides."""
    payload = {
        "passes": True,
        "confidence": 0.9,
        "issues": [],
        "suggestion": "",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _make_planner(llm_response: str) -> ResponsePlanner:
    """A ResponsePlanner with a fake model_manager (no network)."""
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=llm_response)
    return ResponsePlanner(model_manager=mm)


# ---------------------------------------------------------------------------
# Deployed function: ResponsePlanner._parse_review — rejections
# ---------------------------------------------------------------------------


class TestParseReviewStrictContractRejections:

    def test_missing_passes_no_longer_silently_passes(self):
        """Failing-before proof: today `bool(data.get("passes", True))`
        defaults an absent `passes` to True — a malformed review is
        indistinguishable from a genuine pass. Must now reject."""
        raw = json.dumps({"confidence": 0.9, "issues": [], "suggestion": ""})
        assert ResponsePlanner._parse_review(raw) is None

    def test_missing_confidence_rejected(self):
        raw = json.dumps({"passes": True, "issues": [], "suggestion": ""})
        assert ResponsePlanner._parse_review(raw) is None

    def test_both_required_fields_missing_rejected(self):
        assert ResponsePlanner._parse_review("{}") is None

    @pytest.mark.parametrize("bad", ["false", "true", 0, 1, None, "yes"])
    def test_passes_wrong_type_rejected(self, bad):
        assert ResponsePlanner._parse_review(_p(passes=bad)) is None

    @pytest.mark.parametrize("bad", ["0.9", True, False, None, []])
    def test_confidence_wrong_type_rejected(self, bad):
        assert ResponsePlanner._parse_review(_p(confidence=bad)) is None

    @pytest.mark.parametrize("bad", [-0.1, 1.5, -1, 2])
    def test_confidence_out_of_range_rejected(self, bad):
        assert ResponsePlanner._parse_review(_p(confidence=bad)) is None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_confidence_non_finite_rejected(self, bad):
        assert ResponsePlanner._parse_review(_p(confidence=bad)) is None

    @pytest.mark.parametrize("bad", ["x", 5, None, {"a": 1}, ["ok", 5], [None]])
    def test_issues_wrong_type_or_element_rejected(self, bad):
        assert ResponsePlanner._parse_review(_p(issues=bad)) is None

    @pytest.mark.parametrize("bad", [5, ["x"], None, True])
    def test_suggestion_wrong_type_rejected(self, bad):
        assert ResponsePlanner._parse_review(_p(suggestion=bad)) is None

    @pytest.mark.parametrize("raw", [
        "[1, 2]", '"just a string"', "42", "null", "true",
    ])
    def test_top_level_non_object_rejected_without_raising(self, raw):
        # Today `data.get("passes", True)` raises an uncaught AttributeError
        # for any non-dict top level (same BC-21 shape as F03/S01); must
        # return None instead.
        assert ResponsePlanner._parse_review(raw) is None

    def test_malformed_json_rejected(self):
        assert ResponsePlanner._parse_review("not valid json {{{") is None

    def test_empty_string_rejected(self):
        assert ResponsePlanner._parse_review("") is None

    def test_rejection_warning_names_only_field_and_type(self, caplog):
        """Privacy: the one warning names the field/type, never the model's
        own issues/suggestion text (which may quote the response)."""
        secret = "the user's private medical detail"
        raw = _p(passes="false", issues=[secret], suggestion=secret)
        with caplog.at_level(logging.WARNING, logger="response_planner"):
            result = ResponsePlanner._parse_review(raw)
        assert result is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert secret not in warnings[0]
        assert "passes" in warnings[0]


# ---------------------------------------------------------------------------
# Deployed function: ResponsePlanner._parse_review — controls
# ---------------------------------------------------------------------------


class TestParseReviewStrictContractControls:

    def test_valid_full_payload_accepted(self):
        review = ResponsePlanner._parse_review(_p(
            passes=False, confidence=0.42, issues=["a", "b"], suggestion="fix it",
        ))
        assert review is not None
        assert review.passes is False
        assert review.confidence == 0.42
        assert review.issues == ["a", "b"]
        assert review.suggestion == "fix it"

    @pytest.mark.parametrize("value", [True, False])
    def test_valid_true_and_false_passes_accepted(self, value):
        review = ResponsePlanner._parse_review(_p(passes=value))
        assert review is not None
        assert review.passes is value

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_confidence_boundaries_accepted(self, value):
        review = ResponsePlanner._parse_review(_p(confidence=value))
        assert review is not None
        assert review.confidence == value

    def test_int_confidence_accepted_as_number(self):
        review = ResponsePlanner._parse_review(_p(confidence=1))
        assert review is not None
        assert review.confidence == 1.0

    def test_absent_optional_fields_default(self):
        raw = json.dumps({"passes": True, "confidence": 0.5})
        review = ResponsePlanner._parse_review(raw)
        assert review is not None
        assert review.issues == []
        assert review.suggestion == ""

    def test_code_fenced_json_parsed(self):
        fenced = f"```json\n{_p()}\n```"
        review = ResponsePlanner._parse_review(fenced)
        assert review is not None
        assert review.passes is True

    def test_extra_keys_ignored(self):
        raw = json.dumps({
            "passes": True, "confidence": 0.5, "issues": [], "suggestion": "",
            "extra_field_from_model": "chatter",
        })
        assert ResponsePlanner._parse_review(raw) is not None


# ---------------------------------------------------------------------------
# Deployed caller: ResponsePlanner.review_answer()
# ---------------------------------------------------------------------------


class TestReviewAnswerDeployedCallerStrictContract:

    @pytest.mark.asyncio
    async def test_missing_passes_review_answer_returns_none(self):
        planner = _make_planner(json.dumps(
            {"confidence": 0.9, "issues": [], "suggestion": ""}
        ))
        plan = ResponsePlan(key_points=["x"], tone="warm", avoid=[], strategy="y")
        with patch("config.app_config.RESPONSE_REVIEW_MODEL", None), \
             patch("config.app_config.RESPONSE_REVIEW_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_REVIEW_TIMEOUT", 5.0):
            review = await planner.review_answer(plan, "a response long enough", "query")
        assert review is None

    @pytest.mark.asyncio
    async def test_paired_control_valid_payload_returns_review(self):
        planner = _make_planner(_p(passes=False, confidence=0.9))
        plan = ResponsePlan(key_points=["x"], tone="warm", avoid=[], strategy="y")
        with patch("config.app_config.RESPONSE_REVIEW_MODEL", None), \
             patch("config.app_config.RESPONSE_REVIEW_MAX_TOKENS", 200), \
             patch("config.app_config.RESPONSE_REVIEW_TIMEOUT", 5.0):
            review = await planner.review_answer(plan, "a response long enough", "query")
        assert review is not None
        assert review.passes is False
        assert review.confidence == 0.9


# ---------------------------------------------------------------------------
# Deployed review gate: gui/handlers.py handle_submit() (log-only,
# 2026-08-28, handlers.py:4628-4680). Never edited — only imported/read.
# ---------------------------------------------------------------------------


def _capture_submit_contexts(monkeypatch):
    """Spy on gui.handlers.SubmitContext so the test can read the real
    ctx.telemetry the deployed handle_submit() built for this turn, without
    editing gui/handlers.py. patch.stopall() (used by _run_submit) does not
    unwind a monkeypatch fixture setattr, so this survives the whole run."""
    captured = []
    original = _gui_handlers.SubmitContext

    def _spy(*args, **kwargs):
        inst = original(*args, **kwargs)
        captured.append(inst)
        return inst

    monkeypatch.setattr(_gui_handlers, "SubmitContext", _spy)
    return captured


class TestDeployedReviewGateLogOnly:
    """Drives the actual gui/handlers.py handle_submit() review-gate block
    end to end with a REAL ResponsePlanner whose model_manager is a fake
    generate_once — no mocked ReviewResult, no network."""

    # Query that avoids all keyword lists but is long enough to not be
    # casual-skipped (mirrors TestReviewGate in test_handle_submit.py).
    _QUERY = "Can you describe the color of my cat's fur and general appearance in some detail?"
    _INITIAL_RESPONSE = "The cat has some fur. " * 8  # >= 120 chars for the gate to fire

    async def _run(self, monkeypatch, review_json_text):
        captured = _capture_submit_contexts(monkeypatch)
        orch = _make_orchestrator(agentic_enabled=True, streaming_chunks=[self._INITIAL_RESPONSE])
        orch.response_planner = _make_planner(review_json_text)
        orch._current_response_plan = ResponsePlan(
            key_points=["describe fur"], tone="warm", avoid=[], strategy="be descriptive",
        )
        retry_mock = AsyncMock(
            return_value=("A retried answer that must never appear — log-only never swaps.", ""),
        )
        extra = [
            patch("config.app_config.RESPONSE_REVIEW_ENABLED", True),
            patch("config.app_config.RESPONSE_REVIEW_CONFIDENCE_THRESHOLD", 0.5),
            patch("config.app_config.RESPONSE_REVIEW_MODEL", None),
            patch("config.app_config.RESPONSE_REVIEW_MAX_TOKENS", 200),
            patch("config.app_config.RESPONSE_REVIEW_TIMEOUT", 5.0),
            patch("config.app_config.UNCERTAINTY_FALLBACK_ENABLED", False),
            patch(
                "utils.web_search_trigger.analyze_for_web_search_llm",
                new_callable=AsyncMock, return_value=_no_trigger_decision(),
            ),
            patch("gui.handlers._silent_agentic_retry", retry_mock),
        ]
        results = await _run_submit(self._QUERY, orch, extra_patches=extra)
        assert captured, "SubmitContext was never constructed"
        return results, captured[-1], retry_mock

    @pytest.mark.asyncio
    async def test_missing_passes_review_not_recorded_or_acted_on_as_pass(self, monkeypatch):
        """The exact defect: today a missing `passes` defaults to True, so
        this turn's telemetry would read review_passed=True and the debug
        "passed review" branch (handlers.py:4669-4673) would fire —
        indistinguishable from a genuine pass. After the fix the review is
        unavailable: nothing is recorded, and the caller's log-only branches
        never treat it as a pass (or a fail)."""
        malformed = json.dumps({"confidence": 0.99, "issues": [], "suggestion": ""})
        results, ctx, retry_mock = await self._run(monkeypatch, malformed)

        content = _final_content(results)
        assert "The cat has some fur." in content
        assert "must never appear" not in content
        retry_mock.assert_not_awaited()

        # Not recorded as a pass: no review telemetry at all was written for
        # a rejected review — in particular, review_passed is never True.
        assert "review_passed" not in ctx.telemetry
        assert "review_fired" not in ctx.telemetry
        assert ctx.telemetry.get("review_passed") is not True

    @pytest.mark.asyncio
    async def test_paired_control_valid_failing_review_is_recorded_honestly(self, monkeypatch):
        """Paired control: a REAL, fully-typed failing review is recorded as
        a fail (never coerced to a pass) — proving the strict contract
        changes nothing about a correctly-typed payload, and that the
        malformed-review test above is actually exercising the gate."""
        valid_fail = _p(passes=False, confidence=0.99)
        results, ctx, retry_mock = await self._run(monkeypatch, valid_fail)

        content = _final_content(results)
        assert "The cat has some fur." in content  # log-only: still never swapped
        retry_mock.assert_not_awaited()

        assert ctx.telemetry.get("review_fired") is True
        assert ctx.telemetry.get("review_passed") is False
