# /tests/unit/test_search_trigger_strict_parse.py
"""
S01 — strict-contract tests for `LLMSearchTriggerResponse.parse`, a BC-21
sibling of F03's `core/grounding_check.py::_parse_verdict` (A04) and BC-47
(failure collapsed into a valid empty result). Validate the model-authored
JSON's TYPES, never coerce them: `bool("false")` is `True` in Python — the
string `"false"` for `should_search` used to be silently promoted to a
triggered search. A wrong-typed/malformed payload must take the existing
classifier-failure path (`source="fallback"`), never a coerced verdict.

Privacy: search_terms/reason may carry the user's private query. Every
rejection logs at most one WARNING naming only the field and its Python
type (see test_rejection_warning_names_only_field_and_type).

No network: `.parse` is driven with in-memory strings; the deployed
`analyze_for_web_search_llm` is driven with a fake `model_manager`
(`generate_once` stubbed) — the seam tests/test_web_search_trigger.py
already uses.
"""
import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from utils.web_search_trigger import (
    LLMSearchTriggerResponse,
    WebSearchDecision,
    WebSearchDepth,
    analyze_for_web_search_llm,
)


def _valid_payload(**overrides):
    """Fully-typed taught-schema payload — the positive control every
    rejection test mutates one field of."""
    payload = {
        "should_search": True, "confidence": 0.85,
        "reason": "current news query",
        "search_terms": ["term one", "term two"],
        "search_depth": "standard", "num_searches": 2,
        "needs_memory_search": False, "needs_knowledge_search": False,
        "needs_document_generation": False, "needs_pattern_analysis": False,
        "document_topic": "", "document_type": "", "document_source": "",
    }
    payload.update(overrides)
    return payload


def _p(**overrides):
    """Parse a `_valid_payload` with one field overridden."""
    return LLMSearchTriggerResponse.parse(json.dumps(_valid_payload(**overrides)))


class TestParseTriggerStrictContractRejections:
    """Drives the deployed `LLMSearchTriggerResponse.parse` classmethod."""

    def test_string_false_should_search_no_longer_coerced_true(self):
        """Failing-before proof: bool("false") is True in Python."""
        payload = json.dumps(_valid_payload(should_search="false"))
        assert LLMSearchTriggerResponse.parse(payload) is None

    @pytest.mark.parametrize("bad_top", [["a", "b"], "just a string", 42, None])
    def test_top_level_non_object_rejected_without_raising(self, bad_top):
        # Used to raise AttributeError on data.get(...), uncaught.
        assert LLMSearchTriggerResponse.parse(json.dumps(bad_top)) is None

    def test_malformed_or_empty_input_rejected(self):
        assert LLMSearchTriggerResponse.parse("{not valid json") is None
        assert LLMSearchTriggerResponse.parse("") is None

    @pytest.mark.parametrize("field", ["should_search", "search_terms"])
    def test_required_field_missing_rejected(self, field):
        data = _valid_payload()
        del data[field]
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    @pytest.mark.parametrize("bad", ["false", "true", 0, 1, None, "yes"])
    def test_should_search_wrong_type_rejected(self, bad):
        data = _valid_payload(should_search=bad)
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    # "abc": the historical defect — list("abc") -> single-character terms.
    @pytest.mark.parametrize("bad", ["abc", 123, None, {"a": 1}, ["ok", 5], [None]])
    def test_search_terms_wrong_type_or_element_rejected(self, bad):
        data = _valid_payload(search_terms=bad)
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    @pytest.mark.parametrize("field", ["needs_memory_search", "needs_knowledge_search",
                                        "needs_document_generation", "needs_pattern_analysis"])
    @pytest.mark.parametrize("bad", ["true", 1, 0, None])
    def test_optional_bool_field_wrong_type_rejected(self, field, bad):
        data = _valid_payload(**{field: bad})
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    @pytest.mark.parametrize("bad", ["0.9", True, False, None, [0.9],
                                      float("nan"), float("inf"), float("-inf")])
    def test_confidence_wrong_type_or_non_finite_rejected(self, bad):
        data = _valid_payload(confidence=bad)
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    @pytest.mark.parametrize("bad", ["1", True, False, None, 1.5])
    def test_num_searches_wrong_type_rejected(self, bad):
        data = _valid_payload(num_searches=bad)
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    @pytest.mark.parametrize("field", ["search_depth", "document_source",
                                        "reason", "document_topic", "document_type"])
    @pytest.mark.parametrize("bad", [5, True, None, ["x"]])
    def test_str_field_wrong_type_rejected(self, field, bad):
        data = _valid_payload(**{field: bad})
        assert LLMSearchTriggerResponse.parse(json.dumps(data)) is None

    def test_rejection_warning_names_only_field_and_type(self, caplog):
        """The one warning never leaks query/terms/reason text (privacy)."""
        secret = "super secret medical condition search phrase"
        data = _valid_payload(should_search="false", search_terms=[secret], reason=secret)
        with caplog.at_level(logging.WARNING, logger="web_search_trigger"):
            result = LLMSearchTriggerResponse.parse(json.dumps(data))
        assert result is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert secret not in warnings[0]
        assert "should_search" in warnings[0] and "str" in warnings[0]


class TestParseTriggerStrictContractControls:
    """Paired non-triggering controls and documented defaults."""

    def test_valid_true_payload_every_field_present(self):
        result = _p()
        assert result is not None
        assert result.should_search is True
        assert result.search_terms == ["term one", "term two"]

    def test_valid_false_payload_minimal_required_fields_only(self):
        result = LLMSearchTriggerResponse.parse(
            json.dumps({"should_search": False, "search_terms": []}))
        assert result is not None
        assert result.should_search is False
        assert result.search_terms == []

    def test_absent_optional_fields_use_documented_defaults(self):
        result = LLMSearchTriggerResponse.parse(
            json.dumps({"should_search": True, "search_terms": ["x"]}))
        assert result is not None
        assert (result.confidence, result.reason, result.search_depth,
                result.num_searches, result.needs_memory_search,
                result.document_source) == (0.0, "", "quick", 1, False, "")

    def test_confidence_clamped_and_int_accepted(self):
        assert _p(confidence=1.5).confidence == 1.0
        assert _p(confidence=-0.5).confidence == 0.0
        assert _p(confidence=1).confidence == 1.0

    def test_num_searches_clamped(self):
        assert _p(num_searches=10).num_searches == 4
        assert _p(num_searches=0).num_searches == 1

    def test_unknown_whitelist_values_normalized_to_default(self):
        assert _p(search_depth="INVALID").search_depth == "quick"
        assert _p(document_source="somewhere_else").document_source == ""
        assert _p(document_source="Research").document_source == "research"

    def test_code_fenced_json_still_parses(self):
        raw = "```json\n" + json.dumps(_valid_payload()) + "\n```"
        result = LLMSearchTriggerResponse.parse(raw)
        assert result is not None and result.should_search is True


class TestAnalyzeForWebSearchLLMDeployedCaller:
    """Drives the deployed public entry point `analyze_for_web_search_llm`
    with a fake classifier — no network. Proves BC-47: a rejected payload
    must never dispatch a search on its own; it must take the existing
    classifier-failure path and record source="fallback"."""

    QUERY = "what's the latest update on this thing"

    @staticmethod
    def _neutral_heuristic():
        """Non-decisive so the caller consults the LLM, not a short-circuit."""
        return WebSearchDecision(
            should_search=False, depth=WebSearchDepth.QUICK, confidence=0.3,
            reason="ambiguous", matched_keywords=["update"], matched_patterns=[],
        )

    def _mock_manager(self, response_text):
        mgr = MagicMock()
        mgr.generate_once = AsyncMock(return_value=response_text)
        return mgr

    @pytest.mark.asyncio
    async def test_string_false_dispatches_no_search_and_records_fallback(self, monkeypatch):
        """Q13-style BC-21 shape (old code: bool("false") is True -> a
        search would have fired)."""
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()
        monkeypatch.setattr(wst, "should_search_heuristic", lambda q: self._neutral_heuristic())
        monkeypatch.setattr(wst, "LLM_FIRST_ENABLED", True)
        mock_manager = self._mock_manager(json.dumps(_valid_payload(
            should_search="false", confidence=0.95, search_terms=["term"])))
        decision = await analyze_for_web_search_llm(
            query=self.QUERY, model_manager=mock_manager,
            remaining_credits=50.0, web_search_enabled=True,
        )
        assert mock_manager.generate_once.called
        assert decision.should_search is False
        assert decision.source == "fallback"
        assert "Classifier unavailable" in decision.reason
        assert decision.search_terms in ([], None)

    @pytest.mark.asyncio
    async def test_paired_control_real_true_routes_to_search_within_budget(self, monkeypatch):
        """Paired positive control: a REAL JSON boolean true still routes
        to a search within budget."""
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()
        monkeypatch.setattr(wst, "should_search_heuristic", lambda q: self._neutral_heuristic())
        monkeypatch.setattr(wst, "LLM_FIRST_ENABLED", True)
        mock_manager = self._mock_manager(json.dumps(_valid_payload(
            should_search=True, confidence=0.95, search_terms=["current event term"])))
        decision = await analyze_for_web_search_llm(
            query=self.QUERY, model_manager=mock_manager,
            remaining_credits=50.0, web_search_enabled=True,
        )
        assert mock_manager.generate_once.called
        assert decision.should_search is True
        assert decision.source == "llm"
        assert decision.search_terms == ["current event term"]
