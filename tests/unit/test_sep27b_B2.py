# /tests/unit/test_sep27b_B2.py
"""
Lane B, batch B2 (2026-09-27, class: BC-15, BC-58) — "Tier-4 web-trigger LLM
has no email output — a new email request without an email noun still goes
to web" (FOLLOWUPS.md, item under "email intent without an email noun").

Live shape: "Northwind. Search that" after a prior email_search turn — the
gate's narrow Tier-1 arm (`_email_search_cue`) only fires on a word-bounded
e-mail/inbox/gmail/outlook noun IN THE CURRENT MESSAGE, so the turn falls
through to Tier 4. The Tier-4 LLM's decision schema had no email output at
all, so it could only ever guess should_search=web — searching the public
internet for the content of the user's own inbox.

Fix: `utils.web_search_trigger` gained a `needs_email_search` + `email_query`
field on the Tier-4 schema (dataclasses, strict-parse contract, prompt), and
`core.agentic.gate.evaluate_agentic_gate` routes a `needs_email_search` verdict
to tools mode (email_search is one of the tools the agentic loop can pick)
ONLY when `core.email.registry.provider_coverage()` shows a configured
provider — otherwise nothing could search anyway.

BC-15 (taught != parsed): the tests below drive the DEPLOYED
`_build_llm_trigger_prompt` (what the LLM is taught), the DEPLOYED
`LLMSearchTriggerResponse.parse` (what is actually read back), the DEPLOYED
`analyze_for_web_search_llm` (the public entry point, no re-derivation), and
the DEPLOYED `evaluate_agentic_gate` (the actual routing decision) — never a
reimplementation.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agentic.gate import evaluate_agentic_gate
from utils.web_search_trigger import (
    LLMSearchTriggerResponse,
    WebSearchDecision,
    WebSearchDepth,
    _build_llm_trigger_prompt,
    analyze_for_web_search_llm,
)


def _valid_payload(**overrides):
    """Fully-typed taught-schema payload (mirrors
    test_search_trigger_strict_parse.py's `_valid_payload` plus the two new
    B2 keys)."""
    payload = {
        "should_search": False, "confidence": 0.7,
        "reason": "email follow-up, no email noun in this message",
        "search_terms": [], "search_depth": "quick", "num_searches": 0,
        "needs_memory_search": False, "needs_knowledge_search": False,
        "needs_document_generation": False, "needs_pattern_analysis": False,
        "document_topic": "", "document_type": "", "document_source": "",
        "needs_email_search": True, "email_query": "Northwind",
    }
    payload.update(overrides)
    return payload


def _p(**overrides):
    return LLMSearchTriggerResponse.parse(json.dumps(_valid_payload(**overrides)))


# ---------------------------------------------------------------------------
# BC-15 taught-vs-parsed contract: the prompt's OUTPUT JSON block and the
# strict parser must speak the same field names.
# ---------------------------------------------------------------------------

class TestPromptTeachesEmailFields:

    def test_prompt_teaches_needs_email_search_and_email_query(self):
        prompt = _build_llm_trigger_prompt("Northwind. Search that", "2026-09-27")
        assert '"needs_email_search"' in prompt
        assert '"email_query"' in prompt
        assert "EMAIL SEARCH CRITERIA" in prompt

    def test_parser_reads_exactly_the_taught_field_names(self):
        """The parser must accept a payload keyed exactly as the prompt
        teaches it — a renamed field on either side (taught != parsed) would
        silently drop the signal."""
        result = _p()
        assert result is not None
        assert result.needs_email_search is True
        assert result.email_query == "Northwind"


# ---------------------------------------------------------------------------
# Strict-contract parse tests (BC-21 sibling, matches
# test_search_trigger_strict_parse.py's conventions: validate TYPES, never
# coerce; a malformed field rejects the whole payload).
# ---------------------------------------------------------------------------

class TestParseStrictContract:

    def test_missing_new_fields_default_safely(self):
        """An older/unrelated LLM payload that never mentions the new keys
        at all must still parse — OPTIONAL fields keep their defaults."""
        payload = _valid_payload()
        del payload["needs_email_search"]
        del payload["email_query"]
        result = LLMSearchTriggerResponse.parse(json.dumps(payload))
        assert result is not None
        assert result.needs_email_search is False
        assert result.email_query == ""

    @pytest.mark.parametrize("bad", ["true", 1, 0, None, "yes"])
    def test_needs_email_search_wrong_type_rejects_whole_payload(self, bad):
        assert _p(needs_email_search=bad) is None

    @pytest.mark.parametrize("bad", [123, 4.5, ["Northwind"], {"a": 1}])
    def test_email_query_wrong_type_rejects_whole_payload(self, bad):
        assert _p(email_query=bad) is None

    def test_email_query_free_text_not_whitelisted(self):
        """Unlike document_source (a closed whitelist), email_query is free
        text — any string value must survive unmodified."""
        result = _p(email_query="Maren registrar tuition")
        assert result.email_query == "Maren registrar tuition"


# ---------------------------------------------------------------------------
# Deployed public entry point: analyze_for_web_search_llm (no re-derivation,
# no network — fake model_manager.generate_once, same seam as
# test_search_trigger_strict_parse.py).
# ---------------------------------------------------------------------------

class TestAnalyzeForWebSearchLLMPropagatesEmailFields:

    QUERY = "Northwind. Search that"

    @staticmethod
    def _neutral_heuristic(_query):
        """Non-decisive so the caller consults the LLM, not a short-circuit."""
        return WebSearchDecision(
            should_search=False, depth=WebSearchDepth.QUICK, confidence=0.3,
            reason="ambiguous", matched_keywords=[], matched_patterns=[],
        )

    def _mock_manager(self, response_text):
        mgr = MagicMock()
        mgr.generate_once = AsyncMock(return_value=response_text)
        return mgr

    @pytest.mark.asyncio
    async def test_email_intent_without_email_noun_reaches_decision(self, monkeypatch):
        """Failing-before proof: before this batch, WebSearchDecision had no
        needs_email_search/email_query attributes at all, so a conversation-
        resolved email follow-up with no email noun in THIS message could
        only ever surface as an ordinary (or non-) web verdict."""
        import utils.web_search_trigger as wst
        wst._llm_trigger_cache.clear()
        monkeypatch.setattr(wst, "should_search_heuristic", self._neutral_heuristic)
        monkeypatch.setattr(wst, "LLM_FIRST_ENABLED", True)
        mock_manager = self._mock_manager(json.dumps(_valid_payload(
            email_query="Northwind loan")))
        decision = await analyze_for_web_search_llm(
            query=self.QUERY, model_manager=mock_manager,
            remaining_credits=50.0, web_search_enabled=True,
            conversation_context=(
                "User: any email about my student loan?\n"
                "Assistant: ran an email search, nothing in the last 60 days."
            ),
        )
        assert mock_manager.generate_once.called
        assert decision.needs_email_search is True
        assert decision.email_query == "Northwind loan"
        # Not a web search — the LLM was taught to keep at most one of these true.
        assert decision.should_search is False


# ---------------------------------------------------------------------------
# Gate routing (core/agentic/gate.py): the deployed evaluate_agentic_gate,
# driven with a patched analyze_for_web_search_llm — same pattern as
# tests/unit/test_agentic_gate.py's TestContinuationOverride tests.
# ---------------------------------------------------------------------------

class TestGateRoutesEmailIntentToTools:

    QUERY = "Northwind. Search that"

    @staticmethod
    def _email_decision(**overrides):
        base = dict(
            should_search=False, search_terms=[],
            needs_memory_search=False, needs_knowledge_search=False,
            needs_document_generation=False, needs_pattern_analysis=False,
            needs_email_search=True, email_query="Northwind",
            blocked_reason="",
        )
        base.update(overrides)
        return MagicMock(**base)

    @pytest.mark.asyncio
    async def test_configured_provider_routes_to_tools(self):
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock, return_value=self._email_decision(),
        ), patch(
            "core.email.registry.provider_coverage",
            return_value={"searched": ["gmail"], "unconnected": {}},
        ):
            d = await evaluate_agentic_gate(self.QUERY, model_manager=MagicMock())
        assert d.should_trigger is True
        assert "tools" in d.modes
        assert d.search_terms == []

    @pytest.mark.asyncio
    async def test_failed_but_configured_provider_still_routes(self):
        """A provider that is configured but currently broken (revoked
        token, E1's `failed` bucket) still routes to tools — the tool call
        itself (E1) is what renders the honest FAILED notice, never an
        upstream guess that nothing could be tried."""
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock, return_value=self._email_decision(),
        ), patch(
            "core.email.registry.provider_coverage",
            return_value={"searched": [], "unconnected": {},
                          "failed": {"gmail": "Gmail/Google authorization expired"}},
        ):
            d = await evaluate_agentic_gate(self.QUERY, model_manager=MagicMock())
        assert d.should_trigger is True
        assert "tools" in d.modes

    @pytest.mark.asyncio
    async def test_no_configured_provider_does_not_route(self):
        """Failing-before proof (base clone has no needs_email_search
        wiring at all — this scenario silently fell through to whatever
        should_search said, here False): with no email provider connected,
        the turn must not be routed into tools on an email guess either."""
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock, return_value=self._email_decision(),
        ), patch(
            "core.email.registry.provider_coverage",
            return_value={"searched": [], "unconnected": {
                "gmail": "not connected", "outlook": "disabled"}},
        ) as mock_coverage:
            d = await evaluate_agentic_gate(self.QUERY, model_manager=MagicMock())
        assert mock_coverage.called
        assert d.should_trigger is False
        assert "tools" not in d.modes

    @pytest.mark.asyncio
    async def test_web_shaped_request_unchanged(self):
        """Negative test (plan B2): an ordinary web verdict with
        needs_email_search=False must route exactly as before — the new
        elif branch must never fire and must never even consult the email
        registry for a turn that was never about email."""
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock,
            return_value=self._email_decision(
                should_search=True, search_terms=["current news term"],
                needs_email_search=False, email_query="",
            ),
        ), patch(
            "core.email.registry.provider_coverage",
        ) as mock_coverage:
            d = await evaluate_agentic_gate(
                "what's the latest on this thing", model_manager=MagicMock())
        assert d.should_trigger is True
        assert d.modes == ["web_search"]
        assert d.search_terms == ["current news term"]
        mock_coverage.assert_not_called()

    @pytest.mark.asyncio
    async def test_mock_double_missing_attribute_is_mock_safe(self):
        """A test double that predates this batch and never set
        needs_email_search must not auto-vivify a truthy Mock attribute into
        a false trigger (mirrors the needs_pattern_analysis `is True`
        mock-safety convention already in this file)."""
        stale_decision = MagicMock(
            should_search=False, search_terms=[],
            needs_memory_search=False, needs_knowledge_search=False,
            needs_document_generation=False, needs_pattern_analysis=False,
            blocked_reason="",
        )
        # Deliberately do NOT set needs_email_search / email_query.
        with patch(
            "utils.web_search_trigger.analyze_for_web_search_llm",
            new_callable=AsyncMock, return_value=stale_decision,
        ), patch(
            "core.email.registry.provider_coverage",
        ) as mock_coverage:
            d = await evaluate_agentic_gate(self.QUERY, model_manager=MagicMock())
        assert d.should_trigger is False
        mock_coverage.assert_not_called()
