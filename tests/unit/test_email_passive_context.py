"""Tests for passive [RELEVANT EMAILS] prompt section (Step 2).

Step 2: passive email retrieval in prompt builder, cue-gated + distress-suppressed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from core.email.provider import EmailMessage


class TestEmailPassiveContextFireConditions:
    """Fire conditions for passive email retrieval."""

    @pytest.mark.asyncio
    async def test_fire_on_email_cue(self):
        """Fire condition: query contains email cue."""
        # This would be tested via the builder, but at this level we just verify
        # the detection logic exists. Deferred to integration test.
        pass

    @pytest.mark.asyncio
    async def test_fire_on_proper_noun_contact(self):
        """Fire condition: query has rare proper nouns (contact names)."""
        # Same: deferred to integration.
        pass

    @pytest.mark.asyncio
    async def test_no_fire_without_cue_or_contact(self):
        """No fire: query has neither email cue nor contact."""
        # Same: deferred to integration.
        pass


class TestEmailPassiveContextDistressSuppression:
    """Distress suppression for email context."""

    @pytest.mark.asyncio
    async def test_suppressed_on_elevated_tone(self):
        """Email context is suppressed when _distress_active is True."""
        # Deferred to integration test with mock context.
        pass


class TestEmailPassiveContextRelevanceFiltering:
    """Relevance filtering of email results."""

    @pytest.mark.asyncio
    async def test_relevance_bar_filters_results(self):
        """Results below EMAIL_PASSIVE_MIN_RELEVANCE are filtered."""
        # Deferred to integration with mock embedder.
        pass

    @pytest.mark.asyncio
    async def test_cap_limits_results(self):
        """Results are capped at EMAIL_PASSIVE_MAX (default 3)."""
        # Deferred to integration.
        pass


class TestEmailPassiveContextFormatting:
    """Email result formatting in the [RELEVANT EMAILS] section."""

    def test_email_section_header_and_results(self):
        """Section renders as [RELEVANT EMAILS] with numbered entries."""
        # This is tested in the formatter — deferred to test_email_formatter.
        pass


class TestEmailFormatterSection:
    """Formatter [RELEVANT EMAILS] section rendering."""

    def test_formatter_section_code_exists(self):
        """Verify formatter has code to render [RELEVANT EMAILS] section."""
        import inspect
        from core.prompt.formatter import PromptFormatter
        source = inspect.getsource(PromptFormatter)
        # Check that the formatter has email-related rendering code
        assert "relevant_emails" in source
        assert "[RELEVANT EMAILS]" in source


class TestEmailTokenBudgetIntegration:
    """Token budget integration for [RELEVANT EMAILS]."""

    def test_relevant_emails_in_priority_order(self):
        """relevant_emails is in PRIORITY_ORDER for token metering."""
        from core.prompt.token_manager import PRIORITY_ORDER
        priority_names = [name for name, _ in PRIORITY_ORDER]
        assert "relevant_emails" in priority_names

    def test_relevant_emails_priority_is_high(self):
        """relevant_emails has high priority (7) like other real-time context."""
        from core.prompt.token_manager import PRIORITY_ORDER
        priority_dict = dict(PRIORITY_ORDER)
        assert priority_dict.get("relevant_emails") == 7
        # Same tier as google_calendar and web_search_results
        assert priority_dict.get("google_calendar") == 7
        assert priority_dict.get("upcoming_schedule") == 7


class TestEmailBuilderIntegration:
    """Builder integration for email task creation and context assembly."""

    def test_email_task_created_when_enabled(self):
        """Builder creates email task when EMAIL_PASSIVE_CONTEXT_ENABLED."""
        # Deferred to full integration test with builder.
        pass

    def test_email_context_added_to_prompt_ctx(self):
        """Email results added to prompt_ctx['relevant_emails']."""
        # Deferred to full integration.
        pass

    def test_email_task_fails_gracefully(self):
        """Email task failure doesn't block builder (returns [])."""
        # Deferred to integration.
        pass


class TestCoverageAndContactIdentity:
    """2026-09-01 live-smoke findings: (F1) 'no reply from Morgan' was scoped
    to Gmail while the advisor's mail lived in unconnected Outlook — negative
    answers must name what was searched; (F2) the name-string anchor matched
    a Hinge 'Morgan & Luke' marketing mail — sender IDENTITY outranks body
    mentions."""

    def test_provider_coverage_shapes(self, monkeypatch):
        import core.email.registry as reg
        cov = reg.provider_coverage()
        assert set(cov) == {"searched", "unconnected"}
        # coverage_note always renders a sentence
        note = reg.coverage_note()
        assert note.endswith(".")

    def test_coverage_note_unconnected_named(self, monkeypatch):
        import core.email.registry as reg

        class _Fake:
            name = "gmail"
            def is_configured(self):
                return True

        monkeypatch.setattr(reg, "PROVIDERS", {
            "gmail": {"factory": lambda: _Fake(), "enabled": lambda: True},
            "outlook": {"factory": lambda: None, "enabled": lambda: False},
        })
        note = reg.coverage_note()
        assert "Gmail" in note
        assert "Outlook" in note and "disabled" in note

    def _run_gatherer(self, monkeypatch, messages, contacts, query, embedder_factory=None):
        """Drive get_relevant_emails with a fake service + fake contacts +
        deterministic embedder."""
        import asyncio

        import numpy as np

        import core.prompt.gatherer_knowledge as gk
        import core.email.service as svc
        import core.actions.google_contacts as gc

        class _FakeService:
            def __init__(self, msgs):
                self._msgs = msgs
            async def search(self, *a, **k):
                return self._msgs
            async def recent(self, *a, **k):
                return self._msgs

        async def _fake_resolve(name, **k):
            return contacts

        class _FakeEmbedder:
            def encode(self, text, **k):
                # advisor-related text embeds near the query; marketing far
                if "registration" in text.lower() or "advisor" in text.lower() \
                        or "reply" in text.lower():
                    return np.array([1.0, 0.0])
                return np.array([0.0, 1.0])

        monkeypatch.setattr(svc, "get_email_service",
                            lambda: _FakeService(messages))
        monkeypatch.setattr(gc, "resolve_contact", _fake_resolve)
        from models.model_manager import ModelManager
        if embedder_factory is None:
            embedder_factory = lambda: _FakeEmbedder()
        monkeypatch.setattr(ModelManager, "_get_cached_embedder",
                            staticmethod(embedder_factory))

        gatherer = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
        gatherer._distress_active = False
        return asyncio.run(
            gk.KnowledgeRetrievalMixin.get_relevant_emails(gatherer, query))

    def test_resolved_contact_sender_outranks_marketing_mention(self, monkeypatch):
        advisor = EmailMessage(
            provider="gmail", message_id="a1",
            sender="Morgan Reeves <Morgan@gatech.edu>",
            subject="Fall registration", snippet="You are all set for fall.",
            date="2026-08-28T10:00:00")
        hinge = EmailMessage(
            provider="gmail", message_id="h1",
            sender="Hinge Team <hello@hinge.co>",
            subject="Morgan & Luke, we recommend you to each other",
            snippet="View your most compatible. registration reply advisor",
            date="2026-08-09T10:00:00")
        out = self._run_gatherer(
            monkeypatch, [hinge, advisor],
            [{"name": "Morgan Reeves", "email": "Morgan@gatech.edu"}],
            "did Morgan ever reply about registration?")
        assert out, "advisor email must be kept"
        assert out[0]["sender"].startswith("Morgan Reeves")

    def test_sender_name_match_kept_without_semantic_pass(self, monkeypatch):
        # A genuine from-Morgan mail whose subject embeds poorly still surfaces.
        msg = EmailMessage(
            provider="gmail", message_id="m1",
            sender="Morgan <Morgan@example.com>",
            subject="Re:", snippet="ok sounds good",
            date="2026-08-30T10:00:00")
        out = self._run_gatherer(
            monkeypatch, [msg], [], "did Morgan ever reply about registration?")
        assert out and out[0]["sender"].startswith("Morgan")

    def test_body_mention_only_needs_bar(self, monkeypatch):
        # Name only in subject, sender unrelated, embeds FAR from query → dropped.
        msg = EmailMessage(
            provider="gmail", message_id="x1",
            sender="Hinge Team <hello@hinge.co>",
            subject="Morgan & Luke, we recommend you to each other",
            snippet="View your most compatible.",
            date="2026-08-09T10:00:00")
        out = self._run_gatherer(
            monkeypatch, [msg], [], "did Morgan ever reply about registration?")
        assert out == []

    def test_embedding_failure_does_not_inject_unranked_email(self, monkeypatch):
        msg = EmailMessage(
            provider="gmail", message_id="x2",
            sender="Sender <sender@example.com>",
            subject="Unrelated", snippet="Arbitrary inbox content",
            date="2026-08-30T10:00:00")

        def _raise():
            raise RuntimeError("embedder unavailable")

        out = self._run_gatherer(
            monkeypatch,
            [msg],
            [],
            "what is in my inbox?",
            embedder_factory=_raise,
        )
        assert out == []
