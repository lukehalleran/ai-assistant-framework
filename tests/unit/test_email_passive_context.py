"""Tests for the passive [RELEVANT EMAILS] prompt section.

Passive email retrieval in the prompt gatherer: cue-gated, distress-suppressed,
relevance-filtered and capped; rendered by the formatter. 2026-10-27 (lane SRC,
BC-62): the fire-condition / suppression / relevance / cap / formatting classes
were ``pass`` stubs ("deferred to integration") that always passed; they now
drive the DEPLOYED ``KnowledgeRetrievalMixin.get_relevant_emails`` and
``PromptFormatter._assemble_prompt`` with fake email/embedder collaborators.
The three builder-integration stubs were removed: the builder creates the
``relevant_emails`` task inline in ``UnifiedPromptBuilder.build_prompt``'s
~1000-line gather block (no seam to drive without the full builder); the
per-intent ``max_relevant_emails`` override that feeds it is covered by
tests/unit/test_gatherer_knowledge_casual_gating.py, and the section outcome
by tests/unit/test_section_outcomes.py.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.email.provider import EmailMessage

QUERY = "did Morgan ever reply about registration?"
EMAIL_QUERY = "any new email about registration?"
WRAPPED_EMAIL_QUERY = "    any new email\n    about registration?\n   "


def _msg(i, subject="Fall registration", snippet="registration reply advisor",
         sender="Advisor <advisor@wrenfield.example>"):
    return EmailMessage(provider="gmail", message_id=f"m{i}", sender=sender,
                        subject=subject, snippet=snippet,
                        date="2026-08-28T10:00:00")


class _Embedder:
    """registration/advisor/reply text sits ON the query axis; anything else is orthogonal."""

    def encode(self, text, **k):
        low = text.lower()
        if "registration" in low or "advisor" in low or "reply" in low:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])


def _gather(monkeypatch, messages, query, *, distress=False, limit=3):
    import core.email.service as svc
    import core.prompt.gatherer_knowledge as gk
    from models.model_manager import ModelManager

    searches = []

    class _Service:
        async def search(self, terms, **k):
            searches.append(terms)
            return messages

    monkeypatch.setattr(svc, "get_email_service", lambda: _Service())
    monkeypatch.setattr(ModelManager, "_get_cached_embedder",
                        staticmethod(lambda: _Embedder()))
    g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
    g._distress_active = distress
    g.memory_coordinator = SimpleNamespace(graph_memory=None)
    out = asyncio.run(gk.KnowledgeRetrievalMixin.get_relevant_emails(g, query, limit))
    return out, searches


class TestEmailPassiveContextFireConditions:
    @pytest.mark.parametrize("query", [EMAIL_QUERY, WRAPPED_EMAIL_QUERY])
    def test_fire_on_email_cue(self, monkeypatch, query):
        out, searches = _gather(monkeypatch, [_msg(1)], query)
        assert searches, "an email cue must run the search"
        assert len(out) == 1 and out[0]["subject"] == "Fall registration"

    def test_no_fire_without_cue_or_contact(self, monkeypatch):
        out, searches = _gather(monkeypatch, [_msg(1)], "how was the weather today")
        assert searches == []
        assert out == []

    def test_disabled_flag_never_searches(self, monkeypatch):
        import config.app_config as cfg
        monkeypatch.setattr(cfg, "EMAIL_PASSIVE_CONTEXT_ENABLED", False)
        out, searches = _gather(monkeypatch, [_msg(1)], EMAIL_QUERY)
        assert searches == [] and out == []


class TestEmailPassiveContextDistressSuppression:
    @pytest.mark.parametrize("query", [EMAIL_QUERY, WRAPPED_EMAIL_QUERY])
    def test_suppressed_on_elevated_tone(self, monkeypatch, query):
        out, searches = _gather(monkeypatch, [_msg(1)], query, distress=True)
        assert out == []
        assert searches == [], "suppression must precede the provider call"

    def test_same_instance_toggle_restores_results(self, monkeypatch):
        calm, _ = _gather(monkeypatch, [_msg(1)], EMAIL_QUERY, distress=False)
        sad, _ = _gather(monkeypatch, [_msg(1)], EMAIL_QUERY, distress=True)
        assert len(calm) == 1 and sad == []


class TestEmailPassiveContextRelevanceFiltering:
    def test_relevance_bar_filters_results(self, monkeypatch):
        near = _msg(1)
        far = _msg(2, subject="Weekend sale", snippet="discount shoes", sender="Shop <s@shop.example>")
        out, _ = _gather(monkeypatch, [far, near], EMAIL_QUERY)
        assert [m["subject"] for m in out] == ["Fall registration"]
        assert out[0]["relevance"] >= 0.35

    def test_bar_follows_the_live_setting(self, monkeypatch):
        import config.app_config as cfg
        far = _msg(2, subject="Weekend sale", snippet="discount shoes", sender="Shop <s@shop.example>")
        strict, _ = _gather(monkeypatch, [far], EMAIL_QUERY)
        assert strict == []
        monkeypatch.setattr(cfg, "EMAIL_PASSIVE_MIN_RELEVANCE", -1.0)
        lax, _ = _gather(monkeypatch, [far], EMAIL_QUERY)
        assert [m["subject"] for m in lax] == ["Weekend sale"]

    def test_cap_limits_results(self, monkeypatch):
        msgs = [_msg(i, subject=f"Fall registration {i}") for i in range(6)]
        out, _ = _gather(monkeypatch, msgs, EMAIL_QUERY, limit=2)
        assert len(out) == 2


def _formatter():
    from core.prompt.formatter import PromptFormatter
    tm = MagicMock()
    tm.count_tokens = MagicMock(return_value=10)
    f = PromptFormatter(token_manager=tm, time_manager=None)
    f._feature_inventory_cache = None
    return f


def _ctx(**over):
    base = {
        "recent_conversations": [], "memories": [], "facts": [], "summaries": [],
        "reflections": [], "semantic_chunks": [], "wiki": "", "semantic_wiki": "",
        "unresolved_threads": [], "upcoming_schedule": [], "google_calendar": [],
        "proactive_insights": [], "web_search_results": None,
    }
    base.update(over)
    return base


class TestEmailPassiveContextFormatting:
    def test_email_section_header_and_results(self):
        emails = [
            {"date": "2026-08-28T10:00:00", "sender": "Advisor <advisor@wrenfield.example>",
             "subject": "Fall registration", "snippet": "You are all set.", "provider": "gmail"},
            {"date": "2026-08-29T09:00:00", "sender": "Registrar <reg@wrenfield.example>",
             "subject": "Hold lifted", "snippet": "Your hold is cleared.", "provider": "outlook"},
        ]
        out = _formatter()._assemble_prompt(_ctx(relevant_emails=emails), "hello")
        assert "[RELEVANT EMAILS] n=2" in out
        assert "1) 2026-08-28 [G] Advisor" in out
        assert "2) 2026-08-29 [O] Registrar" in out
        assert "Fall registration" in out and "Your hold is cleared." in out

    def test_no_emails_no_section(self):
        assert "[RELEVANT EMAILS]" not in _formatter()._assemble_prompt(_ctx(), "hello")

    def test_failed_search_is_not_rendered_as_an_empty_inbox(self):
        ctx = _ctx(relevant_emails=[], _section_outcomes={
            "relevant_emails": {"status": "failed", "reason": "relevance_unavailable"}})
        out = _formatter()._assemble_prompt(ctx, "hello")
        assert "[RELEVANT EMAILS] n=0" in out
        assert "FAILED" in out and "relevance_unavailable" in out


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


class TestCoverageAndContactIdentity:
    """2026-09-01 live-smoke findings: (F1) 'no reply from Morgan' was scoped
    to Gmail while the advisor's mail lived in unconnected Outlook — negative
    answers must name what was searched; (F2) the name-string anchor matched
    a Hinge 'Morgan & Alex' marketing mail — sender IDENTITY outranks body
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
        # 2026-10-20 (BC-04): the passive contact arm needs a KNOWN PERSON, so
        # these name-anchored scenarios register "Morgan" as a person entity.
        from types import SimpleNamespace
        _graph = SimpleNamespace(
            resolve_entity=lambda m: "morgan" if m.lower() == "morgan" else None,
            get_entity=lambda eid: SimpleNamespace(entity_type="person"),
        )
        gatherer.memory_coordinator = SimpleNamespace(graph_memory=_graph)
        return asyncio.run(
            gk.KnowledgeRetrievalMixin.get_relevant_emails(gatherer, query))

    def test_resolved_contact_sender_outranks_marketing_mention(self, monkeypatch):
        advisor = EmailMessage(
            provider="gmail", message_id="a1",
            sender="Morgan Ashdown <Morgan@wrenfield.example>",
            subject="Fall registration", snippet="You are all set for fall.",
            date="2026-08-28T10:00:00")
        hinge = EmailMessage(
            provider="gmail", message_id="h1",
            sender="Hinge Team <hello@hinge.co>",
            subject="Morgan & Alex, we recommend you to each other",
            snippet="View your most compatible. registration reply advisor",
            date="2026-08-09T10:00:00")
        out = self._run_gatherer(
            monkeypatch, [hinge, advisor],
            [{"name": "Morgan Ashdown", "email": "Morgan@wrenfield.example"}],
            "did Morgan ever reply about registration?")
        assert out, "advisor email must be kept"
        assert out[0]["sender"].startswith("Morgan Ashdown")

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
            subject="Morgan & Alex, we recommend you to each other",
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
