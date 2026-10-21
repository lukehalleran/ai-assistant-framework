"""Passive email context opens on a KNOWN PERSON, not on any capitalized word.

Incident 2026-10-18..20: `[GmailSearch] Found N contacts for 'Python' /
'Process' / 'Blocks' / 'Oct'` — course words and a month abbreviation passed
`extract_rare_proper_nouns` (a keyword-anchor helper) and opened the contact
arm of `KnowledgeRetrievalMixin.get_relevant_emails`, injecting off-topic mail.

Class: BC-04 (classifier missing an anchor).
Drives THE deployed `get_relevant_emails`; the email provider is a recording fake.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import core.actions.google_contacts as gc
import core.email.service as svc
import core.prompt.gatherer_knowledge as gk
from core.email.provider import EmailMessage
from models.model_manager import ModelManager


class _Recorder:
    def __init__(self):
        self.calls = []

    async def search(self, terms, **kw):
        self.calls.append(terms)
        return [EmailMessage(provider="gmail", message_id="m1",
                             sender="Rowan Tamsin <rowan@example.com>",
                             subject="Re: lab", snippet="see you there",
                             date="2026-10-19T10:00:00")]


class _Emb:
    def encode(self, text, **k):
        return np.array([1.0, 0.0])


def _graph(people=(), others=()):
    ents = {n.lower(): "person" for n in people}
    ents.update({n.lower(): "concept" for n in others})
    return SimpleNamespace(
        resolve_entity=lambda m: m.lower() if m.lower() in ents else None,
        get_entity=lambda eid: SimpleNamespace(entity_type=ents[eid]),
    )


def _run(monkeypatch, query, graph, rec):
    monkeypatch.setattr("config.app_config.EMAIL_PASSIVE_CONTEXT_ENABLED", True)
    monkeypatch.setattr(svc, "get_email_service", lambda: rec)

    async def _resolve(name, **k):
        return []

    monkeypatch.setattr(gc, "resolve_contact", _resolve)
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: _Emb()))
    monkeypatch.setattr(gc, "_cache", {})
    g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
    g._distress_active = False
    g.memory_coordinator = SimpleNamespace(graph_memory=graph)
    return asyncio.run(gk.KnowledgeRetrievalMixin.get_relevant_emails(g, query))


def _forms(q):
    """Clean and wrapped/indented rendering of the same live text."""
    # wrapped/indented: hard line breaks after the first word only
    # (a newline before a name is a sentence boundary by design upstream).
    return [q, "  " + q.replace(" ", "\n  ", 1) + "\n"]


NOISE = [
    "how do Python sets work in this module",
    "we covered the Process module and the Blocks template",
    "the report is due Oct 25 for the class",
]


@pytest.mark.parametrize("query", [f for q in NOISE for f in _forms(q)])
def test_capitalized_non_person_words_never_open_the_email_arm(monkeypatch, query):
    rec = _Recorder()
    graph = _graph(people=["Rowan Tamsin"], others=["Python"])
    out = _run(monkeypatch, query, graph, rec)
    assert rec.calls == []
    assert list(out) == []


@pytest.mark.parametrize("query", _forms("so did Rowan Tamsin reply yet"))
def test_known_person_entity_opens_the_arm(monkeypatch, query):
    rec = _Recorder()
    out = _run(monkeypatch, query, _graph(people=["Rowan Tamsin"]), rec)
    assert rec.calls and "Rowan" in rec.calls[0]
    assert out


@pytest.mark.parametrize("query", _forms("check my inbox for the lab thing"))
def test_email_cue_arm_unchanged(monkeypatch, query):
    rec = _Recorder()
    out = _run(monkeypatch, query, _graph(), rec)
    assert rec.calls
    assert out


def test_cached_contact_name_opens_the_arm_without_graph(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("config.app_config.EMAIL_PASSIVE_CONTEXT_ENABLED", True)
    monkeypatch.setattr(svc, "get_email_service", lambda: rec)

    async def _resolve(name, **k):
        return []

    monkeypatch.setattr(gc, "resolve_contact", _resolve)
    monkeypatch.setattr(ModelManager, "_get_cached_embedder", staticmethod(lambda: _Emb()))
    monkeypatch.setattr(gc, "_cache", {("saved", "rowan"): [
        {"name": "Rowan Tamsin", "email": "rowan@example.com", "source": "contacts"}]})
    g = gk.KnowledgeRetrievalMixin.__new__(gk.KnowledgeRetrievalMixin)
    g._distress_active = False
    g.memory_coordinator = None
    asyncio.run(gk.KnowledgeRetrievalMixin.get_relevant_emails(g, "did Tamsin write back"))
    assert rec.calls
