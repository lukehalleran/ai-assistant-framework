"""Topic labels come from grammar, not from chopping letters; attachments are not "this".

Incident date: 2026-10-28. "what's in the news today" became the topic "In The New" (a naive
trailing-"s" chop: "analysis" -> "analysi", "series" -> "serie"), and "make a clean version of
this" + an attached PDF inherited the previous probe's topic. Classes: BC-58, BC-91.
"""
import asyncio

import pytest

from utils.topic_manager import TopicManager, _load_spacy

needs_spacy = pytest.mark.skipif(_load_spacy() is None, reason="spaCy model unavailable")


def _label(text):
    return TopicManager()._extract_primary_from_text(text)


@needs_spacy
@pytest.mark.parametrize("q", ["what's in the news today", "   what's in the news today  "])
def test_news_label_keeps_uncountable_noun_and_drops_function_words(q):
    assert _label(q) == "News"


@needs_spacy
def test_real_plural_is_singularized_by_tag():
    assert _label("tell me about my cats") == "Cat"


@needs_spacy
@pytest.mark.parametrize("q,word", [("tell me about analysis", "Analysis"), ("tell me about the series", "Series")])
def test_singular_nouns_ending_in_s_are_unchanged(q, word):
    assert _label(q) == word


def test_without_spacy_candidate_is_unchanged(monkeypatch):
    import utils.topic_manager as tmod
    monkeypatch.setattr(tmod, "_load_spacy", lambda: None)
    assert TopicManager()._normalize_candidate_grammar("in the news") == "in the news"


class _FakeTopicManager:
    def __init__(self):
        self.last_topic = "Memory Scoring Recency"
        self.fresh_calls = 0

    def get_primary_topic(self, text=None):
        self.fresh_calls += 1
        return "Resume Formatting"


def _pipeline():
    from core.context_pipeline import ContextPipeline
    p = ContextPipeline.__new__(ContextPipeline)
    p.topic_manager = _FakeTopicManager()
    return p


@pytest.mark.parametrize("q", ["make a clean version of this", "   make a clean version of this  "])
def test_attachment_turn_does_not_inherit_previous_topic(q):
    p = _pipeline()
    primary, _ = asyncio.run(p._extract_topics(q, has_attachment=True))
    assert primary == "Resume Formatting" and p.topic_manager.fresh_calls == 1


def test_plain_pronoun_followup_without_attachment_still_inherits():
    p = _pipeline()
    primary, _ = asyncio.run(p._extract_topics("it was maybe 3 years of that"))
    assert primary == "Memory Scoring Recency" and p.topic_manager.fresh_calls == 0
