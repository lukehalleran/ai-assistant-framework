"""Wiki disambiguation-page chunks must not reach the prompt.

Regression for 2026-08-28: the embedded wiki corpus contains disambiguation
pages as plain chunks — "Feel may refer to:\n\nFeeling" and "About may refer
to: About (surname) ..." rendered under [BACKGROUND KNOWLEDGE] on an emotional
turn. The live-API path drops them via page.is_disambiguation, but the local
ChromaDB wiki_knowledge path and the FAISS semantic-chunks path had no filter.
looks_like_disambiguation_text() is the shared text-shape test; these tests pin
its behavior and its wiring into the local wiki path.
"""
from unittest.mock import MagicMock

import pytest

from core.wiki_util import looks_like_disambiguation_text


class TestLooksLikeDisambiguationText:
    # --- the two live junk chunks ---

    def test_feel_chunk_dropped(self):
        assert looks_like_disambiguation_text("Feel may refer to:\n\nFeeling")

    def test_about_chunk_dropped(self):
        text = (
            "About may refer to:\n\nAbout (surname)\nAbout.com, an online "
            "source for original information and advice\nabout.me, a personal "
            "web hosting service"
        )
        assert looks_like_disambiguation_text(text)

    # --- shape variants ---

    def test_may_also_refer_to(self):
        assert looks_like_disambiguation_text("Mercury may also refer to:\nMercury (element)")

    def test_commonly_refers_to(self):
        assert looks_like_disambiguation_text("Java commonly refers to:\nJava (programming language)")

    def test_disambiguation_title_suffix(self):
        assert looks_like_disambiguation_text(
            "Some ordinary-looking lead paragraph.", title="Luke (disambiguation)"
        )

    def test_leading_whitespace_and_newlines(self):
        assert looks_like_disambiguation_text("\n\n  Feel   may\nrefer to:\nFeeling")

    # --- must NOT over-fire ---

    def test_ordinary_article_lead_kept(self):
        text = (
            "Feeling is the nominalization of the verb to feel. The word was "
            "first used in the English language to describe the physical "
            "sensation of touch through either experience or perception."
        )
        assert not looks_like_disambiguation_text(text)

    def test_mid_text_mention_kept(self):
        # The phrase appearing deep in an article body must not match — only
        # the chunk opening is tested.
        text = ("Cariprazine is an atypical antipsychotic. " * 5) + \
            "In some contexts the name may refer to different formulations."
        assert not looks_like_disambiguation_text(text)

    def test_empty_text(self):
        assert not looks_like_disambiguation_text("")
        assert not looks_like_disambiguation_text("", title="")

    def test_ordinary_title_not_dropped(self):
        assert not looks_like_disambiguation_text(
            "Cariprazine is an atypical antipsychotic.", title="Cariprazine"
        )


class TestWikiKnowledgePathFiltered:
    """The local ChromaDB wiki_knowledge path must apply the filter."""

    @pytest.mark.asyncio
    async def test_disambiguation_chunk_filtered_from_local_path(self):
        from core.prompt.context_gatherer import ContextGatherer

        coll = MagicMock()
        coll.count.return_value = 2
        chroma = MagicMock()
        chroma.collections = {"wiki_knowledge": coll}
        chroma.query_collection.return_value = [
            {
                "content": "Feel may refer to:\n\nFeeling",
                "metadata": {"title": "Feel"},
                "relevance_score": 0.9,
            },
            {
                "content": "Feeling is the nominalization of the verb to feel.",
                "metadata": {"title": "Feeling"},
                "relevance_score": 0.8,
            },
        ]

        coordinator = MagicMock()
        coordinator.chroma_store = chroma

        gatherer = ContextGatherer.__new__(ContextGatherer)
        gatherer.memory_coordinator = coordinator

        # NOTE: query must dodge _should_skip_wikipedia's substring patterns
        # ('hi' ⊂ "this", 'no' ⊂ "know", ...) — pre-existing behavior.
        results = await gatherer._get_wiki_content("describe the concept of emotion")
        contents = [r["content"] for r in results]
        assert all("may refer to" not in c for c in contents), contents
        assert any("nominalization" in c for c in contents)
