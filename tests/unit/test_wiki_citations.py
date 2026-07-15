"""Regression: agentic Wikipedia results carry [WIKI_N] citations end-to-end.

Before 2026-07-14, wiki FAISS results were formatted as plain "[1] Wikipedia:"
headers with no citation ids, no source map, and no cite instruction — the
model improvised a bare literal [Wikipedia] with no Sources entry while
[WEB_N] links worked. These tests cover the deployed formatter and the
deployed display linkifier.
"""

from core.agentic.formatters import AgenticFormatter
from gui.handlers import _apply_web_citations


def _wiki_results():
    return [
        {"title": "Pretzel", "section": "History", "text": "twisted dough", "similarity": 0.91},
        {"title": "Lent", "section": "", "text": "fasting season", "similarity": 0.80},
    ]


class TestWikiFormatter:
    def test_headers_carry_wiki_ids(self):
        out = AgenticFormatter().format_wiki_faiss_results(_wiki_results())
        assert "[WIKI_1] Wikipedia: Pretzel / History" in out
        assert "[WIKI_2] Wikipedia: Lent" in out

    def test_start_index_continues_numbering_across_rounds(self):
        out = AgenticFormatter().format_wiki_faiss_results(_wiki_results(), start_index=3)
        assert "[WIKI_4] Wikipedia: Pretzel" in out
        assert "[WIKI_5] Wikipedia: Lent" in out
        assert "[WIKI_1]" not in out


class TestWikiLinkify:
    WIKI_MAP = {
        "WIKI_1": {"title": "Pretzel", "section": "History",
                   "url": "https://en.wikipedia.org/wiki/Pretzel"},
    }
    WEB_MAP = {
        "WEB_1": {"title": "History.com", "url": "https://history.com/pretzel"},
    }

    def test_wiki_marker_linkified_and_in_sources(self):
        text = "Monks made them [WIKI_1]."
        out = _apply_web_citations(text, {}, wiki_map=self.WIKI_MAP)
        assert "[[WIKI_1](https://en.wikipedia.org/wiki/Pretzel)]" in out
        assert "**Sources:**" in out
        assert "[WIKI_1] [Wikipedia: Pretzel](https://en.wikipedia.org/wiki/Pretzel)" in out

    def test_web_and_wiki_share_one_footer(self):
        text = "Origin story [WIKI_1], per historians [WEB_1]."
        out = _apply_web_citations(text, self.WEB_MAP, wiki_map=self.WIKI_MAP)
        assert out.count("**Sources:**") == 1
        assert "[WEB_1] [History.com]" in out
        assert "[WIKI_1] [Wikipedia: Pretzel]" in out

    def test_unmapped_wiki_marker_stripped(self):
        out = _apply_web_citations("Claim [WIKI_9].", {}, wiki_map=self.WIKI_MAP)
        assert "[WIKI_9]" not in out
        assert "**Sources:**" not in out

    def test_idempotent_on_wiki_links(self):
        once = _apply_web_citations("Fact [WIKI_1].", {}, wiki_map=self.WIKI_MAP)
        twice = _apply_web_citations(once, {}, wiki_map=self.WIKI_MAP)
        assert once == twice

    def test_web_only_behavior_unchanged(self):
        out = _apply_web_citations("Per [WEB_1].", self.WEB_MAP)
        assert "[[WEB_1](https://history.com/pretzel)]" in out
        assert "**Sources:**" in out
