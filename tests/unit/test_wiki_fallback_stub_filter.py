"""Fix 1.6 (2026-09-06): wiki "may refer to:" disambiguation stubs leaking
via the live-API fallback path.

Root cause (verified): the 2026-08-28 filter (`core.wiki_util.
looks_like_disambiguation_text`) was applied only on the two local
(ChromaDB/FAISS) wiki paths. The live-API fallback in
`core/prompt/gatherer_knowledge.py::_get_wiki_content` split the raw query on
whitespace with NO stopword filter (`len(word) > 3 and word.isalpha()`) and
fed every long word straight to `_get_wiki_snippet_cached` ->
`core.wiki_util.get_wiki_snippet`, which only checked `page.is_disambiguation`
— and `WikiManager._fetch_extract_action_api` (the Action-API fallback
inside `resolve_and_fetch`) hardcodes `is_disambiguation=False` regardless of
the fetched extract's actual shape. Live result: the query "Give me a
detailed analysis in a table of what my record can establish about
medication gaps." produced "Give may refer to: ..." and "Detail(s) ... may
refer to:" chunks under [BACKGROUND KNOWLEDGE].

These tests exercise the DEPLOYED `core.wiki_util.get_wiki_snippet` and the
DEPLOYED term selector wired into `core/prompt/gatherer_knowledge.py`'s
fallback loop (`knowledge.WikiManager._keywords_from_query`, reused via the
`_wiki_keywords_from_query` alias) — never a re-derivation.
"""
from unittest.mock import patch, MagicMock

from knowledge.WikiManager import WikiPage, STOPWORDS as WIKI_STOPWORDS
from core.wiki_util import get_wiki_snippet
from knowledge.WikiManager import _keywords_from_query as _wiki_keywords_from_query


LIVE_INCIDENT_QUERY = (
    "Give me a detailed analysis in a table of what my record can "
    "establish about medication gaps."
)


class TestFallbackTermSelectionExcludesStopwords:
    def test_live_incident_query_yields_no_stopword_terms(self):
        terms = _wiki_keywords_from_query(LIVE_INCIDENT_QUERY)
        assert terms, "expected at least one probe term"
        for term in terms:
            assert term.lower() not in WIKI_STOPWORDS
            assert len(term) > 1

    def test_give_and_me_never_selected(self):
        # The exact live junk probe: "Give" (-> "Give may refer to:"); "me"/
        # "my" are the raw-split fallback's other short-pronoun leaks.
        terms = [t.lower() for t in _wiki_keywords_from_query(LIVE_INCIDENT_QUERY)]
        assert "give" not in terms
        assert "me" not in terms
        assert "my" not in terms


class TestGetWikiSnippetDropsDisambiguationShapedExtracts:
    def _mock_manager(self, page):
        mgr = MagicMock()
        mgr.resolve_and_fetch = MagicMock(return_value=page)
        return mgr

    def test_disambiguation_shaped_extract_from_live_api_dropped(self):
        # This reproduces WikiManager._fetch_extract_action_api's behavior:
        # is_disambiguation hardcoded False even though the extract text is
        # a disambiguation stub.
        page = WikiPage(
            title="Give",
            url="https://en.wikipedia.org/wiki/Give",
            summary="Give may refer to:\n\nGive (album)\nGive (film)",
            is_disambiguation=False,
        )
        with patch("core.wiki_util._get_manager", return_value=self._mock_manager(page)):
            assert get_wiki_snippet("Give") in ("", None)

    def test_flagged_disambiguation_page_still_dropped(self):
        page = WikiPage(
            title="Mercury",
            url="https://en.wikipedia.org/wiki/Mercury",
            summary="Mercury may refer to several things.",
            is_disambiguation=True,
        )
        with patch("core.wiki_util._get_manager", return_value=self._mock_manager(page)):
            assert get_wiki_snippet("Mercury") in ("", None)

    def test_normal_extract_still_passes(self):
        page = WikiPage(
            title="Cariprazine",
            url="https://en.wikipedia.org/wiki/Cariprazine",
            summary="Cariprazine is an atypical antipsychotic used to treat "
                    "schizophrenia and bipolar disorder.",
            is_disambiguation=False,
        )
        with patch("core.wiki_util._get_manager", return_value=self._mock_manager(page)):
            result = get_wiki_snippet("Cariprazine")
        assert "Cariprazine is an atypical antipsychotic" in result

    def test_no_page_returns_empty(self):
        with patch("core.wiki_util._get_manager", return_value=self._mock_manager(None)):
            assert get_wiki_snippet("Zzzznonexistentqqq") == ""
