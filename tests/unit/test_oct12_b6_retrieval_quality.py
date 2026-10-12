"""B6 (2026-10-12) retrieval quality: generic wiki probe terms, navigation
stub / duplicate wiki chunks, and BC-89 thread-extraction parsing.

Everything drives the DEPLOYED functions (no re-derivation)."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge.WikiManager import _keywords_from_query
from core.agentic.formatters import drop_navigation_and_duplicate_chunks
from core.agentic.tools import ToolExecutor
from memory.thread_extractor import ThreadExtractionError, ThreadExtractor


def _singles(terms):
    return [t for t in terms if " " not in t]


class TestGenericWikiProbeTerms:
    @pytest.mark.parametrize("query,bad", [
        ("what role does the new hire play on that team today", "Role"),
        ("how does the web connect all of these services together", "Web"),
        ("i think we should plan something for this weekend maybe", "We"),
        ("lets talk about how the weekend went for everyone", "Talk"),
    ])
    def test_session_shapes_never_probe_generic_single_token(self, query, bad):
        assert bad not in _keywords_from_query(query)

    def test_capitalized_proper_names_still_probe(self):
        terms = _keywords_from_query("tell me about Northwind and Chicago")
        assert "Northwind" in terms and "Chicago" in terms

    def test_quoted_phrase_still_probes(self):
        terms = _keywords_from_query('can you pull up the "talk radio" article please')
        assert "Talk Radio" in terms
        assert "Talk" in terms  # quoted = the user wrote it as a name

    def test_lowercase_multiword_topic_still_probes(self):
        terms = _keywords_from_query("explain quantum entanglement in detail")
        assert "Quantum Entanglement" in terms

    def test_single_content_token_topic_kept(self):
        assert _keywords_from_query("tell me about photosynthesis please") == ["Photosynthesis"]

    def test_capitalized_function_word_mid_sentence_is_a_name(self):
        assert "May" in _keywords_from_query("what happened to the band May yesterday")


class TestNavigationStubsAndDuplicates:
    def _r(self, section, text, title="France"):
        return {"title": title, "section": section, "text": text, "similarity": 0.5}

    def test_navigation_sections_dropped(self):
        rows = [
            self._r("See also", "* Rail transport in France\n* High-speed rail\n* SNCF"),
            self._r("External links", "https://sncf.com\nOfficial website"),
            self._r("References", "Smith, J. (2001). Railways. Oxford University Press. ISBN 123."),
            self._r("History", "The French rail network was organised around Paris from the 1840s under the Legrand star plan and later nationalised in 1938 as the SNCF."),
        ]
        out = drop_navigation_and_duplicate_chunks(rows)
        assert [r["section"] for r in out] == ["History"]

    def test_prose_under_nav_heading_is_kept(self):
        long = " ".join(["word"] * 30)
        out = drop_navigation_and_duplicate_chunks([self._r("Notes", long)])
        assert len(out) == 1

    def test_identical_chunks_deduped(self):
        a = self._r("History", "Same text   here.")
        b = self._r("History", "same text here.", title="France (continued)")
        assert len(drop_navigation_and_duplicate_chunks([a, b])) == 1

    def test_deployed_agentic_search_applies_it(self):
        rows = [
            self._r("See also", "* A\n* B"),
            self._r("History", "Real prose chunk."),
            self._r("History", "Real prose chunk."),
        ]
        with patch("knowledge.semantic_search.semantic_search_with_neighbors", return_value=rows):
            out = ToolExecutor._search_wiki_faiss(MagicMock(), "q", k=5)
        assert [r["text"] for r in out] == ["Real prose chunk."]


# ---- BC-89: misbehaving reasoning-model fake --------------------------------
_ITEM = {"topic": "Study for exam", "summary": "User needs to study",
         "thread_type": "deadline", "urgency": 0.8,
         "resolution_hint": "done", "deadline_date": None}


class _ReasoningModelFake:
    """Reasoning ON: bracket-laden preamble before the JSON, truncated at the
    budget. Reasoning OFF (disable_reasoning=True): clean JSON."""

    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    async def generate_once(self, prompt, **kw):
        self.calls.append(kw)
        if kw.get("disable_reasoning"):
            return json.dumps([_ITEM])
        if self.mode == "preamble":
            return "Let me think [step 1] about [unfinished] items.\n" + json.dumps([_ITEM])
        if self.mode == "truncated":
            full = json.dumps([_ITEM, dict(_ITEM, topic="Second")])
            return full[: len(full) - 25]
        return ""


_CONVOS = [{"query": "I need to study for my exam next Tuesday", "response": "ok"}]
_SAFE = "Extract.\n{conversation_text}\n{today}\n{tracked_threads}"


class TestThreadExtractionBC89:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["preamble", "truncated"])
    async def test_extraction_requests_no_reasoning_and_parses(self, mode):
        mm = _ReasoningModelFake(mode)
        with patch("memory.thread_extractor.EXTRACTION_PROMPT", _SAFE):
            threads = await ThreadExtractor(model_manager=mm).extract_new_threads(_CONVOS)
        assert mm.calls[0].get("disable_reasoning") is True
        assert mm.calls[0]["max_tokens"] >= 1200
        assert threads

    @pytest.mark.parametrize("mode", ["preamble", "truncated"])
    def test_parser_alone_tolerates_misbehaviour(self, mode):
        from memory.thread_extractor import _parse_json_array_strict
        fake = _ReasoningModelFake(mode)
        import asyncio
        raw = asyncio.run(fake.generate_once("p"))
        items = _parse_json_array_strict(raw)
        assert items and items[0]["topic"] == "Study for exam"

    def test_genuinely_unparseable_still_none_and_empty_array_still_empty(self):
        from memory.thread_extractor import _parse_json_array_strict
        assert _parse_json_array_strict("no array here [oops") is None
        assert _parse_json_array_strict("[]") == []

    @pytest.mark.asyncio
    async def test_resolution_call_also_disables_reasoning(self):
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value="[]")
        from memory.thread_models import OpenThread, ThreadType
        t = OpenThread(thread_id="t1", topic="x", summary="y",
                       thread_type=ThreadType.COMMITMENT, urgency=0.5)
        await ThreadExtractor(model_manager=mm).detect_resolutions(_CONVOS, [t])
        assert mm.generate_once.call_args.kwargs.get("disable_reasoning") is True


class TestDisambiguationMostOftenLead:
    """B6 item 2 (2026-10-12): 'X most often refers to:' leads were kept."""

    def test_most_often_refers_to_dropped(self):
        from core.wiki_util import looks_like_disambiguation_text
        assert looks_like_disambiguation_text("Web most often refers to:\n* World Wide Web")

    def test_most_usually_refers_to_dropped(self):
        from core.wiki_util import looks_like_disambiguation_text
        assert looks_like_disambiguation_text("Talk most usually refers to: conversation")

    def test_ordinary_lead_kept(self):
        from core.wiki_util import looks_like_disambiguation_text
        assert not looks_like_disambiguation_text(
            "Chicago is the most populous city in the U.S. state of Illinois."
        )
