"""
tests/unit/test_routing_keyword_boundaries.py

Deployed-function regression tests for CGR-20260913-005 (dm01_raw_substring,
BC-01/BC-02): four query-routing keyword vocabularies matched by raw `in`
against the lowered query, routed through the utils.trigger_match chokepoint
by batch R06. See docs/execution/generalization/class_guard_responses/
CGR-20260913-005.md for the full disposition.

Anchors:
  #21 memory.memory_retriever.MemoryRetriever._maybe_temporal_window_rerank
      (_RETROSPECTIVE_MARKERS)                  — negation-INSENSITIVE
  #23 memory.user_profile.UserProfile._is_temporal_query
      (TEMPORAL_KEYWORDS)                       — negation-INSENSITIVE
  #28 utils.query_checker.has_thread_break_marker
      (THREAD_BREAK_MARKERS)                    — negation-INSENSITIVE
      (parent fix D5, 2026-09-14; R06 shipped it negation-aware)
  #31 utils.web_search_trigger.quick_prefilter_should_skip
      (EXPLICIT_SEARCH_PHRASES long-paste skip) — negation-AWARE, BC-58
      parity with utils.web_search_trigger._matches_phrase_non_negated
      (the sibling used at the ~842 call site; left unchanged, see response)
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from memory.memory_retriever import MemoryRetriever
from memory.user_profile import UserProfile
from utils.query_checker import has_thread_break_marker
from utils.web_search_trigger import (
    quick_prefilter_should_skip,
    _matches_phrase_non_negated,
    EXPLICIT_SEARCH_PHRASES,
)


def _wrap(text: str) -> str:
    """Line-wrap `text` the way a real client wraps a long paste — a
    newline + indentation inserted at a space, never splitting a token
    (utils/trigger_match.py normalize_ws docstring, 2026-09-10 probe-dump
    doctrine). Used for the BC-64 wrapped/indented fixture pairing."""
    midpoint = text.find(" ", len(text) // 3)
    if midpoint == -1:
        return text
    return text[:midpoint] + "\n  " + text[midpoint + 1:]


# ---------------------------------------------------------------------------
# Filler that must not itself trip SUPPRESSION_PATTERNS, a deictic-followup
# pattern, or any EXPLICIT_SEARCH_PHRASES entry — used to pad queries past
# quick_prefilter_should_skip's long-paste threshold (500 chars).
# ---------------------------------------------------------------------------
_NEUTRAL_FILLER = (
    "This paragraph is an ordinary block of pasted text repeated several "
    "times purely to build up length, and it carries no other meaningful "
    "trigger words of any kind in it at all. "
)


def _long(prefix: str, min_len: int = 520) -> str:
    text = prefix
    while len(text) < min_len:
        text += _NEUTRAL_FILLER
    return text


# ===========================================================================
# Anchor #21 — memory_retriever.MemoryRetriever._maybe_temporal_window_rerank
# ===========================================================================

class _FakeScorer:
    def __init__(self, anchor_hours):
        self._intent_weight_overrides = {"_temporal_anchor_hours": anchor_hours}


def _retriever():
    return MemoryRetriever(
        corpus_manager=None, chroma_store=None, scorer=_FakeScorer(24), time_manager=None
    )


def _ranked_pair():
    """[out-of-window (too recent), in-window] — a genuine retrospective
    marker must promote the in-window item to the front; anything else must
    leave the order untouched (the function's `return ranked` fast path)."""
    now = datetime.now()
    out_of_window = {"id": "recent", "timestamp": now - timedelta(hours=1)}
    in_window = {"id": "retrospective", "timestamp": now - timedelta(hours=20)}
    return [out_of_window, in_window]


class TestRetrospectiveMarkersBoundary:
    def test_containment_yesterdayfilter_does_not_trigger(self):
        """'yesterday' raw-substring-embeds in 'yesterdayfilter' (a made-up
        config name) but is not a retrospective reference — must not
        reorder."""
        ranked = _ranked_pair()
        result = _retriever()._maybe_temporal_window_rerank(
            ranked, "please check the yesterdayfilter setting on my account"
        )
        assert result[0]["id"] == "recent"

    def test_containment_wrapped(self):
        ranked = _ranked_pair()
        query = _wrap("please check the yesterdayfilter setting on my account for me today")
        result = _retriever()._maybe_temporal_window_rerank(ranked, query)
        assert result[0]["id"] == "recent"

    def test_positive_bare_word_yesterday(self):
        ranked = _ranked_pair()
        result = _retriever()._maybe_temporal_window_rerank(ranked, "What happened yesterday afternoon?")
        assert result[0]["id"] == "retrospective"

    def test_positive_phrase_last_night(self):
        ranked = _ranked_pair()
        result = _retriever()._maybe_temporal_window_rerank(ranked, "Tell me about last night")
        assert result[0]["id"] == "retrospective"

    def test_negated_still_counts(self):
        """Negation-INSENSITIVE by contract: a negated retrospective mention
        still refers to that time window and must still trigger the
        rerank."""
        ranked = _ranked_pair()
        result = _retriever()._maybe_temporal_window_rerank(ranked, "I didn't sleep well last night")
        assert result[0]["id"] == "retrospective"

    def test_no_marker_no_reorder(self):
        ranked = _ranked_pair()
        result = _retriever()._maybe_temporal_window_rerank(ranked, "What's the weather like right now?")
        assert result[0]["id"] == "recent"


# ===========================================================================
# Anchor #23 — user_profile.UserProfile._is_temporal_query
# ===========================================================================

@pytest.fixture
def profile():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield UserProfile(path)
    if os.path.exists(path):
        os.remove(path)


class TestTemporalKeywordsBoundary:
    def test_containment_unchanged_does_not_trigger(self, profile):
        assert profile._is_temporal_query("My results look unchanged this week") is False

    def test_containment_prehistory_does_not_trigger(self, profile):
        assert profile._is_temporal_query("Tell me about prehistory") is False

    def test_containment_beforehand_does_not_trigger(self, profile):
        assert profile._is_temporal_query("Let me know beforehand please") is False

    def test_containment_wrapped(self, profile):
        query = _wrap("My overall results look unchanged this week compared to normal")
        assert profile._is_temporal_query(query) is False

    def test_negated_still_counts(self, profile):
        """Negation-INSENSITIVE by contract."""
        assert profile._is_temporal_query("it hasn't changed over time") is True

    def test_positive_controls_preserved(self, profile):
        assert profile._is_temporal_query("What's my squat history?") is True
        assert profile._is_temporal_query("How has my sleep been over time?") is True
        assert profile._is_temporal_query("What did I used to weigh?") is True
        assert profile._is_temporal_query("Show my progress") is True
        assert profile._is_temporal_query("What's my name?") is False
        assert profile._is_temporal_query("How much can I bench?") is False


# ===========================================================================
# Anchor #28 — query_checker.has_thread_break_marker
# ===========================================================================

class TestThreadBreakMarkersBoundary:
    def test_containment_bare_word_unrelated(self):
        """'unrelated' raw-substring-embeds in 'unrelatedid' (a made-up
        field name) but signals no topic switch — must not fire."""
        assert has_thread_break_marker("show every row where unrelatedid is not null") is False

    def test_containment_wrapped(self):
        query = _wrap("please show every row where unrelatedid is not null in the export")
        assert has_thread_break_marker(query) is False

    def test_trailing_comma_preserved_anyway(self):
        """Punctuation-bearing marker 'anyway,': it fails the chokepoint's
        bare-word fullmatch (because of the comma) and so keeps raw-substring
        semantics — the comma requirement survives unchanged, no structural
        check needed at this site."""
        assert has_thread_break_marker("anyway, what about Python?") is True
        assert has_thread_break_marker("anyways what about Python?") is False

    def test_trailing_comma_preserved_by_the_way(self):
        assert has_thread_break_marker("by the way, do you like pizza?") is True
        assert has_thread_break_marker("by the way do you like pizza?") is False

    def test_negated_not_a_break(self):
        """Documented trade-off (parent fix D5, 2026-09-14): the marker check
        is negation-INSENSITIVE, so a negated marker still reads as a break.
        The generic negation cues (never / stop / skip / rather than /
        instead of) routinely PRECEDE a genuine topic change in chat — see
        TestThreadBreakDiscourseCues — and missing an explicit break is the
        costlier error (the old thread's context stays attached). Name kept
        for traceability with R06's failing-before list."""
        assert has_thread_break_marker(
            "I don't think this is unrelated, but let's continue with the current topic"
        ) is True

    def test_positive_controls_preserved(self):
        assert has_thread_break_marker("changing topics, let's discuss") is True
        assert has_thread_break_marker("switching gears now") is True
        assert has_thread_break_marker("this is unrelated to what we were discussing") is True
        assert has_thread_break_marker("continuing the discussion") is False


# ===========================================================================
# Anchor #31 — web_search_trigger.quick_prefilter_should_skip
# (EXPLICIT_SEARCH_PHRASES long-paste skip; BC-58 parity with
#  _matches_phrase_non_negated, the sibling used at the ~842 call site)
# ===========================================================================

class TestExplicitSearchPhrasesBoundary:
    def test_bc76_exactly_one_bare_single_word_in_vocabulary(self):
        """Recorded finding: EXPLICIT_SEARCH_PHRASES contains exactly one
        bare single word ('google'); every other entry is a multi-word or
        punctuated phrase that keeps substring semantics through the
        chokepoint. No phrase added or removed (BC-76)."""
        bare_words = [p for p in EXPLICIT_SEARCH_PHRASES if " " not in p and p.isalpha()]
        assert bare_words == ["google"]

    def test_containment_googlebot_does_not_trigger_skip(self):
        """'google' raw-substring-embeds in 'googlebot' but is not an
        explicit search request — pre-fix this leaves the long-paste skip
        un-triggered (a false 'has explicit phrase' signal); post-fix it
        must skip."""
        text = _long(
            "I'm trying to configure my robots.txt to block googlebot from "
            "crawling certain private pages on my personal blog. "
        )
        assert quick_prefilter_should_skip(text) is True

    def test_containment_wrapped(self):
        prefix = _wrap(
            "I'm trying to configure my robots.txt file to block googlebot "
            "from crawling certain private pages on my personal blog today"
        )
        assert quick_prefilter_should_skip(_long(prefix)) is True

    def test_negated_search_phrase_does_not_stop_the_skip(self):
        text = _long(
            "Don't search for the weather report, just summarize the "
            "following passage for me. "
        )
        assert quick_prefilter_should_skip(text) is True

    def test_non_negated_search_phrase_stops_the_skip(self):
        text = _long(
            "Please search for the latest research papers on this topic "
            "for me. "
        )
        assert quick_prefilter_should_skip(text) is False

    def test_unaffected_long_paste_without_phrase_still_skips(self):
        assert quick_prefilter_should_skip(_long("Some ordinary long paste with no search request in it. ")) is True

    def test_bc58_parity_negated_phrase(self):
        """Paired parity with the negation-aware sibling at web_search_trigger
        ~line 571 (_matches_phrase_non_negated, called at ~842): the SAME
        phrase, negated the SAME way, must agree between the two sites."""
        text = "Don't search for cats, just summarize this article for me."
        has_explicit, matches = _matches_phrase_non_negated(text, EXPLICIT_SEARCH_PHRASES)
        assert (has_explicit, matches) == (False, [])
        assert quick_prefilter_should_skip(_long(text + " ")) is True

    def test_bc58_parity_non_negated_phrase(self):
        text = "Please search for cats for me."
        has_explicit, matches = _matches_phrase_non_negated(text, EXPLICIT_SEARCH_PHRASES)
        assert has_explicit is True
        assert "search for" in matches
        assert quick_prefilter_should_skip(_long(text + " ")) is False


# ===========================================================================
# Parent fix D5 (R06 review, 2026-09-14): thread-break markers are
# negation-INSENSITIVE. The generic negation cues (never / stop / skip /
# rather than / instead of) routinely PRECEDE a genuine topic change in chat,
# and a missed explicit break keeps the previous thread's context attached
# (calculate_thread_continuity_score -> belongs_to_thread -> thread_manager).
# ===========================================================================

class TestThreadBreakDiscourseCues:
    CASES = [
        "never mind, moving on to dinner plans",
        "ok stop, different topic: what's the weather tomorrow",
        "let's skip that. new question: how do I bake bread",
        "rather than that, switching gears to my car insurance",
        "instead of that, changing topics to the garden",
    ]

    @pytest.mark.parametrize("query", CASES)
    def test_discourse_cue_before_marker_is_still_a_break(self, query):
        assert has_thread_break_marker(query) is True

    @pytest.mark.parametrize("query", CASES)
    def test_discourse_cue_before_marker_wrapped(self, query):
        words = query.split(" ")
        wrapped = " ".join(words[:2]) + "\n    " + " ".join(words[2:])
        assert has_thread_break_marker(wrapped) is True

    def test_continuity_score_honours_the_break(self):
        """Deployed consumer: the marker forces 0.0. Paired control: the
        same topic words without a marker score above 0.0, so the 0.0 is
        caused by the break rather than by zero overlap."""
        from utils.query_checker import calculate_thread_continuity_score
        last = "dinner plans tonight with the family"
        control = calculate_thread_continuity_score(
            current_query="dinner plans tonight, any ideas",
            last_query=last, time_diff_seconds=30)
        assert control > 0.0
        broken = calculate_thread_continuity_score(
            current_query="never mind, moving on to dinner plans tonight",
            last_query=last, time_diff_seconds=30)
        assert broken == 0.0
