"""Tests for the 2026-09-04 theme-sweep evidence-coverage fixes.

Live incident: a "gather every correction from 2026-07-15 through today"
theme sweep assembled 77 items across seven ISO weeks; newest-first sort plus
a 12000-char render cap meant only the newest ~3 days (37 items) ever reached
the model, 7 of those were the request turn itself / the reply about it, the
user's own quoted correction cue phrases were never scanned for, and a
pasted third-party email inside a user turn rendered as "you said".
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from core.insight.facets import extract_quoted_phrases
from core.insight.provenance import label_evidence, render_evidence_block
from core.insight.sweep import (
    _finalize,
    exclude_current_request_evidence,
    filter_evidence_by_date_window,
    interleave_evidence_for_coverage,
    run_sweep,
    week_bucket_key,
    window_scan_collection,
)
from core.insight.types import EvidenceItem, FacetPlan, FacetQuery
from memory.corpus_manager import CorpusManager


def _caps(**over):
    caps = {
        "per_facet_cap": 10,
        "total_evidence_cap": 200,
        "evidence_snippet_chars": 280,
        "keyword_scan_max": 50,
        "expand_top_k": 0,
        "expand_window": 2,
        "sweep_timeout_s": 45.0,
    }
    caps.update(over)
    return caps


def _week_item(i, week_offset, collection="conversations", text_len=40):
    """One item in ISO week ``week_offset`` weeks before 2026-09-04."""
    base = datetime(2026, 9, 4) - timedelta(weeks=week_offset, days=(i % 3))
    return EvidenceItem(
        doc_id=f"d{week_offset}-{i}",
        text=("no I mean that's a correction about topic " + str(i)) * (text_len // 10 + 1),
        date=base.date().isoformat(),
        collection=collection,
        speaker="user",
    )


class TestWindowFairRendering:
    """Task 1: 77 items across 7 ISO weeks must not collapse to the newest
    few days behind a 12000-char render cap."""

    def _build_77_items(self):
        items = []
        # 7 distinct ISO weeks, ~11 items each = 77 total.
        for week_offset in range(7):
            for i in range(11):
                items.append(_week_item(i, week_offset))
        return items

    def test_interleave_spans_the_whole_window_under_cap(self):
        items = self._build_77_items()
        assert len(items) == 77
        distinct_weeks = {week_bucket_key(i.date) for i in items}
        assert len(distinct_weeks) == 7

        reordered = interleave_evidence_for_coverage(items)
        assert len(reordered) == 77  # pure reorder, nothing dropped/invented
        assert {i.doc_id for i in reordered} == {i.doc_id for i in items}

        labeled = label_evidence(reordered)
        block = render_evidence_block(labeled, max_chars=12000)

        # render_evidence_block emits one "[E<n>] ..." line per rendered
        # item, in the same order as `labeled`, followed by an omission
        # note + range summary — so the number of "[E" lines identifies the
        # rendered PREFIX of `labeled`.
        rendered_count = sum(1 for ln in block.splitlines() if ln.startswith("[E"))
        assert 0 < rendered_count < len(labeled)
        rendered_weeks = {week_bucket_key(i.date) for i in labeled[:rendered_count]}
        assert len(rendered_weeks) >= 5, (
            f"rendered only {len(rendered_weeks)} distinct weeks: {rendered_weeks}"
        )

    def test_omission_note_names_weeks(self):
        items = self._build_77_items()
        reordered = interleave_evidence_for_coverage(items)
        labeled = label_evidence(reordered)
        block = render_evidence_block(labeled, max_chars=3000)
        assert "omitted for space" in block
        assert "week of" in block

    def test_rendered_range_stated(self):
        items = self._build_77_items()
        labeled = label_evidence(interleave_evidence_for_coverage(items))
        block = render_evidence_block(labeled, max_chars=12000)
        assert "spans" in block

    def test_pure_reorder_never_drops_items(self):
        items = self._build_77_items()
        out = interleave_evidence_for_coverage(items)
        assert sorted(i.doc_id for i in out) == sorted(i.doc_id for i in items)

    def test_undated_items_kept_and_appended_last(self):
        items = self._build_77_items()
        items.append(EvidenceItem(doc_id="undated-1", text="no date here", date=None,
                                   collection="facts"))
        out = interleave_evidence_for_coverage(items)
        assert out[-1].doc_id == "undated-1"
        assert len(out) == 78


class TestRequestSelfExclusion:
    """Task 2: the sweep must not report the request itself, or the reply
    about it, as if it were history — while keeping unrelated same-day
    evidence."""

    REQUEST = (
        "From 2026-07-15 through today, find every correction I made about "
        "the budget numbers in the quarterly finance report we discussed."
    )

    def test_drops_near_duplicate_request_chunk(self):
        chunk = EvidenceItem(
            doc_id="req-echo", text=self.REQUEST, date="2026-09-04",
            collection="conversations", speaker="user",
        )
        unrelated = EvidenceItem(
            doc_id="unrelated", text="Had pizza for lunch today, nothing much else happened.",
            date="2026-09-04", collection="conversations", speaker="user",
        )
        out = exclude_current_request_evidence([chunk, unrelated], self.REQUEST)
        assert [i.doc_id for i in out] == ["unrelated"]

    def test_drops_assistant_reply_about_the_request(self):
        reply = EvidenceItem(
            doc_id="reply-echo",
            text=(
                "User: hey can you actually go do that\n"
                "Assistant: Sure — from 2026-07-15 through today I will find "
                "every correction you made about the budget numbers in the "
                "quarterly finance report we discussed."
            ),
            date="2026-09-04", collection="conversations", speaker="",
        )
        unrelated = EvidenceItem(
            doc_id="unrelated2", text="Went for a walk this afternoon, weather was nice.",
            date="2026-09-04", collection="conversations", speaker="",
        )
        out = exclude_current_request_evidence([reply, unrelated], self.REQUEST)
        assert [i.doc_id for i in out] == ["unrelated2"]

    def test_keeps_unrelated_same_day_items(self):
        unrelated_a = EvidenceItem(doc_id="a", text="Fixed the leaky faucet today.",
                                    date="2026-09-04", collection="facts")
        unrelated_b = EvidenceItem(doc_id="b", text="Watched a documentary about space.",
                                    date="2026-09-04", collection="conversations", speaker="user")
        out = exclude_current_request_evidence([unrelated_a, unrelated_b], self.REQUEST)
        assert {i.doc_id for i in out} == {"a", "b"}

    def test_no_request_text_is_a_noop(self):
        item = EvidenceItem(doc_id="x", text="anything", date="2026-09-04", collection="facts")
        assert exclude_current_request_evidence([item], "") == [item]

    def test_current_turn_timestamp_match_drops(self):
        item = EvidenceItem(doc_id="x", text="completely unrelated content here",
                             date="2026-09-04T12:00:00", collection="facts")
        out = exclude_current_request_evidence(
            [item], "some other request text entirely",
            current_turn_date="2026-09-04T12:00:00",
        )
        assert out == []


class TestQuotedPhraseExtraction:
    """Task 3 (extraction half): double-quoted 2-6 word phrases become corpus
    scan cues; placeholder-letter and all-stopword phrases are skipped."""

    def test_extracts_plain_quoted_phrase(self):
        text = 'Find every time I said "no I mean" in our chats.'
        assert extract_quoted_phrases(text) == ["no I mean"]

    def test_skips_placeholder_letter_phrase(self):
        text = 'Find every time I said "you said X but" or "no I mean".'
        phrases = extract_quoted_phrases(text)
        assert "you said X but" not in phrases
        assert "no I mean" in phrases

    def test_skips_all_stopword_phrase(self):
        text = 'Also check for "the of it" and "that\'s wrong".'
        phrases = extract_quoted_phrases(text)
        assert "the of it" not in phrases
        assert "that's wrong" in phrases

    def test_legit_single_letter_words_not_placeholders(self):
        text = 'Find every time I said "I am wrong" please.'
        assert extract_quoted_phrases(text) == ["I am wrong"]

    def test_length_bounds(self):
        text = 'Too short: "no". Too long: "one two three four five six seven".'
        assert extract_quoted_phrases(text) == []


class TestQuotedPhraseSweepIntegration:
    """Task 3 (sweep half): quoted phrases in the request drive an extra
    corpus scan, merged with the same provenance shape as other corpus
    hits."""

    def _store(self):
        s = MagicMock()
        s.query_collection.return_value = []
        return s

    def test_quoted_phrases_scanned_and_placeholder_skipped(self):
        corpus = MagicMock()
        corpus.search_keyword.return_value = [
            {"timestamp": datetime(2026, 8, 1, 9, 0), "speaker": "user",
             "matched_term": "no I mean", "excerpt": "no I mean that's different",
             "query_preview": "x"},
        ]
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        request_text = 'Find every "no I mean" and every "you said X but" moment.'
        items = asyncio.run(run_sweep(
            plan, chroma_store=self._store(), corpus_manager=corpus,
            caps=_caps(), request_text=request_text,
        ))
        assert corpus.search_keyword.called
        called_terms = corpus.search_keyword.call_args.args[0]
        assert "no I mean" in called_terms
        assert "you said X but" not in called_terms
        assert any(i.facet == "quoted-phrase" and "no I mean" in i.text for i in items)

    def test_no_quotes_means_no_extra_scan(self):
        corpus = MagicMock()
        corpus.search_keyword.return_value = []
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        asyncio.run(run_sweep(
            plan, chroma_store=self._store(), corpus_manager=corpus,
            caps=_caps(), request_text="no quotes in this request at all",
        ))
        corpus.search_keyword.assert_not_called()

    def test_assistant_side_hits_excluded(self):
        corpus = MagicMock()
        corpus.search_keyword.return_value = [
            {"timestamp": datetime(2026, 8, 1, 9, 0), "speaker": "assistant",
             "matched_term": "no I mean", "excerpt": "assistant echoing the phrase",
             "query_preview": "x"},
        ]
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=self._store(), corpus_manager=corpus,
            caps=_caps(), request_text='Find every "no I mean" moment.',
        ))
        assert not any(i.facet == "quoted-phrase" for i in items)


class TestDateWindowFilter:
    """Task 4: an explicit ISO window filters dated sweep items and always
    keeps undated ones."""

    def test_filters_out_of_window_dated_items(self):
        items = [
            EvidenceItem(doc_id="early", text="x", date="2026-07-01", collection="facts"),
            EvidenceItem(doc_id="in-window", text="x", date="2026-07-20", collection="facts"),
            EvidenceItem(doc_id="late", text="x", date="2026-08-15", collection="facts"),
        ]
        out = filter_evidence_by_date_window(items, ("2026-07-15", "2026-08-01"))
        assert [i.doc_id for i in out] == ["in-window"]

    def test_undated_items_always_kept(self):
        items = [
            EvidenceItem(doc_id="undated", text="x", date=None, collection="graph"),
            EvidenceItem(doc_id="out-of-window", text="x", date="2026-01-01", collection="facts"),
        ]
        out = filter_evidence_by_date_window(items, ("2026-07-15", "2026-08-01"))
        assert [i.doc_id for i in out] == ["undated"]

    def test_no_window_is_a_noop(self):
        items = [EvidenceItem(doc_id="a", text="x", date="2026-01-01", collection="facts")]
        assert filter_evidence_by_date_window(items, None) == items

    def test_finalize_applies_date_window(self):
        items = [
            EvidenceItem(doc_id="in", text="content here", date="2026-07-20",
                         collection="facts"),
            EvidenceItem(doc_id="out", text="content here", date="2026-01-01",
                         collection="facts"),
            EvidenceItem(doc_id="nodate", text="content here", date=None,
                         collection="graph"),
        ]
        out = _finalize(items, _caps(), date_window=("2026-07-15", "2026-08-01"))
        ids = {i.doc_id for i in out}
        assert ids == {"in", "nodate"}

    def test_run_sweep_threads_date_window(self):
        store = MagicMock()

        def _q(coll, q, n):
            if coll != "facts":
                return []
            return [
                {"id": "in-window", "content": "some content long enough to survive hygiene checks",
                 "metadata": {"timestamp": "2026-07-20"}},
                {"id": "out-of-window", "content": "some content long enough to survive hygiene checks",
                 "metadata": {"timestamp": "2026-01-01"}},
            ]

        store.query_collection.side_effect = _q
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=store, corpus_manager=None,
            caps=_caps(), date_window=("2026-07-15", "2026-08-01"),
        ))
        ids = {i.doc_id for i in items}
        assert "out-of-window" not in ids
        assert "in-window" in ids


class TestQuotedCorrespondenceProvenance:
    """Task 5: a pasted third-party email inside a user turn must not render
    as "you said" — the quoted block gets its own label; separable framing
    lines stay attributed to the user."""

    EMAIL_TEXT = (
        "Check this out, my dad forwarded it to me:\n\n"
        "Hi there,\n"
        "I wanted to flag that the numbers you sent me were wrong — the "
        "budget total should be $4,200, not $2,400.\n"
        "Best,\n"
        "Jordan\n"
    )

    def test_split_and_labeled_third_party(self):
        item = EvidenceItem(
            doc_id="paste-1", text=self.EMAIL_TEXT, date="2026-08-01",
            collection="conversations", speaker="user",
        )
        labeled = label_evidence([item])
        # Framing line ("Check this out...") stays user-stated; the quoted
        # block gets the third-party label.
        labels = {i.stance_label for i in labeled}
        assert "quoted-correspondence" in labels
        framing_items = [i for i in labeled if i.stance_label != "quoted-correspondence"]
        assert any("Check this out" in i.text for i in framing_items)

    def test_render_shows_third_party_marker(self):
        item = EvidenceItem(
            doc_id="paste-1", text=self.EMAIL_TEXT, date="2026-08-01",
            collection="conversations", speaker="user",
        )
        labeled = label_evidence([item])
        block = render_evidence_block(labeled)
        assert "quoted third-party text inside your message" in block
        assert "you said" in block  # the framing line is still attributed

    def test_plain_user_text_unaffected(self):
        item = EvidenceItem(
            doc_id="plain", text="I think I've been sleeping badly this week.",
            date="2026-08-01", collection="corpus", speaker="user",
        )
        labeled = label_evidence([item])
        assert len(labeled) == 1
        assert labeled[0].stance_label == "user-stated"

    def test_assistant_side_never_split(self):
        item = EvidenceItem(
            doc_id="assistant-1", text=self.EMAIL_TEXT, date="2026-08-01",
            collection="corpus", speaker="assistant",
        )
        labeled = label_evidence([item])
        assert len(labeled) == 1
        assert labeled[0].stance_label == "assistant-inferred"


# ---------------------------------------------------------------------------
# Round 2 (2026-09-04): retrieval-level window-fairness. Live incident: a
# "from 2026-07-15 through today" request's rendered evidence spanned only
# 2026-08-24..09-04 — the 79 ASSEMBLED items had nothing older at all, so
# window-fair RENDERING (round 1) had nothing to render. Root cause (found
# via a read-only harness against the real corpus, see the coordinator
# report): NOT search_keyword's own cap/newest-first (verified: in isolation
# it already returns every in-window match well under its cap) — it was
# `_finalize`'s per-collection proportional quota, which shares the single
# "corpus" collection label between every facet's ordinary keyword scan AND
# the quoted-phrase scan and slices `group[:quota]` newest-first: a broad/
# common facet keyword floods "corpus" with recent hits and silently drops
# the quoted-phrase scan's older, explicitly in-window items even while the
# total item count stays at/under total_evidence_cap the whole time.
# ---------------------------------------------------------------------------


class _EmptyChroma:
    def query_collection(self, coll, q, n):
        return []

    def _get_collection(self, name):
        return None


class TestPhraseScanWindowFairRetrieval:
    """~120-turn corpus across 8 ISO weeks: 6 "no I mean" turns scattered
    across 5 OLDER weeks (all inside the request window), ~110 unrelated
    filler turns, and the newest 3 turns are same-day near-repeats of the
    live request itself (the reported symptom: today's repeat-turns of the
    request survived while 07-20..08-18 corrections vanished)."""

    REQUEST_TEXT = (
        'Evidence sweep request. From 2026-07-15 through today, find every '
        'turn where I corrected something you said — for example when I '
        'said "no I mean" or "I meant" — and show me the full list.'
    )
    TODAY = datetime(2026, 9, 4, 12, 0, 0)
    WINDOW = ("2026-07-15", "2026-09-04")

    @pytest.fixture
    def corpus(self, tmp_path):
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        # 6 targeted "no I mean" turns across 5 distinct OLDER ISO weeks —
        # all still inside the 2026-07-15..09-04 request window.
        target_week_offsets = [7, 6, 5, 4, 3, 3]
        for idx, w in enumerate(target_week_offsets):
            ts = self.TODAY - timedelta(weeks=w, hours=idx)
            cm.add_entry(
                f"No I mean the schedule really did change on this one, item {idx}.",
                "Got it, thanks for the correction.",
                timestamp=ts,
            )
        # ~110 unrelated filler turns spread across the whole range.
        for i in range(110):
            ts = self.TODAY - timedelta(hours=3 * i)
            cm.add_entry(
                f"Just a routine unrelated update about topic {i} today.",
                "Noted.",
                timestamp=ts,
            )
        # Newest 3 turns: same-day near-repeats of the LIVE request.
        for i in range(3):
            ts = self.TODAY - timedelta(minutes=5 * i)
            cm.add_entry(self.REQUEST_TEXT, "Working on it now.", timestamp=ts)
        return cm

    def test_target_weeks_survive_and_repeats_are_excluded(self, corpus):
        plan = FacetPlan(facets=[FacetQuery(name="theme", query_text="corrections", keywords=[])])
        items = asyncio.run(run_sweep(
            plan, chroma_store=_EmptyChroma(), corpus_manager=corpus,
            caps=_caps(keyword_scan_max=50), request_text=self.REQUEST_TEXT,
            date_window=self.WINDOW,
        ))
        qp_items = [i for i in items if i.facet == "quoted-phrase"]
        weeks = {week_bucket_key(i.date) for i in qp_items}
        assert len(weeks) >= 5, f"only {len(weeks)} distinct weeks: {weeks}"
        # None of the three same-day request-repeat turns survive.
        assert not any("Evidence sweep request" in i.text for i in qp_items)
        assert not any("show me the full list" in i.text for i in qp_items)


class TestPerWeekCap:
    """Direct cap test: with more matches available than the cap, each
    represented ISO week keeps at least 2 items (when it has that many) —
    round-robin interleaving before truncation, not a bare newest-first cut."""

    def test_at_least_two_per_week_when_available(self):
        items = []
        for w in range(5):
            for i in range(5):
                items.append(EvidenceItem(
                    doc_id=f"w{w}-{i}", text=f"content {w}-{i}",
                    date=(datetime(2026, 9, 4) - timedelta(weeks=w, days=i)).date().isoformat(),
                    collection="corpus",
                ))
        capped = interleave_evidence_for_coverage(items)[:10]
        counts: dict = {}
        for it in capped:
            key = week_bucket_key(it.date)
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 5, counts
        assert all(c >= 2 for c in counts.values()), counts

    def test_cap_smaller_than_week_count_still_spreads_newest_first(self):
        items = []
        for w in range(8):
            items.append(EvidenceItem(
                doc_id=f"w{w}", text=f"content {w}",
                date=(datetime(2026, 9, 4) - timedelta(weeks=w)).date().isoformat(),
                collection="corpus",
            ))
        capped = interleave_evidence_for_coverage(items)[:4]
        weeks = {week_bucket_key(i.date) for i in capped}
        assert len(weeks) == 4  # one per week, newest 4 weeks


class _FakeCollection:
    def __init__(self, rows):
        self._rows = rows

    def get(self, include=None):
        return {
            "documents": [r["content"] for r in self._rows],
            "metadatas": [r["metadata"] for r in self._rows],
            "ids": [r["id"] for r in self._rows],
        }


class _DateRangeChroma:
    def __init__(self, conversations=(), summaries=()):
        self._conv = _FakeCollection(list(conversations))
        self._summary = _FakeCollection(list(summaries))

    def query_collection(self, coll, q, n):
        return []

    def _get_collection(self, name):
        if name == "conversations":
            return self._conv
        if name == "summaries":
            return self._summary
        return None


class TestDateRangeArm:
    """The chroma date-range arm (reusing window_scan_collection) must
    contribute dated items for weeks with no strong semantic hit."""

    def _dated_rows(self, prefix, n_weeks=7):
        rows = []
        for w in range(n_weeks):
            date = (datetime(2026, 9, 4) - timedelta(weeks=w)).date().isoformat()
            rows.append({
                "id": f"{prefix}-{w}",
                "content": (
                    f"dated content from week offset {w}, long enough to "
                    "pass the hygiene length checks without being junk."
                ),
                "metadata": {"note_date": date},
            })
        return rows

    def test_date_range_arm_covers_weeks_with_no_semantic_hit(self):
        chroma = _DateRangeChroma(conversations=self._dated_rows("conv"))
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q", keywords=[])])
        items = asyncio.run(run_sweep(
            plan, chroma_store=chroma, corpus_manager=None,
            caps=_caps(), request_text="", date_window=("2026-07-15", "2026-09-04"),
        ))
        date_range_items = [i for i in items if i.facet == "date-range"]
        assert date_range_items, "expected the date-range arm to contribute items"
        weeks = {week_bucket_key(i.date) for i in date_range_items}
        assert len(weeks) >= 5, f"only {len(weeks)} distinct weeks: {weeks}"

    def test_no_date_window_means_no_scan(self):
        chroma = _DateRangeChroma(conversations=self._dated_rows("conv"))
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q", keywords=[])])
        items = asyncio.run(run_sweep(
            plan, chroma_store=chroma, corpus_manager=None,
            caps=_caps(), request_text="", date_window=None,
        ))
        assert not any(i.facet == "date-range" for i in items)

    def test_window_scan_collection_shared_with_handlers(self):
        # gui.handlers._window_scan_collection is a thin delegating alias —
        # both call sites must produce the same result for the same inputs.
        import gui.handlers as handlers
        chroma = _DateRangeChroma(conversations=self._dated_rows("conv", n_weeks=2))
        direct = window_scan_collection(chroma, "conversations", ("2026-07-15", "2026-09-04"), 10)
        via_handlers = handlers._window_scan_collection(
            chroma, "conversations", ("2026-07-15", "2026-09-04"), 10)
        assert direct == via_handlers


class TestSameDayTightenedExclusion:
    """Fix #3: same-day items with >= 30% shingle overlap are dropped (the
    any-day bar stays 60%) — the previous-day-repeat-turn class from the
    live incident often lands in that 30-60% band."""

    REQUEST = (
        "From 2026-07-15 through today, find every correction I made about "
        "the budget numbers in the quarterly finance report we discussed."
    )
    # Calibrated to ~33% 8-word-shingle overlap with REQUEST — below the
    # any-day 60% bar, above the tightened same-day 30% bar.
    MODERATE_OVERLAP_TEXT = (
        "Wait can you find every correction I made about the budget "
        "numbers please."
    )

    def test_same_day_moderate_overlap_dropped(self):
        item = EvidenceItem(
            doc_id="x", collection="conversations", speaker="user",
            date="2026-09-04T10:59:00", text=self.MODERATE_OVERLAP_TEXT,
        )
        out = exclude_current_request_evidence(
            [item], self.REQUEST, current_turn_date="2026-09-04T12:00:00",
        )
        assert out == []

    def test_any_day_moderate_overlap_kept(self):
        # No current_turn_date supplied: only the any-day 60% bar applies.
        item = EvidenceItem(
            doc_id="x", collection="conversations", speaker="user",
            date="2026-09-04T10:59:00", text=self.MODERATE_OVERLAP_TEXT,
        )
        out = exclude_current_request_evidence([item], self.REQUEST)
        assert out == [item]

    def test_different_day_moderate_overlap_kept(self):
        item = EvidenceItem(
            doc_id="x", collection="conversations", speaker="user",
            date="2026-08-01T10:59:00", text=self.MODERATE_OVERLAP_TEXT,
        )
        out = exclude_current_request_evidence(
            [item], self.REQUEST, current_turn_date="2026-09-04T12:00:00",
        )
        assert out == [item]


class TestFinalizeWindowFairPerCollection:
    """Root-cause regression lock: _finalize's per-collection quota must
    select window-fairly (not blind newest-first) when a date_window is
    active, or one recency-skewed facet's corpus hits crowd the quoted-
    phrase scan's older hits out of the SAME "corpus" collection bucket."""

    def _corpus_items(self, n_per_week, n_weeks, prefix):
        items = []
        for w in range(n_weeks):
            for i in range(n_per_week):
                items.append(EvidenceItem(
                    doc_id=f"{prefix}-{w}-{i}", text=f"{prefix} content {w}-{i}",
                    date=(datetime(2026, 9, 4) - timedelta(weeks=w, hours=i)).isoformat(),
                    collection="corpus", facet=prefix,
                ))
        return items

    def test_older_weeks_survive_the_per_collection_quota_with_date_window(self):
        # A "you"-style broad facet dumps 60 recent-only corpus hits; the
        # quoted-phrase scan contributes 8 older, week-spread hits. Total
        # (68) exceeds total_evidence_cap, so the proportional quota engages.
        recent_noise = self._corpus_items(n_per_week=60, n_weeks=1, prefix="you-said")
        older_signal = self._corpus_items(n_per_week=1, n_weeks=8, prefix="quoted-phrase")
        for it in older_signal:
            it.facet = "quoted-phrase"
        out = _finalize(
            recent_noise + older_signal,
            _caps(total_evidence_cap=40),
            date_window=("2026-07-01", "2026-09-04"),
        )
        qp_weeks = {week_bucket_key(i.date) for i in out if i.facet == "quoted-phrase"}
        assert len(qp_weeks) >= 5, f"only {len(qp_weeks)} distinct weeks survived: {qp_weeks}"

    def test_no_date_window_keeps_legacy_newest_first_behavior(self):
        recent_noise = self._corpus_items(n_per_week=60, n_weeks=1, prefix="you-said")
        older_signal = self._corpus_items(n_per_week=1, n_weeks=8, prefix="quoted-phrase")
        for it in older_signal:
            it.facet = "quoted-phrase"
        out = _finalize(recent_noise + older_signal, _caps(total_evidence_cap=40), date_window=None)
        # Legacy behavior (no explicit window): pure newest-first per
        # collection — the older signal is NOT guaranteed to survive.
        qp_weeks = {week_bucket_key(i.date) for i in out if i.facet == "quoted-phrase"}
        assert len(qp_weeks) <= 2


# ---------------------------------------------------------------------------
# Round 3, fix A (2026-09-04): prior-sweep-output feedback-loop guard. A live
# run cited [E1] — the ASSISTANT'S REPLY from a PREVIOUS insight-mode sweep
# ("...Current memory: reflects the corrected version...") — as fresh
# evidence and minted it into a NEW "correction #5"; a misattribution from
# two runs ago was propagating run to run.
# ---------------------------------------------------------------------------

class TestPriorSweepOutputExclusion:
    def test_insight_assembly_reply_excluded_ordinary_reply_kept(self, tmp_path):
        """The facet-level corpus scan checks BOTH sides of an entry — this
        is the exact mechanism that let a PRIOR sweep's own assistant reply
        (not just its request) leak back in as evidence."""
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        now = datetime(2026, 9, 4, 12, 0, 0)
        cm.add_entry(
            "gather every correction I've made",
            "Current memory: reflects the corrected version, no I mean the "
            "fixed close time is gone.",
            timestamp=now - timedelta(days=3),
            response_mode="insight-assembly",
        )
        cm.add_entry(
            "No I mean the meeting moved to Friday, not Thursday.",
            "Got it, thanks for clarifying.",
            timestamp=now - timedelta(days=2),
            response_mode="enhanced",
        )
        plan = FacetPlan(facets=[
            FacetQuery(name="theme", query_text="corrections", keywords=["mean"]),
        ])
        items = asyncio.run(run_sweep(
            plan, chroma_store=_EmptyChroma(), corpus_manager=cm, caps=_caps(),
        ))
        texts = [i.text for i in items if i.collection == "corpus"]
        assert not any("Current memory" in t for t in texts)
        assert any("meeting moved to Friday" in t for t in texts)

    def test_quoted_phrase_scan_drops_prior_sweep_request(self, tmp_path):
        cm = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
        now = datetime(2026, 9, 4, 12, 0, 0)
        cm.add_entry(
            'gather everything where I said "no I mean" before',
            "Working on it.",
            timestamp=now - timedelta(days=3),
            response_mode="insight-assembly",
        )
        cm.add_entry(
            "No I mean the report is due Friday, not Monday.",
            "Noted.",
            timestamp=now - timedelta(days=2),
            response_mode="enhanced",
        )
        request_text = 'Find every time I said "no I mean" in our chats.'
        plan = FacetPlan(facets=[FacetQuery(name="theme", query_text="q", keywords=[])])
        items = asyncio.run(run_sweep(
            plan, chroma_store=_EmptyChroma(), corpus_manager=cm, caps=_caps(),
            request_text=request_text,
        ))
        qp_texts = [i.text for i in items if i.facet == "quoted-phrase"]
        assert not any("gather everything" in t for t in qp_texts)
        assert any("report is due Friday" in t for t in qp_texts)

    def test_chroma_conversations_response_mode_dropped(self):
        store = MagicMock()

        def _q(coll, q, n):
            if coll != "conversations":
                return []
            return [
                {"id": "prior-sweep", "content": "no I mean the corrected version stands",
                 "metadata": {"timestamp": "2026-09-01", "response_mode": "doc-generation"}},
                {"id": "ordinary", "content": "no I mean we should meet Tuesday instead",
                 "metadata": {"timestamp": "2026-09-01"}},
            ]

        store.query_collection.side_effect = _q
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=store, corpus_manager=None, caps=_caps(),
        ))
        texts = [i.text for i in items if i.collection == "conversations"]
        assert not any("corrected version stands" in t for t in texts)
        assert any("meet Tuesday instead" in t for t in texts)

    def test_scaffolding_marker_backstop_for_legacy_entries(self):
        store = MagicMock()

        def _q(coll, q, n):
            if coll != "conversations":
                return []
            return [
                {"id": "legacy-sweep", "content": "Denominator caveat: this record over-samples hard days [E3]",
                 "metadata": {"timestamp": "2026-09-01"}},
                {"id": "ordinary", "content": "no I mean we should meet Tuesday instead",
                 "metadata": {"timestamp": "2026-09-01"}},
            ]

        store.query_collection.side_effect = _q
        plan = FacetPlan(facets=[FacetQuery(name="f", query_text="q")])
        items = asyncio.run(run_sweep(
            plan, chroma_store=store, corpus_manager=None, caps=_caps(),
        ))
        texts = [i.text for i in items if i.collection == "conversations"]
        assert not any("Denominator caveat" in t for t in texts)
        assert any("meet Tuesday instead" in t for t in texts)
