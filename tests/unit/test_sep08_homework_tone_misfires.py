"""2026-09-08 homework-session tone/retrieval misfires (follow-up batch B7).

A day-long R/statistics homework session (a live-probe retest after commit
21a914e) tripped six independent bare-substring / over-eager-teaching bugs:

R1. `utils.query_checker._is_heavy_topic_heuristic` counted HEAVY_KEYWORDS
    hits with bare substring — "ice" ⊂ "Price" (the homework's response
    variable) flipped is_heavy_topic True on pasted R script turns.
R2. `utils.tone_detector._recent_distress_from_history` trusted a fresh
    row's stored `is_heavy_topic=True` (poisoned by R1) plus a first-person
    marker as distress evidence, floor-latching CONCERN/LIGHT SUPPORT onto
    11 homework turns.
R3. `utils.tone_detector.calculate_harm_score` (`_calculate_harm_score`)
    also scanned HIGH/MEDIUM/CONCERN/EVENT lists via bare substring; the
    MEDIUM phrase "need to use" matched inside "...id just need to use CDF
    to find vals..." and scored CONCERN via harm score.
R4. `utils.need_detector.detect_need_type` taught a "need" exemplar on
    every high-confidence keyword fast-path hit regardless of shape,
    learning 13 "perspective" exemplars from homework/paste text, one
    carrying an injected `[ACTIVE DOCUMENT — ...]` passage verbatim.
R5. `core.response_parser.ResponseParser._SINGLE_LETTER_ABBREV_RE` exempted
    ANY single letter before ".e", so "it's all base R.e" (R the stats
    language, not an abbreviation) kept its kimi-3 trailing-artifact 'e'.
R6. `core.prompt.gatherer_knowledge.get_user_uploads` admitted three
    identical lecture-transcript chunks persisted under three different
    temp-file titles (pre-2026-09-04, before attachment_display_name()).

Each fix narrows/corrects the deployed function directly; tests drive those
functions, never a re-derivation. Synthetic names only.
"""

from datetime import datetime, timedelta

import pytest

import utils.query_checker as qc
import utils.tone_detector as td
import utils.need_detector as nd
from core.response_parser import ResponseParser
import core.prompt.gatherer_knowledge as gk


PASTED_R_SCRIPT = (
    "yeah its not working\n"
    "#read csv data file\n"
    'used_car_data <- read.csv("UsedCars.csv")\n'
    "model <- lm(Price ~ ., data = used_car_data)"
)

AFTERNOON_SENTENCE = (
    "yeah just a hair over. ok this i can probably do 5. Calculate the "
    "p-value for each beta using the defining formula p = 2*Pr(t < "
    "-|tstat|). (Hint: use pt() function for the cdf of t distribution.) "
    "i have the ts in two places now, and then id just need to use CDF "
    "to find vals of prob up to abs(that b"
)


# =============================================================================
# R1 — heavy_keyword_hits / _is_heavy_topic_heuristic word boundary
# =============================================================================

class TestHeavyKeywordWordBoundary:
    @pytest.mark.parametrize("text", [
        "Price", "prices", "office", "notice", "device", "a nice day",
    ])
    def test_no_ice_substring_hit(self, text):
        assert qc.heavy_keyword_hits(text) == []

    def test_ice_raids_still_hits(self):
        hits = qc.heavy_keyword_hits("ICE raids downtown")
        assert "ice" in hits
        assert any(h in ("raid", "raids") for h in hits)

    def test_pasted_r_script_not_heavy(self):
        assert qc._is_heavy_topic_heuristic(PASTED_R_SCRIPT) is False

    def test_protest_riot_still_heavy(self):
        # Two genuine hits ("protest", "riot") — the >=2-hit branch.
        assert qc._is_heavy_topic_heuristic("protest turned into a riot") is True

    def test_discriminat_stem_still_matches(self):
        # Stem matching (left boundary only) must survive the word-bound fix.
        assert "discriminat" in qc.heavy_keyword_hits(
            "there was discrimination and also a protest"
        )

    def test_spills_does_not_hit_pills(self):
        assert "pills" not in qc.heavy_keyword_hits("the spills all over the floor")
        assert "pills" in qc.heavy_keyword_hits("I need my pills")


# =============================================================================
# R2 — tone_detector history re-check (heavy_keyword_hits + first-person on
# the SAME code-stripped text; stored is_heavy_topic no longer sufficient)
# =============================================================================

class TestHistoryDistressRecheck:
    def _fresh_ts(self):
        return datetime.now().isoformat()

    def _stale_ts(self):
        return (datetime.now() - timedelta(minutes=45)).isoformat()

    def test_pasted_script_row_not_distress(self):
        row = {
            "query": PASTED_R_SCRIPT,
            "is_heavy_topic": True,
            "timestamp": self._fresh_ts(),
        }
        assert td._recent_distress_from_history([row]) is False

    def test_real_distress_row_still_counts(self):
        row = {
            "query": "I can't cope with any of this anymore",
            "is_heavy_topic": True,
            "timestamp": self._fresh_ts(),
        }
        assert td._recent_distress_from_history([row]) is True

    def test_real_distress_row_stale_does_not_count(self):
        row = {
            "query": "I can't cope with any of this anymore",
            "is_heavy_topic": True,
            "timestamp": self._stale_ts(),
        }
        assert td._recent_distress_from_history([row]) is False

    def test_textless_row_keeps_fail_closed_behavior(self):
        row = {"is_heavy_topic": True, "timestamp": self._fresh_ts()}
        assert td._recent_distress_from_history([row]) is True

    def test_heavy_keyword_hit_without_first_person_still_excluded(self):
        # "police" is a real HEAVY_KEYWORDS word-bounded hit, but no
        # first-person marker — must still fail (both checks required).
        row = {
            "query": "Were any UK or US politicians charged with crimes "
                     "this week? What did the police or courts announce?",
            "is_heavy_topic": True,
            "timestamp": self._fresh_ts(),
        }
        assert td._recent_distress_from_history([row]) is False


# =============================================================================
# R3 — harm score: word-bounded matchers + _SUBSTANCE_USE_RE clause context
# =============================================================================

class TestHarmScoreSubstanceUseClauseContext:
    def test_afternoon_sentence_scores_zero(self):
        score, matched, _ = td._calculate_harm_score(AFTERNOON_SENTENCE)
        assert score == 0
        assert not any("MEDIUM" in m for m in matched)

    def test_need_to_use_mid_clause_does_not_score(self):
        score, matched, _ = td._calculate_harm_score(
            "need to use the pt() function"
        )
        assert score == 0
        assert matched == []

    def test_really_need_to_use_again_scores_medium(self):
        score, matched, cats = td._calculate_harm_score(
            "I really need to use again"
        )
        assert score >= 5
        assert cats["medium"] >= 1
        assert any("need/want to use" in m for m in matched)

    def test_want_to_use_period_scores(self):
        score, matched, cats = td._calculate_harm_score("i want to use.")
        assert score >= 5
        assert cats["medium"] >= 1

    @pytest.mark.asyncio
    async def test_end_to_end_afternoon_sentence_is_conversational(self, monkeypatch):
        def fake_semantic(message, conversation_history=None, model_manager=None,
                           force_escalation=False):
            return (
                td.CrisisLevel.CONVERSATIONAL, 0.9,
                {"high": 0.05, "medium": 0.08, "concern": 0.1, "conversational": 0.7},
            )
        monkeypatch.setattr(td, "_semantic_crisis_detection", fake_semantic)

        result = await td.detect_crisis_level(
            AFTERNOON_SENTENCE, conversation_history=None,
            model_manager=None, previous_tone=None,
        )
        assert result.level == td.CrisisLevel.CONVERSATIONAL


class TestHarmScoreWordBoundary:
    def test_shutdown_does_not_hit_down(self):
        assert "down" in td.CONCERN_KEYWORDS
        score, matched, _ = td._calculate_harm_score(
            "The system shutdown happened at midnight"
        )
        assert not any("down" == m.split(": ", 1)[-1] for m in matched)

    def test_uncontrolling_does_not_hit_controlling(self):
        assert "controlling" in td.MEDIUM_CRISIS_KEYWORDS
        score, matched, _ = td._calculate_harm_score("uncontrolling remote")
        assert not any("controlling" == m.split(": ", 1)[-1] for m in matched)

    def test_controlling_word_still_hits(self):
        _, matched, cats = td._calculate_harm_score("he is so controlling")
        assert cats["medium"] >= 1


# =============================================================================
# R4 — need_detector teaching skip on paste/technical shapes
# =============================================================================

class TestNeedTeacherSkip:
    def test_skips_code_shaped_message(self):
        assert nd._skip_need_teaching(
            'used_car_data <- read.csv("UsedCars.csv")'
        ) is True

    def test_skips_active_document_marker(self):
        assert nd._skip_need_teaching(
            "ok next q please [ACTIVE DOCUMENT — Homework1-1.pdf, Task 2]"
        ) is True

    def test_skips_attachment_note_marker(self):
        assert nd._skip_need_teaching(
            "here you go [ATTACHMENT NOTE: Housing.csv not attached]"
        ) is True

    def test_skips_deadline_note_marker(self):
        assert nd._skip_need_teaching(
            "due soon [DEADLINE NOTE] 11:59 PM Eastern = 10:59 PM Central"
        ) is True

    def test_skips_long_message(self):
        long_msg = "I need help with this " + " ".join(["word"] * 60)
        assert nd._skip_need_teaching(long_msg) is True

    def test_short_first_person_perspective_still_skips_nothing(self):
        assert nd._skip_need_teaching(
            "I just need someone to listen, I don't want advice"
        ) is False

    def test_fast_path_teaches_for_short_message(self, monkeypatch):
        from unittest.mock import patch
        strong = nd.NeedAnalysis(
            need_type=nd.NeedType.PRESENCE, confidence=0.9, trigger="keyword",
            raw_scores={}, explanation="",
        )
        calls = []

        class _FakeStore:
            def record(self, *a, **kw):
                calls.append((a, kw))

        with patch.object(nd, "_keyword_need_detection", return_value=strong), \
             patch("utils.adaptive_exemplars.get_store", return_value=_FakeStore()):
            nd.detect_need_type("I really just need you here with me tonight")
        assert len(calls) == 1

    def test_fast_path_skips_teaching_for_code_shaped_message(self, monkeypatch):
        from unittest.mock import patch
        strong = nd.NeedAnalysis(
            need_type=nd.NeedType.PERSPECTIVE, confidence=0.9, trigger="keyword",
            raw_scores={}, explanation="",
        )
        calls = []

        class _FakeStore:
            def record(self, *a, **kw):
                calls.append((a, kw))

        code_msg = (
            "#####MGT HW 1 PT 1 #QUESTION 1 #read csv data file into data "
            'frame used_car_data <- read.csv("UsedCars.csv")'
        )
        with patch.object(nd, "_keyword_need_detection", return_value=strong), \
             patch("utils.adaptive_exemplars.get_store", return_value=_FakeStore()):
            nd.detect_need_type(code_msg)
        assert calls == []


# =============================================================================
# R5 — response_parser abbreviation guard narrowed to "i.e"
# =============================================================================

class TestAbbreviationGuard:
    def test_base_r_artifact_stripped(self):
        assert ResponseParser.strip_trailing_stream_artifact(
            "it's all base R.e"
        ) == "it's all base R."

    def test_ie_lowercase_preserved(self):
        text = "see the docs i.e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_ie_uppercase_preserved(self):
        assert ResponseParser.strip_trailing_stream_artifact("I.e") == "I.e"

    def test_existing_ie_regression_case(self):
        text = "the first smoothing parameter, i.e"
        assert ResponseParser.strip_trailing_stream_artifact(text) == text

    def test_other_single_letters_now_strip(self):
        # A single letter OTHER than "i" is no longer exempt.
        assert ResponseParser.strip_trailing_stream_artifact(
            "check appendix A.e"
        ) == "check appendix A."


# =============================================================================
# R6 — upload content dedupe
# =============================================================================

class TestUploadContentDedupe:
    def _doc(self, title, content, relevance=0.5):
        return {
            "content": content,
            "metadata": {"title": title, "timestamp": "2026-08-01T00:00:00"},
            "relevance_score": relevance,
        }

    def test_identical_content_three_titles_keeps_one(self):
        content = "Lecture transcript about linear regression. " * 5
        docs = [
            self._doc("upload:tmps4dvg5t1.txt", content, 0.9),
            self._doc("upload:tmpabc123.txt", content, 0.8),
            self._doc("upload:tmpxyz999.txt", content, 0.7),
        ]
        out = gk._dedupe_upload_content(docs)
        assert len(out) == 1
        assert out[0]["metadata"]["title"] == "upload:tmps4dvg5t1.txt"

    def test_order_preserved_with_distinct_content(self):
        docs = [
            self._doc("upload:a.txt", "content A"),
            self._doc("upload:b.txt", "content B"),
            self._doc("upload:c.txt", "content A"),  # dup of first
        ]
        out = gk._dedupe_upload_content(docs)
        assert [d["metadata"]["title"] for d in out] == ["upload:a.txt", "upload:b.txt"]

    def test_empty_content_never_collides(self):
        docs = [
            self._doc("upload:roster", ""),
            self._doc("upload:also-empty", ""),
        ]
        out = gk._dedupe_upload_content(docs)
        assert len(out) == 2

    def test_whitespace_normalization_still_dedupes(self):
        docs = [
            self._doc("upload:a.txt", "line one\nline two   line three"),
            self._doc("upload:b.txt", "line one line two line three"),
        ]
        out = gk._dedupe_upload_content(docs)
        assert len(out) == 1
