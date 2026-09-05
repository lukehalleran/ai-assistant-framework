"""
2026-09-05 heavy-history first-person guard.

Live defect (daemon_debug.log, 13:10): turn 1 was a public-news question
("Were any UK or US politicians charged with crimes this week? What did the
police or courts announce?") whose own tone was CONVERSATIONAL, but the LLM
heavy-topic classifier flagged it `is_heavy_topic=True`. Turn 2 was a
77-word logistics request about reading an Outlook email for deadlines;
harm score and semantic scores both landed CONVERSATIONAL, but
`_recent_distress_from_history` counted turn 1's heavy flag and the sticky
floor forced CONCERN — a LIGHT SUPPORT reply ("don't offer unsolicited
advice") on an explicit task request. Doctrine (same as the 2026-08-15
vent-shape narrowing of the agentic veto): heavy SUBJECT MATTER about the
outside world is not distress evidence; only the user's own first-person
material is. A fresh heavy history row now counts toward session distress
only when its own text carries a first-person marker (rows with no text
field at all still count, fail-closed, for legacy rows).
"""

from datetime import datetime, timedelta

import pytest

from utils.tone_detector import (
    CrisisLevel,
    _recent_distress_from_history,
    _session_in_distress,
    detect_crisis_level,
)

PUBLIC_NEWS_TEXT = (
    "Were any UK or US politicians charged with crimes this week? "
    "What did the police or courts announce?"
)
OUTLOOK_QUERY = (
    "Can you read the last email I received from them in outlook and tell "
    "me what deadlines it mentioned?"
)


def _fresh_ts() -> str:
    return datetime.now().isoformat()


def _stale_ts() -> str:
    return (datetime.now() - timedelta(hours=6)).isoformat()


class TestPublicNewsHeavyRowIsNotDistress:
    def test_public_news_row_not_distress_evidence(self):
        row = {
            "query": PUBLIC_NEWS_TEXT,
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        assert _recent_distress_from_history([row]) is False
        assert _session_in_distress(CrisisLevel.CONVERSATIONAL, [row]) is False


class TestFirstPersonHeavyRowIsDistress:
    def test_first_person_row_is_distress_evidence(self):
        row = {
            "query": "I stopped the meds and the pain is back",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        assert _recent_distress_from_history([row]) is True
        assert _session_in_distress(CrisisLevel.CONVERSATIONAL, [row]) is True


class TestLegacyRowWithNoTextFailsClosed:
    def test_no_text_fields_counts_as_before(self):
        row = {"timestamp": _fresh_ts(), "is_heavy_topic": True}
        assert _recent_distress_from_history([row]) is True


class TestStaleFirstPersonRowStillGated:
    def test_first_person_but_stale_is_not_distress(self):
        row = {
            "query": "I stopped the meds and the pain is back",
            "timestamp": _stale_ts(),
            "is_heavy_topic": True,
        }
        assert _recent_distress_from_history([row]) is False


class TestMixedListOrderAgnostic:
    def _rows(self):
        observational = {
            "query": "the government announced sanctions today",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        first_person = {
            "query": "I stopped the meds and the pain is back",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        return observational, first_person

    def test_forward_order(self):
        observational, first_person = self._rows()
        assert _recent_distress_from_history([observational, first_person]) is True

    def test_reversed_order(self):
        observational, first_person = self._rows()
        assert _recent_distress_from_history([first_person, observational]) is True


class TestPossessiveCounts:
    def test_possessive_marker_is_distress_evidence(self):
        row = {
            "query": "my dad's surgery is tomorrow and the doctors sound worried",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        assert _recent_distress_from_history([row]) is True


class TestWordBoundary:
    @pytest.mark.parametrize("text", [
        "Imagine the minister resigned over the war crimes report",
        "Did they charge him with war crimes?",
    ])
    def test_substring_hits_do_not_match(self, text):
        row = {"query": text, "timestamp": _fresh_ts(), "is_heavy_topic": True}
        assert _recent_distress_from_history([row]) is False


class TestFallbackKeys:
    def test_user_key_fallback_first_person(self):
        row = {
            "user": "I can't sleep since the raid",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        assert _recent_distress_from_history([row]) is True

    def test_content_key_fallback_observational(self):
        row = {
            "content": "the government announced sanctions",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        assert _recent_distress_from_history([row]) is False


# The real semantic stage always returns all four level scores; the borderline
# stage downstream takes their max. These are the live 2026-09-05 13:10 scores.
_LIVE_SCORES = {"high": 0.046, "medium": 0.084, "concern": 0.243, "conversational": 0.069}


class TestEndToEndDetectCrisisLevel:
    @pytest.mark.asyncio
    async def test_public_news_history_does_not_latch_floor(self, monkeypatch):
        import utils.tone_detector as td

        def fake_semantic(message, conversation_history=None, model_manager=None,
                           force_escalation=False):
            return (CrisisLevel.CONVERSATIONAL, 0.9, _LIVE_SCORES)

        monkeypatch.setattr(td, "_semantic_crisis_detection", fake_semantic)

        row = {
            "query": PUBLIC_NEWS_TEXT,
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        result = await detect_crisis_level(
            OUTLOOK_QUERY,
            conversation_history=[row],
            model_manager=None,
            previous_tone=CrisisLevel.CONVERSATIONAL,
        )
        assert result.level == CrisisLevel.CONVERSATIONAL
        assert result.trigger != "distress_sticky_floor"

    @pytest.mark.asyncio
    async def test_control_first_person_history_still_floors(self, monkeypatch):
        import utils.tone_detector as td

        def fake_semantic(message, conversation_history=None, model_manager=None,
                           force_escalation=False):
            return (CrisisLevel.CONVERSATIONAL, 0.9, _LIVE_SCORES)

        monkeypatch.setattr(td, "_semantic_crisis_detection", fake_semantic)

        row = {
            "query": "I stopped the meds and the pain is back",
            "timestamp": _fresh_ts(),
            "is_heavy_topic": True,
        }
        result = await detect_crisis_level(
            OUTLOOK_QUERY,
            conversation_history=[row],
            model_manager=None,
            previous_tone=CrisisLevel.CONVERSATIONAL,
        )
        assert result.trigger == "distress_sticky_floor"
        assert result.level == CrisisLevel.CONCERN
