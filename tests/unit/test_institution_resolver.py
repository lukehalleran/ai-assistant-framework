"""Institution injection for web search queries (2026-08-27).

Incident (drop-date thread, turn 3): "confirm the drop date" produced the
search terms "college drop date August 2026" / "school withdrawal deadline
August 2026". The trigger LLM first attached the user's CITY ("Springfield,
Illinois college drop date" — the wrong-college class, correctly stripped by
strip_unjustified_location) and then had nothing to name the school with,
even though the profile knows school=Georgia Tech at confidence 1.0. The
generic queries burned Tavily credits on generic pages.

utils/institution_resolver.py mirrors location_resolver: env override →
profile facts (school-family relations, is_current, institution-shaped
values) → quick_profile. apply_institution() is the deterministic backstop
behind the LLM prompt guidance, scoped by the wrong-college doctrine
inverted: academic-logistics queries only, never over a DIFFERENT named
school, under-fires by design.
"""
import json
import os

import pytest

from utils.institution_resolver import (
    InstitutionResolver,
    _INSTITUTION_VALUE_RE,
    apply_institution,
    query_is_academic_logistics,
)


LIVE_QUERY = "Can we do a web search and attempt to confirm the specific drop date"
LIVE_TERMS = ["college drop date August 2026", "school withdrawal deadline August 2026"]


def _write_profile(tmp_path, profile):
    p = tmp_path / "user_profile.json"
    p.write_text(json.dumps(profile), encoding="utf-8")
    return str(p)


def _fact(relation, value, is_current=True, confidence=1.0):
    return {
        "relation": relation, "value": value,
        "is_current": is_current, "confidence": confidence,
    }


# ===========================================================================
# Value shape validation
# ===========================================================================

class TestInstitutionValueShape:

    @pytest.mark.parametrize("value", [
        "Georgia Tech",
        "MIT",
        "University of Wisconsin-Madison",
        "Georgia Institute of Technology",
        "St. Olaf College",
    ])
    def test_institution_shaped_values_accepted(self, value):
        assert _INSTITUTION_VALUE_RE.match(value)

    @pytest.mark.parametrize("value", [
        "in third best grad program in nation",   # real profile junk shape
        "get into school stuff",
        "my school",
        "",
        "a really long sentence about the school that goes on and on forever",
    ])
    def test_sentence_shaped_junk_rejected(self, value):
        assert not _INSTITUTION_VALUE_RE.match(value)


# ===========================================================================
# Profile extraction
# ===========================================================================

class TestProfileExtraction:

    def test_school_relation_extracted(self, tmp_path):
        path = _write_profile(tmp_path, {
            "categories": {"education": [_fact("school", "Georgia Tech")]},
        })
        assert InstitutionResolver(path).get_institution() == "Georgia Tech"

    def test_school_beats_stale_university_fact(self, tmp_path):
        """The live profile carries a past school under `university`
        (is_current=True, conf 0.85) beside school=Georgia Tech (conf 1.0) —
        relation rank must win, not insertion order."""
        path = _write_profile(tmp_path, {
            "categories": {"education": [
                _fact("university", "University of Wisconsin-Madison", confidence=0.85),
                _fact("school", "Georgia Tech", confidence=1.0),
            ]},
        })
        assert InstitutionResolver(path).get_institution() == "Georgia Tech"

    def test_non_current_facts_ignored(self, tmp_path):
        path = _write_profile(tmp_path, {
            "categories": {"education": [
                _fact("school", "Old School Academy", is_current=False),
            ]},
        })
        assert InstitutionResolver(path).get_institution() is None

    def test_junk_shaped_values_skipped(self, tmp_path):
        path = _write_profile(tmp_path, {
            "categories": {"education": [
                _fact("school", "in third best grad program in nation"),
            ]},
        })
        assert InstitutionResolver(path).get_institution() is None

    def test_quick_profile_fallback(self, tmp_path):
        path = _write_profile(tmp_path, {
            "categories": {},
            "quick_profile": {"school": "Georgia Tech"},
        })
        assert InstitutionResolver(path).get_institution() == "Georgia Tech"

    def test_missing_profile_returns_none(self, tmp_path):
        r = InstitutionResolver(str(tmp_path / "nope.json"))
        assert r.get_institution() is None

    def test_corrupt_profile_returns_none(self, tmp_path):
        p = tmp_path / "user_profile.json"
        p.write_text("{not json", encoding="utf-8")
        assert InstitutionResolver(str(p)).get_institution() is None

    def test_mtime_cache_refreshes_on_change(self, tmp_path):
        path = _write_profile(tmp_path, {
            "categories": {"education": [_fact("school", "Georgia Tech")]},
        })
        r = InstitutionResolver(path)
        assert r.get_institution() == "Georgia Tech"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"categories": {"education": [_fact("school", "New College")]}}, f)
        os.utime(path, (1e9, 1e9))  # force a distinct mtime
        assert r.get_institution() == "New College"

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "utils.institution_resolver.INSTITUTION_OVERRIDE", "Override U"
        )
        path = _write_profile(tmp_path, {
            "categories": {"education": [_fact("school", "Georgia Tech")]},
        })
        assert InstitutionResolver(path).get_institution() == "Override U"

    def test_disabled_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "utils.institution_resolver.INSTITUTION_ENABLED", False
        )
        path = _write_profile(tmp_path, {
            "categories": {"education": [_fact("school", "Georgia Tech")]},
        })
        assert InstitutionResolver(path).get_institution() is None


# ===========================================================================
# Academic-logistics query detection
# ===========================================================================

class TestAcademicCue:

    @pytest.mark.parametrize("query", [
        LIVE_QUERY,
        "when is the withdrawal deadline",
        "how do I contact the registrar",
        "academic calendar fall 2026",
        "tuition refund policy",
        "when does registration open",
    ])
    def test_logistics_queries_detected(self, query):
        assert query_is_academic_logistics(query)

    @pytest.mark.parametrize("query", [
        "how does an SVM work",          # coursework, not logistics
        "what's the weather in Atlanta",
        "best pizza near me",
        "",
    ])
    def test_non_logistics_queries_rejected(self, query):
        assert not query_is_academic_logistics(query)


# ===========================================================================
# apply_institution backstop
# ===========================================================================

class TestApplyInstitution:

    def test_live_turn_reproduction(self):
        """The exact terms from the 2026-08-27 turn gain the user's school."""
        out = apply_institution(LIVE_TERMS, LIVE_QUERY, "Georgia Tech")
        assert out == [
            "Georgia Tech drop date August 2026",
            "Georgia Tech withdrawal deadline August 2026",
        ]

    def test_academic_term_without_generic_word_prepended(self):
        out = apply_institution(
            ["withdrawal deadline fall 2026"], LIVE_QUERY, "Georgia Tech"
        )
        assert out == ["Georgia Tech withdrawal deadline fall 2026"]

    def test_non_academic_query_untouched(self):
        terms = ["college football scores"]
        assert apply_institution(terms, "who won the game", "Georgia Tech") == terms

    def test_different_named_school_untouched(self):
        """'When is Harvard University's drop deadline' must stay Harvard's —
        injecting the user's school would misdirect the search (the inverted
        wrong-college incident)."""
        terms = ["Harvard University drop deadline 2026"]
        out = apply_institution(
            terms, "when is Harvard University's drop deadline", "Georgia Tech"
        )
        assert out == terms

    def test_users_own_named_school_still_applies(self):
        out = apply_institution(
            ["school drop date 2026"],
            "when is Georgia Tech University drop date",  # names the USER's school
            "Georgia Tech",
        )
        assert out == ["Georgia Tech drop date 2026"]

    def test_term_already_naming_institution_untouched(self):
        terms = ["Georgia Tech drop date August 2026"]
        assert apply_institution(terms, LIVE_QUERY, "Georgia Tech") == terms

    def test_mixed_terms_only_academic_touched(self):
        """A weather sub-query in a mixed request stays untouched."""
        out = apply_institution(
            ["college drop date 2026", "weather in Atlanta today"],
            "what's the drop date and the weather",
            "Georgia Tech",
        )
        assert out == ["Georgia Tech drop date 2026", "weather in Atlanta today"]

    def test_no_institution_is_noop(self):
        assert apply_institution(LIVE_TERMS, LIVE_QUERY, None) == LIVE_TERMS
        assert apply_institution(LIVE_TERMS, LIVE_QUERY, "  ") == LIVE_TERMS

    def test_empty_terms_is_noop(self):
        assert apply_institution([], LIVE_QUERY, "Georgia Tech") == []


# ===========================================================================
# Trigger-prompt wiring
# ===========================================================================

class TestTriggerPromptWiring:

    def _prompt(self, institution=None):
        from utils.web_search_trigger import _build_llm_trigger_prompt
        return _build_llm_trigger_prompt(
            query=LIVE_QUERY,
            current_date="2026-08-27",
            user_institution=institution,
        )

    def test_prompt_names_school_when_known(self):
        prompt = self._prompt("Georgia Tech")
        assert "User's school: Georgia Tech" in prompt
        assert "SCHOOL-LOGISTICS QUERIES" in prompt
        # The scope guards ride along in the guideline.
        assert "DIFFERENT school" in prompt

    def test_prompt_clean_without_institution(self):
        prompt = self._prompt(None)
        assert "User's school" not in prompt
        assert "SCHOOL-LOGISTICS" not in prompt
