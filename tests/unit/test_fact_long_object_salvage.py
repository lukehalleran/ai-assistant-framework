"""Tests for the long-object cap + salutation salvage (2026-08-26).

Incident: a pasted email became a fact OBJECT wholesale —
`user | email_sent | "Hi Morgan, I wanted to reach out …~700 chars…"`
(stored 2026-06-15 via the LLM path, which had NO length cap; the regex
path's 300-char cap would have silently dropped it). Either way the
recipient name was invisible to embeddings and entity resolution, so no
`Morgan` graph node was ever created. The salvage recovers the recipient
from the salutation on BOTH deployed paths; over-long objects without a
salutation still drop.
"""

import pytest

from memory.fact_extractor import (
    _clean_triple,
    _fact_object_max_chars,
    _salvage_long_object,
)
from memory.llm_fact_extractor import _normalize_triple

JUNE_EMAIL = (
    "Hi Morgan,\nI wanted to reach out about my standing this semester. I'm "
    "enrolled in two courses and have fallen significantly behind in one due "
    "to a series of health issues — I was sick at the start of the term, then "
    "had a stretch of about a week where I had a lapse in medication I take "
    "for ADHD, which cost me roughly 7 working days total. Carrying two "
    "classes has made it hard to recover the lost ground. I'm trying to "
    "understand my options before the late drop deadline. Thanks for your "
    "help, Luke"
)

LONG_RANT = "so basically what happened was " + "really long text " * 30


class TestSalvage:
    def test_recipient_recovered_from_real_incident_email(self):
        assert _salvage_long_object(JUNE_EMAIL) == "Morgan"

    def test_two_word_name(self):
        assert _salvage_long_object("Dear Morgan Reeves,\nlong body here") == (
            "Morgan Reeves"
        )

    def test_no_salutation_returns_none(self):
        assert _salvage_long_object(LONG_RANT) is None

    def test_lowercase_word_after_greeting_is_not_a_name(self):
        assert _salvage_long_object("hi there, " + "x" * 400) is None

    def test_salutation_verb_case_insensitive(self):
        assert _salvage_long_object("HEY Sam, quick question about") == "Sam"


class TestRegexPath:
    def test_email_object_salvaged_to_recipient(self):
        assert _clean_triple("user", "email_sent", JUNE_EMAIL) == (
            "user", "email_sent", "morgan"
        )

    def test_long_object_without_salutation_dropped(self):
        assert _clean_triple("user", "said", LONG_RANT) is None

    def test_normal_object_untouched(self):
        assert _clean_triple("user", "has_advisor", "Morgan Reeves") == (
            "user", "has_advisor", "morgan reeves"
        )

    def test_env_cap_override(self, monkeypatch):
        monkeypatch.setenv("FACT_OBJECT_MAX_CHARS", "50")
        assert _fact_object_max_chars() == 50
        # 60-char non-salutation object now exceeds the cap and drops
        obj = "a perfectly ordinary but slightly long descriptive object x"
        assert len(obj) > 50
        assert _clean_triple("user", "described", obj) is None

    def test_env_cap_bad_value_falls_back(self, monkeypatch):
        monkeypatch.setenv("FACT_OBJECT_MAX_CHARS", "not-a-number")
        assert _fact_object_max_chars() == 300


class TestLLMPath:
    def test_email_object_salvaged_to_recipient(self):
        out = _normalize_triple(
            {"subject": "user", "relation": "email_sent", "object": JUNE_EMAIL}
        )
        assert out is not None
        assert out["object"] == "Morgan"

    def test_long_object_without_salutation_dropped(self):
        assert _normalize_triple(
            {"subject": "user", "relation": "said", "object": LONG_RANT}
        ) is None

    def test_normal_object_untouched(self):
        out = _normalize_triple(
            {"subject": "user", "relation": "has_advisor", "object": "Morgan"}
        )
        assert out is not None
        assert out["object"] == "Morgan"
