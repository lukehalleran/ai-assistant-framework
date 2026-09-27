"""E3 (2026-09-27, PLAN_20260927_session_audit_fixes.md "Shutdown fact
extraction resolves dates against the message's own time"):

Live incident (plan Evidence #5): at 21:13 on 2026-09-26 the user said
"finish ... tommorrow"; the shutdown LLM extractor's prompt told the model
"Today's date is <extraction time>" (whenever the shutdown run happened to
fire, sometimes hours later) and had the MODEL resolve the relative date
itself against that WRONG anchor — the live fact stored
``needs_to_do=... Mon 2026-09-28`` (one day off), and facts rendered the
extraction time "[2026-09-27 02:20 (today)]" instead of when it was said.

Fix (memory/llm_fact_extractor.py):
  - Each rendered prompt line now carries the message's OWN turn timestamp
    ("- [Sat 2026-09-26 21:13] User: ..."), and the TEMPORAL rule tells the
    model to copy a relative date word VERBATIM rather than resolve it.
  - `_attach_source_excerpts` (the provenance join against
    memory.fact_source.find_supporting_user_span, already computing
    `observed_at` from the matched turn's own timestamp since the 2026-09-06
    B2 work) now forwards that turn time as the triple's `timestamp` — only
    when the join actually found a real `turn_id`; otherwise it logs and
    leaves `timestamp` unset (extraction-time fallback, unchanged).

Fix (memory/user_profile.py):
  - `UserProfile.add_fact` used to parse a STRING `timestamp` into a
    `datetime` AFTER already deciding, via `isinstance(timestamp, datetime)`,
    whether to resolve relative words against it — so a string timestamp
    (exactly what the LLM path now forwards) silently fell through to
    `datetime.now()` for date resolution. The parse step now runs first.

Fix (memory/shutdown_processor.py):
  - The triple's forwarded `timestamp` also overrides chroma_store.add_fact's
    default now()-stamp for the `facts` collection, mirroring the profile path.

All of this is exercised through the DEPLOYED functions
(`_normalize_triple`, `LLMFactExtractor._attach_source_excerpts`,
`UserProfile.add_fact` / `add_facts_batch`) — never a re-derivation
(CLAUDE.md "validation must call the deployed function").

class: BC-58, BC-51
"""
from __future__ import annotations

from datetime import date, datetime

from memory.llm_fact_extractor import (
    LLMFactExtractor,
    _format_turn_timestamp,
    _normalize_triple,
)
from memory.user_profile import UserProfile
from memory.user_profile_schema import ProfileCategory


def _messages_section(prompt: str) -> str:
    section = prompt.split("USER MESSAGES (newest last):\n", 1)[1]
    return section.split("\n\nJSON:")[0]


def _find_fact(profile: UserProfile, relation: str):
    """Locate a stored fact by relation across every category, sidestepping
    get_category's TTL filtering (irrelevant to what this module tests)."""
    for facts in profile.profile["categories"].values():
        for f in facts:
            if isinstance(f, dict) and f.get("relation") == relation:
                return f
    return None


# ---------------------------------------------------------------------------
# 1. Prompt rendering: per-message timestamps + verbatim-copy TEMPORAL rule
# ---------------------------------------------------------------------------

class TestFormatTurnTimestamp:
    def test_iso_string(self):
        assert _format_turn_timestamp("2026-09-26T21:13:00") == "Sat 2026-09-26 21:13"

    def test_datetime_object(self):
        assert _format_turn_timestamp(datetime(2026, 9, 26, 21, 13)) == "Sat 2026-09-26 21:13"

    def test_none_is_none(self):
        assert _format_turn_timestamp(None) is None

    def test_unparseable_string_is_none(self):
        assert _format_turn_timestamp("not-a-date") is None


class TestPromptMessageTimestamps:
    def _extractor(self, max_chars=4000):
        return LLMFactExtractor(model_manager=None, max_input_chars=max_chars)

    def test_entry_with_turn_id_gets_timestamp_bracket(self):
        pairs = [{"query": "finish the remaining problems tomorrow",
                  "turn_id": "2026-09-26T21:13:00"}]
        prompt = self._extractor()._build_prompt(pairs)
        section = _messages_section(prompt).strip()
        assert section == "- [Sat 2026-09-26 21:13] User: finish the remaining problems tomorrow"

    def test_entry_without_timestamp_renders_bare(self):
        prompt = self._extractor()._build_prompt(["I love hiking"])
        section = _messages_section(prompt).strip()
        assert section == "- User: I love hiking"

    def test_temporal_rule_tells_model_to_copy_verbatim_not_resolve(self):
        prompt = self._extractor()._build_prompt(["placeholder message"])
        assert "Today's date is" not in prompt
        assert "VERBATIM" in prompt
        assert "do NOT resolve them to an absolute date yourself" in prompt

    def test_multiple_timestamped_entries_stay_chronological_with_brackets(self):
        pairs = [
            {"query": "first thing", "turn_id": "2026-09-26T09:00:00"},
            {"query": "second thing", "turn_id": "2026-09-26T21:13:00"},
        ]
        section = _messages_section(self._extractor()._build_prompt(pairs))
        # Search within the rendered messages section only — the TEMPORAL
        # rule's own worked example elsewhere in the prompt happens to reuse
        # a similar literal timestamp and must not be confused with it.
        assert section.index("[Sat 2026-09-26 09:00]") < section.index("[Sat 2026-09-26 21:13]")
        assert section.index("first thing") < section.index("second thing")

    def test_existing_coverage_prompt_tests_unaffected_by_untimestamped_entries(self):
        """Regression guard: entries with no turn_id/timestamp (the wizard's
        bare-string onboarding call, and existing_facts-only fixtures) must
        render exactly as before — no stray bracket, same budget arithmetic
        as test_learned_relations.py::TestCoveragePromptBuild."""
        pairs = [{"query": f"message number {i} " + "x" * 300, "response": ""}
                 for i in range(10)]
        prompt = LLMFactExtractor(model_manager=None, max_input_chars=1200)._build_prompt(pairs)
        assert "message number 9" in prompt
        assert "message number 0" not in prompt
        assert "- [" not in prompt


# ---------------------------------------------------------------------------
# 2. Provenance join: the triple's `timestamp` is the SOURCE MESSAGE's turn
#    time, falling back (logged) only when unjoinable.
# ---------------------------------------------------------------------------

class TestTripleTimestampProvenance:
    def test_timestamp_set_from_matched_turn(self):
        raw = {"subject": "user", "relation": "exam_date",
               "object": "the remaining problem set tomorrow", "confidence": 0.85}
        norm = _normalize_triple(raw)
        assert norm is not None
        triples = [norm]
        msg = {"query": "the remaining problem set tomorrow",
               "timestamp": "2026-09-26T21:13:00"}
        LLMFactExtractor._attach_source_excerpts(triples, [msg])
        assert triples
        assert triples[0]["timestamp"] == "2026-09-26T21:13:00"
        assert triples[0]["source_turn_id"] == "2026-09-26T21:13:00"

    def test_no_turn_id_leaves_timestamp_unset_and_logs_fallback(self, caplog):
        raw = {"subject": "user", "relation": "hobby", "object": "chess", "confidence": 0.8}
        norm = _normalize_triple(raw)
        assert norm is not None
        triples = [norm]
        with caplog.at_level("INFO", logger="llm_facts"):
            LLMFactExtractor._attach_source_excerpts(triples, ["I've been playing chess lately."])
        assert triples
        assert "timestamp" not in triples[0]
        assert any("falls back to extraction time" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. End-to-end: UserProfile.add_fact resolves against the MESSAGE's own
#    time (via the deployed add_facts_batch path), never extraction time.
# ---------------------------------------------------------------------------

class TestEndToEndMessageTimeResolution:
    def _profile(self, tmp_path):
        return UserProfile(profile_path=str(tmp_path / "profile.json"))

    def test_relative_date_resolves_against_a_fixed_past_message_time(self, tmp_path):
        """A message timestamp far from the real wall clock on ANY day this
        suite runs: if resolution ever fell back to datetime.now(), the
        resolved date would land near the actual current date, never
        2024-01-02. This is the discriminating regression guard for the
        add_fact string-timestamp ordering bug — it fails without the
        memory/user_profile.py reorder (reasoned below, not re-run: with the
        old ordering, `isinstance(timestamp, datetime)` sees a str at the
        point the relative-reference check runs, so `ref_date` falls back to
        `datetime.now()` and "tomorrow" resolves to the REAL current date's
        tomorrow instead of 2024-01-02)."""
        raw = {"subject": "user", "relation": "exam_date",
               "object": "the remaining problem set tomorrow", "confidence": 0.85}
        norm = _normalize_triple(raw)
        triples = [norm]
        msg = {"query": "the remaining problem set tomorrow",
               "timestamp": "2024-01-01T09:00:00"}
        LLMFactExtractor._attach_source_excerpts(triples, [msg])
        assert triples[0]["timestamp"] == "2024-01-01T09:00:00"

        profile = self._profile(tmp_path)
        added = profile.add_facts_batch(triples)
        assert added == 1

        fact = _find_fact(profile, "exam_date")
        assert fact is not None
        expected_weekday = date(2024, 1, 2).strftime("%a")
        assert f"{expected_weekday} 2024-01-02" in fact["value"]
        assert "tomorrow" not in fact["value"].lower()
        # The fact's own timestamp is the message time, not extraction time.
        assert fact["timestamp"].startswith("2024-01-01T09:00:00")

    def test_live_incident_scenario_message_stamped_saturday_night(self, tmp_path):
        """Mirrors the live 09-26 21:13 incident literally: a message says
        "tomorrow" at 21:13 on Saturday 2026-09-26 -- resolves to that
        Saturday's actual tomorrow (Sunday 2026-09-27, the plan's stated
        acceptance value), never Monday the 28th (what the live bug stored)."""
        raw = {"subject": "user", "relation": "exam_date",
               "object": "the remaining problem set tomorrow", "confidence": 0.85}
        norm = _normalize_triple(raw)
        triples = [norm]
        msg = {"query": "the remaining problem set tomorrow",
               "timestamp": "2026-09-26T21:13:00"}
        LLMFactExtractor._attach_source_excerpts(triples, [msg])

        profile = self._profile(tmp_path)
        added = profile.add_facts_batch(triples)
        assert added == 1

        fact = _find_fact(profile, "exam_date")
        assert fact is not None
        assert "Sun 2026-09-27" in fact["value"]
        assert "Mon 2026-09-28" not in fact["value"]
        assert fact["timestamp"].startswith("2026-09-26T21:13:00")

    def test_string_timestamp_ordering_bug_fixed_in_add_fact_directly(self, tmp_path):
        """Targets the exact ordering bug in UserProfile.add_fact: a STRING
        timestamp (as forwarded by the LLM shutdown path) must be parsed to a
        datetime BEFORE the relative-word resolution step runs, not after."""
        profile = self._profile(tmp_path)
        ok = profile.add_fact(
            "exam_date", "the remaining problem set tomorrow", confidence=0.8,
            timestamp="2024-01-01T09:00:00",
        )
        assert ok
        fact = _find_fact(profile, "exam_date")
        assert fact is not None
        expected_weekday = date(2024, 1, 2).strftime("%a")
        assert f"{expected_weekday} 2024-01-02" in fact["value"]

    def test_datetime_object_timestamp_unaffected_by_the_reorder(self, tmp_path):
        """Callers that already pass a real datetime object (every existing
        caller in the test suite, e.g. test_user_profile.py) must resolve
        exactly as before the reorder."""
        profile = self._profile(tmp_path)
        ok = profile.add_fact(
            "exam_date", "the remaining problem set tomorrow", confidence=0.8,
            timestamp=datetime(2024, 1, 1, 9, 0, 0),
        )
        assert ok
        fact = _find_fact(profile, "exam_date")
        assert fact is not None
        expected_weekday = date(2024, 1, 2).strftime("%a")
        assert f"{expected_weekday} 2024-01-02" in fact["value"]

    def test_category_is_study_confirming_exam_date_mapping(self, tmp_path):
        """Sanity check that the relation choice above lands where expected
        (RELATION_CATEGORY_MAP direct hit -- avoids the embedding-similarity
        categorization layer entirely, keeping this test deterministic and
        offline)."""
        profile = self._profile(tmp_path)
        profile.add_fact("exam_date", "Fri 2026-10-02", confidence=0.8,
                          timestamp=datetime(2026, 9, 26, 21, 13))
        current = profile.get_category(ProfileCategory.STUDY, include_historical=True)
        assert len(current) == 1
