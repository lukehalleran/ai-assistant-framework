"""B1 replay (2026-09-06 Phase B contracts): does the provenance join, across
all three extraction paths, ever cite a same-day DENIAL as evidence for an
affirmative dose claim -- or admit a "today" object using only a PRIOR-day
statement as if that supported "today"?

Synthetic scenario (generalized, never the owner's own wording): the user
reports taking an extra dose of a fictional medication ("Zelphex" -- the
repo's standing fictional med name, see test_fact_source_negation_scope.py
and test_cross_deduplicator.py) YESTERDAY, and explicitly denies taking any
dose TODAY, in the same turn. A downstream extractor might still propose a
triple whose object reads as a TODAY dose.

Three joins are driven directly against the DEPLOYED functions (never a
re-derivation):
  (a) memory.fact_source.find_supporting_user_span
  (b) the per-turn path, MemoryStorage.extract_and_store_facts, fed a stub
      fact_extractor whose output is built with the SAME deployed
      memory.fact_source.supporting_excerpt call the real regex extractor's
      _to_node() makes (memory/fact_extractor.py:1939) -- so the stub
      reproduces what the regex path would actually attach, not a fabrication
  (c) the shutdown LLM path's normalize + provenance join:
      memory.llm_fact_extractor._normalize_triple +
      LLMFactExtractor._attach_source_excerpts (which internally calls
      find_supporting_user_span, same as (a))

STOP CONDITION RESULT (verified against HEAD before writing any fix): all
three paths ALREADY avoid citing the today-denial clause as excerpt for any
of the required variants (object phrasing "5mg today" / "extra 5mg" /
"5mg Zelphex"; reversed clause order; "Didn't"/"did not" contraction;
implicit-subject phrasing). fact_source's clause-level negation
(_clause_is_negated, 2026-09-06) plus the object-grounding requirement (a
denial clause that never mentions the object token cannot be selected as
"best") already prevents the today-clause from ever becoming the stored
excerpt. These tests are kept as a REGRESSION SUITE only -- no parser
changes were made for B1 (see docs/HANDOFF_20260906_phaseAB_contracts.md
Phase B / B1).

Note (out of scope for B1, reported to the referee): a triple whose OBJECT
itself claims "today" (e.g. "5mg today") is not rejected outright -- it is
admitted using the yesterday clause as evidence, which is a genuine
temporal mismatch (the object says "today", the evidence describes
yesterday). B2's claim_kind/event_date classifier is the general fix for
this at READ time (a fact whose evidence excerpt classifies as a past EVENT
is excluded from "current" state regardless of what the object text
literally says) -- see test_claim_temporal_kind.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.fact_source import find_supporting_user_span, supporting_excerpt
from memory.llm_fact_extractor import LLMFactExtractor, _normalize_triple
from memory.memory_storage import MemoryStorage

YESTERDAY_CLAUSE = "Took an extra 5mg of Zelphex yesterday."
TODAY_DENIAL_CLAUSE = "Didn't take any today."

BASE_MSG = f"{YESTERDAY_CLAUSE} {TODAY_DENIAL_CLAUSE}"
REVERSED_MSG = f"{TODAY_DENIAL_CLAUSE} {YESTERDAY_CLAUSE}"
DID_NOT_MSG = f"{YESTERDAY_CLAUSE} Did not take any today."

# A distinct yesterday-affirmation wording (no medication name mentioned) --
# exercises the object-grounding path separately from the "Zelphex" variants
# above; its own yesterday clause differs from YESTERDAY_CLAUSE.
IMPLICIT_YESTERDAY_CLAUSE = "Had the extra 5mg yesterday."
IMPLICIT_SUBJECT_MSG = f"Skipped it today. {IMPLICIT_YESTERDAY_CLAUSE}"

# The literal implicit-subject wording named in the handoff (bare "5", no
# unit) -- exercised separately below since it is too weak a token to ground
# at all (documents an existing, unrelated under-fire, not a fix target).
IMPLICIT_SUBJECT_BARE_DIGIT_MSG = "Skipped it today. Had the extra 5 yesterday."

BAD_TODAY_OBJECTS = ("5mg today",)
# (message, yesterday-clause-text, [supported objects for that message])
YESTERDAY_MESSAGE_CASES = (
    (BASE_MSG, YESTERDAY_CLAUSE, ("extra 5mg", "5mg Zelphex")),
    (REVERSED_MSG, YESTERDAY_CLAUSE, ("extra 5mg", "5mg Zelphex")),
    (DID_NOT_MSG, YESTERDAY_CLAUSE, ("extra 5mg", "5mg Zelphex")),
    (IMPLICIT_SUBJECT_MSG, IMPLICIT_YESTERDAY_CLAUSE, ("extra 5mg",)),
)


def _triple(obj: str) -> dict:
    return {"subject": "user", "relation": "medication_dose", "object": obj}


def _never_denial(ev) -> None:
    """A returned EvidenceSpan must never be (or contain only) the same-day
    denial -- it must never carry the denial's own words."""
    if ev is None:
        return
    low = ev.text.lower()
    assert "didn't" not in low
    assert "did not" not in low
    assert "skipped" not in low
    assert ev.text.strip() not in {TODAY_DENIAL_CLAUSE, "Skipped it today."}


# ---------------------------------------------------------------------------
# (a) memory.fact_source.find_supporting_user_span directly
# ---------------------------------------------------------------------------

class TestPathA_FindSupportingUserSpan:
    @pytest.mark.parametrize("msg", [BASE_MSG, REVERSED_MSG, DID_NOT_MSG, IMPLICIT_SUBJECT_MSG])
    @pytest.mark.parametrize("obj", BAD_TODAY_OBJECTS)
    def test_today_object_never_gets_denial_as_excerpt(self, msg, obj):
        ev = find_supporting_user_span(_triple(obj), [msg])
        _never_denial(ev)

    @pytest.mark.parametrize("msg,yesterday_clause,objects", YESTERDAY_MESSAGE_CASES)
    def test_yesterday_object_carries_yesterday_clause(self, msg, yesterday_clause, objects):
        for obj in objects:
            ev = find_supporting_user_span(_triple(obj), [msg])
            assert ev is not None, f"expected support for {obj!r} in {msg!r}"
            _never_denial(ev)
            assert ev.text == yesterday_clause

    def test_bare_digit_object_is_ungrounded_and_rejected(self):
        """The literal handoff wording ('Had the extra 5 yesterday') uses a
        bare digit with no unit -- too weak a token to ground on its own, so
        an object built the same way ('5 today') is rejected outright rather
        than accidentally matched. Documents an existing, unrelated
        under-fire; not something B1 needs to fix."""
        ev = find_supporting_user_span(_triple("5 today"), [IMPLICIT_SUBJECT_BARE_DIGIT_MSG])
        assert ev is None

    def test_reversed_order_still_prefers_yesterday_clause(self):
        ev = find_supporting_user_span(_triple("extra 5mg"), [REVERSED_MSG])
        assert ev is not None
        assert ev.text == YESTERDAY_CLAUSE


# ---------------------------------------------------------------------------
# (b) the per-turn path: MemoryStorage.extract_and_store_facts
# ---------------------------------------------------------------------------

class TestPathB_MemoryStorageExtractAndStoreFacts:
    """Stub fact_extractor whose output is built with the DEPLOYED
    memory.fact_source.supporting_excerpt (the same call
    memory/fact_extractor.py's _to_node makes at line 1939) -- this exercises
    the real storage-forwarding plumbing without re-deriving fact_source's
    logic."""

    def _run(self, msg: str, obj: str):
        excerpt = supporting_excerpt(msg, obj, 400)
        mock_chroma = MagicMock()
        mock_chroma.add_fact.return_value = "test-id"
        mock_extractor = MagicMock()
        mock_extractor.extract_facts = AsyncMock(return_value=[{
            "content": f"user | medication_dose | {obj}",
            "metadata": {
                "subject": "user",
                "relation": "medication_dose",
                "object": obj,
                "confidence": 0.8,
                "source": "conversation",
                "source_excerpt": excerpt,
                "source_support": "regex_span",
            },
        }])
        storage = MemoryStorage(
            corpus_manager=MagicMock(),
            chroma_store=mock_chroma,
            fact_extractor=mock_extractor,
        )
        asyncio.run(storage.extract_and_store_facts(msg, "", 0.8))
        mock_chroma.add_fact.assert_called_once()
        return mock_chroma.add_fact.call_args.kwargs["source"]

    @pytest.mark.parametrize("obj", BAD_TODAY_OBJECTS)
    def test_today_object_forwarded_excerpt_never_denial(self, obj):
        source = self._run(BASE_MSG, obj)
        assert "didn't" not in source["source_excerpt"].lower()
        assert source["source_excerpt"].strip() != TODAY_DENIAL_CLAUSE

    @pytest.mark.parametrize("obj", ("extra 5mg", "5mg Zelphex"))
    def test_yesterday_object_forwarded_excerpt_is_yesterday_clause(self, obj):
        source = self._run(BASE_MSG, obj)
        assert source["source_excerpt"] == YESTERDAY_CLAUSE


# ---------------------------------------------------------------------------
# (c) the shutdown LLM path's normalize + provenance join
# ---------------------------------------------------------------------------

class TestPathC_LLMExtractorNormalizeAndAttach:
    """Simulates a stub LLM output (the raw JSON-shaped dict an LLM call
    would have returned) through the REAL _normalize_triple and
    _attach_source_excerpts -- no LLM call is made."""

    def _run(self, msg: str, obj: str):
        raw = {"subject": "user", "relation": "medication_dose", "object": obj,
               "confidence": 0.8}
        norm = _normalize_triple(raw)
        assert norm is not None
        triples = [norm]
        LLMFactExtractor._attach_source_excerpts(triples, [msg])
        return triples

    @pytest.mark.parametrize("msg", [BASE_MSG, REVERSED_MSG, DID_NOT_MSG, IMPLICIT_SUBJECT_MSG])
    @pytest.mark.parametrize("obj", BAD_TODAY_OBJECTS)
    def test_today_object_never_attaches_denial_excerpt(self, msg, obj):
        triples = self._run(msg, obj)
        if not triples:
            return  # dropped entirely -- also acceptable, never a false claim
        excerpt = triples[0]["source_excerpt"].lower()
        assert "didn't" not in excerpt
        assert "did not" not in excerpt
        assert "skipped" not in excerpt

    @pytest.mark.parametrize("msg,yesterday_clause,objects", YESTERDAY_MESSAGE_CASES)
    def test_yesterday_object_attaches_yesterday_excerpt(self, msg, yesterday_clause, objects):
        for obj in objects:
            triples = self._run(msg, obj)
            assert triples, f"expected {obj!r} to survive the provenance join for {msg!r}"
            assert triples[0]["source_excerpt"] == yesterday_clause
