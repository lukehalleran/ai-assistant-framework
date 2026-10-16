"""A recurrence claim needs recurrence evidence (2026-10-16).

Incident: a 10-15 reply said "you've earned the food order" from
``eating_habit = orders food after completing homework`` (minted from ONE
evening, empty excerpt).  Sibling live facts: ``sleep_pattern`` /
``study_habit`` (kind ``unknown``) and ``social_habit`` (kind ``habit`` through
ANOTHER clause's "every day").  Mechanisms: nothing checked a recurrence-NAMED
relation against its evidence (BC-51), and ``classify_claim_time`` treated bare
"every" as recurrence, let a habit cue beat future framing, and classified the
sentence instead of the winning clause (BC-04).

Everything here drives the DEPLOYED functions (classify_claim_time,
classify_claim_time_for_object, find_supporting_user_span,
LLMFactExtractor._attach_source_excerpts, FactExtractor._to_node,
ShutdownProcessor._extract_llm_facts).  Every live-text fixture is asserted in
BOTH clean and wrapped/indented form.  Names and values are invented.

class: BC-51, BC-04, BC-75 (also BC-58: sibling callers of the classifier,
BC-62/63: sink parity with a misbehaving fake model)
"""
import json
import textwrap
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import memory.learned_relations as lr
from memory.fact_source import (
    classify_claim_time,
    classify_claim_time_for_object,
    find_supporting_user_span,
    is_recurrence_relation,
    object_clause_for,
    recurrence_claim_supported,
)
from memory.llm_fact_extractor import LLMFactExtractor
from memory.shutdown_processor import ShutdownProcessor

OBSERVED = datetime(2026, 10, 16, 12, 0)


def wrapped(text: str) -> str:
    """The same words, hard-wrapped with an indented continuation (workflow §3)."""
    lines = textwrap.wrap(text, width=30)
    return "  " + "\n    ".join(lines)


def indented(text: str) -> str:
    """The claim on ONE indented line below a preamble line (an accept-case
    wrap: ``_claim_spans`` reads prose per line, so a habit statement hard-wrapped
    across lines is a deliberate fail-closed false reject — see MEASUREMENTS)."""
    return "Long day, honestly.\n    " + text + "\n"


def both(text: str):
    return [text, wrapped(text)]


def both_accept(text: str):
    return [text, indented(text)]


def _t(relation, obj, subject="user"):
    return {"subject": subject, "relation": relation, "object": obj}


def _find(relation, obj, text):
    return find_supporting_user_span(_t(relation, obj), [text])


# ---------------------------------------------------------------------------
# CLS-1: the classifier chokepoint
# ---------------------------------------------------------------------------

class TestClassifierTable:
    def test_plan_framing_beats_a_recurrence_cue(self):
        # base: "habit"
        ct = classify_claim_time("I plan to run every day starting tomorrow.", observed_at=OBSERVED)
        assert ct.kind == "plan"

    @pytest.mark.parametrize("text", [
        "I reviewed every chapter yesterday.",
        "I answered each question carefully.",
        "I checked every problem on the sheet.",
    ])
    def test_quantifier_every_is_not_recurrence(self, text):
        # base: "habit" for all three
        assert classify_claim_time(text, observed_at=OBSERVED).kind != "habit"

    def test_pizza_tonight_is_not_a_habit(self):
        assert classify_claim_time("I'm eating pizza tonight.", observed_at=OBSERVED).kind != "habit"

    @pytest.mark.parametrize("text", [
        "I always go to bed around 11.",
        "I usually study at the library on weekdays.",
        "I work Monday Wednesday Friday.",
        "I take it twice a day.",
        "I walk every day.",
        "I call my sister each Sunday.",
        "I check email every other day.",
        "I stretch every 3 days.",
        "I swim every single morning.",
        "I lift every weekday.",
    ])
    def test_real_recurrence_stays_habit(self, text):
        assert classify_claim_time(text, observed_at=OBSERVED).kind == "habit"

    def test_intent_framing_is_same_clause_only(self):
        # The habit lives in its own, un-framed clause: stays a habit here
        # (the sentence-level frame check is the GATE's job, below).
        ct = classify_claim_time("I will see Rowan tomorrow, and I walk every day.", observed_at=OBSERVED)
        assert ct.kind == "habit"

    def test_need_to_framing_is_a_plan(self):
        # fail-closed shape recorded in MEASUREMENTS (reuses the plan-cue vocabulary)
        assert classify_claim_time("I need to run every day.", observed_at=OBSERVED).kind == "plan"

    def test_historical_is_not_a_global_kind(self):
        # R2-4 option A: no new kind; the frame check lives in the gate only.
        ct = classify_claim_time("I used to run every day.", observed_at=OBSERVED)
        assert ct.kind in ("habit", "unknown", "plan", "event", "state")
        assert not recurrence_claim_supported(
            "exercise_habit", "I used to run every day.", "I used to run every day.", observed_at=OBSERVED)


# ---------------------------------------------------------------------------
# CLS-2: clause scope
# ---------------------------------------------------------------------------

def _scoped(sentence, obj):
    clause = object_clause_for(sentence, obj)
    return clause, classify_claim_time_for_object(sentence, clause, observed_at=OBSERVED)


class TestClauseScope:
    def test_elliptical_clause_inherits_the_shared_recurrence(self):
        _, ct = _scoped("I always run and swim", "swim")
        assert ct.kind == "habit"

    def test_own_today_beats_the_leading_yesterday(self):
        # base: dated YESTERDAY (the whole-sentence anchor)
        clause, ct = _scoped("Yesterday I worked late and today I saw Rowan", "Rowan")
        assert ct.kind == "event"
        assert ct.event_date == OBSERVED.date()

    def test_clause_with_own_subject_inherits_the_sentence_anchor(self):
        _, ct = _scoped("Yesterday I worked late and I saw Rowan", "Rowan")
        assert ct.kind == "event"
        assert ct.event_date == datetime(2026, 10, 15).date()

    def test_past_elliptical_head_inherits_nothing(self):
        # "walked" is -ed at the head of an elliptical clause: a separate completed act
        _, ct = _scoped("I always run and walked home", "walked home")
        assert ct.kind != "habit"
        _, ct = _scoped("I always run, and walked home", "walked home")
        assert ct.kind != "habit"

    def test_neighbour_recurrence_does_not_make_a_one_off_a_habit(self):
        # the live social_habit shape (invented names): base kind "habit"
        clean, hard_wrapped = both("I walk every day, and saw Tamsin for 2 hours yesterday")
        ev = _find("social_note", "saw Tamsin for 2 hours yesterday", clean)
        assert ev is not None and ev.claim_kind == "event" and ev.event_date
        # Hard-wrapped: _claim_spans splits at the newline, so the verb and its
        # object land in different spans (pre-existing); the recurrence cue
        # must still not reach the object.
        ev = _find("social_note", "saw Tamsin for 2 hours yesterday", hard_wrapped)
        assert ev is not None and ev.claim_kind != "habit"

    def test_unclassifiable_past_verb_is_still_not_habit(self):
        # "hung" is not in the closed past-tense set (R2-s1: recall out of scope)
        ev = _find("social_note", "hung with Tamsin for 2 hours Sat", "I walk every day, and hung with Tamsin for 2 hours Sat")
        assert ev is not None and ev.claim_kind != "habit"

    def test_winner_clause_not_last_loop_clause_decides(self):
        # Two messages mention the object; the EXACT-object one wins and a LATER
        # losing message carries a recurrence cue.  Classification follows the
        # winner's own clause (never the loop's last clause).
        msgs = ["Yesterday I worked late and I saw Rowan",
                "Rowan comes by every day"]
        ev = find_supporting_user_span(_t("social_note", "saw Rowan"), msgs)
        assert ev is not None and ev.turn_index == 0
        assert ev.claim_kind == "event" and ev.event_date


class TestTrailingAnchor:
    """Referee round 1: a trailing event anchor reaches earlier clauses (an
    `unknown` past event would render as current forever)."""

    ONE_OFFS = [
        ("the gym", "I went to the gym and did legs yesterday."),
        ("Rowan", "I saw Rowan and we got lunch yesterday."),
        ("Rowan", "I worked late, then I saw Rowan yesterday."),
        ("the museum", "I visited the museum, and Tamsin came along last night."),
        ("the report", "I finished the report, then went home yesterday."),
        ("pasta", "We ate pasta, and I slept early last night."),
        ("the park", "I went to the park, saw Rowan, and ate lunch yesterday."),
        ("the clinic", "I came to the clinic, and I left at noon yesterday."),
    ]

    @pytest.mark.parametrize("obj,text", ONE_OFFS)
    def test_trailing_anchor_dates_the_event(self, obj, text):
        msg = {"query": text, "timestamp": "2026-10-16T12:00:00"}
        ev = find_supporting_user_span(_t("visited", obj), [msg])
        assert ev is not None
        assert (ev.claim_kind, ev.event_date) == ("event", "2026-10-15")
        wrapped_msg = {"query": indented(text), "timestamp": "2026-10-16T12:00:00"}
        ev = find_supporting_user_span(_t("visited", obj), [wrapped_msg])
        assert ev is not None and (ev.claim_kind, ev.event_date) == ("event", "2026-10-15")

    def test_own_anchor_still_wins_over_a_following_one(self):
        _, ct = _scoped("I saw Rowan today and left yesterday", "Rowan")
        assert ct.event_date == OBSERVED.date()

    def test_recurrence_is_never_inherited_from_a_following_clause(self):
        _, ct = _scoped("I swam, and I walk every day", "swam")
        assert ct.kind != "habit"


# ---------------------------------------------------------------------------
# CLS-2/3: frame check + inheritance, each under a family relation
# ---------------------------------------------------------------------------

class TestFrameAndInheritanceGate:
    @pytest.mark.parametrize("relation,text,obj,accepted", [
        ("exercise_habit", "I plan to run and swim every day", "swim every day", False),
        ("exercise_habit", "I used to run and swim every day", "swim every day", False),
        ("exercise_habit", "I always run and walked home", "walked home", False),
        ("exercise_habit", "I always run and swim", "swim", True),
        ("exercise_habit", "I no longer run every day", "run every day", False),
        ("exercise_habit", "I stopped running every day", "running every day", False),
        ("exercise_habit", "I'm going to start running every morning", "running every morning", False),
        ("exercise_habit", "I want to run every morning", "run every morning", False),
        ("bedtime_habit", "I always go to bed around 11", "go to bed around 11", True),
        ("shift_pattern", "I work Monday and Friday", "Monday and Friday", True),
    ])
    def test_gate_through_the_provenance_join(self, relation, text, obj, accepted):
        for variant in (both_accept(text) if accepted else both(text)):
            ev = _find(relation, obj, variant)
            assert (ev is not None) is accepted, variant
            if accepted:
                assert ev.claim_kind == "habit"

    def test_false_reject_is_deliberate_and_fail_closed(self):
        # MEASUREMENTS: a real current habit behind a historical frame is rejected.
        assert _find("exercise_habit", "swim every day", "I used to run but now I swim every day") is None

    def test_non_family_relation_is_untouched(self):
        assert _find("likes", "swimming", "I used to love swimming") is not None
        assert recurrence_claim_supported("likes", "I used to love swimming", "I used to love swimming")

    def test_family_membership_is_token_based(self):
        for rel in ("eating_habit", "sleep_pattern", "morning_routine", "habits", "study_patterns"):
            assert is_recurrence_relation(rel), rel
        for rel in ("likes", "works_at", "shift_pattern_x_y"):
            pass
        assert not is_recurrence_relation("patterned")
        assert not is_recurrence_relation("likes")


# ---------------------------------------------------------------------------
# CLS-3: the three live excerpts, through _attach_source_excerpts
# ---------------------------------------------------------------------------

LIVE = [
    ("eating_habit", "orders food after completing homework",
     "Finished the problem set tonight, so I ordered food after completing homework."),
    ("sleep_pattern", "slept heavy til noon",
     "Slept heavy til noon today after the late night."),
    ("study_habit", "completed 5 out of 6 underlying problems",
     "Completed 5 out of 6 underlying problems on the review sheet."),
    ("social_habit", "hung with Tamsin for 2 hours Sat",
     "I walk every day, and hung with Tamsin for 2 hours Sat"),
]


class TestAttachSourceExcerpts:
    @pytest.mark.parametrize("relation,obj,text", LIVE)
    def test_live_one_off_claims_are_rejected(self, relation, obj, text):
        for variant in both(text):
            triples = [{"subject": "user", "relation": relation, "object": obj, "confidence": 0.8}]
            LLMFactExtractor._attach_source_excerpts(triples, [variant])
            assert triples == [], variant

    def test_real_habit_is_accepted_with_an_excerpt(self):
        for variant in both_accept("I always go to bed around 11"):
            triples = [{"subject": "user", "relation": "bedtime_habit",
                        "object": "goes to bed around 11", "confidence": 0.8}]
            LLMFactExtractor._attach_source_excerpts(triples, [variant])
            assert len(triples) == 1, variant
            assert triples[0]["claim_kind"] == "habit"
            assert "11" in triples[0]["source_excerpt"]

    @pytest.mark.parametrize("text,obj", [
        ("I'm eating pizza tonight after homework", "eating pizza tonight after homework"),       # state
        ("I am going to order food after homework tomorrow", "order food after homework tomorrow"),  # plan
        ("I used to order food after homework", "order food after homework"),                     # historical
        ("Ordered food after homework last night", "ordered food after homework"),                # event
    ])
    def test_non_habit_kinds_fail_under_a_family_relation(self, text, obj):
        for variant in both(text):
            triples = [{"subject": "user", "relation": "eating_habit", "object": obj, "confidence": 0.8}]
            LLMFactExtractor._attach_source_excerpts(triples, [variant])
            assert triples == [], variant


# ---------------------------------------------------------------------------
# R2-3: the regex producer
# ---------------------------------------------------------------------------

class TestRegexProducer:
    @pytest.fixture(scope="class")
    def fx(self):
        from memory.fact_extractor import FactExtractor
        return FactExtractor(use_rebel=False, use_regex=True)

    def test_possessive_routine_without_recurrence_is_rejected(self, fx):
        for variant in both("my routine is gym then coffee"):
            assert fx._to_node("user", "routine", "gym then coffee", 0.8, "regex", variant) is None, variant

    def test_possessive_routine_with_recurrence_is_kept(self, fx):
        for variant in both_accept("my routine is gym every morning"):
            node = fx._to_node("user", "routine", "gym every morning", 0.8, "regex", variant)
            assert node is not None, variant
            assert node.metadata["claim_kind"] == "habit"

    def test_schedule_producer_is_exempt_by_tag_not_by_name(self, fx):
        text = "My shift is nights"
        kept = fx._to_node("user", "shift_pattern", "nights", 0.8, "schedule_regex", text)
        assert kept is not None
        # the SAME relation from a non-schedule producer is gated like any other
        assert fx._to_node("user", "shift_pattern", "nights", 0.8, "regex", text) is None
        # and a generic-possessive routine from the regex path is gated
        assert fx._to_node("user", "sleep_pattern", "bad", 0.8, "regex", "my sleep pattern is bad") is None

    def test_entity_facts_are_out_of_scope(self, fx):
        node = fx._to_node("rowan", "morning_routine", "runs", 0.8, "regex", "Rowan runs.")
        assert node is not None

    @pytest.mark.asyncio
    async def test_end_to_end_extract_facts(self, fx):
        out = await fx.extract_facts("my routine is gym then coffee", "")
        assert not [n for n in out if n.metadata.get("relation") == "routine"]
        out = await fx.extract_facts("my routine is gym every morning", "")
        assert [n for n in out if n.metadata.get("relation") == "routine"]


# ---------------------------------------------------------------------------
# Sink parity (BC-62/63): a misbehaving model proposes both facts
# ---------------------------------------------------------------------------

class TestSinkParity:
    @pytest.mark.asyncio
    async def test_profile_chroma_and_graph_receive_only_the_supported_triple(self):
        proposal = json.dumps([
            {"subject": "user", "relation": "eating_habit",
             "object": "orders food after completing homework", "confidence": 0.9},
            {"subject": "user", "relation": "bedtime_habit",
             "object": "goes to bed around 11", "confidence": 0.9},
        ])
        mm = MagicMock()
        mm.generate_once = AsyncMock(return_value=proposal)
        chroma = MagicMock()
        chroma.add_fact.side_effect = lambda fact, source, confidence=0.75: f"id-{abs(hash(fact)) % 1000}"
        profile = MagicMock()
        profile.profile = {"categories": {}}
        storage = MagicMock()
        storage.fact_verifier = None
        storage.graph_memory = MagicMock()
        storage.entity_resolver = MagicMock()
        proc = ShutdownProcessor(
            corpus_manager=MagicMock(), chroma_store=chroma, consolidator=MagicMock(),
            fact_extractor=MagicMock(), model_manager=mm, user_profile=profile,
            storage=storage, session_start=datetime(2026, 10, 16, 0, 0),
        )
        convs = [
            {"query": "Finished the problem set tonight, so I ordered food after completing homework.",
             "response": "Nice work.", "timestamp": "2026-10-16T09:00:00"},
            {"query": "I always go to bed around 11", "response": "Good.",
             "timestamp": "2026-10-16T09:05:00"},
        ]
        with patch("memory.memory_storage._get_graph_enabled", return_value=True):
            await proc._extract_llm_facts(convs)

        chroma_texts = [c.kwargs.get("fact") or c.args[0] for c in chroma.add_fact.call_args_list]
        assert len(chroma_texts) == 1 and "bedtime_habit" in chroma_texts[0], chroma_texts

        graph_rels = [c.kwargs.get("rel") for c in storage._ingest_fact_to_graph.call_args_list]
        assert graph_rels == ["bedtime_habit"], graph_rels

        batches = [c.args[0] for c in profile.add_facts_batch.call_args_list]
        sent = [t["relation"] for batch in batches for t in batch]
        assert sent == ["bedtime_habit"], sent


# ---------------------------------------------------------------------------
# CLS-4: learned relations count accepted relations, on source dates
# ---------------------------------------------------------------------------

class _FixedMM:
    def __init__(self, payload):
        self.payload = payload

    async def generate_once(self, **kwargs):
        return self.payload


@pytest.fixture
def lr_store(tmp_path, monkeypatch):
    path = str(tmp_path / "learned_relations.json")
    monkeypatch.setattr(lr, "_STORE_PATH", path)
    monkeypatch.setattr(lr, "_store", None)
    store = lr.get_learned_relation_store()
    yield store
    monkeypatch.setattr(lr, "_store", None)


def _days(store, relation):
    return list(store._data["relations"].get(relation, {}).get("days", []))


class TestLearnedRelationOrdering:
    @pytest.mark.asyncio
    async def test_unsupported_triple_records_nothing(self, lr_store):
        payload = json.dumps([{"subject": "user", "relation": "snack_habit",
                               "object": "ate crackers", "confidence": 0.9}])
        ex = LLMFactExtractor(_FixedMM(payload))
        msgs = [{"query": "Ate some crackers tonight", "timestamp": "2026-10-10T20:00:00"}]
        out = await ex.extract_triples(msgs)
        assert out == []
        assert _days(lr_store, "snack_habit") == []   # base: recorded before the join

    @pytest.mark.asyncio
    async def test_supported_triple_records_on_the_source_day_not_extraction_day(self, lr_store):
        payload = json.dumps([{"subject": "user", "relation": "therapy_schedule",
                               "object": "biweekly with Harper", "confidence": 0.9}])
        ex = LLMFactExtractor(_FixedMM(payload))
        msgs = [{"query": "I see Harper for therapy biweekly with Harper",
                 "timestamp": "2026-10-10T20:00:00"}]
        for _ in range(2):          # the same turn replayed on two later shutdowns
            out = await ex.extract_triples(msgs)
            assert len(out) == 1
        assert _days(lr_store, "therapy_schedule") == ["2026-10-10"]

    @pytest.mark.asyncio
    async def test_no_source_time_means_no_learning(self, lr_store):
        payload = json.dumps([{"subject": "user", "relation": "therapy_schedule",
                               "object": "biweekly with Harper", "confidence": 0.9}])
        ex = LLMFactExtractor(_FixedMM(payload))
        out = await ex.extract_triples(["I see Harper for therapy biweekly with Harper"])
        assert len(out) == 1
        assert _days(lr_store, "therapy_schedule") == []

    def test_opaque_id_with_valid_timestamp_uses_the_timestamp(self):
        msg = {"query": "I always go to bed around 11", "id": "turn-abc-123",
               "timestamp": "2026-10-09T22:30:00"}
        ev = find_supporting_user_span(_t("bedtime_habit", "go to bed around 11"), [msg])
        assert ev is not None
        assert ev.turn_id == "turn-abc-123"
        assert ev.source_time == "2026-10-09T22:30:00"

    def test_opaque_id_without_timestamp_has_no_source_time(self):
        msg = {"query": "I always go to bed around 11", "id": "turn-abc-123"}
        ev = find_supporting_user_span(_t("bedtime_habit", "go to bed around 11"), [msg])
        assert ev is not None and ev.source_time == ""
        # observed_at still falls back to now() for claim-time anchoring only
        assert ev.observed_at
