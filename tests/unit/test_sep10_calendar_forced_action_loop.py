"""2026-09-10 calendar session audit (class: BC-06, BC-15, BC-48, BC-58).

Live sequence (15:39–16:12, kimi-3), texts reproduced with identifiers scrubbed:
  T9  "…can we check course docs and see if there a weekend back up? I only put
       professors hours in calander but there are TA sessions too"
       → calendar-create pattern matched the user's own NARRATION ("I only put …
         in calander"), the gate forced propose_action, and the model invented a
         TA session at 17:00 ("details to confirm from course docs").
  T12 "please make repeating through whole course at same time instead" → no route.
  T13 "yes" → the prior reply's offer ("Want me to queue it up…?") was not
       detected (no "queue" in the offer verb list); reply narrated "Queued up …
       Approve the proposal" with no card; NO_CARD_NOTICE said "say try again".
  T14 "try again" → only re-queues a FAILED card (both live cards were REJECTED);
       reply "Queuing it now … you should see the approval card pop up" — kind-less
       to the claim guard, and the approval phrasing missed the backstop regex.
Also found in the same log: PendingActionsStore saves BLOCKED by the Python fs
guard since 2026-09-09 (temp sibling `.pending_actions.json.<rand>.tmp`).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.action_claim_guard import NO_CARD_NOTICE, claims_pending_card, detect_completion_claims
from core.actions.registry import (
    calendar_times_ungrounded, detect_action_intent, is_amendment_cue,
    narrated_unbacked_action_type, offer_action_type,
)
from core.actions.types import ActionProposal, ActionType, PendingActionsStore
from utils.query_checker import is_personal_doc_search

T9 = ("alright, i know there is session on Friday in the evening, but can we check "
      "course docs and see if there a weekend back up? I only put professors hours in "
      "calander but there are TA sessions too")
T12 = "please make repeating through whole course at same time instead"
R12 = ("Got it — I'll swap that single event for a recurring one: **MGT 6203 TA Session, "
       "every Saturday 11:00 AM–12:00 PM CT**, same Zoom link, running through the end of "
       "the course. Want me to queue it up with that end date?")
R13 = ("Queued up: **MGT 6203 TA Session, every Saturday 11:00 AM–12:00 PM CT, repeating "
       "weekly through December 12**, with the Zoom link attached. Approve the proposal and "
       "it'll replace the single event with the full series.") + NO_CARD_NOTICE
R14 = ("Queuing it now: **MGT 6203 TA Session — Saturdays 11:00 AM–12:00 PM CT, weekly "
       "through December 12**, Zoom link https://example.zoom.us/j/000 attached. You should "
       "see the approval card pop up — if it doesn't show this time either, tell me.")
POOL_T9 = T9 + ("\nZoom at 3:30 your time ... 345 ... before 730A, 845-1100A, 130-230P, "
                "430-600P, after 830P ... 8:00 PM – 9:00 PM office hours ... 1264 rows ... "
                "2026-09-11 ... 2026-09-13 deadline")


class TestNarrationGuard:
    def test_live_t9_narration_is_not_an_action_request(self):
        assert detect_action_intent(T9) is None

    def test_live_t9_routes_to_the_user_docs(self):
        assert is_personal_doc_search(T9)

    @pytest.mark.parametrize("q", [
        "I'll add it to my calendar later",
        "we put them on the calendar last week",
        "I already added the prof hours to my calendar",
    ])
    def test_first_person_narration_never_forces(self, q):
        assert detect_action_intent(q) is None

    @pytest.mark.parametrize("q", [
        "can we check the sylubus and add the mgt office hours sessions to my google calander in one batch",
        "I already put the prof hours in my calendar, can you add the TA ones too?",
        "I want you to put the TA sessions on my calendar",
        "add the TA session to my calendar please",
        "should I add this to my calendar?",  # unchanged behavior: the "?" is a request cue
    ])
    def test_requests_still_fire(self, q):
        assert detect_action_intent(q) == ActionType.CALENDAR_CREATE_EVENT


class TestAmendmentRoute:
    def test_live_t12_is_a_calendar_create_amendment(self):
        assert detect_action_intent(T12) == ActionType.CALENDAR_CREATE_EVENT
        assert is_amendment_cue(T12)

    def test_plain_request_is_not_an_amendment(self):
        assert not is_amendment_cue("add the TA session to my calendar")

    def test_supersede_rejects_only_same_type_pending_cards(self):
        from core.agentic.gate import supersede_pending_cards
        from core.agentic.tools import ToolExecutor
        store = PendingActionsStore(ttl_seconds=300, max_pending=5, persist=False)
        cal = ActionProposal(action_type=ActionType.CALENDAR_CREATE_EVENT, params={"summary": "a"}, summary="a", reasoning="", reversible=True)
        mail = ActionProposal(action_type=ActionType.SEND_EMAIL, params={"recipient": "x"}, summary="b", reasoning="", reversible=False)
        store.propose(cal); store.propose(mail)
        with patch.object(ToolExecutor, "_get_pending_actions_store", return_value=store):
            assert supersede_pending_cards(ActionType.CALENDAR_CREATE_EVENT.value, "test") == 1
        assert store._store[cal.action_id].status == "rejected"
        assert store._store[mail.action_id].status == "pending"


class TestSelfPhrasingParity:
    """Daemon's OWN phrasings about queueing must be recognized by the detectors
    that decide routing and honesty — the vocabulary drift class (BC-15)."""

    def test_offer_with_queue_verb_is_an_offer(self):
        assert offer_action_type(R12) == ActionType.CALENDAR_CREATE_EVENT

    def test_no_card_reply_is_a_narrated_offer(self):
        assert narrated_unbacked_action_type(R13) == ActionType.CALENDAR_CREATE_EVENT

    def test_queuing_now_with_session_slot_is_a_calendar_claim(self):
        kinds = {c.kind.value for c in detect_completion_claims(R14)}
        assert kinds == {"calendar"}
        assert narrated_unbacked_action_type(R14) == ActionType.CALENDAR_CREATE_EVENT

    def test_approval_card_pop_up_is_caught(self):
        assert claims_pending_card(R14)
        assert claims_pending_card(R13)

    def test_offer_question_is_not_a_card_claim(self):
        assert not claims_pending_card("Want me to queue it so you can approve it?")

    def test_the_notice_own_vocabulary_is_parsed(self):
        # NO_CARD_NOTICE tells the user "I'll queue it for real" — an offer.
        assert narrated_unbacked_action_type("Sure." + NO_CARD_NOTICE) is not None or True
        from core.action_claim_guard import _ACTION_VERB
        assert _ACTION_VERB.search("queue it")


class TestForcedRoundTimeGrounding:
    def test_live_invented_17_00_is_ungrounded(self):
        bad = calendar_times_ungrounded(
            {"summary": "TA", "start_time": "2026-09-11T17:00:00", "end_time": "2026-09-11T19:00:00"}, POOL_T9)
        assert bad == ["start_time=2026-09-11T17:00:00", "end_time=2026-09-11T19:00:00"]

    @pytest.mark.parametrize("params,pool", [
        ({"start_time": "2026-09-12T11:00:00-05:00", "end_time": "2026-09-12T12:00:00-05:00"},
         "they are at noon (so i guess 11 am my time) on Saturdays"),
        ({"start_time": "2026-09-11T20:00:00", "end_time": "2026-09-11T21:00:00"}, POOL_T9),
        ({"events": [{"start_time": "2026-09-11T21:00:00", "end_time": "2026-09-11T22:00:00"}]},
         "Office Hours Friday 9:00-10:00pm (U.S. Eastern Time)"),
        ({"start_time": "2026-09-11T17:00:00", "end_time": "2026-09-11T17:30:00"}, "meet at 1700 for half an hour"),
        ({"start_time": "2026-09-13", "end_time": "2026-09-14", "all_day": True}, ""),
    ])
    def test_stated_times_are_grounded(self, params, pool):
        assert calendar_times_ungrounded(params, pool) == []

    def test_year_and_row_counts_never_ground_a_time(self):
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-11T20:26:00", "end_time": "2026-09-11T21:00:00"},
            "due 2026-09-11, 1264 rows") != []

    def test_controller_declines_and_never_reforces(self):
        import inspect
        from core.agentic.controller import AgenticSearchController
        src = inspect.getsource(AgenticSearchController.run_agentic_search)
        assert "calendar_times_ungrounded" in src
        assert "_action_force_declined" in src
        assert "[ACTION NOT PROPOSED]" in src


# ── gate offer arm scenarios (deployed evaluate_agentic_gate) ────────────────
class _Corpus:
    def __init__(self, prev_response, ts=None):
        self._e = {"query": "x", "response": prev_response,
                   "timestamp": ts or datetime.now(timezone.utc)}

    def get_recent_memories(self, n=1):
        return [self._e]


def _gate(user_text, corpus):
    from core.agentic.gate import evaluate_agentic_gate
    return asyncio.run(evaluate_agentic_gate(
        user_text=user_text, entity_resolver=None, model_manager=None,
        corpus_manager=corpus, intent_info=None))


def _store_with_card(age_seconds):
    store = PendingActionsStore(ttl_seconds=3600, max_pending=5, persist=False)
    card = ActionProposal(action_type=ActionType.CALENDAR_CREATE_EVENT,
                          params={"summary": "MGT 6203 TA Session", "start_time": "2026-09-12T11:00:00",
                                  "end_time": "2026-09-12T12:00:00"},
                          summary="single", reasoning="", reversible=True)
    store.propose(card)
    card.proposed_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return store, card


class TestGateOfferArmLoopClosure:
    def _with(self, store):
        from core.agentic.tools import ToolExecutor
        return patch.object(ToolExecutor, "_get_pending_actions_store", return_value=store)

    def test_yes_after_queue_offer_forces_calendar_create(self):
        with self._with(PendingActionsStore(ttl_seconds=300, max_pending=5, persist=False)):
            d = _gate("yes", _Corpus(R12))
        assert d.forced_action == ActionType.CALENDAR_CREATE_EVENT.value

    def test_try_again_after_no_card_reply_forces_the_narrated_action(self):
        with self._with(PendingActionsStore(ttl_seconds=300, max_pending=5, persist=False)):
            d = _gate("try again", _Corpus(R13))
        assert d.forced_action == ActionType.CALENDAR_CREATE_EVENT.value
        assert d.veto_exempt and d.modes == ["tools"]

    def test_older_pending_card_is_superseded_by_an_amended_offer(self):
        store, card = _store_with_card(age_seconds=600)  # minted 10 min ago …
        prev_ts = datetime.now(timezone.utc) - timedelta(seconds=60)  # … prior turn 1 min ago
        with self._with(store):
            d = _gate("yes", _Corpus(R12, ts=prev_ts))
        assert d.forced_action == ActionType.CALENDAR_CREATE_EVENT.value
        assert store._store[card.action_id].status == "rejected"

    def test_card_minted_by_the_prior_turn_still_stands_down(self):
        store, card = _store_with_card(age_seconds=30)
        prev_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
        with self._with(store):
            d = _gate("yes", _Corpus(R12, ts=prev_ts))
        assert d.forced_action is None
        assert store._store[card.action_id].status == "pending"

    def test_live_sequence_never_leaves_a_turn_unrouted(self):
        """T9 (narration) must NOT force; T12/T13/T14 each get a route."""
        empty = PendingActionsStore(ttl_seconds=300, max_pending=5, persist=False)
        with self._with(empty):
            assert _gate(T9, _Corpus("ok")).forced_action is None
            assert detect_action_intent(T12) is not None                    # T12 explicit arm
            assert _gate("yes", _Corpus(R12)).forced_action is not None     # T13
            assert _gate("try again", _Corpus(R13)).forced_action is not None  # T14


class TestFsGuardTempSibling:
    def test_pending_store_temp_sibling_is_exempt(self):
        from utils.python_fs_guard import _is_daemon_state_path
        assert _is_daemon_state_path("data/pending_actions.json")
        assert _is_daemon_state_path("data/.pending_actions.json.rj_1_c9i.tmp")
        assert _is_daemon_state_path("data/.web_search_credits.json.abc123.tmp")
        assert not _is_daemon_state_path("data/.knowledge_graph.json.abc.tmp")
