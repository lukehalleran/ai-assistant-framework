"""2026-09-10 referee follow-ups on commit 0aeffb7 (calendar action loop closure).

A referee review of the calendar forced-action-loop batch found three
follow-ups, all deterministic:

1. Fix 1 (class: BC-06 over-fire) — `core.action_claim_guard._KIND_PATTERNS`
   added "weekly|sessions?" to the CALENDAR pattern on 2026-09-10 so a
   kind-less calendar queue claim ("weekly through December 12") would be
   caught. But CALENDAR is checked before NOTE in priority order, so an
   ordinary NOTE offer that happens to mention "this session" ("Want me to
   save a note from this session?") or "weekly" ("add that to your weekly
   note?") now resolves to CALENDAR instead of NOTE — and
   `core.actions.registry.offer_action_type` would force a calendar-create
   round off a "yes".

2. Fix 2 (class: BC-30 prompt-only grounding, sibling-shape gap) —
   `core.actions.registry._pool_hours` (feeding `calendar_times_ungrounded`)
   missed two time shapes: (a) a bare hour after a time preposition with no
   meridiem/colon ("office hours at 3 on Fridays", "meets at 11", "from 9 to
   10", "9-10 on Saturdays"), and (b) an ISO timestamp embedded directly in
   the pool text ("start_time=2026-09-11T17:00:00", as rendered into action
   digests and the controller's own "[ACTION NOT PROPOSED]" note) — the "T"
   immediately before the digits defeats the `h2` arm's leading \\b.

3. Fix 3 (class: BC-30 prompt-only grounding gate too narrow) — the forced-
   round time-grounding check in `core.agentic.controller.run_agentic_search`
   was gated on `_this_round_forced_type` only. After a decline
   (`session._action_force_declined = True`) the loop continues UNFORCED and
   a later calendar_create_event decision in the SAME session was never
   checked against the pool again — the model could re-propose the same
   guessed time unchecked. `_should_ground_calendar_times(session,
   this_round_forced_type)` extracts the gate predicate so it can be tested
   directly.

Every test below calls the deployed function — no re-derivation.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.action_claim_guard import ActionKind, detect_kind
from core.actions.registry import calendar_times_ungrounded, offer_action_type
from core.agentic.controller import _should_ground_calendar_times

# Reuse the exact live fixtures from the 2026-09-10 batch so Fix 2 stays
# consistent with the already-shipped negative/positive cases.
T9 = ("alright, i know there is session on Friday in the evening, but can we check "
      "course docs and see if there a weekend back up? I only put professors hours in "
      "calander but there are TA sessions too")
POOL_T9 = T9 + ("\nZoom at 3:30 your time ... 345 ... before 730A, 845-1100A, 130-230P, "
                "430-600P, after 830P ... 8:00 PM – 9:00 PM office hours ... 1264 rows ... "
                "2026-09-11 ... 2026-09-13 deadline")


# ---------------------------------------------------------------------------
# Fix 1 — CALENDAR weak-word shadow over NOTE offers
# ---------------------------------------------------------------------------
class TestCalendarWeakWordDoesNotShadowNote:
    def test_note_from_this_session_is_a_note(self):
        assert detect_kind("Want me to save a note from this session?") == ActionKind.NOTE

    def test_weekly_note_is_a_note(self):
        assert detect_kind("Want me to add that to your weekly note?") == ActionKind.NOTE

    def test_jot_it_down_for_this_session_is_a_note(self):
        assert detect_kind("Want me to jot that down as a note for this session?") == ActionKind.NOTE

    def test_note_offer_with_session_word_is_not_forced_external(self):
        # NOTE offers are handled by the (internal, self-repairable)
        # PendingProposalStore path, never by the external forced-action
        # route — offer_action_type must not resolve this to CALENDAR.
        assert offer_action_type("Want me to jot that down as a note for this session?") is None

    def test_still_calendar_named_session_slot(self):
        # The exact live 2026-09-10 shape this pattern was added FOR must
        # keep working: a named recurring session/office-hours slot with no
        # "calendar"/"event" word, and no NOTE word either.
        r14 = ("Queuing it now: **MGT 6203 TA Session — Saturdays 11:00 AM–12:00 PM CT, "
               "weekly through December 12**, Zoom link https://example.zoom.us/j/000 "
               "attached. You should see the approval card pop up — if it doesn't show "
               "this time either, tell me.")
        assert detect_kind(r14) == ActionKind.CALENDAR

    def test_still_calendar_recurring_event_offer(self):
        assert detect_kind("Want me to make it a recurring event?") == ActionKind.CALENDAR

    def test_calendar_word_alone_still_wins_over_absent_note(self):
        assert detect_kind("Want me to add that appointment to your calendar?") == ActionKind.CALENDAR


# ---------------------------------------------------------------------------
# Fix 2 — bare-hour / range / ISO-in-pool grounding gaps
# ---------------------------------------------------------------------------
class TestBareHourAndIsoPoolGrounding:
    def test_bare_hour_after_at_grounds_both_readings(self):
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-12T15:00:00", "end_time": "2026-09-12T15:30:00"},
            "office hours at 3 on Fridays") == []
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-12T03:00:00"},
            "office hours at 3 on Fridays") == []

    def test_meets_at_11_grounds_both_readings(self):
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-12T11:00:00"}, "meets at 11") == []
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-12T23:00:00"}, "meets at 11") == []

    def test_from_9_to_10_grounds_both_ends_and_pm_readings(self):
        pool = "from 9 to 10"
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-11T09:00:00", "end_time": "2026-09-11T10:00:00"}, pool) == []
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-11T21:00:00", "end_time": "2026-09-11T22:00:00"}, pool) == []

    def test_bare_range_with_hyphen_grounds_both_ends(self):
        pool = "9-10 on Saturdays"
        assert calendar_times_ungrounded({"start_time": "2026-09-13T09:00:00"}, pool) == []
        assert calendar_times_ungrounded({"start_time": "2026-09-13T21:00:00"}, pool) == []

    def test_iso_timestamp_embedded_in_pool_text_grounds_itself(self):
        pool = "action digest: forced calendar_create_event start_time=2026-09-11T17:00:00 end_time=2026-09-11T19:00:00"
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-11T17:00:00", "end_time": "2026-09-11T19:00:00"}, pool) == []

    def test_action_not_proposed_note_shape_grounds_the_declined_time(self):
        # Exact shape the controller appends via _append_accumulated when it
        # declines a guessed time — a LATER unforced round proposing the
        # SAME time (now legitimately sourced from that note) must ground.
        note = ("[ACTION NOT PROPOSED] forced calendar_create_event not proposed: "
                "start_time=2026-09-11T17:00:00 appears nowhere in the request or "
                "gathered context — a guessed time is worse than no card")
        assert calendar_times_ungrounded({"start_time": "2026-09-11T17:00:00"}, note) == []

    @pytest.mark.parametrize("pool", [
        "1264 rows",
        "2026",
        "30 mg",
        "in 5 days",
    ])
    def test_unit_suffixed_numbers_and_bare_years_never_ground(self, pool):
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-11T20:26:00", "end_time": "2026-09-11T21:00:00"}, pool) != []

    def test_live_pool_t9_still_flags_the_invented_17_00(self):
        # Regression guard: the new arms must not accidentally ground the
        # exact invented time this whole feature exists to catch.
        bad = calendar_times_ungrounded(
            {"summary": "TA", "start_time": "2026-09-11T17:00:00", "end_time": "2026-09-11T19:00:00"},
            POOL_T9)
        assert bad == ["start_time=2026-09-11T17:00:00", "end_time=2026-09-11T19:00:00"]

    def test_year_and_row_counts_still_never_ground_a_time(self):
        assert calendar_times_ungrounded(
            {"start_time": "2026-09-11T20:26:00", "end_time": "2026-09-11T21:00:00"},
            "due 2026-09-11, 1264 rows") != []


# ---------------------------------------------------------------------------
# Fix 3 — grounding gate must also fire post-decline, unforced
# ---------------------------------------------------------------------------
class TestShouldGroundCalendarTimesGate:
    def test_forced_round_always_grounds(self):
        session = SimpleNamespace(_action_force_declined=False)
        assert _should_ground_calendar_times(session, "calendar_create_event") is True

    def test_unforced_round_with_no_prior_decline_does_not_ground(self):
        session = SimpleNamespace(_action_force_declined=False)
        assert _should_ground_calendar_times(session, None) is False

    def test_unforced_round_missing_the_decline_attr_does_not_ground(self):
        session = SimpleNamespace()
        assert _should_ground_calendar_times(session, None) is False

    def test_unforced_round_after_a_prior_decline_still_grounds(self):
        session = SimpleNamespace(_action_force_declined=True)
        assert _should_ground_calendar_times(session, None) is True

    def test_forced_round_after_a_prior_decline_still_grounds(self):
        session = SimpleNamespace(_action_force_declined=True)
        assert _should_ground_calendar_times(session, "calendar_create_event") is True

    def test_wired_into_run_agentic_search(self):
        import inspect
        from core.agentic.controller import AgenticSearchController
        src = inspect.getsource(AgenticSearchController.run_agentic_search)
        assert "_should_ground_calendar_times(session" in src
        assert "calendar_times_ungrounded" in src
        assert "_action_force_declined" in src
        assert "[ACTION NOT PROPOSED]" in src
