"""Figure provenance check: every in-scope number/time is copied, derived or unsupported.

Incident 2026-10-25 16:07 (lane FIGB): "how wrecked you were feeling an hour ago" said
one minute after the reply that used the word. Class: BC-103 (the model is left to compute
a quantity the code already knows), BC-46, BC-58. Fixture values are invented.
"""
from datetime import datetime, timedelta

import pytest

import core.figure_check as fc

NOW = datetime(2026, 10, 25, 16, 7)   # a Sunday


def _item(text, age, role="assistant", id_=None):
    return fc.SourceItem(id_ or f"i{abs(hash((text, age))) % 10**6}", text, role,
                         None if age is None else NOW - age, "conversation")


CLOCK = fc.SourceItem("clock", "", "clock", NOW, "clock")


def _forms(text):
    """Clean AND wrapped/indented form of a live-text fixture."""
    wrapped = "    " + text.replace(" feeling ", "\n    feeling ", 1)
    return [text, wrapped]


def _check(reply, pool):
    report = fc.check_reply(reply, pool + [CLOCK], NOW)
    return report, fc.apply_corrections(reply, report)


@pytest.mark.parametrize("reply", _forms("Given how wrecked you were feeling an hour ago, take care of Rowan."))
def test_incident_hour_ago_against_one_minute_old_item_is_corrected(reply):
    pool = [_item("I'm sorry you feel wrecked, Tamsin.", timedelta(minutes=1))]
    report, (revised, n) = _check(reply, pool)
    assert n == 1
    assert "a minute ago" in revised and "an hour ago" not in revised
    assert [f.cls for f in report.findings] == ["elapsed_precise"]


def test_capitalisation_is_preserved():
    reply = "An hour ago you felt wrecked."
    pool = [_item("You feel wrecked tonight.", timedelta(minutes=1))]
    revised, n = _check(reply, pool)[1]
    assert n == 1 and revised.startswith("A minute ago")
    pool = [_item("You feel wrecked tonight.", timedelta(minutes=4))]
    assert _check(reply, pool)[1][0].startswith("Four minutes ago")


@pytest.mark.parametrize("reply", _forms("Given how wrecked you were feeling an hour ago, take care."))
def test_consistent_hour_is_untouched(reply):
    pool = [_item("I'm sorry you feel wrecked.", timedelta(minutes=55))]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].cls == "derived:elapsed"


def test_two_referents_far_apart_is_ambiguous_and_unchanged():
    reply = "Given how wrecked you were feeling an hour ago, take care."
    pool = [_item("You sound wrecked.", timedelta(minutes=1), id_="new"),
            _item("You were wrecked.", timedelta(days=3), id_="old")]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].cls == "elapsed_unconfirmed"
    assert report.findings[0].reason == "ambiguous_referents"


def test_vague_this_morning_against_evening_item_is_logged_not_edited():
    reply = "You said you felt wrecked this morning."
    pool = [_item("I felt wrecked.", timedelta(hours=1), role="user")]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].cls == "elapsed_vague"


def test_world_fact_is_out_of_scope():
    reply = "Withdrawal typically resolves within 7 days for most people."
    report, (revised, n) = _check(reply, [_item("You took 40 mg.", timedelta(hours=2), role="user")])
    assert (revised, n) == (reply, 0)
    assert {f.cls for f in report.findings} == {"out_of_scope"}


@pytest.mark.parametrize("reply", _forms("You said you took 45 mg of it feeling fine."))
def test_dose_copied_wrong_is_logged_not_edited(reply):
    pool = [_item("I took 40 mg of it today.", timedelta(hours=2), role="user")]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert "copied_mismatch" in [f.cls for f in report.findings]


def test_dose_copied_right_and_unit_converted_are_not_findings():
    pool = [_item("I took 40 mg of it today.", timedelta(hours=2), role="user")]
    assert [f.cls for f in _check("You said you took 40 mg of it.", pool)[0].findings] == ["copied"]
    assert [f.cls for f in _check("You said you took 0.04 g of it.", pool)[0].findings] == ["derived:unit_conversion"]


def test_derived_gap_between_two_items():
    reply = "That's 3 days between your two sleep notes."
    pool = [_item("Sleep note: slept badly.", timedelta(days=10), role="user"),
            _item("Sleep note: slept fine.", timedelta(days=7), role="user")]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].cls == "derived:elapsed"


def test_derived_sum_of_two_pool_values():
    pool = [_item("I took 40 mg in the morning.", timedelta(hours=8), role="user"),
            _item("I took 20 mg at night.", timedelta(hours=2), role="user")]
    report, _ = _check("You took 60 mg today.", pool)
    assert report.findings[0].cls == "derived:sum_difference"


def test_derived_list_count():
    pool = [fc.SourceItem("count:tasks", "", "meta", None, "tasks", count=4)]
    report, _ = _check("You have 4 tasks open.", pool)
    assert report.findings[0].cls == "derived:list_count"


@pytest.mark.parametrize("reply", [
    "Your deadline is Tuesday October 26, 2026.",
    "    Your deadline is Tuesday\n    October 26, 2026.",
])
def test_weekday_date_mismatch_is_corrected(reply):
    report, (revised, n) = _check(reply, [])
    assert n == 1
    assert "Tuesday" not in revised and "Monday" in revised
    assert [f.cls for f in report.findings] == ["weekday_date"]


def test_correct_weekday_date_is_left_alone():
    reply = "Your deadline is Monday October 26, 2026."
    assert fc.check_reply(reply, [CLOCK], NOW).findings == []


def test_referent_that_says_the_same_phrase_makes_it_a_copy():
    reply = "You told me you took the tablet two hours ago."
    pool = [_item("I took the tablet two hours ago.", timedelta(minutes=30), role="user")]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].cls == "copied"


def test_only_classes_in_correct_classes_are_edited():
    reply = "Given how wrecked you were feeling an hour ago, take care."
    pool = [_item("I'm sorry you feel wrecked.", timedelta(minutes=1))]
    report = fc.check_reply(reply, pool + [CLOCK], NOW)
    assert fc.apply_corrections(reply, report, correct_classes=())[1] == 0
    assert fc.apply_corrections(reply, report, correct_classes=("weekday_date",))[1] == 0


def test_no_nlp_degrades_but_still_corrects_incident():
    reply = "Given how wrecked you were feeling an hour ago, take care."
    pool = [_item("I'm sorry you feel wrecked.", timedelta(minutes=1))]
    report = fc.check_reply(reply, pool + [CLOCK], NOW, nlp=None)
    assert report.nlp is False
    assert fc.apply_corrections(reply, report)[1] == 0   # attachment is unknown without a parse: never edited


def test_build_source_pool_reads_timestamped_conversation_rows():
    ctx = {"recent_conversations": [{"query": "rough night", "response": "I'm sorry you feel wrecked.",
                                     "timestamp": (NOW - timedelta(minutes=1)).isoformat()}],
           "facts": ["likes tea", "walks daily"]}
    pool = fc.build_source_pool(ctx, [], "tool says 3 results", NOW, user_text="hello")
    roles = {i.role for i in pool}
    assert {"user", "assistant", "profile", "tool", "clock", "meta"} <= roles
    reply = "Given how wrecked you were feeling an hour ago, take care."
    assert fc.apply_corrections(reply, fc.check_reply(reply, pool, NOW))[1] == 1


def test_weak_evidence_never_corrects():
    """Measured false corrections: one shared lemma, no person in the clause, a referent that narrates times."""
    one_lemma = [_item("You sound wrecked.", timedelta(minutes=1))]
    reply = "Given how wrecked you were feeling an hour ago, take care."
    report, (revised, n) = _check(reply, one_lemma)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].reason == "weak_binding"
    no_person = "The wrecked feeling from an hour ago is common."
    report, (revised, n) = _check(no_person, [_item("You feel wrecked.", timedelta(minutes=1))])
    assert n == 0 and report.findings[0].reason == "elapsed_not_about_user"
    narrates = [_item("You felt wrecked yesterday.", timedelta(minutes=1))]
    report, (revised, n) = _check("Given how wrecked you were feeling an hour ago, take care.", narrates)
    assert n == 0 and report.findings[0].reason == "referent_narrates_time"


REAL_SHAPE_USER = "Honestly I feel awful, couldn't sleep and my head is pounding."
REAL_SHAPE_ASSISTANT = (
    "I'm sorry, that sounds rough, especially when you're already wrecked. Headache and fatigue "
    "typically start 12\u201324 hours out and resolve within about a week. Rest if you can.")


@pytest.mark.parametrize("reply", [
    "Hey \u2014 good to hear from you. Given how wrecked you were feeling an hour ago, how are you holding up now?",
    "    Hey \u2014 good to hear from you. Given how wrecked you\n    were feeling an hour ago, how are you holding up now?",
])
def test_real_incident_shape_with_durations_in_referent_is_corrected(reply):
    stamp = NOW - timedelta(seconds=72)
    pool = [fc.SourceItem("u", REAL_SHAPE_USER, "user", stamp, "conversation"),
            fc.SourceItem("a", REAL_SHAPE_ASSISTANT, "assistant", stamp, "conversation")]
    report, (revised, n) = _check(reply, pool)
    assert n == 1, [(f.cls, f.reason, f.overlap) for f in report.findings]
    assert "an hour ago" not in revised and "a minute ago" in revised


def test_time_units_and_number_words_never_bind():
    nlp = fc._get_nlp()
    doc = nlp("You felt wrecked three hours and two days ago, Rowan.")
    lem = fc._content_lemmas_nlp(doc, [])
    assert not lem & {"hour", "day", "three", "two"} and "wreck" in lem


def test_only_event_anchored_times_make_a_referent_narrate():
    assert fc._event_anchored_times("start 12\u201324 hours out, resolve within about a week, for 3 days") == []
    assert fc._event_anchored_times("It began yesterday") and fc._event_anchored_times("On Monday we met")


def test_f5_false_correction_shapes_stay_uncorrected():
    """Measured false corrections on stored replies: a 'yesterday' bound to a 65-minute-old turn,
    and clauses without a first/second-person subject."""
    old = [_item("We talked about moving the boxes into storage.", timedelta(minutes=65))]
    reply = "You mentioned moving the boxes into storage yesterday."
    assert _check(reply, old)[1] == (reply, 0)
    for text in ("The storage boxes moved a minute ago.", "The update shipped yesterday for the storage boxes."):
        assert _check(text, [_item("Storage boxes arrived; moving them.", timedelta(hours=22))])[1][1] == 0


CLAUSE_REFERENT = ("Then yesterday's dose isn't a wild spike, more like your workout-day dose landing "
                   "on a rest day when you're already wrecked. Keep an eye on how you feel.")


@pytest.mark.parametrize("reply", [
    "Hey. Given how wrecked you were feeling an hour ago, how are you holding up now?",
    "    Hey. Given how wrecked you\n    were feeling an hour ago, how are you holding up now?",
])
def test_time_in_another_clause_of_the_referent_does_not_block(reply):
    stamp = NOW - timedelta(seconds=72)
    pool = [fc.SourceItem("u", "I feel awful and my dose confuses me.", "user", stamp, "conversation"),
            fc.SourceItem("a", CLAUSE_REFERENT, "assistant", stamp, "conversation")]
    report, (revised, n) = _check(reply, pool)
    assert n == 1, [(f.cls, f.reason, f.overlap) for f in report.findings]
    assert "a minute ago" in revised


def test_time_in_the_same_clause_as_a_bound_lemma_blocks():
    stamp = NOW - timedelta(seconds=72)
    pool = [fc.SourceItem("a", "You felt wrecked yesterday and still do.", "assistant", stamp, "conversation"),
            fc.SourceItem("u", "I feel awful.", "user", stamp, "conversation")]
    reply = "Given how wrecked you were feeling an hour ago, take care."
    report, (revised, n) = _check(reply, pool)
    assert n == 0 and report.findings[0].reason == "referent_narrates_time"


@pytest.mark.parametrize("reply", [
    "You may have mislabeled it in your notes from yesterday, Rowan.",
    "    You may have mislabeled it in your notes\n    from yesterday, Rowan.",
    "You've been grinding on the storage boxes since yesterday.",
    "    You've been grinding on the\n    storage boxes since yesterday.",
])
def test_noun_attached_and_duration_boundary_times_are_never_corrected(reply):
    pool = [_item("Rowan, your storage boxes and notes need labels; you mislabeled the notes while grinding.",
                  timedelta(minutes=4))]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)


def test_day_scale_deictic_contradicted_is_reported_not_edited():
    reply = "Given how wrecked you were feeling yesterday, take care."
    pool = [_item("You feel wrecked.", timedelta(minutes=3))]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0)
    assert report.findings[0].cls == "elapsed_unconfirmed" and report.findings[0].reason == "day_scale_not_edited"


def test_moment_phrase_is_never_rewritten_to_a_day_scale_age():
    reply = "You said you felt wrecked just now, Rowan."
    pool = [_item("You feel wrecked, Rowan.", timedelta(hours=22))]
    report, (revised, n) = _check(reply, pool)
    assert (revised, n) == (reply, 0) and report.findings[0].reason == "age_not_moment_scale"
