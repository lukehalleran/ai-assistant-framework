"""figure_check log-only siblings: cross_owner_date, unsupported_detail, stale_day_count.

Incidents: 2026-10-14 evening ("HW4 due the 25th" -- a date from another course's
calendar entry re-cited as the discussed assignment's deadline) and 2026-09-21
("one page on the career fair site", "from your phone" -- circumstances nobody
supplied riding on a supported user action); S1 day-count hand-off (2026-10-27).
Classes: BC-51, BC-46, BC-103. Fixture content is invented. Every reply fixture is
asserted in the clean AND the soft-wrapped/indented form.

Drives THE deployed ``figure_check.check_reply`` and the deployed
``handlers._apply_delivery_revisions`` (log rows only; the reply is never edited).
"""
from datetime import date, datetime
from types import SimpleNamespace

import pytest

import core.figure_check as fc
import gui.handlers as handlers
from config import app_config
from utils.streak_claims import DayCountClaim, episode_anchors

NOW = datetime(2026, 10, 21, 12, 0)
CLOCK = fc.SourceItem("clock", "", "clock", NOW, "clock")


def _wrap(text):
    """Soft-wrapped + indented form of a one-paragraph reply."""
    words = text.split(" ")
    mid = len(words) // 2
    return "    " + " ".join(words[:mid]) + "\n    " + " ".join(words[mid:])


def _forms(text):
    return [text, _wrap(text)]


def _doc(text, id_="src_attachment_syllabus", section="conversation", role="document"):
    return fc.SourceItem(id_, text, role, None, section)


def _user(text, id_="u1"):
    return fc.SourceItem(id_, text, "user", NOW, "conversation")


def _classes(reply, pool, anchors=None):
    report = fc.check_reply(reply, pool + [CLOCK], NOW, day_anchors=anchors)
    return [f.cls for f in report.findings if f.cls in
            ("cross_owner_date", "unsupported_detail", "stale_day_count")], report


# --------------------------------------------------------------------- cross_owner_date
SYLLABUS = _doc("CS 6400 Syllabus\nHW4 is due October 20.\nHW5 is due November 3.")
CALENDAR = _doc("CS 7000 Midterm review session October 25", id_="cal1", section="google_calendar")


@pytest.mark.parametrize("reply", _forms("HW4 is due the 25th, so start early, Rowan."))
def test_date_from_another_owner_is_logged(reply):
    got, report = _classes(reply, [SYLLABUS, CALENDAR])
    assert got == ["cross_owner_date"]
    f = next(f for f in report.findings if f.cls == "cross_owner_date")
    assert "date_from_other_owner" in f.reason and f.figure.text.strip().endswith("25th")


@pytest.mark.parametrize("reply", _forms("HW4 is due October 20, so start early, Rowan."))
def test_matching_owner_date_is_silent(reply):
    assert _classes(reply, [SYLLABUS, CALENDAR])[0] == []


@pytest.mark.parametrize("reply", _forms("HW4 is due the 25th, so start early, Rowan."))
def test_two_owners_sharing_the_token_is_unknown(reply):
    other = _doc("CS 7000 Syllabus\nHW4 is due October 22.", id_="src_attachment_other")
    assert _classes(reply, [SYLLABUS, other, CALENDAR])[0] == []


@pytest.mark.parametrize("reply", _forms("HW9 is due the 25th, so start early, Rowan."))
def test_token_absent_from_every_source_is_unknown(reply):
    assert _classes(reply, [SYLLABUS, CALENDAR])[0] == []


@pytest.mark.parametrize("reply", _forms("Is HW4 due the 25th? I can check, Rowan."))
def test_question_and_offer_are_out_of_scope(reply):
    assert _classes(reply, [SYLLABUS, CALENDAR])[0] == []


@pytest.mark.parametrize("reply", _forms("For CS 6400, HW4 is due the 25th, so start early."))
def test_reply_naming_the_course_restricts_to_that_owner(reply):
    # the reply names CS 6400; its HW4 row says Oct 20; the 25th belongs to CS 7000 -> logged
    assert _classes(reply, [SYLLABUS, CALENDAR])[0] == ["cross_owner_date"]
    assert _classes(reply, [CALENDAR])[0] == []   # no CS 6400 HW4 row at all -> unknown


def test_assistant_rows_are_not_sources():
    wrong = fc.SourceItem("a1", "HW4 is due the 25th.", "assistant", NOW, "conversation")
    got, _ = _classes("HW4 is due the 25th, Rowan.", [SYLLABUS, CALENDAR, wrong])
    assert got == ["cross_owner_date"]   # the earlier wrong reply cannot vouch for itself


# --------------------------------------------------------------------- unsupported_detail
BASE_USER = _user("I finished the application form yesterday, Tamsin helped me.")


@pytest.mark.parametrize("reply", _forms("You submitted the form on the career fair site from your phone, Rowan."))
def test_invented_location_and_medium_are_logged(reply):
    got, report = _classes(reply, [BASE_USER])
    assert got == ["unsupported_detail", "unsupported_detail"]
    texts = {" ".join(f.figure.text.split()) for f in report.findings if f.cls == "unsupported_detail"}
    assert texts == {"on the career fair site", "from your phone"}


@pytest.mark.parametrize("reply", _forms("You submitted the form on the career fair site from your phone, Rowan."))
def test_supported_circumstances_are_silent(reply):
    pool = [_user("I submitted it from my phone on the career fair site last night.")]
    assert _classes(reply, pool)[0] == []


@pytest.mark.parametrize("reply", _forms("You submitted the form on the career fair site from your phone, Rowan."))
def test_support_must_sit_in_one_source(reply):
    # "career fair" in one source and "site" in another does not support "the career fair site"
    pool = [_user("The career fair was fun, from my phone I took pictures.", "u1"),
            _user("The site was down.", "u2")]
    assert _classes(reply, pool)[0] == ["unsupported_detail"]


@pytest.mark.parametrize("reply", [
    "Could you submit it on the career fair site from your phone?",
    "You could submit it on the career fair site from your phone.",
    "Most students submit forms on the career fair site.",
    "Submit it on the career fair site from your phone.",
    "If you applied on the career fair site, you are set.",
    "You did well at the end.",
])
@pytest.mark.parametrize("wrap", [False, True])
def test_questions_offers_world_facts_imperatives_are_out_of_scope(reply, wrap):
    text = _wrap(reply) if wrap else reply
    assert _classes(text, [BASE_USER])[0] == []


# --------------------------------------------------------------------- stale_day_count
def _anchors():
    claim = DayCountClaim(count=23, stated_on=date(2026, 10, 11), subject="the gym streak",
                          lemmas=frozenset({"gym", "streak"}), snippet="23 days into the gym streak")
    return episode_anchors([claim])   # projected on 2026-10-21 -> day 33


@pytest.mark.parametrize("reply", _forms("You are 23 days into the gym streak, Rowan."))
def test_stale_count_is_logged(reply):
    got, report = _classes(reply, [], _anchors())
    assert got == ["stale_day_count"]
    f = next(f for f in report.findings if f.cls == "stale_day_count")
    assert "stated:23 projected:33" in f.reason


@pytest.mark.parametrize("n", [32, 33, 34])
@pytest.mark.parametrize("wrap", [False, True])
def test_projected_count_within_one_is_derived(n, wrap):
    reply = f"You are {n} days into the gym streak, Rowan."
    assert _classes(_wrap(reply) if wrap else reply, [], _anchors())[0] == []


def test_no_anchor_for_the_activity_is_silent():
    assert _classes("You are 23 days into the reading plan, Rowan.", [], _anchors())[0] == []
    assert _classes("You are 23 days into the gym streak, Rowan.", [], None)[0] == []


# --------------------------------------------------------------------- deployed path / config
def _ctx():
    return SimpleNamespace(
        user_text="status?", telemetry={}, personal_claim_mode="log_only",
        raw_context={"recent_conversations": [{"query": "I did the form", "response": "ok"}]},
        history=[], orchestrator=None)


@pytest.fixture
def seen(monkeypatch):
    got = []

    async def grounding(ctx, body, source_material=""):
        return None, ""

    async def personal(ctx, body):
        return None

    monkeypatch.setattr(handlers, "_apply_grounding_check_for_delivery", grounding)
    monkeypatch.setattr(handlers, "_apply_personal_claim_check_for_delivery", personal)
    monkeypatch.setattr(handlers.checker_findings, "record_finding",
                        lambda checker, **kw: got.append((checker, kw)) or True)
    return got


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap", [False, True])
async def test_deployed_pipeline_logs_but_never_edits(seen, wrap):
    body = "You submitted the form from your phone, Rowan."
    body = _wrap(body) if wrap else body
    final = await handlers._apply_delivery_revisions(_ctx(), body)
    assert final == body
    rows = [kw for c, kw in seen if c == "figure_check" and kw["verdict"] == "unsupported_detail"]
    assert rows and rows[0]["mode"] == "log_only"
    assert "phone" in rows[0]["claim_span"]


@pytest.mark.asyncio
async def test_toggle_same_instance(seen, monkeypatch):
    body = "You submitted the form from your phone, Rowan."
    monkeypatch.setattr(app_config, "FIGURE_CHECK_LOG_CLASSES", ("unsupported",))
    await handlers._apply_delivery_revisions(_ctx(), body)
    assert not [1 for _, kw in seen if kw["verdict"] == "unsupported_detail"]
    monkeypatch.setattr(app_config, "FIGURE_CHECK_LOG_CLASSES", ("unsupported", "unsupported_detail"))
    await handlers._apply_delivery_revisions(_ctx(), body)
    assert [1 for _, kw in seen if kw["verdict"] == "unsupported_detail"]


def test_deployed_defaults_and_effective_correct_classes():
    for cls in ("cross_owner_date", "unsupported_detail", "stale_day_count"):
        assert cls in app_config.FIGURE_CHECK_LOG_CLASSES
        assert cls not in app_config.FIGURE_CHECK_CORRECT_CLASSES
    assert tuple(app_config.FIGURE_CHECK_CORRECT_CLASSES) == ("elapsed_precise", "weekday_date")


def test_day_anchor_reader_is_read_only_and_disabled_by_toggle(monkeypatch):
    monkeypatch.setattr(app_config, "DAY_COUNTS_ENABLED", False)
    assert handlers._figure_check_day_anchors() == ()
