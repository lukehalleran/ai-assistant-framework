"""Claim-ledger extractor prompt + deterministic validator (memory/claim_ledger.py).

Incident 2026-10-20: the user's CONDITIONAL "if I can get 6 done today then I
am only 6 days behind" was later restated as "after being six days behind", and
the 10-18 ESTIMATE "12 days i think?" was superseded on 10-20 by ~fifteen days
yet still quoted. Classes: BC-75, BC-52 (BC-84: one bad element never discards
the rest).

Fake model JSON only; wording is invented but has the same shape. Every text is
asserted in BOTH its clean and its wrapped/indented form.
"""
from __future__ import annotations

import json

import pytest

from memory import claim_ledger as cl

CONDITIONAL_CLEAN = ("Quick update on the module. If I can get 6 done today then I am only 6 days "
                     "behind on the reading.")
CONDITIONAL_WRAPPED = ("Quick update on the module. If I can get 6 done\n    today then I am only 6 "
                       "days\n    behind on the reading.")
ESTIMATE_CLEAN = "Covered 7 weeks of the material over the last like. 12 days i think?"
ESTIMATE_WRAPPED = "Covered 7 weeks of the material\n   over the last like.\n   12 days i think?"
WORKED_CLEAN = "Worked it out properly: so about fifteen days of work in total."
WORKED_WRAPPED = "Worked it out\n   properly: so about\n   fifteen days of work in total."


def payload(*claims):
    return json.dumps({"claims": list(claims)})


def raw(kind="duration", key="days_behind", value="6", unit="days", modality="asserted",
        quote="then I am only 6 days behind", new_subject=True, description="days behind on the reading"):
    return {"kind": kind, "subject_key": key, "new_subject": new_subject, "description": description,
            "value": value, "unit": unit, "modality": modality, "quote": quote}


@pytest.mark.parametrize("text", [CONDITIONAL_CLEAN, CONDITIONAL_WRAPPED], ids=["clean", "wrapped"])
def test_conditional_incident_model_says_asserted_validator_forces_conditional(text):
    claims, drops = cl.validate_claims(payload(raw()), text, "2026-10-17T10:00:00")
    assert drops == []
    assert [(c.value, c.modality) for c in claims] == [("6", "conditional")]


@pytest.mark.parametrize("text", ["I am only 6 days behind on the reading.",
                                  "I am only 6 days\n    behind on the reading."], ids=["clean", "wrapped"])
def test_same_figure_stated_plainly_stays_asserted(text):
    claims, _ = cl.validate_claims(payload(raw(quote="I am only 6 days behind on the reading")),
                                   text, "2026-10-17T10:00:00")
    assert [c.modality for c in claims] == ["asserted"]


def test_validator_never_upgrades_a_model_conditional_or_estimate():
    text = "I am only 6 days behind on the reading."
    for modality in ("conditional", "estimate"):
        claims, _ = cl.validate_claims(payload(raw(modality=modality, quote="I am only 6 days behind")),
                                       text, "2026-10-17T10:00:00")
        assert claims[0].modality == modality


@pytest.mark.parametrize("est,worked", [(ESTIMATE_CLEAN, WORKED_CLEAN), (ESTIMATE_WRAPPED, WORKED_WRAPPED)],
                         ids=["clean", "wrapped"])
def test_estimate_then_worked_out_value_current_is_fifteen(est, worked):
    ledger = cl.ClaimLedger()
    c1, d1 = cl.validate_claims(
        payload(raw(key="sprint_span", value="12", modality="estimate", quote="12 days i think?",
                    description="length of the sprint so far")),
        est, "2026-10-18T09:00:00")
    assert d1 == [] and c1[0].modality == "estimate"
    for c in c1:
        ledger.add(c)
    subjects = ledger.open_subjects(as_of="2026-10-20T02:47:00")
    c2, d2 = cl.validate_claims(
        payload(raw(key="sprint_span", value="15", unit="days", new_subject=False,
                    quote="about fifteen days of work in total", description="length of the sprint so far")),
        worked, "2026-10-20T02:47:00", subjects)
    assert d2 == [] and c2[0].value == "15" and c2[0].modality == "asserted"
    assert ledger.add(c2[0]) == [c1[0].id]
    assert ledger.current("sprint_span").value == "15"


def test_quote_not_in_text_is_dropped():
    claims, drops = cl.validate_claims(payload(raw(quote="I am 6 days behind")), CONDITIONAL_CLEAN,
                                       "2026-10-17T10:00:00")
    assert claims == [] and [d["reason"] for d in drops] == ["quote_not_in_text"]


def test_value_not_in_quote_is_dropped():
    claims, drops = cl.validate_claims(payload(raw(value="9")), CONDITIONAL_CLEAN, "2026-10-17T10:00:00")
    assert claims == [] and [d["reason"] for d in drops] == ["value_not_in_quote"]


def test_unknown_reused_key_becomes_a_new_subject():
    claims, drops = cl.validate_claims(
        payload(raw(key="no_such_key", new_subject=False, quote="I am only 6 days behind")),
        "I am only 6 days behind.", "2026-10-17T10:00:00",
        [{"subject_key": "other_subject", "description": "x"}])
    assert drops == [] and claims[0].subject_key == "no_such_key"


def test_one_bad_claim_does_not_drop_the_good_one():
    text = "I am only 6 days behind on the reading and I have 3 chapters left."
    good = raw(quote="I am only 6 days behind")
    bad_quote = raw(key="chapters_left", kind="count", value="3", unit="chapters", quote="4 chapters left")
    bad_kind = dict(raw(key="x_thing"), kind="mood")
    good2 = raw(key="chapters_left", kind="count", value="3", unit="chapters", quote="3 chapters left",
                description="chapters left")
    claims, drops = cl.validate_claims(payload(bad_quote, good, bad_kind, "junk", good2), text,
                                       "2026-10-17T10:00:00")
    assert sorted(c.subject_key for c in claims) == ["chapters_left", "days_behind"]
    assert sorted(d["reason"] for d in drops) == ["bad_kind", "not_an_object", "quote_not_in_text"]


@pytest.mark.parametrize("bad", ["", "not json", '{"claims": 3}', '{"stuff": []}', "[]", '{"claims": [',
                                 "```json\n{\"claims\": []"])
def test_malformed_model_output_yields_no_claims_and_a_shape_drop(bad):
    claims, drops = cl.validate_claims(bad, CONDITIONAL_CLEAN, "2026-10-17T10:00:00")
    assert claims == [] and drops and drops[0]["reason"].startswith("shape=")


def test_fenced_json_is_accepted():
    claims, _ = cl.validate_claims("```json\n" + payload(raw(quote="I am only 6 days behind")) + "\n```",
                                   "I am only 6 days behind.", "2026-10-17T10:00:00")
    assert len(claims) == 1


def test_date_value_resolved_against_said_on():
    text = "HW 7 is due Friday so I have to finish it by then."
    claims, drops = cl.validate_claims(
        payload(raw(kind="deadline", key="hw7_due", value="2026-10-23", unit="", quote="HW 7 is due Friday")),
        text, "2026-10-20T10:00:00")
    # "Friday" has no resolver rule (no weekday-only expression): the value is NOT located -> dropped
    assert claims == [] and drops[0]["reason"] == "value_not_in_quote"
    text2 = "HW 7 is due October 23 so I have to finish it by then."
    claims, drops = cl.validate_claims(
        payload(raw(kind="deadline", key="hw7_due", value="2026-10-23", unit="", quote="HW 7 is due October 23")),
        text2, "2026-10-20T10:00:00")
    assert drops == [] and claims[0].value == "2026-10-23"


def test_prepare_user_text_excludes_probes_and_framing_survives():
    reason, _ = cl.prepare_user_text("[test]I am only 6 days behind.[/test]")
    assert reason == "test_probe"
    assert cl.prepare_user_text("")[0] == "empty"
    reason, text = cl.prepare_user_text(CONDITIONAL_WRAPPED)
    assert reason == "" and "6" in text


def test_extraction_prompt_lists_subjects_and_asks_for_strict_json():
    system, user = cl.build_extraction_prompt(
        CONDITIONAL_CLEAN, "2026-10-17T10:00:00",
        [{"subject_key": "sprint_span", "description": "length of the sprint", "value": "15", "unit": "days",
          "modality": "asserted", "stated_at": "2026-10-20T02:47:00"}])
    assert '{"claims"' in system and "conditional" in system and "estimate" in system
    assert '"key":"sprint_span"' in user and '"what":"length of the sprint"' in user
    assert "<<<" in user and "MESSAGE DATE: 2026-10-17" in user


# --- AMEND1 (2026-10-20 trial): identifier vs quantity, units, one claim per figure, quote fold ---------


def wrap(text: str) -> str:
    """The wrapped/indented form a client sends: line breaks + indentation between words."""
    words = text.split(" ")
    return "\n    ".join(" ".join(words[i:i + 3]) for i in range(0, len(words), 3))


def _count(quote, value, unit, key="assignments_finished", modality="asserted", kind="count"):
    return raw(kind=kind, key=key, value=value, unit=unit, modality=modality, quote=quote,
               description="items finished")


@pytest.mark.parametrize("fixture", [
    ("I finished HW 4 and it went fine.", "HW 4", "4", "assignments"),
    ("Week 5 Homework is posted already.", "Week 5 Homework", "5", "homework"),
    ("Question 3 was the hard one today.", "Question 3", "3", "questions"),
], ids=["hw", "week", "question"])
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_a1_a_number_that_names_an_item_is_not_a_count(fixture, form):
    text, quote, value, unit = fixture
    text = text if form == "clean" else wrap(text)
    claims, drops = cl.validate_claims(payload(_count(quote, value, unit)), text, "2026-10-17T10:00:00")
    assert claims == [] and [d["reason"] for d in drops] == ["identifier_number"]


@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_a1_a_real_count_with_the_same_number_is_admitted(form):
    text = "I finished 4 assignments this week and it went fine."
    text = text if form == "clean" else wrap(text)
    claims, drops = cl.validate_claims(payload(_count("I finished 4 assignments this week", "4", "assignments")),
                                       text, "2026-10-17T10:00:00")
    assert drops == [] and [(c.value, c.unit_norm) for c in claims] == [("4", "assignment")]


@pytest.mark.parametrize("fixture", [
    ("I have this so far 10.", "I have this so far 10", "10", "problems"),
    ("I am only 6 behind on the reading.", "I am only 6 behind", "6", "days"),
    ("I answered 10 of them.", "I answered 10", "10", ""),
], ids=["no_unit_after", "unit_absent", "empty_unit"])
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_a1_count_needs_its_unit_after_the_number(fixture, form):
    text, quote, value, unit = fixture
    text = text if form == "clean" else wrap(text)
    claims, drops = cl.validate_claims(payload(_count(quote, value, unit)), text, "2026-10-17T10:00:00")
    assert claims == [] and [d["reason"] for d in drops] == ["unit_not_in_quote"]


def test_a1_unit_must_come_after_the_number_not_before():
    # "days" appears BEFORE the number only: not a unit of this figure
    claims, drops = cl.validate_claims(
        payload(raw(quote="days are tough, I have 6 left", value="6", unit="days")),
        "days are tough, I have 6 left.", "2026-10-17T10:00:00")
    assert claims == [] and drops[0]["reason"] == "unit_not_in_quote"


def test_a2_unit_is_lemmatised_and_dates_carry_no_unit():
    claims, _ = cl.validate_claims(
        payload(raw(quote="I am only 6 days behind", unit="Days")), "I am only 6 days behind.",
        "2026-10-17T10:00:00")
    assert (claims[0].unit, claims[0].unit_norm) == ("days", "day")
    claims, drops = cl.validate_claims(
        payload(raw(kind="deadline", key="hw7_due", value="2026-10-23", unit="days",
                    quote="HW 7 is due October 23")),
        "HW 7 is due October 23 so I have to finish.", "2026-10-20T10:00:00")
    assert drops == [] and (claims[0].unit, claims[0].unit_norm) == ("", "")


def test_a2_extraction_prompt_names_the_measured_quantity_with_three_subjects():
    system, user = cl.build_extraction_prompt(
        CONDITIONAL_CLEAN, "2026-10-17T10:00:00",
        [{"subject_key": "days_behind_class", "description": "days behind", "kind": "duration",
          "value": "6", "unit": "days", "modality": "asserted", "stated_at": "2026-10-17T10:00:00"}])
    assert "WHAT is measured" in system and "days_behind_reading" in system
    assert "days_to_catch_up" in system and "chapters_finished" in system
    assert "different units are never the same subject" in system
    assert "HW 4" in system and "identifier" in system
    assert '"key":"days_behind_class"' in user and '"kind":"duration"' in user


@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_a3_one_claim_per_figure_each_validated_alone(form):
    text = ESTIMATE_CLEAN if form == "clean" else ESTIMATE_WRAPPED
    material = raw(kind="duration", key="material_covered", value="7", unit="weeks",
                   quote="Covered 7 weeks of the material", description="material covered")
    spent = raw(kind="duration", key="days_spent_on_material", value="12", unit="days", modality="estimate",
                quote="12 days i think?", description="days spent")
    claims, drops = cl.validate_claims(payload(material, spent), text, "2026-10-18T09:00:00")
    assert drops == []
    assert [(c.subject_key, c.value, c.unit_norm, c.modality) for c in claims] == [
        ("material_covered", "7", "week", "asserted"), ("days_spent_on_material", "12", "day", "estimate")]
    # one bad figure does not take the other down
    bad = dict(spent, value="13")
    claims, drops = cl.validate_claims(payload(material, bad), text, "2026-10-18T09:00:00")
    assert [c.subject_key for c in claims] == ["material_covered"]
    assert [d["reason"] for d in drops] == ["value_not_in_quote"]
    system, _ = cl.build_extraction_prompt(text, "2026-10-18T09:00:00", [])
    assert "One claim per figure" in system and "material_covered" in system and "days_spent_on_material" in system


FOLD_CASES = {
    "apostrophe": ("I don't know but I'm only 6 days behind", "I’m only 6 days behind"),
    "double_quotes": ('He said "only 6 days behind" to me', "“only 6 days behind”"),
    "dash": ("it was 6 days - maybe less", "6 days — maybe less"),
    "ellipsis": ("It took 6 days... maybe less", "6 days… maybe less"),
    "case": ("I am Only 6 Days Behind today", "i am only 6 days behind"),
}


@pytest.mark.parametrize("name", sorted(FOLD_CASES))
@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_a6_quote_fold_accepts_typography_and_case_but_stays_exact(name, form):
    text, model_quote = FOLD_CASES[name]
    shown = text if form == "clean" else wrap(text)
    item = raw(quote=model_quote.strip('"“”'), value="6", unit="days")
    claims, drops = cl.validate_claims(payload(item), shown, "2026-10-17T10:00:00")
    assert drops == [], (name, drops)
    assert claims[0].quote in cl.normalize_ws(shown)          # stored quote is the ORIGINAL-case span
    # mutation of one word is still dropped: no fuzzy matching
    broken = dict(item, quote=item["quote"].replace("6 days", "6 daze"))
    claims, drops = cl.validate_claims(payload(broken), shown, "2026-10-17T10:00:00")
    assert claims == [] and drops[0]["reason"] == "quote_not_in_text"


def test_a6_fold_text_maps_back_to_original_case():
    base, folded, idx = cl.fold_text("A ‘Quoted’   … Span")
    assert base == "A ‘Quoted’ … Span"
    assert folded == "a 'quoted' ... span" and len(idx) == len(folded)
    found = cl.locate_quote("A ‘Quoted’   … Span", "'quoted' ... SPAN")
    assert found is not None and found[2] == "‘Quoted’ … Span"


# --- AMEND2 B1 (2026-10-20 trial 2): the subject catalogue must not be quotable ------------------------------

CATALOGUE = [{"subject_key": "hw6_deadline", "description": "deadline for finishing homework 6", "kind": "deadline",
              "value": "2026-10-18", "unit": "", "modality": "asserted", "stated_at": "2026-10-17T10:00:00"},
             {"subject_key": "sprint_span", "description": "length of the sprint", "kind": "duration",
              "value": "15", "unit": "days", "modality": "asserted", "stated_at": "2026-10-20T02:47:00"}]


def test_b1_rendered_catalogue_has_no_values_dates_or_equals_sign():
    rendered = cl._render_open_subjects(CATALOGUE)
    rows = json.loads(rendered)
    assert [sorted(r) for r in rows] == [["key", "kind", "unit", "what"]] * 2
    assert "=" not in rendered and "2026-10-18" not in rendered and "15" not in rendered
    assert "asserted" not in rendered
    assert cl._render_open_subjects([]) == "[]"
    for system, user in (cl.build_extraction_prompt("I am only 6 days behind.", "2026-10-20", CATALOGUE),
                         cl.build_reply_check_prompt("About fifteen days.", "2026-10-20", CATALOGUE)):
        assert "SUBJECT CATALOGUE" in user and "NOT" in user and "never quote" in user
        assert "between <<< and >>> only" in system and "never from the subject catalogue" in system
        assert "=" not in user.split("<<<")[0].split("CATALOGUE", 1)[1]


@pytest.mark.parametrize("form", ["clean", "wrapped"])
def test_b1_a_quote_echoed_from_the_catalogue_still_drops(form):
    text = "Worked on homework today and it went fine, I am only 6 days behind."
    text = text if form == "clean" else wrap(text)
    echo = raw(kind="deadline", key="hw6_deadline", value="2026-10-18", unit="",
               quote="deadline for finishing homework 6 = 2026-10-18")
    good = raw(quote="I am only 6 days behind", unit="days")
    claims, drops = cl.validate_claims(payload(echo, good), text, "2026-10-17T10:00:00", CATALOGUE)
    assert [d["reason"] for d in drops] == ["quote_not_in_text"]
    assert [c.value for c in claims] == ["6"]
