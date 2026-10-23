"""Claim ledger: the stored quote covers its modality marker; the render keeps the condition.

Incident 2026-10-22: the owner's real ledger rendered "days behind: 6 days —
CONDITIONAL, you said 'I am only 6 days behind'": the model's quote kept only
the consequent of "if I can get 6 done today then I am only 6 days behind", so
the line's only sign of the condition was the label. Over-long quotes were cut
head-only, and 8 of 12 lines carried an empty "; earlier: none" (one repeated
the current value as "earlier"). Classes: BC-75, BC-52.

Drives the deployed validate_claims / render_stated_figures with fake model
JSON. Every text is asserted in clean and wrapped/indented form.
"""
from __future__ import annotations

import json

import pytest

from memory import claim_ledger as cl

SAID = "2026-10-17T10:00:00"
COND_CLEAN = "Quick update. If I can get 6 done today then I am only 6 days behind."
COND_WRAPPED = "Quick update. If I can get 6 done\n    today then I am only 6\n    days behind."
EST_CLEAN = "Covered most of it. i think it was about 12 days."
EST_WRAPPED = "Covered most of it.\n    i think it was\n    about 12 days."
ASSERT_CLEAN = "Quick update. I am only 6 days behind."
ASSERT_WRAPPED = "Quick update.\n    I am only 6 days\n    behind."


def payload(quote, modality, value="6"):
    return json.dumps({"claims": [{
        "kind": "duration", "subject_key": "days_behind", "new_subject": True,
        "description": "days behind", "value": value, "unit": "days",
        "modality": modality, "quote": quote}]})


def one(text, quote, modality, value="6"):
    claims, drops = cl.validate_claims(payload(quote, modality, value), text, SAID)
    assert drops == [] and len(claims) == 1
    return claims[0]


@pytest.mark.parametrize("text", [COND_CLEAN, COND_WRAPPED], ids=["clean", "wrapped"])
def test_conditional_quote_widens_to_cover_the_antecedent(text):
    c = one(text, "I am only 6 days behind", "asserted")
    assert c.modality == "conditional"
    assert " ".join(c.quote.split()).lower() == "if i can get 6 done today then i am only 6 days behind"
    assert cl.locate_quote(text, c.quote) is not None       # still an exact span of the text


@pytest.mark.parametrize("text", [EST_CLEAN, EST_WRAPPED], ids=["clean", "wrapped"])
def test_estimate_quote_widens_to_cover_the_hedge(text):
    c = one(text, "12 days", "estimate", value="12")
    assert c.modality == "estimate"
    assert " ".join(c.quote.split()).lower() == "i think it was about 12 days"
    assert cl.locate_quote(text, c.quote) is not None


@pytest.mark.parametrize("text", [ASSERT_CLEAN, ASSERT_WRAPPED], ids=["clean", "wrapped"])
def test_asserted_quote_is_unchanged(text):
    c = one(text, "I am only 6 days behind", "asserted")
    assert c.modality == "asserted" and c.quote == "I am only 6 days behind" or \
        " ".join(c.quote.split()) == "I am only 6 days behind"


@pytest.mark.parametrize("text", [COND_CLEAN, COND_WRAPPED], ids=["clean", "wrapped"])
def test_quote_that_already_has_the_marker_is_unchanged(text):
    q = "If I can get 6 done today then I am only 6 days behind"
    c = one(text, q, "conditional")
    assert " ".join(c.quote.split()) == q


def test_overlong_sentence_keeps_the_original_quote():
    filler = "and so on " * 25
    text = f"If I can get 6 done today {filler}then I am only 6 days behind."
    c = one(text, "I am only 6 days behind", "asserted")
    assert c.modality == "conditional" and c.quote == "I am only 6 days behind"


# --- render ------------------------------------------------------------------------------------------
NOW = "2026-10-20T20:30:00"
KW = dict(max_lines=12, standing_window_days=30, deadline_window_days=14)


def lc(value, modality="asserted", at="2026-10-20T09:00:00", quote=None, key="k"):
    quote = quote or f"{value} days"
    return cl.LedgerClaim(
        id=cl.make_claim_id(key, "duration", at, value, quote + modality), subject_key=key,
        description="subject", kind="duration", value=value, unit="days", modality=modality,
        stated_at=at, quote=quote)


def render(*claims):
    ledger = cl.ClaimLedger()
    for c in claims:
        ledger.add(c)
    return cl.render_stated_figures(ledger, NOW, **KW)


@pytest.mark.parametrize("modality", ["conditional", "estimate"])
def test_long_non_asserted_quote_keeps_head_clause_and_figure(modality):
    q = ("if I can get the whole first module of reading and exercises finished before the weekend "
         "then I am only 6 days behind")
    (line,) = render(lc("6", modality, quote=q))
    assert "if I can get the whole" in line and "only 6 days behind" in line and "…" in line


def test_long_asserted_quote_keeps_the_head_only_cut():
    q = "I am keeping a long running log of everything that matters and the running total so far this week is about 6 days behind"
    (line,) = render(lc("6", quote=q))
    assert "I am keeping a long" in line and line.count("…") == 1 and "6 days behind" not in line


def test_no_line_says_earlier_none_and_equal_older_values_are_not_shown():
    (single,) = render(lc("6", at="2026-10-20T09:00:00"))
    assert "earlier" not in single
    (same,) = render(lc("6", at="2026-10-18T09:00:00"), lc("6", at="2026-10-19T09:00:00"),
                     lc("7", at="2026-10-20T09:00:00"))
    assert "earlier: 6 days (Mon 10-19)" in same and same.count("6 days (") == 1
    (equal,) = render(lc("6", at="2026-10-19T09:00:00"), lc("6", at="2026-10-20T09:00:00"))
    assert "earlier" not in equal


def test_two_different_older_values_are_still_listed():
    (line,) = render(lc("9", at="2026-10-16T09:00:00"), lc("10", at="2026-10-17T09:00:00"),
                     lc("12", at="2026-10-18T09:00:00"), lc("15", at="2026-10-20T09:00:00"))
    assert "earlier: 12 days (Sun 10-18), 10 days (Sat 10-17)" in line
