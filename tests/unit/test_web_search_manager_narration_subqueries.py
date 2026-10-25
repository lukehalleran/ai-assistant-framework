"""decompose_query drops narration-derived sub-queries (2026-10-25, WEB2 W1).

Class: BC-05 (missing guard), BC-38 (paid call admitted without need).
PR #102 stopped first-person check-ins at the trigger; the decomposer could
still turn a narration clause into a paid sub-query. Drives THE deployed
decompose_query with a fake model.
"""
import asyncio
from unittest.mock import MagicMock, patch

from knowledge.web_search_manager import WebSearchManager

CLEAN = "I had coffee and couldn't focus all afternoon, anyway"
WRAPPED = "I had coffee and couldn't\n    focus all afternoon, anyway"
MIXED = "I had coffee and couldn't focus. Compare Tesla Model 3 and Rivian R1T range"
MIXED_WRAPPED = "I had coffee and couldn't\n   focus. Compare Tesla Model 3 and\n   Rivian R1T range"
LEGIT = "Compare Tesla Model 3 versus Rivian R1T range and price"
LEGIT_WRAPPED = "Compare Tesla Model 3 versus\n    Rivian R1T range and price"


def _decompose(query, subs):
    reply = ("SHOULD_SPLIT: yes\nCONFIDENCE: 0.9\nREASON: multi\nSUB_QUERIES:\n"
             + "\n".join(f"- {s}" for s in subs))
    mgr = WebSearchManager.__new__(WebSearchManager)
    mgr.rate_limiter = MagicMock()
    mgr.rate_limiter.get_remaining_credits.return_value = 50
    mgr.link_selector_model = "m"
    mgr._get_user_location = lambda: ""
    mgr._get_user_institution = lambda: ""
    fake = MagicMock()
    fake.generate.return_value = reply
    with patch("models.model_manager.ModelManager", return_value=fake):
        return asyncio.run(mgr.decompose_query(query))


def test_narration_fragment_subqueries_all_drop_falls_back_to_original():
    for q in (CLEAN, WRAPPED):
        r = _decompose(q, ["how to improve focus after coffee", "coffee caffeine focus tips"])
        assert r.sub_queries == []
        assert r.should_decompose is False
        assert r.original_query == q


def test_legit_multi_part_keeps_all_parts():
    for q in (LEGIT, LEGIT_WRAPPED):
        subs = ["Tesla Model 3 range price", "Rivian R1T range price"]
        r = _decompose(q, subs)
        assert r.should_decompose and r.sub_queries == subs


def test_mixed_message_keeps_only_world_part():
    for q in (MIXED, MIXED_WRAPPED):
        r = _decompose(q, ["how to improve focus after coffee",
                           "Tesla Model 3 range", "Rivian R1T range"])
        assert r.sub_queries == ["Tesla Model 3 range", "Rivian R1T range"]


def test_lookup_cue_clause_is_not_narration():
    from knowledge.web_search_manager import drop_narration_subqueries
    q = "I had coffee, look up how caffeine affects focus"
    subs = ["caffeine effects on focus"]
    assert drop_narration_subqueries(q, subs) == subs


TOPIC_POINTERS = [
    ("I am referring to voting", ["Vermont voting information", "voting issues"]),
    ("I'm asking about the new tax law", ["new tax law summary", "tax law changes"]),
    ("I was talking about the election results", ["election results", "results by county"]),
    ("I mean the voting rules in Vermont", ["Vermont voting rules", "voting rules"]),
]


def _wrap(q):
    w = q.split()
    return " ".join(w[:2]) + "\n    " + " ".join(w[2:])


def test_topic_pointer_clauses_keep_their_subqueries():
    for q, subs in TOPIC_POINTERS:
        for form in (q, _wrap(q)):
            r = _decompose(form, subs)
            assert r.sub_queries == subs, form
            assert r.should_decompose is True


def test_narration_control_still_drops_beside_pointer_logic():
    for form in (CLEAN, WRAPPED):
        r = _decompose(form, ["how to improve focus after coffee", "coffee caffeine focus tips"])
        assert r.sub_queries == []
