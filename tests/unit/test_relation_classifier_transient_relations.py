"""Transient profile relations expire; durable ones do not.

Incident 2026-10-20 (class: BC-26 read-time TTL gap): live profile relations
`showered`, `rested_until` and `leaving_for_*` classified durable (TTL None),
so stale one-off lines rendered forever. Three categories close it, all in
``memory.relation_classifier`` (time-bounded-state tail, departure/arrival
movement prefix, bare past-tense ACTION verb decided by the spaCy parse).
Drives THE deployed ``ephemeral_ttl_hours`` / ``is_ephemeral_relation``.
"""
import pytest

from memory import relation_classifier as rc


@pytest.mark.parametrize("rel", [
    # (a) time-bounded state: last token names the END of the state
    "rested_until", "off_work_till", "away_until",
    # (b) departure / arrival movement
    "leaving_for_work", "leaving_for_the_airport", "heading_to_class",
    "going_to_the_gym", "arriving_late", "arriving_at_noon",
    # (c) one bare completed daily/bodily act
    "showered", "napped", "ate", "slept", "cooked", "exercised",
])
def test_transient_relations_get_a_ttl(rel):
    assert rc.ephemeral_ttl_hours(rel) == 24.0
    assert rc.is_ephemeral_relation(rel)
    # case / padding forms reach the same answer
    assert rc.ephemeral_ttl_hours("  " + rel.upper() + " ") == 24.0


@pytest.mark.parametrize("rel", [
    # durable relations that must NOT change class
    "graduated_from", "worked_at", "married_to", "diagnosed_with",
    "lives_in", "works_at", "attended", "completed", "friends_since",
    "lived_in_since", "married_since",
    # bare participles that are life-course status or standing states
    "graduated", "married", "divorced", "retired", "adopted", "enrolled",
    "diagnosed", "moved", "employed", "disabled", "licensed",
    # nouns / identifiers that merely END in -ed or look verb-like
    "bed", "hundred", "shed", "speed", "pandoc",
    # present-tense or base forms are not completed acts
    "shower", "nap", "sleeps",
])
def test_durable_relations_stay_durable(rel):
    assert rc.ephemeral_ttl_hours(rel) is None, rel
    assert not rc.is_ephemeral_relation(rel)


def test_unavailable_parser_leaves_bare_past_relation_durable(monkeypatch):
    """No spaCy -> unknown relation stays durable (never a guess)."""
    rc._is_past_action_verb.cache_clear()
    monkeypatch.setattr("memory.fact_extractor._get_nlp", lambda: None)
    try:
        assert rc.ephemeral_ttl_hours("showered") is None
        # the structural categories do not need the parser
        assert rc.ephemeral_ttl_hours("rested_until") == 24.0
    finally:
        rc._is_past_action_verb.cache_clear()


def test_health_transient_horizon_still_wins():
    # an illness relation keeps the longer health horizon, unchanged
    assert rc.ephemeral_ttl_hours("recovering_until") == rc._health_transient_ttl_hours()


@pytest.mark.parametrize("rel", [
    "dropped", "withdrew", "withdrawn", "unenrolled", "deferred", "joined",
    "quit", "resigned", "transferred", "left", "switched",
])
def test_membership_change_relations_stay_durable(rel):
    """Incident 2026-10-21 (class: BC-26): `dropped | <course>` was a 24h
    ephemeral relation, so a standing enrollment change vanished."""
    assert rc.ephemeral_ttl_hours(rel) is None, rel
    assert not rc.is_ephemeral_relation(rel)
    assert rc.ephemeral_ttl_hours("  " + rel.upper() + " ") is None


@pytest.mark.parametrize("rel", [
    "showered", "eaten", "packed", "took", "started", "rested_until",
    "leaving_for_work",
])
def test_membership_exclusion_does_not_widen_to_acts(rel):
    assert rc.ephemeral_ttl_hours(rel) == 24.0, rel
