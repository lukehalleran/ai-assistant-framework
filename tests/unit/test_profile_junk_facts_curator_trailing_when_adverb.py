"""The profile junk curator proposes a noun + temporal-adverb object.

Incident 2026-09-15 (class: BC-04): ``took = "meds recently"`` sat in the quick
profile; ``_is_junk_object`` only rejects a when-word as the WHOLE object, so the
curator never proposed it.  The curator's structural test (spaCy ADV tail that is
a bare when-word per graph_utils.is_temporal_deictic) closes it; doses, schedule
relations and ordinary objects are left alone.  Drives THE deployed
``ProfileJunkFactCurator.scan``.
"""
from memory.curation.curators import ProfileJunkFactCurator
from memory.curation.engine import StoreBundle


class _Profile:
    def __init__(self, facts):
        self.profile = {"categories": {"health": facts}}


def _flagged(*pairs):
    facts = [{"fact_id": f"f{i}", "relation": r, "value": v, "is_current": True}
             for i, (r, v) in enumerate(pairs)]
    props = ProfileJunkFactCurator().scan(StoreBundle(user_profile=_Profile(facts)))
    ids = {i.doc_id for p in props for i in p.items}
    return {pairs[int(i[1:])] for i in ids}


def test_noun_plus_temporal_adverb_is_proposed():
    got = _flagged(("took", "meds recently"), ("took", "ibuprofen earlier"))
    assert got == {("took", "meds recently"), ("took", "ibuprofen earlier")}


def test_wrapped_value_is_proposed_too():
    assert _flagged(("took", "  meds\n recently ")) == {("took", "  meds\n recently ")}


def test_dose_schedule_and_plain_objects_are_left_alone():
    assert _flagged(
        ("took", "5 mg recently"),
        ("day_off", "weekday recently"),
        ("likes", "pizza"),
        ("took", "melatonin"),
        ("works_at", "Northwind"),
    ) == set()


def test_sentinels_still_pass():
    assert all(s.passed for s in ProfileJunkFactCurator().sentinels(StoreBundle()))
