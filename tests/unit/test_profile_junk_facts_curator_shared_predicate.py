"""The profile junk curator and the extractor share ONE trailing-when-adverb predicate.

Incident date: 2026-10-27 (wave-2 follow-up: the curator kept its own copy of the
structural test that `memory/fact_extractor.py` also defines).  Class: BC-58 (two
definitions of one predicate drift apart).

Drives THE deployed ``ProfileJunkFactCurator.scan``; the extractor's function is
replaced with a recording stub -- if the curator still parsed on its own the stub
would not change its verdict.
"""
import memory.fact_extractor as fx
from memory.curation.curators import ProfileJunkFactCurator
from memory.curation.engine import StoreBundle


class _Profile:
    def __init__(self, facts):
        self.profile = {"categories": {"health": facts}}


def _flagged(value, relation="took"):
    facts = [{"fact_id": "f0", "relation": relation, "value": value, "is_current": True}]
    props = ProfileJunkFactCurator().scan(StoreBundle(user_profile=_Profile(facts)))
    return bool({i.doc_id for p in props for i in p.items})


def test_curator_delegates_to_the_extractor_predicate(monkeypatch):
    seen = []
    # a value the extractor's real predicate would never flag ("sourdough starter")
    monkeypatch.setattr(fx, "_trailing_when_adverb_fragment", lambda v: (seen.append(v), True)[1])
    import memory.curation.curators.profile_junk_facts as pj
    monkeypatch.setattr(pj, "_extractor_when_adverb_fragment", fx._trailing_when_adverb_fragment)
    assert _flagged("sourdough starter") is True
    assert seen == ["sourdough starter"]


def test_schedule_relation_exemption_stays_in_the_curator(monkeypatch):
    import memory.curation.curators.profile_junk_facts as pj
    monkeypatch.setattr(pj, "_extractor_when_adverb_fragment", lambda v: True)
    assert _flagged("weekday recently", relation="day_off") is False


def test_real_predicate_clean_and_wrapped():
    assert _flagged("meds recently") is True
    assert _flagged("  meds\n  recently ") is True
    assert _flagged("5 mg recently") is False
