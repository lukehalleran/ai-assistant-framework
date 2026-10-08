"""ResidenceConflictCurator (2026-10-02): stored living-situation facts under
old relation names (living_arrangement, roommate...) that conflict with a newer
fact are proposed for reversible supersession. class: BC-58, BC-76."""
from __future__ import annotations

from memory.curation.curators import ALL_CURATORS, ProfileJunkFactCurator
from memory.curation.curators.residence_conflicts import ResidenceConflictCurator
from memory.curation.engine import CurationEngine, StoreBundle
from memory.curation.journal import CurationJournal
from memory.curation.types import Confidence, Instrument, ProposalStatus


class FakeProfile:
    def __init__(self, facts):
        self.profile = {"categories": {"living_situation": list(facts)}}
        self.saves = 0

    def save(self, *, raise_on_error=False):
        self.saves += 1


def _fact(fid, rel, val, ts, current=True):
    f = {"relation": rel, "value": val, "is_current": current}
    if fid:
        f["fact_id"] = fid
    if ts:
        f["timestamp"] = ts
    return f


OLD = _fact("f_old", "living_arrangement", "live on my own", "2026-02-01T10:00:00")
NEW = _fact("f_new", "living_with", "mother", "2026-08-01T10:00:00")


def _engine(tmp_path, profile):
    eng = CurationEngine(
        StoreBundle(user_profile=profile),
        queue_path=str(tmp_path / "q.json"),
        journal=CurationJournal(str(tmp_path / "a.jsonl")),
    )
    eng.register(ResidenceConflictCurator())
    return eng


def test_registered_and_sentinels_pass():
    assert ResidenceConflictCurator in ALL_CURATORS
    assert all(s.passed for s in ResidenceConflictCurator().sentinels(StoreBundle()))


def test_profile_junk_curator_has_dwelling_sentinel():
    names = {s.name: s.passed for s in ProfileJunkFactCurator().sentinels(StoreBundle())}
    assert names["bare_dwelling_residence_flags"] is True


def test_pair_yields_one_proposal_for_older():
    props = ResidenceConflictCurator().scan(StoreBundle(user_profile=FakeProfile([NEW, OLD])))
    assert len(props) == 1
    p = props[0]
    assert [i.doc_id for i in p.items] == ["f_old"]
    assert p.items[0].change_type == "supersede_profile_fact"
    assert p.items[0].after["reason"] == "superseded by newer living_with fact f_new (2026-08-01)"
    assert p.instrument == Instrument.METADATA
    assert p.confidence == Confidence.DETERMINISTIC  # honest tier; queue ceiling keeps it for review


def test_single_fact_and_no_profile_none():
    assert ResidenceConflictCurator().scan(StoreBundle(user_profile=FakeProfile([OLD]))) == []
    assert ResidenceConflictCurator().scan(StoreBundle()) == []


def test_skips_noncurrent_unaddressable_unrelated_and_untimed():
    facts = [
        OLD,
        _fact("f_dead", "living_with", "dad", "2026-09-01T00:00:00", current=False),
        _fact(None, "living_with", "mom", "2026-09-02T00:00:00"),  # no id
        _fact("f_like", "likes", "pizza", "2026-01-01T00:00:00"),
        _fact("f_like2", "likes", "tacos", "2026-09-01T00:00:00"),
    ]
    assert ResidenceConflictCurator().scan(StoreBundle(user_profile=FakeProfile(facts))) == []
    # a group member without a parseable timestamp -> newest unknown -> skip
    untimed = [OLD, _fact("f_x", "living_with", "mother", None)]
    assert ResidenceConflictCurator().scan(StoreBundle(user_profile=FakeProfile(untimed))) == []


def test_three_in_group_keeps_only_newest():
    mid = _fact("f_mid", "roommate", "mom", "2026-05-01T00:00:00")
    props = ResidenceConflictCurator().scan(StoreBundle(user_profile=FakeProfile([OLD, NEW, mid])))
    assert {i.doc_id for i in props[0].items} == {"f_old", "f_mid"}


def test_multi_valued_relation_skipped(monkeypatch):
    import memory.curation.curators.residence_conflicts as mod
    monkeypatch.setattr(mod, "is_multi_valued_relation", lambda r: r == "living_with")
    assert ResidenceConflictCurator().scan(StoreBundle(user_profile=FakeProfile([OLD, NEW]))) == []


def test_engine_queue_apply_undo_roundtrip(tmp_path):
    profile = FakeProfile([dict(OLD), dict(NEW)])
    eng = _engine(tmp_path, profile)
    report = eng.run_scan()
    assert report.proposals_queued == 1 and not report.sentinel_failures
    p = eng.pending()[0]
    assert p.status == ProposalStatus.PENDING
    eng.apply(p.proposal_id, actor="human")
    by_id = {f["fact_id"]: f for f in profile.profile["categories"]["living_situation"]}
    assert by_id["f_old"]["is_current"] is False
    assert by_id["f_new"]["is_current"] is True
    assert eng.run_scan().proposals_queued == 0
    eng.undo(p.proposal_id)
    assert by_id["f_old"]["is_current"] is True
    assert "curation_stale_reason" not in by_id["f_old"]


def test_through_deployed_user_profile_stored_old_names(tmp_path):
    """Stored facts built via the real profile shape (old names) are found."""
    from memory.user_profile import UserProfile
    up = UserProfile.__new__(UserProfile)
    up.profile = {"categories": {"living_situation": [dict(OLD), dict(NEW)]}}
    props = ResidenceConflictCurator().scan(StoreBundle(user_profile=up))
    assert len(props) == 1 and props[0].items[0].doc_id == "f_old"
