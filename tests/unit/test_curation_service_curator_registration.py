"""init_engine registers every curator in ALL_CURATORS (BC-10 dead wiring).

2026-10-27: ResidenceConflictCurator was defined and listed in ALL_CURATORS but
init_engine hand-listed six curators, so it never ran in the live daemon.
class: BC-10, BC-71.
"""
from __future__ import annotations

import memory.curation.engine as engine_mod
import memory.curation.service as service
from memory.curation.curators import ALL_CURATORS, ResidenceConflictCurator
from memory.curation.types import ProposalStatus

# A curator deliberately NOT run live must be named here with the reason.
DELIBERATELY_UNREGISTERED: dict = {}


class FakeProfile:
    def __init__(self, facts):
        self.profile = {"categories": {"living_situation": list(facts)}}

    def save(self, *, raise_on_error=False):
        pass


OLD = {"fact_id": "f_old", "relation": "living_arrangement", "value": "live on my own",
       "is_current": True, "timestamp": "2026-02-01T10:00:00"}
NEW = {"fact_id": "f_new", "relation": "living_with", "value": "mother",
       "is_current": True, "timestamp": "2026-08-01T10:00:00"}


def _init(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(engine_mod, "_DEFAULT_QUEUE_PATH", str(tmp_path / "q.json"))
    from memory.curation import journal
    monkeypatch.setattr(journal, "_DEFAULT_JOURNAL_PATH", str(tmp_path / "a.jsonl"), raising=False)
    monkeypatch.setattr(service, "_engine", None)
    eng = service.init_engine(user_profile=profile)
    assert eng is not None
    return eng


def test_init_engine_registers_residence_curator(tmp_path, monkeypatch):
    eng = _init(tmp_path, monkeypatch, FakeProfile([]))
    assert any(isinstance(c, ResidenceConflictCurator) for c in eng._curators)


def test_every_curator_in_all_curators_is_registered(tmp_path, monkeypatch):
    eng = _init(tmp_path, monkeypatch, FakeProfile([]))
    registered = {type(c) for c in eng._curators}
    missing = {c.__name__ for c in ALL_CURATORS if c not in registered} - set(DELIBERATELY_UNREGISTERED)
    assert not missing, f"ALL_CURATORS not registered by init_engine: {sorted(missing)}"


def test_residence_conflict_is_queued_not_applied(tmp_path, monkeypatch):
    profile = FakeProfile([dict(OLD), dict(NEW)])
    eng = _init(tmp_path, monkeypatch, profile)
    assert eng.max_mode.value == "queue"
    eng.run_scan()
    mine = [p for p in eng.pending() if p.curator == "residence_conflicts"]
    assert len(mine) == 1 and mine[0].status == ProposalStatus.PENDING
    facts = {f["fact_id"]: f for f in profile.profile["categories"]["living_situation"]}
    assert facts["f_old"]["is_current"] is True  # queued only
