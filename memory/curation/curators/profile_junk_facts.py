"""Supersede junk facts in the quick profile (data/user_profile.json).

The chroma-side JunkFactCurator quarantines `facts` docs, but the quick
profile — the [USER PROFILE] section every prompt renders — kept its own
copies, and the only path to clean it was the terminal
(scripts/purge_profile_facts.py + a curated id file). 2026-09-05: the
owner had to purge `has_doctor="Rowan is cautious about drinking…"` and
`works_on="this assistant"` by hand; a read-only scan of the live profile
found 45 more current facts the deployed guard rejects (`time_off_work=
today`, `job_fair_date=Thursday`, `food_status=in the oven`…).

Selection uses THE deployed fact_extractor._is_junk_object (never a
re-derivation). Instrument: supersession (is_current=False + reason),
reversible through the profile adapter — never deletion. Facts without a
fact_id can't be addressed reversibly and are left alone.
"""

from typing import List

from memory.curation.engine import StoreBundle, new_proposal_id
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
    SentinelResult,
)
from memory.fact_extractor import _is_junk_object


class ProfileJunkFactCurator:
    name = "profile_junk_facts"

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        return [
            SentinelResult(name="temporal_deictic_flags",
                           passed=_is_junk_object("today", "time_off_work")),
            SentinelResult(name="weekday_object_flags",
                           passed=_is_junk_object("on thursday", "texted")),
            SentinelResult(name="care_team_clause_flags",
                           passed=_is_junk_object(
                               "Rowan is cautious about drinking due to past accident",
                               "has_doctor")),
            SentinelResult(name="demonstrative_flags",
                           passed=_is_junk_object("this assistant", "works_on")),
            SentinelResult(name="real_object_passes",
                           passed=not _is_junk_object("pizza", "likes")),
            SentinelResult(name="schedule_relation_keeps_weekday",
                           passed=not _is_junk_object("thursday", "day_off")),
            SentinelResult(name="negation_exempt_relation_passes",
                           passed=not _is_junk_object("no patient portal",
                                                      "doctor_communication")),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        profile = stores.user_profile
        if profile is None:
            return []
        cats = (getattr(profile, "profile", None) or {}).get("categories", {})
        items: List[ItemChange] = []
        examples: List[str] = []
        for facts_list in cats.values():
            if not isinstance(facts_list, list):
                continue
            for fact in facts_list:
                if not isinstance(fact, dict):
                    continue
                if not fact.get("is_current", True):
                    continue
                fid = fact.get("fact_id")
                if not fid:
                    continue  # unaddressable → leave alone
                rel = str(fact.get("relation") or "")
                val = str(fact.get("value") or "")
                if not rel or not _is_junk_object(val, rel):
                    continue
                items.append(ItemChange(
                    store="profile", doc_id=str(fid),
                    change_type="supersede_profile_fact",
                    after={"reason": "junk_object"},
                ))
                if len(examples) < 3:
                    examples.append(f"{rel}={val[:60]}")
        if not items:
            return []
        ex = "; ".join(examples)
        return [CurationProposal(
            proposal_id=new_proposal_id(),
            curator=self.name,
            instrument=Instrument.METADATA,
            confidence=Confidence.DETERMINISTIC,
            batch=True,
            title=f"Retire {len(items)} junk profile facts",
            evidence=(
                "Profile facts whose value the deployed fact_extractor._is_junk_object "
                "guard rejects (bare when-words, weekday names, prepositional "
                "fragments, unresolved demonstratives, clause objects on has_<clinician>). "
                "Supersession only — each fact stays in the profile as history and "
                f"can be restored with Undo. {ex}"
            ),
            items=items,
        )]
