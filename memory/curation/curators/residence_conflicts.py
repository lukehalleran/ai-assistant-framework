"""Supersede stale living-situation facts that conflict with a newer one.

2026-10-02: `living_arrangement="live on my own"` (Feb) stayed current beside
`living_with=mother` (Aug) in the quick profile. UserProfile.add_fact now
canonicalizes household relations to `living_with` (SAFE_RELATION_ALIASES), so
a NEW write supersedes the old one — but facts stored under the old relation
names never conflict-checked. This curator handles those stored conflicts.

Selection: current, addressable (fact_id) profile facts whose relation is a
living-situation relation (memory.user_profile_schema.is_living_situation_relation),
grouped by canonicalize_profile_relation. Multi-valued relations are skipped.
In a group with >=2 current facts every fact older than the newest is proposed
for supersession. A group is skipped when any member lacks a parseable
timestamp (the newest can't be established). Instrument: reversible profile
supersession only — never deletion. Selection is the same deterministic
"newest single-valued fact wins" rule UserProfile.add_fact applies on write, so
the evidence tier is DETERMINISTIC; what keeps it for owner review in the
Curation Center is the engine's queue ceiling (curation.max_mode), not the label.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from memory.curation.engine import StoreBundle, new_proposal_id
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
    SentinelResult,
)
from memory.relation_classifier import is_multi_valued_relation
from memory.user_profile_schema import (
    canonicalize_profile_relation,
    is_living_situation_relation,
)


def _ts(fact: dict) -> Optional[str]:
    raw = fact.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=None).isoformat()


def _conflicts(facts: List[dict]) -> List[Tuple[dict, dict, str]]:
    """Return (stale_fact, newest_fact, canonical_relation) triples."""
    groups: Dict[str, List[Tuple[str, int, dict]]] = defaultdict(list)
    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict) or not fact.get("is_current", True):
            continue
        if not fact.get("fact_id"):
            continue  # unaddressable → leave alone
        rel = str(fact.get("relation") or "")
        if not rel or not is_living_situation_relation(rel):
            continue
        canon = canonicalize_profile_relation(rel, str(fact.get("value") or ""))
        if is_multi_valued_relation(canon) or is_multi_valued_relation(rel):
            continue
        groups[canon].append((_ts(fact) or "", idx, fact))
    out: List[Tuple[dict, dict, str]] = []
    for canon, members in groups.items():
        if len(members) < 2 or any(not m[0] for m in members):
            continue  # nothing to conflict, or newest not establishable
        members.sort(key=lambda m: (m[0], m[1]))
        newest = members[-1][2]
        for _, _, fact in members[:-1]:
            out.append((fact, newest, canon))
    return out


class ResidenceConflictCurator:
    name = "residence_conflicts"

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        old = {"fact_id": "s_old", "relation": "living_arrangement",
               "value": "live on my own", "is_current": True,
               "timestamp": "2026-02-01T10:00:00"}
        new = {"fact_id": "s_new", "relation": "living_with",
               "value": "mother", "is_current": True,
               "timestamp": "2026-08-01T10:00:00"}
        pair = _conflicts([new, old])
        return [
            SentinelResult(
                name="old_alias_fact_superseded_by_newer",
                passed=len(pair) == 1 and pair[0][0]["fact_id"] == "s_old"
                and pair[0][1]["fact_id"] == "s_new"),
            SentinelResult(name="single_fact_no_conflict",
                           passed=_conflicts([old]) == []),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        profile = stores.user_profile
        if profile is None:
            return []
        cats = (getattr(profile, "profile", None) or {}).get("categories", {})
        facts: List[dict] = []
        for facts_list in cats.values():
            if isinstance(facts_list, list):
                facts.extend(f for f in facts_list if isinstance(f, dict))
        items: List[ItemChange] = []
        examples: List[str] = []
        for stale, newest, canon in _conflicts(facts):
            date = str(newest.get("timestamp") or "")[:10]
            items.append(ItemChange(
                store="profile", doc_id=str(stale["fact_id"]),
                change_type="supersede_profile_fact",
                after={"reason": (f"superseded by newer {canon} fact "
                                  f"{newest['fact_id']} ({date})")},
            ))
            if len(examples) < 3:
                examples.append(
                    f"{stale.get('relation')}={str(stale.get('value'))[:40]} "
                    f"-> {newest.get('relation')}={str(newest.get('value'))[:40]}")
        if not items:
            return []
        ex = "; ".join(examples)
        return [CurationProposal(
            proposal_id=new_proposal_id(),
            curator=self.name,
            instrument=Instrument.METADATA,
            confidence=Confidence.DETERMINISTIC,
            batch=True,
            title=f"Retire {len(items)} outdated living-situation fact(s)",
            evidence=(
                "Current living-situation profile facts that share one canonical "
                "relation with a newer fact; the older statement is proposed for "
                "supersession. Supersession only — each fact stays in the profile "
                f"as history and can be restored with Undo. {ex}"
            ),
            items=items,
        )]
