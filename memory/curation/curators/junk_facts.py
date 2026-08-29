"""Quarantine junk fact triples in the chroma `facts` collection.

Replaces the terminal path of scripts/purge_junk_facts.py — quarantine flag
instead of delete. Selection uses THE deployed extraction guard
fact_extractor._is_junk_object (adverbial/temporal/negation fragments that
predate the 2026-08-02/03 inflow guards). A doc that doesn't parse as
"subject | relation | object" is left alone — never junk by absence.
"""

from typing import List, Optional, Tuple

from memory.curation.curators._scan import already_quarantined, iter_collection
from memory.curation.engine import StoreBundle, new_proposal_id
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
    SentinelResult,
)
from memory.fact_extractor import _is_junk_object


def _parse_triple(content: str) -> Optional[Tuple[str, str, str]]:
    parts = [p.strip() for p in (content or "").split(" | ", 2)]
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


class JunkFactCurator:
    name = "junk_facts"

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        return [
            SentinelResult(name="fragment_flags",
                           passed=_is_junk_object("for a bit", "dad_show_up")),
            SentinelResult(name="real_object_passes",
                           passed=not _is_junk_object("pizza", "likes")),
            SentinelResult(
                # Negation IS the content for communication/status relations
                # (the 08-05 "no patient portal" lesson) — exempt must hold.
                name="negation_exempt_relation_passes",
                passed=not _is_junk_object("no patient portal",
                                           "doctor_communication"),
            ),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        if stores.chroma_store is None:
            return []
        items: List[ItemChange] = []
        examples: List[str] = []
        for doc in iter_collection(stores.chroma_store, "facts"):
            if already_quarantined(doc["metadata"]):
                continue
            triple = _parse_triple(doc.get("document") or "")
            if triple is None:
                continue
            _subj, rel, obj = triple
            if _is_junk_object(obj, rel):
                items.append(ItemChange(
                    store="chroma:facts", doc_id=doc["id"],
                    change_type="quarantine",
                    after={"curation_quarantine_reason": "junk_object"},
                ))
                if len(examples) < 3:
                    examples.append(f"{rel}={obj}")
        if not items:
            return []
        ex = "; ".join(examples)
        return [CurationProposal(
            proposal_id=new_proposal_id(),
            curator=self.name,
            instrument=Instrument.METADATA,
            confidence=Confidence.DETERMINISTIC,
            batch=True,
            title=f"Quarantine {len(items)} junk fact fragments",
            evidence=(
                "Objects flagged by the deployed fact_extractor._is_junk_object "
                "guard (adverbial/temporal/negation fragments stored before the "
                f"inflow guards existed). Quarantine only, no deletion. {ex}"
            ),
            items=items,
        )]
