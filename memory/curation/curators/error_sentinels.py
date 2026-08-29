"""Quarantine API-error-sentinel / junk conversation docs and summaries.

Replaces the terminal path of scripts/purge_error_memories.py — but flips
the quarantine flag instead of deleting (deletion never happens through the
engine). Selection uses THE deployed retrieval predicates
memory.utils.is_junk_conversation_doc / is_junk_summary, so a quarantined
doc is by definition one retrieval already refuses to surface; the flag
adds an explicit inventory (Quarantine view) and a cheap metadata filter.
"""

from typing import List

from memory.curation.curators._scan import already_quarantined, iter_collection
from memory.curation.engine import StoreBundle, new_proposal_id
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
    SentinelResult,
)
from memory.utils import is_junk_conversation_doc, is_junk_summary


class ErrorSentinelCurator:
    name = "error_sentinels"

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        # Known-junk must flag; ordinary content must not. Pure predicate
        # checks — no store access needed.
        junk = is_junk_conversation_doc(
            content="User: hey\nAssistant: [API unavailable] upstream error"
        )
        clean = is_junk_conversation_doc(
            content="User: how was the gym\nAssistant: Solid session — "
                    "you hit the squat PR you were chasing."
        )
        junk_sum = is_junk_summary("[API Error] 402 insufficient credits")
        clean_sum = is_junk_summary(
            "- Luke confirmed the enrollment decision and called the bursar."
        )
        return [
            SentinelResult(name="junk_conv_flags", passed=bool(junk)),
            SentinelResult(name="clean_conv_passes", passed=not clean),
            SentinelResult(name="junk_summary_flags", passed=bool(junk_sum)),
            SentinelResult(name="clean_summary_passes", passed=not clean_sum),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        if stores.chroma_store is None:
            return []
        items: List[ItemChange] = []
        examples: List[str] = []

        for doc in iter_collection(stores.chroma_store, "conversations"):
            if already_quarantined(doc["metadata"]):
                continue
            meta = doc["metadata"]
            if is_junk_conversation_doc(
                content=doc.get("document") or "",
                query=meta.get("query", ""),
                response=meta.get("response", ""),
            ):
                items.append(ItemChange(
                    store="chroma:conversations", doc_id=doc["id"],
                    change_type="quarantine",
                    after={"curation_quarantine_reason": "api_error_or_test_doc"},
                ))
                if len(examples) < 3:
                    examples.append((doc.get("document") or "")[:80])

        for doc in iter_collection(stores.chroma_store, "summaries"):
            if already_quarantined(doc["metadata"]):
                continue
            if is_junk_summary(doc.get("document") or ""):
                items.append(ItemChange(
                    store="chroma:summaries", doc_id=doc["id"],
                    change_type="quarantine",
                    after={"curation_quarantine_reason": "junk_summary"},
                ))
                if len(examples) < 3:
                    examples.append((doc.get("document") or "")[:80])

        if not items:
            return []
        ex = "; ".join(f"e.g. {e!r}" for e in examples)
        return [CurationProposal(
            proposal_id=new_proposal_id(),
            curator=self.name,
            instrument=Instrument.METADATA,
            confidence=Confidence.DETERMINISTIC,
            batch=True,
            title=f"Quarantine {len(items)} error/test docs the retrieval filter already hides",
            evidence=(
                "Selected by the deployed is_junk_conversation_doc / "
                "is_junk_summary predicates (API-error sentinels, connectivity "
                f"tests). Quarantine only — nothing is deleted. {ex}"
            ),
            items=items,
        )]
