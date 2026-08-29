"""Repair stream artifacts in stored docs via THE deployed strips.

Replaces the terminal path of scripts/strip_special_token_artifacts.py:
edge `<|sep|>`-class special-token runs and the kimi trailing-'e' glued to
terminal punctuation, in conversations + summaries documents. Content
repair with pre-image (reversible), never quarantine or delete — these are
good docs wearing an artifact.
"""

from typing import List

from core.response_parser import ResponseParser
from memory.curation.curators._scan import iter_collection
from memory.curation.engine import StoreBundle, new_proposal_id
from memory.curation.types import (
    Confidence,
    CurationProposal,
    Instrument,
    ItemChange,
    SentinelResult,
)


def _strip(text: str) -> str:
    return ResponseParser.strip_trailing_stream_artifact(text)


class StreamArtifactCurator:
    name = "stream_artifacts"

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        sep = _strip("<|sep|>That's a good question.")
        trailing = _strip("Did it land by the way?e")
        abbrev = _strip("Use the tool, i.e")           # abbreviation guard holds
        mid = _strip("the <|sep|> token appears mid-text here.")
        return [
            SentinelResult(name="sep_stripped",
                           passed=sep == "That's a good question."),
            SentinelResult(name="trailing_e_stripped",
                           passed=trailing == "Did it land by the way?"),
            SentinelResult(name="abbreviation_kept",
                           passed=abbrev == "Use the tool, i.e"),
            SentinelResult(name="mid_text_token_kept",
                           passed="<|sep|>" in mid),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        if stores.chroma_store is None:
            return []
        items: List[ItemChange] = []
        examples: List[str] = []
        for collection in ("conversations", "summaries"):
            for doc in iter_collection(stores.chroma_store, collection):
                text = doc.get("document") or ""
                if not text:
                    continue
                repaired = _strip(text)
                # Refuse any repair that guts the doc (prime directive):
                # the strip is edge-only, so a large delta means something
                # unexpected — skip rather than risk it.
                if repaired == text or not repaired.strip():
                    continue
                if len(text) - len(repaired) > 32:
                    continue
                items.append(ItemChange(
                    store=f"chroma:{collection}", doc_id=doc["id"],
                    change_type="replace_content",
                    after={"document": repaired},
                ))
                if len(examples) < 3:
                    examples.append(text[:60])
        if not items:
            return []
        ex = "; ".join(f"e.g. {e!r}" for e in examples)
        return [CurationProposal(
            proposal_id=new_proposal_id(),
            curator=self.name,
            instrument=Instrument.METADATA,
            confidence=Confidence.DETERMINISTIC,
            batch=True,
            title=f"Repair stream artifacts in {len(items)} stored docs",
            evidence=(
                "Edge special-token runs (<|sep|>) and trailing stream "
                "letters removed by the deployed "
                "ResponseParser.strip_trailing_stream_artifact — the same "
                f"transform live turns already get. Pre-images kept. {ex}"
            ),
            items=items,
        )]
