"""Quarantine knowledge-graph nodes that are bare when-words.

2026-09-05: "today"/"tomorrow" in a casual plans update resolved to live
graph nodes (`today` carried 11 in-edges: `user|dad|today`,
`user|feels_better|today`, `auggie|talked_about|today` …) and [KNOWLEDGE
GRAPH] rendered nine nonsense lines. The ingestion guard now refuses such
nodes and GraphMemory.edge_is_suppressed hides their edges at read time,
but legacy nodes stay on disk and the only cleanup path was the terminal
(scripts/graph_junk_cleanup.py + a curated id file).

Selection uses THE deployed memory.graph_utils.is_temporal_deictic.
Instrument: node-level `curation_quarantined` metadata (reversible through
the graph adapter; read-time suppression honours it) — never deletion.
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
from memory.graph_utils import is_temporal_deictic


class GraphTemporalNodeCurator:
    name = "graph_temporal_nodes"

    def sentinels(self, stores: StoreBundle) -> List[SentinelResult]:
        return [
            SentinelResult(name="today_flags", passed=is_temporal_deictic("today")),
            SentinelResult(name="prefixed_weekday_flags", passed=is_temporal_deictic("on_thursday")),
            SentinelResult(name="pet_name_passes", passed=not is_temporal_deictic("biscuit")),
            SentinelResult(name="place_passes", passed=not is_temporal_deictic("planet fitness")),
            SentinelResult(name="sun_not_a_deictic", passed=not is_temporal_deictic("sun")),
        ]

    def scan(self, stores: StoreBundle) -> List[CurationProposal]:
        gm = stores.graph_memory
        if gm is None:
            return []
        g = getattr(gm, "graph", None)
        if g is None:
            return []
        items: List[ItemChange] = []
        examples: List[str] = []
        for nid, attrs in g.nodes(data=True):
            meta = attrs.get("metadata") if isinstance(attrs, dict) else None
            if isinstance(meta, dict) and meta.get("curation_quarantined"):
                continue
            if not is_temporal_deictic(str(nid)):
                continue
            items.append(ItemChange(
                store="graph", doc_id=str(nid), change_type="quarantine_node",
                after={"curation_quarantine_reason": "temporal_deictic"},
            ))
            if len(examples) < 4:
                deg = g.degree(nid) if hasattr(g, "degree") else 0
                examples.append(f"{nid} (deg {deg})")
        if not items:
            return []
        return [CurationProposal(
            proposal_id=new_proposal_id(),
            curator=self.name,
            instrument=Instrument.METADATA,
            confidence=Confidence.DETERMINISTIC,
            batch=True,
            title=f"Quarantine {len(items)} when-word graph nodes",
            evidence=(
                "Graph nodes that are bare temporal words (today, tomorrow, on thursday…) "
                "per the deployed graph_utils.is_temporal_deictic — never entities. "
                "Quarantine flags the node; every edge touching it stops rendering. "
                f"Reversible with Undo. {'; '.join(examples)}"
            ),
            items=items,
        )]
