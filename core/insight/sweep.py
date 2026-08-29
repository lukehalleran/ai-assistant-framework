"""
core/insight/sweep.py

Module Contract
- Purpose: Run the ungated evidence sweep for an insight FacetPlan across ALL
  stores: six Chroma collections (semantic), the corpus (exact keyword — the
  raw-text channel that defeats the fact extractor's triple-shape bias), the
  knowledge graph (1-hop from resolved entities), plus temporal expansion
  around the strongest conversation hits.
- Inputs: FacetPlan + live components (chroma_store, corpus_manager,
  graph_memory, entity_resolver, memory_expander) + caps dict (defaults from
  INSIGHT_* config).
- Outputs: deduped, date-sorted (newest first), snippet-clipped list of
  EvidenceItem, proportionally trimmed to the total cap.
- Key behaviors:
  * NO cosine gate anywhere — generous caps replace gating (the memory gate's
    per-doc threshold is structurally unable to pass a collective-signal
    evidence set; that is the whole reason this mode exists).
  * Graph traversal skips the 'user' star hub and any node with degree ≥
    GRAPH_EXPANSION_HUB_DEGREE, and honors read-time TTL
    (_edge_is_stale_transient) — same discipline as query expansion.
  * Expansion only around top conversation hits (EXPANDABLE_COLLECTIONS;
    threads are NEVER expanded).
  * Wall-clock bounded: on sweep_timeout_s the accumulated partial evidence
    is returned, never an exception.
  * Chroma/corpus calls run in threads (asyncio.to_thread) and facets run
    concurrently.
- Side effects: read-only against every store.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Optional

from core.insight.types import EvidenceItem, FacetPlan, FacetQuery
from utils.logging_utils import get_logger

logger = get_logger("insight_sweep")

SWEEP_COLLECTIONS = (
    "conversations", "summaries", "reflections",
    "facts", "obsidian_notes", "threads",
)


def default_caps() -> dict:
    from config import app_config as cfg
    return {
        "per_facet_cap": cfg.INSIGHT_PER_FACET_CAP,
        "total_evidence_cap": cfg.INSIGHT_TOTAL_EVIDENCE_CAP,
        "evidence_snippet_chars": cfg.INSIGHT_EVIDENCE_SNIPPET_CHARS,
        "keyword_scan_max": cfg.INSIGHT_KEYWORD_SCAN_MAX,
        "expand_top_k": cfg.INSIGHT_EXPAND_TOP_K,
        "expand_window": cfg.INSIGHT_EXPAND_WINDOW,
        "sweep_timeout_s": cfg.INSIGHT_SWEEP_TIMEOUT_S,
    }


def _meta_date(metadata: dict) -> Optional[str]:
    for key in ("timestamp", "date", "created_at", "last_seen"):
        v = (metadata or {}).get(key)
        if v:
            return str(v)
    return None


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _dedupe_key(item: EvidenceItem) -> str:
    if item.doc_id:
        return f"id:{item.doc_id}"
    return "tx:" + hashlib.sha1(item.text[:160].lower().encode()).hexdigest()


async def run_sweep(
    plan: FacetPlan,
    *,
    chroma_store,
    corpus_manager,
    graph_memory=None,
    entity_resolver=None,
    memory_expander=None,
    caps: Optional[dict] = None,
) -> list[EvidenceItem]:
    """Run the full sweep. Returns partial evidence on timeout, never raises."""
    caps = {**default_caps(), **(caps or {})}
    collected: list[EvidenceItem] = []  # shared accumulator — survives timeout

    async def _sweep_facet(facet: FacetQuery) -> None:
        # --- Chroma semantic, all six collections, NO gate ---
        async def _query(coll: str):
            try:
                return await asyncio.to_thread(
                    chroma_store.query_collection, coll, facet.query_text,
                    caps["per_facet_cap"],
                )
            except Exception as e:
                logger.debug(f"[Insight] query_collection({coll}) failed: {e}")
                return []

        chroma_results = await asyncio.gather(*[_query(c) for c in SWEEP_COLLECTIONS])
        conversation_hits: list[dict] = []
        for coll, rows in zip(SWEEP_COLLECTIONS, chroma_results):
            for row in rows or []:
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                collected.append(EvidenceItem(
                    doc_id=row.get("id"),
                    text=content,
                    date=_meta_date(row.get("metadata")),
                    collection=coll,
                    speaker="",
                    facet=facet.name,
                ))
                if coll == "conversations":
                    conversation_hits.append(row)

        # --- Corpus exact-keyword (raw-text channel, speaker-attributed) ---
        if facet.keywords and corpus_manager is not None:
            try:
                hits = await asyncio.to_thread(
                    lambda: corpus_manager.search_keyword(
                        facet.keywords, max_results=caps["keyword_scan_max"],
                        context_chars=caps["evidence_snippet_chars"],
                    )
                )
            except Exception as e:
                logger.debug(f"[Insight] corpus search_keyword failed: {e}")
                hits = []
            for h in hits:
                ts = h.get("timestamp")
                collected.append(EvidenceItem(
                    doc_id=None,
                    text=h.get("excerpt", ""),
                    date=ts.isoformat() if isinstance(ts, datetime) else (str(ts) if ts else None),
                    collection="corpus",
                    speaker=h.get("speaker", ""),
                    facet=facet.name,
                ))

        # --- Graph 1-hop from resolved entities (hub-aware, TTL-aware) ---
        if facet.entities and graph_memory is not None:
            _sweep_graph(facet, collected)

        # --- Temporal expansion around top conversation hits ---
        if memory_expander is not None and caps["expand_top_k"] > 0:
            for row in conversation_hits[:caps["expand_top_k"]]:
                try:
                    exp = await asyncio.to_thread(
                        memory_expander.expand, row.get("id"),
                        caps["expand_window"], "conversations",
                    )
                except Exception as e:
                    logger.debug(f"[Insight] expansion failed: {e}")
                    continue
                for turn in (exp or {}).get("turns", []):
                    content = (turn.get("content") or "").strip()
                    if not content:
                        continue
                    collected.append(EvidenceItem(
                        doc_id=turn.get("id"),
                        text=content,
                        date=_meta_date(turn.get("metadata")) or _meta_date(turn),
                        collection="conversations",
                        speaker="",
                        facet=facet.name,
                    ))

    def _sweep_graph(facet: FacetQuery, out: list[EvidenceItem]) -> None:
        from config.app_config import GRAPH_EXPANSION_HUB_DEGREE
        from memory.stance_classifier import effective_stance

        now = datetime.now()
        for mention in facet.entities:
            try:
                entity_id = (
                    entity_resolver.resolve(mention) if entity_resolver else mention.lower()
                )
                if not entity_id or entity_id == "user":
                    continue  # never fan out from the user star hub
                node = graph_memory.get_entity(entity_id)
                if node is None:
                    continue
                try:
                    degree = graph_memory.graph.degree(entity_id)
                except Exception:
                    degree = 0
                if degree >= GRAPH_EXPANSION_HUB_DEGREE:
                    logger.debug(f"[Insight] Skipping hub entity {entity_id} (deg={degree})")
                    continue
                for edge in graph_memory.get_relations(entity_id, direction="both"):
                    if graph_memory._edge_is_stale_transient(edge, now):
                        continue
                    src = graph_memory.get_entity(edge.source_id)
                    tgt = graph_memory.get_entity(edge.target_id)
                    sentence = edge.to_natural_language(
                        src.display_name if src else edge.source_id,
                        tgt.display_name if tgt else edge.target_id,
                    )
                    ts = edge.last_seen or edge.first_seen
                    out.append(EvidenceItem(
                        doc_id=f"edge:{edge.edge_key()}",
                        text=sentence,
                        date=ts.isoformat() if isinstance(ts, datetime) else None,
                        collection="graph",
                        speaker="",
                        is_appraisal=effective_stance(edge.metadata) == "appraisal",
                        facet=facet.name,
                    ))
            except Exception as e:
                logger.debug(f"[Insight] graph sweep for {mention!r} failed: {e}")

    async def _run_all() -> None:
        await asyncio.gather(*[_sweep_facet(f) for f in plan.facets])

    try:
        await asyncio.wait_for(_run_all(), timeout=caps["sweep_timeout_s"])
    except asyncio.TimeoutError:
        logger.warning(
            f"[Insight] Sweep timed out at {caps['sweep_timeout_s']}s — "
            f"returning {len(collected)} partial evidence items"
        )
    except Exception as e:
        logger.warning(f"[Insight] Sweep error: {e} — returning partial evidence")

    return _finalize(collected, caps)


def _finalize(items: list[EvidenceItem], caps: dict) -> list[EvidenceItem]:
    """Dedupe → snippet-clip → date-sort (newest first) → proportional trim."""
    seen: set[str] = set()
    deduped: list[EvidenceItem] = []
    for item in items:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        item.text = _clip(item.text, caps["evidence_snippet_chars"])
        deduped.append(item)

    deduped.sort(key=lambda i: i.date or "", reverse=True)

    total_cap = caps["total_evidence_cap"]
    if len(deduped) <= total_cap:
        return deduped

    # Proportional per-collection allocation (floor 1) so one chatty store
    # can't crowd out the graph/notes channels entirely.
    by_coll: dict[str, list[EvidenceItem]] = {}
    for item in deduped:
        by_coll.setdefault(item.collection, []).append(item)
    total = len(deduped)
    kept: list[EvidenceItem] = []
    for coll, group in by_coll.items():
        quota = max(1, int(round(total_cap * len(group) / total)))
        kept.extend(group[:quota])
    kept.sort(key=lambda i: i.date or "", reverse=True)
    dropped = len(deduped) - len(kept[:total_cap])
    if dropped > 0:
        logger.info(f"[Insight] Evidence trimmed: kept {total_cap} of {len(deduped)} ({dropped} dropped)")
    return kept[:total_cap]
