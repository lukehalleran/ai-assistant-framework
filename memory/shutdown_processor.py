# memory/shutdown_processor.py
"""
Module Contract
- Purpose: End-of-session memory consolidation: block summaries, fact extraction (rule + LLM),
  user profile updates, procedural skills, code proposals, thread extraction/resolution,
  implementation tracking, synthesis dreaming (cross-domain candidate generation + filtering),
  knowledge graph save, claim index save, cross-collection dedup (dry-run preview only),
  and session reflections.
- Class: ShutdownProcessor(corpus_manager, chroma_store, model_manager, user_profile,
    thread_store, claim_index, fact_verifier)
- Key methods:
  - process_shutdown_memory(session_conversations, topic, **kwargs) -> Dict[str, Any]
    Main entry: runs all consolidation phases and returns results summary.
    Phases are parallelized via asyncio.gather where independent:
      Phase A (parallel): facts + LLM facts + behavioral patterns + skills
        (LLM facts also feed the knowledge graph via storage._ingest_fact_to_graph —
         same hook as the per-turn path; entity–entity triples become lateral edges)
      Phase B (parallel): proposals + impl tracking + threads + wiki enrichment
        + wikidata enrichment (anchored typed edges from the offline cache)
      Phase C (sequential): graph save, category cache, cross-dedup
  - run_synthesis_dreaming() -> None
    Standalone step (NOT part of process_shutdown_memory): cross-domain candidate
    generation + filter pipeline. Driven separately by main.py under its own
    SYNTHESIS_DREAM_TIMEOUT_S budget so its slow per-candidate LLM filter isn't
    cancelled by the shared SHUTDOWN_TASK_TIMEOUT_S reflection/fact budget. Auto-
    halts if audit FP rate > SYNTHESIS_AUDIT_FP_HALT_THRESHOLD (checked via
    synthesis_memory.get_audit_stats() before generation).
  - run_shutdown_reflection(session_conversations, topic, session_context) -> Optional[str]
    Generates topic-specific session reflections with entity-rich content
    (SESSION TOPIC, KEY ENTITIES, WHAT HAPPENED, PATTERNS & INSIGHTS).
    Stored as retrieval text via _extract_reflection_retrieval_text() for embedding,
    with structured metadata (primary_topic, entities, themes, emotional_tone, project_area).
- Internal phases (called by process_shutdown_memory):
  - _generate_block_summaries(entries, conversations, topic) — fixed-size block summaries
  - _store_summary(summary_text, N, b, start, end, block) — persists with source_doc_ids backlinks,
    claim extraction, staleness_ratio, and temporal anchors
  - _extract_session_facts(session_conversations) — rule-based fact extraction with fact verification gate;
    forwards source_excerpt from MemoryNode metadata to ChromaDB
  - _extract_llm_facts(session_conversations) — LLM-assisted triple extraction with fact verification gate;
    injects existing profile facts so LLM reuses relation names for updates/cancellations;
    forwards source_excerpt (claim-bearing span) + source_role/turn_id/turn_index/support/anchor
    provenance from triples to ChromaDB and UserProfile (2026-09-02: responses are never extraction input);
    2026-08-23: forwards each triple's "stance" and a capture_tone derived by
    _capture_tone_for_triple (joins the triple's object/subject back to the
    session corpus entry's is_heavy_topic flag; unmatched → "unknown") into
    fact metadata + graph edge metadata;
    2026-09-27 (E3, BC-58): conv_pairs are stamped with each turn's own
    "turn_id" so LLMFactExtractor can render per-message timestamps and
    resolve relative dates ("tomorrow") against the message's own time, not
    shutdown-extraction time; each surviving triple's "timestamp" (the
    matched source message's turn time) is forwarded to both ChromaDB and
    UserProfile.add_fact, which resolves the relative word deterministically
    against it — falls back to extraction time (logged) only when the
    provenance join found no turn_id to anchor to
  - _extract_procedural_skills(session_conversations) — adaptive workflow extraction
  - _generate_proposals(session_conversations) — goal-directed code change proposals
  - _check_implementation_tracking() — lightweight proposal implementation detection
  - _process_open_threads(session_conversations) — resolution detection → extraction → cap enforcement
  - _save_knowledge_graph() — flush graph + aliases to disk
  - _run_cross_collection_dedup() — dry-run preview only (never auto-deletes)
- Note: Entity facts (non-user subjects) go to ChromaDB only, NOT to UserProfile.
  Only facts with subject="user" are passed to add_facts_batch().
- Summary backlinks: _store_summary() captures source conversation doc IDs for expand_memory drill-down.
- Dependencies:
  - memory.storage.multi_collection_chroma_store, memory.fact_extractor, memory.llm_fact_extractor,
    memory.fact_verification, memory.claim_tracker, memory.thread_store, memory.thread_extractor,
    knowledge.proposal_generator, knowledge.implementation_detector, memory.cross_deduplicator
- Side effects:
  - ChromaDB writes (summaries, facts, skills, proposals, reflections)
  - User profile updates (user-only facts batch)
  - Knowledge graph + alias + claim index JSON persistence
  - LLM API calls for summarization, fact extraction, skill extraction, proposals, reflections
  - Cross-collection dedup preview logging (dry-run only, never auto-deletes)
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from utils.logging_utils import get_logger
from memory.fact_source import authored_text, is_test_turn
import json
import time

from memory import summary_coverage as _cov

logger = get_logger("shutdown_processor")


def _log_shutdown_llm_failure(step: str, exc: BaseException) -> None:
    """LOUD, greppable marker for shutdown-time LLM failures.

    A provider outage during shutdown silently loses the session's
    facts/summaries/reflections — the 2026-08-24 18:59 shutdown hit three
    OpenRouter 402s that were only visible at DEBUG level, so nothing
    surfaced that the session's memory processing had failed (2026-08-28
    retrospective). grep '[SHUTDOWN-LLM-FAILURE]' in daemon_debug*.log.
    """
    text = str(exc)
    lowered = text.lower()
    if "402" in text or "payment required" in lowered or "credit" in lowered:
        logger.error(
            f"[SHUTDOWN-LLM-FAILURE] {step}: API credits/payment failure — "
            f"this session's {step} output is LOST: {text[:200]}"
        )
    else:
        logger.error(f"[SHUTDOWN-LLM-FAILURE] {step}: {text[:200]}")

# --- Environment-driven configuration ---
REFLECTIONS_ENABLED = os.getenv("REFLECTIONS_ENABLED", "1").strip() not in ("0", "false", "no", "off")
REFLECTION_MAX_TOKENS = int(os.getenv("REFLECTION_MAX_TOKENS", "300"))
REFLECTION_MODEL_ALIAS = os.getenv("LLM_REFLECTION_ALIAS", os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini"))
REFLECTION_MIN_EXCHANGES = int(os.getenv("REFLECTION_MIN_EXCHANGES", "4"))
# Max proposals implementation-tracked per shutdown (oldest-checked first);
# bounds the phase-B wall time under the shared SHUTDOWN_TASK_TIMEOUT_S.
IMPL_TRACKING_MAX_PER_SHUTDOWN = 40
SUMMARY_MAX_BLOCKS_PER_SHUTDOWN = int(os.getenv("SUMMARY_MAX_BLOCKS_PER_SHUTDOWN", "1"))
REFLECTION_MAX_EXCERPTS = int(os.getenv("REFLECTION_MAX_EXCERPTS", "0") or 0)
REFLECTION_MAX_SUMMARIES = int(os.getenv("REFLECTION_MAX_SUMMARIES", "0") or 0)
REFLECTION_FALLBACK_RECENT = int(os.getenv("REFLECTION_FALLBACK_RECENT", "0") or 0)


class ShutdownProcessor:
    """
    Handles end-of-session memory consolidation and reflection.

    Responsibilities:
    - Block-based summary generation (fixed-size blocks, N items each)
    - Fact extraction from session conversations
    - LLM-assisted triple extraction
    - User profile updates
    - Procedural skill extraction (adaptive workflows)
    - Open thread extraction and resolution detection (via thread_store)
    - Session-end reflection generation

    Args (constructor):
        thread_store: Optional ThreadStore for persisting open threads
            (commitments, deadlines, unanswered questions) extracted at shutdown.
    """

    def __init__(
        self,
        corpus_manager,
        chroma_store,
        consolidator,
        fact_extractor,
        model_manager,
        user_profile,
        storage,
        session_start: datetime,
        memory_coordinator=None,
        thread_store=None,
        claim_index=None,
    ):
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.consolidator = consolidator
        self.fact_extractor = fact_extractor
        self.model_manager = model_manager
        self.user_profile = user_profile
        self._storage = storage
        self.session_start = session_start
        self.memory_coordinator = memory_coordinator
        self.thread_store = thread_store
        self.claim_index = claim_index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_summary(e: Dict) -> bool:
        typ = (e.get('type') or '').lower()
        tags = e.get('tags') or []
        return ("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)

    @staticmethod
    def _is_reflection(e: Dict) -> bool:
        typ = (e.get('type') or '').lower()
        tags = [str(t).lower() for t in (e.get('tags') or [])]
        return (typ == 'reflection') or ('type:reflection' in tags)

    @staticmethod
    def _ts(e):
        ts = e.get('timestamp')
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.min
        if not isinstance(ts, datetime):
            ts = datetime.min
        return ts

    # ------------------------------------------------------------------
    # process_shutdown_memory
    # ------------------------------------------------------------------

    async def process_shutdown_memory(
        self, session_conversations: list[dict] | None = None
    ):
        """Create any due summaries in fixed-size blocks and run fact extraction.

        Behavior:
          - Let N = self.consolidator.consolidation_threshold (e.g., 3, 10, 20)
          - Let T = total non-summary conversation entries in the corpus
          - Let S = number of consolidator-produced summaries already stored
          - Target summaries = floor(T / N)
          - Generate (Target - S) new summaries, each summarizing a disjoint block
            of size N, from oldest to newest (so we never duplicate prior work).
        """
        # An explicit list is filtered at the entry so every downstream method inherits it.
        if session_conversations:
            session_conversations = [e for e in session_conversations if not is_test_turn(e)]
        try:
            # 1) Summary blocks (ONE method shared with scripts/backfill_summaries.py)
            await self.summarize_uncovered_blocks(max(1, SUMMARY_MAX_BLOCKS_PER_SHUTDOWN))

            # ── Parallel Phase A: Extraction ──────────────────────────
            # Facts, LLM facts, and skills are
            # independent LLM calls that can overlap.
            phase_a = await asyncio.gather(
                self._extract_session_facts(session_conversations),
                self._extract_llm_facts(session_conversations),
                self._extract_procedural_skills(session_conversations),
                return_exceptions=True,
            )
            for r in phase_a:
                if isinstance(r, Exception):
                    _log_shutdown_llm_failure("extraction", r)

            # ── Parallel Phase B: Generation + maintenance ────────────
            # Proposals, threads, and wiki enrichment are independent of each
            # other and of Phase A results. Synthesis dreaming is deliberately
            # NOT here: its per-candidate LLM filter is too slow for the shared
            # SHUTDOWN_TASK_TIMEOUT_S budget (it was cancelled mid-flight on
            # every exit). It now runs as a separate, longer-bounded step driven
            # by main.py via run_synthesis_dreaming(), outside this budget.
            phase_b = await asyncio.gather(
                self._generate_proposals(session_conversations),
                self._check_implementation_tracking(),
                self._process_open_threads(session_conversations),
                self._run_wiki_enrichment(),
                self._run_wikidata_enrichment(),
                return_exceptions=True,
            )
            for r in phase_b:
                if isinstance(r, Exception):
                    _log_shutdown_llm_failure("generation", r)

            # ── Sequential Phase C: Persistence ───────────────────────
            # Graph save must follow wiki enrichment (Phase B).
            self._save_knowledge_graph()
            try:
                from memory.user_profile_schema import save_category_cache
                save_category_cache()
            except Exception as e:  # degrades: category cache not saved, relations re-categorized next session
                logger.debug(f"[Shutdown] Category cache save failed (non-fatal): {e}")

            # Dedup runs last after all writes complete.
            await self._run_cross_collection_dedup()

            # Fire-and-forget: spawn detached agent_branch work (queued objectives
            # and/or goal-driven). NEVER blocks shutdown (agent_branch is minutes-long).
            self._maybe_spawn_agent_branch()

            logger.info("[Shutdown] Memory processing complete")

        except Exception as e:
            logger.error(f"Shutdown processing error: {e}")

    def _maybe_spawn_agent_branch(self) -> None:
        """If AGENT_BRANCH_SHUTDOWN_ENABLED, fire-and-forget DETACHED agent_branch
        work and return immediately — agent_branch runs are minutes-long (clones +
        containers + LLM) and must NEVER block shutdown. Two opt-in modes, both
        default OFF:
          - the objective QUEUE (explicit objectives) — runs only if non-empty.
          - GOAL-DRIVEN (AGENT_BRANCH_SHUTDOWN_GOALS): each lens derives its OWN
            objective from its goals + GOALS.md, implements + regression-gates it.
        Both ingest survivors into the proposal store for review next session.
        Requires podman + an LLM key; otherwise logs and skips. Cost is bounded
        (queue cap / N lenses) and nothing runs unless explicitly enabled."""
        if not os.getenv("AGENT_BRANCH_SHUTDOWN_ENABLED"):
            return
        try:
            import shutil
            if shutil.which("podman") is None:
                logger.info("[Shutdown] agent_branch: podman unavailable, skipping")
                return
            if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
                logger.info("[Shutdown] agent_branch: no LLM key, skipping")
                return
            import subprocess
            import sys
            from pathlib import Path
            Path("logs").mkdir(exist_ok=True)

            def _spawn(argv, logname, desc):
                log = open(f"logs/{logname}", "a", encoding="utf-8")  # noqa: SIM115
                subprocess.Popen(  # noqa: S603 — fixed argv, detached, never awaited
                    [sys.executable, "-m", *argv],
                    stdout=log, stderr=log, start_new_session=True)
                logger.info("[Shutdown] spawned detached %s; survivors appear in the "
                            "Proposals tab next session", desc)

            # explicit objective queue (runs only if something was queued)
            from agent_branch.queue import ObjectiveQueue
            if ObjectiveQueue.load().pending():
                m = os.getenv("AGENT_BRANCH_SHUTDOWN_MAX", "2")
                _spawn(["agent_branch.run_queue", "--max", str(m)],
                       "agent_branch_queue.log", f"agent_branch queue run (max={m})")

            # goal-driven: each lens derives + runs its own objective from its goals
            if os.getenv("AGENT_BRANCH_SHUTDOWN_GOALS"):
                lenses = os.getenv("AGENT_BRANCH_SHUTDOWN_LENSES",
                                   "reliability,coverage,capability")
                _spawn(["agent_branch.goal_runner", "--lenses", lenses],
                       "agent_branch_goals.log", f"goal-driven run (lenses={lenses})")
        except Exception as e:  # noqa: BLE001 — must never block shutdown
            logger.warning("[Shutdown] agent_branch spawn failed: %s", e)

    def _select_uncovered(self, cap: int):
        """THE selection for summary blocks: (non_summ, N, watermark, blocks), or None
        when coverage is unreadable (callers must not summarize blind). Pure read."""
        corpus = list(self.corpus_manager.derivable_entries())

        non_summ = [
            e for e in corpus
            if (not self._is_summary(e)) and (not self._is_reflection(e))
        ]
        non_summ.sort(key=self._ts)

        N = max(1, int(getattr(self.consolidator, 'consolidation_threshold', 10)))

        # Coverage is judged by CONTENT time (summary anchors), never by a
        # positional block index into the rolling corpus (BC-95).
        ranges = _cov.covered_ranges(self.chroma_store)
        if ranges is None:
            return None
        watermark = _cov.coverage_watermark(ranges)
        blocks = _cov.uncovered_blocks(non_summ, watermark, N, cap, _cov.entry_time)
        return non_summ, N, watermark, blocks

    def plan_uncovered_blocks(self, cap: int) -> list[dict]:
        """Pure-read plan of the blocks `summarize_uncovered_blocks(cap)` would summarize
        (no LLM, no writes): [{index, n_entries, start, end}], oldest first."""
        sel = self._select_uncovered(cap)
        if sel is None:
            return []
        _, _, _, blocks = sel
        plan = []
        for i, block in enumerate(blocks):
            times = [_cov.entry_time(e) for e in block if _cov.entry_time(e) != datetime.min]
            plan.append({
                'index': i,
                'n_entries': len(block),
                'start': min(times) if times else None,
                'end': max(times) if times else None,
            })
        return plan

    async def summarize_uncovered_blocks(self, cap: int) -> int:
        """Summarize up to ``cap`` full uncovered blocks (oldest first); returns the
        number of summaries created. The shutdown and the offline backfill both call
        this (one chokepoint: selection, prompt and storage are never re-derived)."""
        sel = self._select_uncovered(cap)
        if sel is None:
            logger.info(
                "[Shutdown] Summary coverage unreadable — skipping summary blocks this shutdown"
            )
            return 0
        non_summ, N, watermark, blocks = sel
        if blocks:
            return await self._generate_block_summaries(blocks, N, non_summ, watermark)
        pending = _cov.remaining_full_blocks(non_summ, watermark, N, _cov.entry_time)
        newer = len([e for e in non_summ if watermark is None or _cov.entry_time(e) > watermark])
        since_last = newer % N
        logger.info(
            "[Shutdown] No new summaries due (N=%s, T=%s); since_last=%s, remaining=%s, backlog=%s",
            N, len(non_summ), since_last, (N - since_last) if since_last else N, pending
        )
        return 0

    async def _generate_block_summaries(self, blocks, N, non_summ=None, watermark=None):
        """Summarize each uncovered block (lists of corpus entries, oldest first)."""
        created = 0
        for block in blocks:
            if not block:
                continue
            content_list = []
            for c in block:
                q = (c.get('query') or '').strip()
                a = (c.get('response') or '').strip()
                if not (q or a):
                    continue
                content_list.append({'content': f"User: {q}\nAssistant: {a}", 'q': q, 'a': a})
            if not content_list:
                continue

            summary_text = await self._consolidate_block(content_list)
            summary_text = (summary_text or '').strip()
            # Stop at the first block that did not land: coverage is a watermark, so a
            # later block stored after a failed one would move it past the gap and the
            # failed block would never be summarized (multi-block runs, 2026-10-15).
            if not summary_text or not self._store_summary(summary_text, block):
                logger.warning(
                    "[Shutdown] Summary block not stored (empty/junk/duplicate) — stopping this "
                    "run so later blocks cannot move the watermark past it"
                )
                break
            created += 1

        # Backlog after this shutdown = full blocks still newer than the content
        # just summarized (the blocks handed in were the oldest uncovered ones).
        remaining = 0
        if non_summ is not None:
            covered = {id(e) for blk in blocks for e in blk}
            rest = [e for e in non_summ
                    if id(e) not in covered and (watermark is None or _cov.entry_time(e) > watermark)]
            remaining = len(rest) // N
        logger.info(
            f"[Shutdown] Created {created} summary block(s) (N={N}); backlog={remaining}"
        )
        return created

    async def _consolidate_block(self, content_list: list) -> Optional[str]:
        """Compress one block of exchanges into an extractive summary.

        Delegates to MemoryConsolidator.consolidate_memories() \u2014 the single
        consolidation path \u2014 so the extractive prompt lives in exactly one
        place (memory_consolidator.EXTRACTIVE_SUMMARY_PROMPT). Tiny blocks and
        the missing-model-manager fallback are handled inside that path.
        """
        try:
            summary_node = await self.consolidator.consolidate_memories(content_list)
            if summary_node and getattr(summary_node, 'content', None):
                return summary_node.content.strip()
        except Exception as e:
            logger.warning(f"[Shutdown] Block consolidation failed: {e}")
        return None

    def _store_summary(self, summary_text: str, block: list) -> bool:
        """Store summary into corpus and Chroma. True only when the Chroma copy — the one
        summary coverage reads its anchors from — was written (2026-10-15)."""
        # Third storage path (2026-08-22): this writes chroma via raw
        # add_to_collection, BYPASSING chroma_store.add_summary — which is
        # where the stream-artifact strip and junk check live. Two summaries
        # created at 14:27/14:47 landed with kimi-3's leading "<|sep|>"
        # despite the fix being live. Sanitize + junk-reject here so all
        # three consumers (corpus, chroma, claim index) see clean text.
        try:
            from core.response_parser import ResponseParser
            summary_text = ResponseParser.sanitize_for_storage(summary_text or "")
            from memory.utils import is_junk_summary
            if is_junk_summary(summary_text):
                logger.warning("[Shutdown] Rejected junk summary at _store_summary")
                return False
        except ImportError:
            pass
        try:
            existing_texts = set()
            try:
                for s in self.corpus_manager.get_summaries(500):
                    txt = (s.get('content') or '').strip()
                    if txt:
                        existing_texts.add(txt)
            except Exception as e:
                logger.warning(f"[Shutdown] Could not retrieve summaries for dedup check: {e}")
            if summary_text in existing_texts:
                return False
            entry = {
                'content': summary_text,
                'timestamp': datetime.now(),
                'type': 'summary',
                'tags': ['summary:consolidated', 'source:consolidator'],
            }
            # Content span (BC-96): corpus readers sort/render by this, not by 'timestamp'.
            block_times = [_cov.entry_time(e) for e in block if _cov.entry_time(e) != datetime.min]
            if block_times:
                entry['covers_start'] = min(block_times).isoformat()
                entry['covers_end'] = max(block_times).isoformat()
            self.corpus_manager.add_summary(entry)
        except (AttributeError, TypeError) as e:
            logger.debug(f"[Shutdown] Failed to add summary to corpus: {e}")

        try:
            md = {
                'timestamp': datetime.now().isoformat(),
                'type': 'summary',
                'importance_score': 0.7,
                'tags': 'summary:consolidated,source:consolidator',
                'memory_count': len(block),
                'staleness_ratio': 0.0,
            }
            # Add temporal anchors from the block's timestamp range
            try:
                block_times = [_cov.entry_time(e) for e in block if _cov.entry_time(e) != datetime.min]
                if block_times:
                    md['temporal_anchor_start'] = min(block_times).isoformat()
                    md['temporal_anchor_end'] = max(block_times).isoformat()
            except Exception:  # degrades: summary lacks temporal anchor metadata, so it sorts without coverage dates
                pass

            # Capture source conversation doc IDs for expand_memory drill-down
            try:
                ts_start = md.get('temporal_anchor_start')
                ts_end = md.get('temporal_anchor_end')
                if ts_start and ts_end and hasattr(self.chroma_store, 'get_ids_by_timestamp_range'):
                    source_ids = self.chroma_store.get_ids_by_timestamp_range(
                        'conversations', ts_start, ts_end
                    )
                    if source_ids:
                        md['source_doc_ids'] = ','.join(source_ids)
                        logger.debug(f"[Shutdown] Summary linked to {len(source_ids)} source conversations")
            except Exception as se:  # degrades: summary stored without source conversation ids, so drill-down is unavailable
                logger.debug(f"[Shutdown] Could not capture source doc IDs: {se}")

            doc_id = None
            if hasattr(self.chroma_store, 'add_to_collection'):
                doc_id = self.chroma_store.add_to_collection('summaries', summary_text, md)

            # Extract and register claims in the ClaimIndex
            if doc_id and self.claim_index:
                try:
                    from config.app_config import STALENESS_ENABLED
                    if STALENESS_ENABLED:
                        from memory.claim_tracker import extract_claims_from_text
                        entity_resolver = None
                        mc = self.memory_coordinator
                        if mc and getattr(mc, 'entity_resolver', None):
                            entity_resolver = mc.entity_resolver
                        claims = extract_claims_from_text(summary_text, entity_resolver)
                        if claims:
                            self.claim_index.add_claims(doc_id, 'summaries', claims)
                            # Store claim hashes in metadata for cross-reference
                            claim_hashes = ",".join(c.claim_hash for c in claims)
                            self.chroma_store.update_metadata('summaries', doc_id, {
                                'embedded_claims': claim_hashes,
                            })
                            logger.debug(f"[Staleness] Registered {len(claims)} claims for summary {doc_id}")
                except Exception as ce:  # degrades: summary stored without embedded claim hashes, so staleness tracking skips it
                    logger.debug(f"[Staleness] Claim extraction for summary failed: {ce}")
            return bool(doc_id)

        except (AttributeError, TypeError) as e:
            logger.debug(f"[Shutdown] Failed to add summary to chroma: {e}")
        return False

    async def _extract_session_facts(self, session_conversations):
        """Extract facts from this session's user turns."""
        session_recent: list[Dict]
        if isinstance(session_conversations, list) and session_conversations:
            session_recent = list(session_conversations)
        else:
            try:
                corpus = list(self.corpus_manager.derivable_entries())
            except (AttributeError, TypeError):
                corpus = []
            session_recent = [
                e for e in corpus
                if (not self._is_summary(e)) and (not self._is_reflection(e))
            ]
            try:
                session_recent = [e for e in session_recent if self._ts(e) >= self.session_start]
            except (ValueError, TypeError):
                pass
            session_recent.sort(key=self._ts, reverse=True)

        for conv in session_recent[:10]:
            try:
                # authored_text: an attachment-only turn (user_text == "")
                # has no authored words — never extract from the blob.
                q = authored_text(conv).strip()
                if not q:
                    continue
                r = (conv.get('response') or '').strip()
                # Skip API error responses
                if r.startswith("[API Error]"):
                    r = ""
                _observed = self._ts(conv)
                # THE admission chokepoint (2026-10-20, BC-58): same skip
                # reason / source reduction / paste guard as the per-turn path.
                from memory.memory_storage import (  # lazy import: patch-point
                    prepare_fact_extraction_source, admit_extracted_facts,
                )
                _skip, _src = prepare_fact_extraction_source(q)
                if _skip:
                    logger.info(f"[Shutdown] Fact extraction skipped: {_skip}")
                    continue
                facts = await self.fact_extractor.extract_facts(
                    _src, r, observed_at=_observed if _observed != datetime.min else None,
                )
                facts = admit_extracted_facts(q, facts or [])
            except (AttributeError, RuntimeError, ValueError) as e:
                logger.debug(f"[Shutdown] Fact extraction failed: {e}")
                facts = []
            for fact in (facts or []):
                try:
                    fact_content = getattr(fact, 'content', str(fact))
                    # Verification gate (if available via storage)
                    verifier = getattr(self._storage, 'fact_verifier', None)
                    if verifier:
                        try:
                            from memory.fact_verification import FactVerdict
                            from memory.cross_deduplicator import CrossCollectionDeduplicator
                            s, p, o = CrossCollectionDeduplicator._extract_triple({
                                "content": fact_content, "metadata": {},
                            })
                            vr = await verifier.verify(
                                subject=s, predicate=p, object_val=o,
                                fact_text=fact_content,
                                source="shutdown_extraction", confidence=0.7,
                            )
                            if vr.verdict == FactVerdict.REJECT:
                                logger.debug(f"[Shutdown] Fact rejected by verifier: {fact_content[:80]}")
                                continue
                            if vr.verdict == FactVerdict.STORE_AND_FLAG:
                                for cand in vr.conflicting_candidates:
                                    if cand.doc_id:
                                        try:
                                            supersede_md = {"superseded_by": fact_content[:200]}
                                            supersede_md.update(vr.metadata_updates)
                                            self.chroma_store.update_metadata("facts", cand.doc_id, supersede_md)
                                        except Exception:  # degrades: superseded fact not flagged in the facts collection, old value may resurface
                                            pass
                        except Exception as ve:  # degrades: superseded fact not flagged in the facts collection, old value may resurface
                            logger.debug(f"[Shutdown] Verification failed, proceeding: {ve}")

                    # Forward source_excerpt if available in metadata
                    src_dict = {
                        "source": "shutdown_extraction",
                        "confidence": 0.7,
                    }
                    fact_md = getattr(fact, 'metadata', None) or {}
                    src_exc = fact_md.get("source_excerpt", "")
                    if src_exc:
                        src_dict["source_excerpt"] = src_exc[:200]
                    # Claim temporal kind (2026-09-06): forward to the facts
                    # collection like the per-turn path does — new keys only.
                    for _k in ("claim_kind", "event_date", "observed_at"):
                        if fact_md.get(_k):
                            src_dict[_k] = fact_md[_k]
                    result = self.chroma_store.add_fact(
                        fact=fact_content,
                        source=src_dict,
                        confidence=0.7
                    )
                    if result is None:
                        logger.debug(f"[Shutdown] Fact skipped as duplicate: {fact_content}")
                except (AttributeError, TypeError, ValueError):
                    continue

    async def _extract_llm_facts(self, session_conversations):
        """Optional LLM-assisted triple extraction over recent user messages."""
        try:
            _enabled = int(os.getenv("LLM_FACTS_ENABLED", "1")) == 1
        except (ValueError, TypeError):
            _enabled = False

        if not _enabled or not self.model_manager:
            return

        # Coverage defaults raised 2026-08-05 (were 12 turns / 10 triples /
        # 6000 chars): with FULL responses in each entry the old budget fit
        # ~4-5 pairs and dropped the newest — weeks of "my doctor doesn't
        # respond" never reached the extractor. Responses are EXCLUDED from
        # the prompt since 2026-09-02 (not evidence), so 9000 chars covers ~30 turns.
        try:
            last_turns = int(os.getenv("LLM_FACTS_LAST_TURNS", "30"))
        except (ValueError, TypeError):
            last_turns = 30
        try:
            max_triples = int(os.getenv("LLM_FACTS_MAX_TRIPLES", "16"))
        except (ValueError, TypeError):
            max_triples = 16
        try:
            max_chars = int(os.getenv("LLM_FACTS_MAX_INPUT_CHARS", "9000"))
        except (ValueError, TypeError):
            max_chars = 9000
        model_alias = os.getenv("LLM_FACTS_MODEL", "gpt-4o-mini")

        if isinstance(session_conversations, list) and session_conversations:
            sess_items = session_conversations
        else:
            try:
                corpus = list(self.corpus_manager.derivable_entries())
            except (AttributeError, TypeError):
                corpus = []
            try:
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                    and self._ts(e) >= self.session_start
                ]
            except (ValueError, TypeError):
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                ]

        # Build conversation pairs. The extractor reads ONLY the user query
        # (2026-09-02: Daemon responses are not evidence about the user); the
        # response rides along for the capture-tone join, and the timestamp
        # becomes the evidence span's turn_id for provenance. It is ALSO
        # forwarded as the message's own ``timestamp`` (the source time the
        # learned-relation day is read from; without it nothing is learned).
        conv_pairs = []
        for e in sess_items:
            q = (e.get('query') or '').strip()
            r = (e.get('response') or '').strip()
            if q:
                pair = {"query": q, "response": r}
                # Raw user text for attachment turns (2026-09-05) — the
                # extractor and provenance join prefer it over the merged blob.
                ut = e.get('user_text')
                # Keep it even when EMPTY (attachment-only turn): the empty
                # key is what tells the extractor "no authored words" instead
                # of letting it fall back to the merged attachment blob.
                if isinstance(ut, str):
                    pair["user_text"] = ut.strip()
                ts = e.get('timestamp')
                if ts:
                    ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    pair["turn_id"] = ts_iso
                    pair["timestamp"] = ts_iso
                conv_pairs.append(pair)
        conv_tail = conv_pairs[-max(1, last_turns):]

        if not conv_tail:
            return

        from memory.llm_fact_extractor import LLMFactExtractor
        llm_ex = LLMFactExtractor(
            self.model_manager,
            model_alias=model_alias,
            max_input_chars=max_chars,
            max_triples=max_triples,
        )

        # Gather existing profile facts so LLM can reuse relation names for updates
        # Deduplicate by canonical relation, keep newest per relation, sort newest first
        existing_facts = None
        if self.user_profile:
            try:
                from memory.user_profile_schema import canonicalize_profile_relation
                current_view = self.user_profile.get_current_view()
                raw_facts = []
                for cat_name, facts_list in current_view.items():
                    for f in facts_list:
                        if isinstance(f, dict) and f.get("relation") and f.get("value"):
                            raw_facts.append({
                                "relation": f["relation"],
                                "value": f["value"],
                                "category": cat_name,
                                "timestamp": f.get("timestamp", ""),
                            })
                # Deduplicate by canonical relation — keep newest per canonical name
                seen_rels = {}
                for f in raw_facts:
                    canonical = canonicalize_profile_relation(f["relation"], f.get("value"))
                    ts = f.get("timestamp", "")
                    if canonical not in seen_rels or ts > seen_rels[canonical].get("timestamp", ""):
                        # Store with canonical relation name so LLM sees clean names
                        seen_rels[canonical] = {
                            "relation": canonical,
                            "value": f["value"],
                            "category": f["category"],
                            "timestamp": ts,
                        }
                existing_facts = sorted(
                    seen_rels.values(),
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )[:60]
            except Exception as pf_err:  # degrades: LLM extractor sees no existing profile facts, risking duplicate triples
                logger.debug(f"[Shutdown] Failed to gather profile facts for LLM context: {pf_err}")

        triples = await llm_ex.extract_triples(conv_tail, existing_facts=existing_facts)

        # Verification gate: batch-verify before storage
        verifier = getattr(self._storage, 'fact_verifier', None)
        verification_results = {}
        if verifier and triples:
            try:
                from memory.fact_verification import FactVerdict
                batch = [
                    {
                        "subject": t.get("subject", ""),
                        "predicate": t.get("relation", ""),
                        "object": t.get("object", ""),
                        "fact_text": f"{t.get('subject','')} | {t.get('relation','')} | {t.get('object','')}",
                        "source": "llm_shutdown",
                        "confidence": 0.75,
                        "fact_scope": t.get("fact_scope", "user"),
                    }
                    for t in triples
                    if t.get("subject") and t.get("relation") and t.get("object")
                ]
                vr_list = await verifier.verify_batch(batch)
                for b, vr in zip(batch, vr_list):
                    verification_results[b["fact_text"]] = vr
            except Exception as ve:  # degrades: LLM facts stored without batch verification this shutdown
                logger.debug(f"[Shutdown] Batch verification failed, proceeding: {ve}")

        kept = 0
        for t in triples:
            subj = t.get('subject')
            rel = t.get('relation')
            obj = t.get('object')
            if not subj or not rel or not obj:
                continue
            fact_text = f"{subj} | {rel} | {obj}"

            # Apply verification verdict if available
            if fact_text in verification_results:
                from memory.fact_verification import FactVerdict
                vr = verification_results[fact_text]
                if vr.verdict == FactVerdict.REJECT:
                    logger.debug(f"[Shutdown] LLM fact rejected by verifier: {fact_text[:80]}")
                    continue
                if vr.verdict == FactVerdict.STORE_AND_FLAG:
                    for cand in vr.conflicting_candidates:
                        if cand.doc_id:
                            try:
                                supersede_md = {"superseded_by": fact_text[:200]}
                                supersede_md.update(vr.metadata_updates)
                                self.chroma_store.update_metadata("facts", cand.doc_id, supersede_md)
                            except Exception:  # degrades: superseded fact not flagged in the facts collection, old value may resurface
                                pass

            try:
                src_dict = {
                    "source": "llm_shutdown",
                    "confidence": 0.75,
                }
                # Stance + capture tone (2026-08-23): stance was set by
                # _normalize_triple (deterministic-over-LLM); capture tone
                # joins the triple back to the session corpus entry that
                # mentions its object/subject and reads its is_heavy_topic
                # flag. Unmatched → "unknown" (settledness requires EXPLICIT
                # non-elevated evidence, so unknown under-fires — safe).
                if t.get("stance"):
                    src_dict["stance"] = t["stance"]
                src_dict["capture_tone"] = self._capture_tone_for_triple(t, sess_items)
                src_exc = t.get("source_excerpt", "")
                if src_exc:
                    src_dict["source_excerpt"] = src_exc[:200]
                # Provenance (2026-09-02): which user turn + what kind of
                # support joined this triple to the user's own words.
                for _pk in ("source_role", "source_turn_id", "source_support", "source_anchor"):
                    if t.get(_pk) not in (None, ""):
                        src_dict[_pk] = str(t[_pk])[:80]
                # E3 (2026-09-27, BC-58): the triple's own turn time (set by
                # LLMFactExtractor._attach_source_excerpts from the matched
                # source message, never shutdown extraction time) overrides
                # chroma_store.add_fact's default now()-stamp, same as the
                # profile-facts path below (add_facts_batch reads this key).
                if t.get("timestamp"):
                    src_dict["timestamp"] = t["timestamp"]
                if t.get("source_turn_index") is not None:
                    try:
                        src_dict["source_turn_index"] = int(t["source_turn_index"])
                    except (TypeError, ValueError):
                        pass
                result = self.chroma_store.add_fact(
                    fact=fact_text,
                    source=src_dict,
                    confidence=0.75,
                )
                if result is not None:
                    kept += 1
                    # Feed knowledge graph — same hook the per-turn path uses.
                    # Without this, LLM-extracted triples (the richest source
                    # of entity-entity relations) never became graph edges.
                    try:
                        from memory.memory_storage import _get_graph_enabled
                        if getattr(self._storage, 'graph_memory', None) and \
                                getattr(self._storage, 'entity_resolver', None) and \
                                _get_graph_enabled():
                            self._storage._ingest_fact_to_graph(
                                subj=subj, rel=rel, obj=obj,
                                fact_id=str(result),
                                confidence=0.75,
                                stance=t.get("stance", ""),
                                capture_tone=src_dict.get("capture_tone", ""),
                            )
                    except Exception as graph_err:  # degrades: extracted fact stored but its knowledge-graph edge is missing
                        logger.debug(f"[Shutdown] Graph ingestion failed: {graph_err}")
                else:
                    logger.debug(f"[Shutdown] LLM fact skipped as duplicate: {fact_text}")
            except (AttributeError, TypeError, ValueError):
                continue
        logger.info(f"[LLM Facts] kept={kept} (model={model_alias})")

        # Update user profile with user-scoped facts only (entity facts stay in ChromaDB only)
        if triples and self.user_profile:
            try:
                user_triples = [
                    t for t in triples
                    if t.get("subject", "user").lower() == "user"
                ]
                if user_triples:
                    added = self.user_profile.add_facts_batch(user_triples)
                    logger.info(f"[Shutdown] Added {added} user facts to profile (skipped {len(triples) - len(user_triples)} entity facts)")
                else:
                    logger.debug("[Shutdown] No user-scoped facts to add to profile")
            except Exception as profile_err:
                logger.warning(f"[Shutdown] Failed to update user profile: {profile_err}")

    @staticmethod
    def _capture_tone_for_triple(t, sess_items) -> str:
        """Join a triple back to the session corpus entry that mentions its
        object (or non-user subject) and map that entry's is_heavy_topic flag
        to a capture-tone regime. Newest match wins; no match → "unknown"
        (2026-08-23 stance layer — settledness treats unknown as never
        counting, the conservative direction)."""
        try:
            obj = str(t.get("object") or "").strip().lower()
            subj = str(t.get("subject") or "").strip().lower()
            needles = [n for n in (obj, subj) if n and n != "user" and len(n) >= 3]
            if not needles:
                return "unknown"
            for e in reversed(sess_items or []):
                if not isinstance(e, dict):
                    continue
                text = ((e.get("query") or "") + " " + (e.get("response") or "")).lower()
                if any(n in text for n in needles):
                    heavy = e.get("is_heavy_topic")
                    if heavy is None:
                        return "unknown"
                    return "elevated" if heavy else "non_elevated"
        except Exception:  # degrades: capture tone recorded as unknown, so settledness does not count it
            pass
        return "unknown"

    # ------------------------------------------------------------------
    # Procedural skill extraction
    # ------------------------------------------------------------------

    async def _extract_procedural_skills(self, session_conversations):
        """Extract reusable problem-solving patterns from session conversations.

        Uses LLM to identify 'How-To' patterns (trigger -> action) that could
        be useful for future similar situations.  Stores them via MemoryStorage
        with semantic deduplication.
        """
        try:
            from config.app_config import PROCEDURAL_SKILLS_ENABLED
            if not PROCEDURAL_SKILLS_ENABLED:
                return
        except ImportError:
            return

        if not self.model_manager or not hasattr(self.model_manager, 'generate_once'):
            return

        # Gather session conversations
        if isinstance(session_conversations, list) and session_conversations:
            sess_items = session_conversations
        else:
            try:
                corpus = list(self.corpus_manager.derivable_entries())
            except (AttributeError, TypeError):
                return
            try:
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                    and self._ts(e) >= self.session_start
                ]
            except (ValueError, TypeError):
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                ]

        # Need enough conversation to extract patterns
        if len(sess_items) < 3:
            return

        # Build conversation excerpt
        excerpts = []
        for e in sess_items[-12:]:
            q = (e.get('query') or '').strip()
            a = (e.get('response') or '').strip()
            if q or a:
                lines = []
                if q:
                    lines.append(f"User: {q[:300]}")
                if a:
                    lines.append(f"Assistant: {a[:400]}")
                excerpts.append("\n".join(lines))

        if not excerpts:
            return

        conversation_text = "\n\n".join(excerpts)

        prompt = (
            "You are a pattern analyst. Review the conversation below and extract 0-3 "
            "reusable problem-solving patterns (procedural skills). Only extract patterns "
            "that are GENERALIZABLE -- not specific to a single instance.\n\n"
            "For each pattern, output EXACTLY this JSON format (one per line):\n"
            '{"trigger": "situation description (10-500 chars)", '
            '"action_pattern": "abstract solution steps (20-1000 chars)", '
            '"category": "one of: debugging|workflow|prompt_engineering|interpersonal|architectural|optimization|testing", '
            '"confidence": 0.0-1.0, '
            '"tags": ["keyword1", "keyword2"]}\n\n'
            "Rules:\n"
            "- trigger = the SITUATION that calls for this pattern\n"
            "- action_pattern = the ABSTRACT approach, not specific code\n"
            "- confidence = how generalizable (0.5 = niche, 1.0 = universal)\n"
            "- Output ONLY valid JSON lines, no other text\n"
            "- If no generalizable patterns exist, output nothing\n\n"
            f"CONVERSATION:\n{conversation_text}\n\n"
            "Patterns (JSON lines only):"
        )

        model_alias = os.getenv("LLM_SKILLS_MODEL", REFLECTION_MODEL_ALIAS)
        api_models = getattr(self.model_manager, "api_models", {}) or {}
        skills_model = model_alias if model_alias in api_models else None

        try:
            # Explicit model_name (no switch_model) so this is safe to run
            # concurrently with other shutdown LLM calls.
            raw = await self.model_manager.generate_once(
                prompt, model_name=skills_model, max_tokens=600
            )
        except Exception as e:
            logger.warning(f"[Shutdown] Skill extraction LLM call failed: {e}")
            return

        if not raw or not raw.strip():
            return

        # Parse JSON lines
        from memory.procedural_skill import ProceduralSkill, SkillCategory

        kept = 0
        failed = 0  # StoreWriteError count [F10b] -- distinct from a dedup skip
        session_id = self.session_start.isoformat() if isinstance(self.session_start, datetime) else str(self.session_start)

        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                skill = ProceduralSkill(
                    trigger=data.get("trigger", ""),
                    action_pattern=data.get("action_pattern", ""),
                    category=SkillCategory(data.get("category", "workflow")),
                    confidence=float(data.get("confidence", 0.6)),
                    tags=data.get("tags", [])[:10],
                    source_session_id=session_id,
                    created_at=time.time(),
                )
            except (ValueError, KeyError) as e:
                logger.debug(f"[Shutdown] Skipping invalid skill: {e}")
                continue

            from utils.retrieval_outcome import StoreWriteError
            try:
                doc_id = await self._storage.store_skill(skill)
            except StoreWriteError:
                failed += 1
                continue
            if doc_id:
                kept += 1

        if kept or failed:
            logger.info(f"[Shutdown] Extracted {kept} procedural skill(s), {failed} failed")

    # ------------------------------------------------------------------
    # Code proposal generation
    # ------------------------------------------------------------------

    async def _generate_proposals(self, session_conversations):
        """Generate goal-directed code proposals from session conversations.

        ENHANCED: Routes through the Daemon retrieval pipeline for rich context
        when memory_coordinator is available. Falls back to cold generation
        (truncated excerpts + file reads only) otherwise.
        """
        try:
            from config.app_config import CODE_PROPOSALS_ENABLED
            if not CODE_PROPOSALS_ENABLED:
                return
        except ImportError:
            return

        if not self.model_manager or not hasattr(self.model_manager, 'generate_once'):
            return

        # Gather session conversations
        if isinstance(session_conversations, list) and session_conversations:
            sess_items = session_conversations
        else:
            try:
                corpus = list(self.corpus_manager.derivable_entries())
            except (AttributeError, TypeError):
                return
            try:
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                    and self._ts(e) >= self.session_start
                ]
            except (ValueError, TypeError):
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                ]

        if len(sess_items) < 3:
            return

        try:
            from knowledge.proposal_generator import GoalDirectedGenerator
            from memory.proposal_store import ProposalStore
            # Read at call time so the Settings slider applies without restart
            # (was unwired — the generator always used its class default of 5)
            from config.app_config import CODE_PROPOSALS_MAX_PER_SESSION

            generator = GoalDirectedGenerator(
                model_manager=self.model_manager,
                repo_path=".",
                max_proposals=CODE_PROPOSALS_MAX_PER_SESSION,
            )
            proposal_store = ProposalStore(chroma_store=self.chroma_store)
            dedup_context = proposal_store.get_for_dedup()

            # Try pipeline-enriched generation first
            if self.memory_coordinator:
                try:
                    rich_context = await self._gather_proposal_context(sess_items)
                    if dedup_context:
                        rich_context += f"\n\n## Existing Proposals (avoid duplicates)\n{dedup_context}"
                    proposals = await generator.generate_proposals_with_context(
                        pipeline_context=rich_context,
                    )
                    logger.info("[Shutdown] Used pipeline-enriched proposal generation")
                except Exception as e:
                    logger.warning(f"[Shutdown] Pipeline proposal generation failed, falling back: {e}")
                    proposals = await self._generate_proposals_cold(
                        sess_items, generator, dedup_context
                    )
            else:
                proposals = await self._generate_proposals_cold(
                    sess_items, generator, dedup_context
                )

            kept = 0
            failed = 0  # StoreWriteError count [F11b-1b] -- distinct from a dedup skip
            for proposal in proposals:
                existing_id = proposal_store.check_similarity(proposal)
                if existing_id:
                    logger.debug(f"[Shutdown] Proposal skipped (duplicate of {existing_id}): {proposal.title}")
                    continue
                from utils.retrieval_outcome import StoreWriteError
                try:
                    doc_id = proposal_store.store_proposal(proposal)
                except StoreWriteError:
                    failed += 1
                    continue
                if doc_id:
                    kept += 1

            if kept or failed:
                logger.info(f"[Shutdown] Generated {kept} proposal(s), {failed} failed")

        except Exception as e:
            logger.warning(f"[Shutdown] Proposal generation failed: {e}")

    async def _generate_proposals_cold(self, sess_items, generator, dedup_context):
        """Original cold proposal generation (fallback).

        Uses only truncated conversation excerpts + file reads (skeleton, goals,
        git log). No retrieval pipeline context.
        """
        excerpts = []
        for e in sess_items[-8:]:
            q = (e.get('query') or '').strip()
            a = (e.get('response') or '').strip()
            if q or a:
                lines = []
                if q:
                    lines.append(f"User: {q[:300]}")
                if a:
                    lines.append(f"Assistant: {a[:400]}")
                excerpts.append("\n".join(lines))

        if not excerpts:
            return []

        extra_context = "## Recent Conversation\n" + "\n\n".join(excerpts)
        if dedup_context:
            extra_context += f"\n\n## Existing Proposals (avoid duplicates)\n{dedup_context}"

        return await generator.generate_proposals(extra_context=extra_context)

    async def _gather_proposal_context(self, sess_items: list) -> str:
        """Gather rich context from the Daemon retrieval pipeline for proposals.

        Calls MemoryCoordinator retrieval methods in parallel to build a
        comprehensive context string with semantic memories, summaries,
        reflections, procedural skills, user facts, git commits, and
        user profile info.
        """
        mc = self.memory_coordinator

        # Build synthetic query from session topics for semantic retrieval
        queries = [
            (e.get('query') or '').strip()
            for e in sess_items
            if (e.get('query') or '').strip()
        ]
        topic_summary = " ".join(q[:100] for q in queries[-5:])
        synthetic_query = f"suggest improvements and features based on: {topic_summary}"

        # Parallel retrieval with reduced limits (focused for proposals)
        tasks = {
            "memories": mc.get_memories(synthetic_query, limit=5),
            "skills": mc.get_skills(synthetic_query, limit=3),
            "facts": mc.get_facts(synthetic_query, limit=5),
            "reflections": mc.get_reflections_hybrid(synthetic_query, limit=2),
        }

        # Sync methods — call directly (not via asyncio.gather)
        summaries = []
        try:
            summaries = mc.get_summaries_hybrid(synthetic_query, limit=3)
        except Exception as e:  # degrades: session reflection prompt lacks related summaries
            logger.debug(f"[Shutdown] Summary retrieval failed: {e}")

        git_commits = []
        try:
            git_commits = self.chroma_store.query_collection(
                "procedural", synthetic_query, n_results=5
            )
        except Exception as e:  # degrades: session reflection prompt lacks related git commits
            logger.debug(f"[Shutdown] Git commit retrieval failed: {e}")

        profile_text = ""
        try:
            if self.user_profile:
                profile_text = self.user_profile.get_context_injection(
                    max_tokens=300, query=synthetic_query
                )
        except Exception as e:  # degrades: session reflection prompt lacks user profile context
            logger.debug(f"[Shutdown] User profile retrieval failed: {e}")

        # Await async tasks
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:  # degrades: session reflection prompt lacks one retrieval source
                logger.debug(f"[Shutdown] {key} retrieval failed: {e}")
                results[key] = []

        memories = results.get("memories", [])
        skills = results.get("skills", [])
        facts = results.get("facts", [])
        reflections = results.get("reflections", [])

        # Format into structured context sections
        sections = []

        # Recent conversation (from session items, richer than cold approach)
        conv_lines = []
        for e in sess_items[-8:]:
            q = (e.get('query') or '').strip()
            a = (e.get('response') or '').strip()
            if q or a:
                parts = []
                if q:
                    parts.append(f"User: {q[:300]}")
                if a:
                    parts.append(f"Assistant: {a[:400]}")
                conv_lines.append("\n".join(parts))
        if conv_lines:
            sections.append("## Recent Conversation\n" + "\n\n".join(conv_lines))

        # Semantic memories
        if memories:
            lines = []
            for m in memories[:5]:
                content = m.get('content') or m.get('response') or ''
                if content:
                    lines.append(f"- {content[:300]}")
            if lines:
                sections.append("## Relevant Memories\n" + "\n".join(lines))

        # Session summaries
        if summaries:
            lines = [
                f"- {(s.get('content') or s.get('text') or '')[:200]}"
                for s in summaries[:3]
                if s.get('content') or s.get('text')
            ]
            if lines:
                sections.append("## Session Summaries\n" + "\n".join(lines))

        # Reflections
        if reflections:
            lines = [
                f"- {(r.get('content') or r.get('text') or '')[:200]}"
                for r in reflections[:2]
                if r.get('content') or r.get('text')
            ]
            if lines:
                sections.append("## Past Reflections\n" + "\n".join(lines))

        # Procedural skills
        if skills:
            lines = []
            for sk in skills[:3]:
                meta = sk.get('metadata', {})
                trigger = meta.get('trigger', '')
                action = meta.get('action_pattern', '')
                if trigger and action:
                    lines.append(f"- When: {trigger[:150]} → Do: {action[:200]}")
            if lines:
                sections.append("## Problem-Solving Patterns\n" + "\n".join(lines))

        # User facts
        if facts:
            lines = [
                f"- {(f.get('content') or f.get('text') or '')[:150]}"
                for f in facts[:5]
                if f.get('content') or f.get('text')
            ]
            if lines:
                sections.append("## Known User Facts\n" + "\n".join(lines))

        # User profile
        if profile_text and profile_text.strip():
            sections.append(f"## User Profile\n{profile_text[:400]}")

        # Git commits
        if git_commits:
            lines = [
                f"- {(c.get('content') or '')[:150]}"
                for c in git_commits[:5]
                if c.get('content')
            ]
            if lines:
                sections.append("## Recent Git Activity\n" + "\n".join(lines))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Lightweight implementation tracking
    # ------------------------------------------------------------------

    async def _check_implementation_tracking(self):
        """
        Lightweight file-existence check for pending proposals (Stage 1 only).

        Runs at shutdown to keep implementation_status metadata fresh.
        ~50ms per proposal (pure os.path.exists calls, no git/LLM).
        """
        try:
            from config.app_config import IMPL_TRACKING_ENABLED, IMPL_TRACKING_AT_SHUTDOWN

            if not IMPL_TRACKING_ENABLED or not IMPL_TRACKING_AT_SHUTDOWN:
                return

            from knowledge.implementation_detector import ImplementationDetector
            from memory.proposal_store import ProposalStore

            chroma_store = self.chroma_store
            if not chroma_store:
                return

            def _track_sync():
                # Blocking chroma/file calls (list_all per update) — runs on a
                # worker thread so it can't stall the phase-B gather's other
                # tasks (2026-10-10, BC-69: 233 proposals blocked the loop
                # ~40 s and the shared 60 s budget cancelled the thread pass).
                store = ProposalStore(chroma_store=chroma_store)
                proposals = store.get_pending_and_approved()
                if not proposals:
                    return 0, 0
                # Oldest-checked first (never-tracked = 0), capped per run;
                # the rest are picked up on later shutdowns.
                proposals.sort(key=lambda p: getattr(p, "last_tracked_at", None) or 0.0)
                total = len(proposals)
                if total > IMPL_TRACKING_MAX_PER_SHUTDOWN:
                    logger.info(
                        f"[Shutdown] Implementation tracking: capping {total} "
                        f"proposals to {IMPL_TRACKING_MAX_PER_SHUTDOWN} (oldest-checked first)"
                    )
                    proposals = proposals[:IMPL_TRACKING_MAX_PER_SHUTDOWN]
                detector = ImplementationDetector(repo_path=".")
                done = 0
                for proposal in proposals:
                    result = detector.detect_single(proposal, lightweight=True)
                    if not result.skipped_reason:
                        if store.update_tracking_metadata(proposal.id, result):
                            done += 1
                return done, total

            updated, total = await asyncio.to_thread(_track_sync)
            if updated:
                logger.info(f"[Shutdown] Implementation tracking: updated {updated}/{total} proposals")

        except Exception as e:
            logger.warning(f"[Shutdown] Implementation tracking failed: {e}")

    # ------------------------------------------------------------------
    # Open thread extraction
    # ------------------------------------------------------------------

    async def _process_open_threads(self, session_conversations):
        """Extract open threads from session and detect resolutions of existing ones.

        Phase 1: Get existing open threads → run detect_resolutions() → mark
                 resolved, then cascade each resolution to near-identical
                 duplicate threads (topics_equivalent)
        Phase 2: Run extract_new_threads() (told what's already tracked) →
                 store new threads, dropping duplicates of just-resolved or
                 still-open threads (threads_duplicate; open twins get their
                 last_referenced refreshed instead)
        Phase 3: Enforce cap (THREAD_MAX_OPEN)
        """
        try:
            from config.app_config import THREAD_SURFACING_ENABLED
            if not THREAD_SURFACING_ENABLED:
                return
        except ImportError:
            return

        if not self.thread_store:
            return

        if not self.model_manager or not hasattr(self.model_manager, "generate_once"):
            return

        # Gather session conversations
        if isinstance(session_conversations, list) and session_conversations:
            sess_items = session_conversations
        else:
            try:
                corpus = list(self.corpus_manager.derivable_entries())
            except (AttributeError, TypeError):
                return
            try:
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                    and self._ts(e) >= self.session_start
                ]
            except (ValueError, TypeError):
                sess_items = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                ]

        if len(sess_items) < 2:
            return

        from memory.thread_extractor import ThreadExtractor

        extractor = ThreadExtractor(model_manager=self.model_manager)

        # Phase 1 (resolution detection) and Phase 2 (new-thread extraction) each
        # make an independent LLM call with no dependency on the other — run them
        # concurrently, then apply their ChromaDB writes sequentially afterward.
        existing_open = []
        from utils.retrieval_outcome import RetrievalError
        try:
            existing_open = self.thread_store.list_open_threads()
        except RetrievalError as e:
            # Typed degrade (not a broad except): a failed read must not
            # fall through to extraction, or every extracted thread gets
            # stored again as a duplicate (nothing here to de-dup against).
            logger.warning(f"[Shutdown] Listing open threads failed: {e}")
            return

        async def _detect_resolutions():
            if not existing_open:
                return []
            return await extractor.detect_resolutions(sess_items, existing_open)

        resolutions, new_threads = await asyncio.gather(
            _detect_resolutions(),
            extractor.extract_new_threads(sess_items, open_threads=existing_open),
            return_exceptions=True,
        )

        from memory.thread_store import threads_duplicate, topics_equivalent

        # Phase 1 result: apply resolutions
        resolved_threads = []
        resolved_ids = set()
        if isinstance(resolutions, Exception):
            logger.warning(f"[Shutdown] Thread resolution detection failed: {resolutions}")
        elif resolutions:
            open_by_id = {t.thread_id: t for t in existing_open}
            for thread_id, resolution in resolutions:
                self.thread_store.resolve_thread(thread_id, resolution)
                resolved_ids.add(thread_id)
                if thread_id in open_by_id:
                    resolved_threads.append(open_by_id[thread_id])
            logger.info(f"[Shutdown] Resolved {len(resolutions)} thread(s)")

            # Cascade: a task accumulates duplicate thread rows across sessions;
            # resolving one must also close its near-identical siblings, or they
            # keep surfacing the task as still-pending after the user finished it.
            for candidate in existing_open:
                if candidate.thread_id in resolved_ids:
                    continue
                twin = next(
                    (r for r in resolved_threads if topics_equivalent(candidate, r)),
                    None,
                )
                if twin is not None:
                    self.thread_store.resolve_thread(
                        candidate.thread_id,
                        f"Duplicate of resolved thread '{twin.topic}'",
                    )
                    resolved_ids.add(candidate.thread_id)
                    resolved_threads.append(candidate)
                    logger.info(
                        f"[Shutdown] Cascade-resolved duplicate thread "
                        f"'{candidate.topic}' (twin of '{twin.topic}')"
                    )

        # Phase 2 result: store new threads. The extractor sees the whole
        # session including pre-completion talk, so it can re-create a task the
        # user finished later that same session — drop anything that duplicates
        # a thread resolved this pass or one still open (refreshing the latter).
        if isinstance(new_threads, Exception):
            logger.warning(f"[Shutdown] Thread extraction failed: {new_threads}")
        elif new_threads:
            still_open = [t for t in existing_open if t.thread_id not in resolved_ids]
            stored = 0
            failed = 0
            for thread in new_threads:
                twin = next(
                    (r for r in resolved_threads if threads_duplicate(thread, r)),
                    None,
                )
                if twin is not None:
                    logger.info(
                        f"[Shutdown] Dropped new thread '{thread.topic}' — "
                        f"duplicate of thread resolved this pass ('{twin.topic}')"
                    )
                    continue
                twin = next(
                    (o for o in still_open if threads_duplicate(thread, o)),
                    None,
                )
                if twin is not None:
                    self.thread_store.touch_thread(twin)
                    logger.info(
                        f"[Shutdown] Dropped new thread '{thread.topic}' — "
                        f"already tracked as '{twin.topic}' (refreshed)"
                    )
                    continue
                from utils.retrieval_outcome import StoreWriteError
                try:
                    doc_id = self.thread_store.store_thread(thread)
                except StoreWriteError:
                    failed += 1
                    continue
                if doc_id:
                    stored += 1
            if stored or failed:
                logger.info(f"[Shutdown] Stored {stored} new thread(s), {failed} failed")

        # Phase 3: Enforce cap
        try:
            self.thread_store.enforce_cap()
        except Exception as e:  # degrades: thread cap not enforced this shutdown, store may exceed its limit
            logger.debug(f"[Shutdown] Thread cap enforcement failed: {e}")

    # ------------------------------------------------------------------
    # Synthesis dreaming (cross-domain candidate generation)
    # ------------------------------------------------------------------

    async def run_synthesis_dreaming(self):
        """Public entry: run synthesis dreaming as a standalone shutdown step.

        Pulled out of process_shutdown_memory's Phase B so it runs under its own
        SYNTHESIS_DREAM_TIMEOUT_S budget (driven by main.py) instead of competing
        with reflection + fact extraction inside SHUTDOWN_TASK_TIMEOUT_S. Self-
        gating (config-disabled / auto-halt / no-candidates all return quietly).
        """
        await self._run_synthesis_dreaming()

    async def _run_synthesis_dreaming(self):
        """Step 6.8: Generate and filter cross-domain synthesis candidates.

        Runs all three generators with independent quotas for head-to-head
        comparison. All candidates pooled into a single filter pass.
        Generators: RetrievalSynthesisGenerator, GraphWalkGenerator, SynthesisGenerator.

        Auto-halt: if audit queue FP rate exceeds threshold with sufficient
        graded data, synthesis is skipped until FP rate is addressed.

        The audit check is fail-closed: a typed audit-read failure skips
        dreaming instead of proceeding as if the audit were healthy.
        """
        try:
            from config.app_config import (
                SYNTHESIS_GENERATOR_ENABLED,
                SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION,
                SYNTHESIS_POOLED_ENABLED,
                SYNTHESIS_POOLED_CANDIDATES_PER_SESSION,
                SYNTHESIS_POOLED_LLM_CONCURRENCY,
                SYNTHESIS_RETRIEVAL_ENABLED,
                GRAPH_WALK_ENABLED,
                GRAPH_WALK_MAX_CANDIDATES,
                GRAPH_WALK_MIN_BRIDGE_EDGES,
                SYNTHESIS_AUDIT_ENABLED,
            )
            # Dreaming runs if EITHER the pooled discovery generator OR the legacy
            # personal->wiki generators are enabled. (SYNTHESIS_GENERATOR_ENABLED alone
            # no longer gates dreaming — it now only gates the retired Tier-2 fallback.)
            if not (SYNTHESIS_GENERATOR_ENABLED or SYNTHESIS_POOLED_ENABLED):
                return

            # Auto-halt check: skip synthesis if FP rate too high. Fail-closed:
            # a typed RetrievalError (audit stats unreadable) skips dreaming
            # with one labels-only warning. Any OTHER failure here is no
            # longer swallowed -- it propagates to this method's outer
            # except below, which ends dreaming non-fatally. Dreaming never
            # runs without a readable audit.
            if SYNTHESIS_AUDIT_ENABLED:
                from utils.retrieval_outcome import RetrievalError
                try:
                    from memory.synthesis_memory import SynthesisMemory
                    audit_mem = SynthesisMemory(self.chroma_store)
                    audit_stats = audit_mem.get_audit_stats()
                    if audit_stats.get("auto_halt", False):
                        logger.warning(
                            "[Shutdown] Synthesis HALTED by audit queue — "
                            "FP rate %.1f%% exceeds threshold %.1f%% "
                            "(graded: %d, min: %d). Fix quality before re-enabling.",
                            audit_stats["fp_rate"] * 100,
                            audit_stats["fp_halt_threshold"] * 100,
                            audit_stats["total_graded"],
                            audit_stats["min_graded_for_halt"],
                        )
                        return
                except RetrievalError as e:
                    logger.warning(
                        "[Shutdown] Synthesis dreaming skipped: audit stats unavailable (%s: %s)",
                        e.source, e.reason,
                    )
                    return

            from knowledge.synthesis_filter import SynthesisFilter
            from memory.synthesis_memory import SynthesisMemory

            mc = self.memory_coordinator
            graph_memory = getattr(mc, "graph_memory", None) if mc else None
            entity_resolver = getattr(mc, "entity_resolver", None) if mc else None

            candidates = []

            if SYNTHESIS_POOLED_ENABLED:
                # PRIMARY discovery generator: prominent curated concepts paired in the
                # non-obvious cosine band (validated 2026-06-30: ~17% accept / ~46%
                # MODERATE+STRONG vs ~0 for the retired personal->wiki tiers). When on,
                # this is the ONLY generator — the lever is concept prominence.
                from knowledge.synthesis_pooled_generator import PooledConceptSynthesisGenerator
                pooled_gen = PooledConceptSynthesisGenerator(
                    model_manager=self.model_manager,
                    concurrency=SYNTHESIS_POOLED_LLM_CONCURRENCY,
                )
                candidates = await pooled_gen.generate_candidates(
                    count=SYNTHESIS_POOLED_CANDIDATES_PER_SESSION,
                )
                logger.info(
                    "[Shutdown] Pooled discovery generator: %d candidates", len(candidates),
                )
            else:
                # RETIRED personal->wiki path (Tier 0/1/2), preserved behind the flags.
                # Tier 0: Retrieval-based generator (primary)
                if SYNTHESIS_RETRIEVAL_ENABLED:
                    from knowledge.synthesis_retriever import RetrievalSynthesisGenerator
                    retrieval_gen = RetrievalSynthesisGenerator(
                        chroma_store=self.chroma_store,
                        model_manager=self.model_manager,
                        graph_memory=graph_memory,
                        entity_resolver=entity_resolver,
                    )
                    retrieval_candidates = await retrieval_gen.generate_candidates(
                        count=SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION,
                    )
                    candidates.extend(retrieval_candidates)
                    logger.info(
                        "[Shutdown] Retrieval generator: %d candidates",
                        len(retrieval_candidates),
                    )

                # Tier 1: Graph walk generator (if bridge edges sufficient)
                if GRAPH_WALK_ENABLED and graph_memory and entity_resolver:
                    bridge_count = graph_memory.count_bridge_edges()
                    if bridge_count >= GRAPH_WALK_MIN_BRIDGE_EDGES:
                        from knowledge.graph_walk_generator import GraphWalkGenerator
                        walk_gen = GraphWalkGenerator(
                            graph_memory=graph_memory,
                            entity_resolver=entity_resolver,
                            model_manager=self.model_manager,
                        )
                        walk_candidates = await walk_gen.generate_candidates(
                            count=GRAPH_WALK_MAX_CANDIDATES,
                        )
                        candidates.extend(walk_candidates)
                        logger.info(
                            "[Shutdown] Graph walk generator: %d candidates (bridges=%d)",
                            len(walk_candidates), bridge_count,
                        )
                    else:
                        logger.info(
                            "[Shutdown] Graph walk skipped: %d bridges < %d threshold",
                            bridge_count, GRAPH_WALK_MIN_BRIDGE_EDGES,
                        )

                # Tier 2: Cross-store random sampling (fallback)
                if SYNTHESIS_GENERATOR_ENABLED:
                    from knowledge.synthesis_generator import SynthesisGenerator
                    remaining = SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION - len(candidates)
                    if remaining > 0:
                        generator = SynthesisGenerator(
                            chroma_store=self.chroma_store,
                            model_manager=self.model_manager,
                            graph_memory=graph_memory,
                            entity_resolver=entity_resolver,
                        )
                        fallback_candidates = await generator.generate_candidates(count=remaining)
                        candidates.extend(fallback_candidates)

            if not candidates:
                logger.info("[Shutdown] Synthesis dreaming: no candidates generated")
                return

            synthesis_memory = SynthesisMemory(self.chroma_store)
            filter_pipeline = SynthesisFilter(
                chroma_store=self.chroma_store,
                model_manager=self.model_manager,
                synthesis_memory=synthesis_memory,
                graph_memory=graph_memory,
                entity_resolver=entity_resolver,
            )
            results = await filter_pipeline.process_batch(candidates)

            logger.info(
                "[Shutdown] Synthesis dreaming: generated=%d, accepted=%d, rejected=%d",
                len(candidates),
                results.get("accepted", 0),
                results.get("rejected", 0),
            )

        except Exception as e:
            logger.warning("[Shutdown] Synthesis dreaming failed (non-fatal): %s", e)

    async def _run_wiki_enrichment(self):
        """Step 6.9: Enrich knowledge graph with Wikipedia articles accessed this session."""
        try:
            from config.app_config import WIKI_ENRICHMENT_ENABLED, WIKI_ENRICHMENT_TIMEOUT_S
            if not WIKI_ENRICHMENT_ENABLED:
                return

            mc = self.memory_coordinator
            graph_memory = getattr(mc, "graph_memory", None) if mc else None
            entity_resolver = getattr(mc, "entity_resolver", None) if mc else None

            if not graph_memory or not entity_resolver:
                return

            import asyncio
            from knowledge.wiki_enrichment import WikiGraphEnricher
            enricher = WikiGraphEnricher(graph_memory, entity_resolver)

            result = await asyncio.wait_for(
                enricher.enrich_from_session(),
                timeout=WIKI_ENRICHMENT_TIMEOUT_S,
            )

            if result.get("articles_added", 0) > 0:
                logger.info(
                    "[Shutdown] Wiki enrichment: %d articles added, %d relations created",
                    result.get("articles_added", 0),
                    result.get("relations_added", 0),
                )
        except TimeoutError:
            logger.warning("[Shutdown] Wiki enrichment timed out (non-fatal)")
        except Exception as e:
            logger.warning("[Shutdown] Wiki enrichment failed (non-fatal): %s", e)

    async def _run_wikidata_enrichment(self):
        """Step 6.95: Add Wikidata typed edges to personal graph entities.

        Anchored + capped (see knowledge/wikidata_enrichment.py) — pure local
        work over the offline cache, no LLM or network calls.
        """
        try:
            # Test-isolation guard: root-level integration tests build shutdown
            # processors around default-path components; without this, a test
            # run enriches + saves the LIVE data/knowledge_graph.json (observed
            # 2026-07-14). Same convention as scripts/bin/usercustomize.py.
            if os.environ.get("PYTEST_CURRENT_TEST"):
                logger.debug("[Shutdown] Wikidata enrichment skipped under pytest")
                return

            from config.app_config import (
                WIKIDATA_ENRICHMENT_ENABLED,
                WIKIDATA_PERSIST_PATH,
            )
            if not WIKIDATA_ENRICHMENT_ENABLED:
                return

            mc = self.memory_coordinator
            graph_memory = getattr(mc, "graph_memory", None) if mc else None
            entity_resolver = getattr(mc, "entity_resolver", None) if mc else None
            if not graph_memory or not entity_resolver:
                return

            from knowledge.wikidata_enrichment import WikidataGraphEnricher
            enricher = WikidataGraphEnricher(
                graph_memory, entity_resolver, cache_path=WIKIDATA_PERSIST_PATH,
            )
            result = enricher.enrich()

            if result.get("edges_added", 0) > 0:
                logger.info(
                    "[Shutdown] Wikidata enrichment: %d entities matched, "
                    "%d edges added, %d nodes created",
                    result.get("matched", 0),
                    result.get("edges_added", 0),
                    result.get("nodes_created", 0),
                )
        except Exception as e:
            logger.warning("[Shutdown] Wikidata enrichment failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Knowledge graph persistence
    # ------------------------------------------------------------------

    def _save_knowledge_graph(self):
        """Flush knowledge graph, entity aliases, and claim index to disk."""
        try:
            mc = self.memory_coordinator
            if mc and getattr(mc, "graph_memory", None):
                mc.graph_memory.save()
                logger.info(
                    "[Shutdown] Knowledge graph saved: %d nodes, %d edges",
                    mc.graph_memory.node_count(),
                    mc.graph_memory.edge_count(),
                )
            if mc and getattr(mc, "entity_resolver", None):
                mc.entity_resolver.save_external_aliases()
        except Exception as e:
            logger.warning("[Shutdown] Knowledge graph save failed (non-fatal): %s", e)

        # Save claim index (staleness tracking)
        try:
            if self.claim_index:
                self.claim_index.save()
        except Exception as e:
            logger.warning("[Shutdown] Claim index save failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Cross-collection deduplication
    # ------------------------------------------------------------------

    _dedup_ran = False  # class-level guard against double-run

    async def _run_cross_collection_dedup(self):
        """Run cross-collection deduplication after all extractions.

        Runs as a non-blocking maintenance pass. Failures are logged
        but never block shutdown. Guarded to run at most once per process.
        """
        if ShutdownProcessor._dedup_ran:
            logger.debug("[Shutdown] Cross-dedup already ran this session, skipping")
            return
        ShutdownProcessor._dedup_ran = True

        from config.app_config import (
            CROSS_DEDUP_ENABLED, CROSS_DEDUP_ON_SHUTDOWN, CROSS_DEDUP_AUTO_EXECUTE,
        )

        if not CROSS_DEDUP_ENABLED or not CROSS_DEDUP_ON_SHUTDOWN:
            logger.debug("[Shutdown] Cross-collection dedup disabled, skipping")
            return

        try:
            from memory.cross_deduplicator import CrossCollectionDeduplicator

            dedup = CrossCollectionDeduplicator(self.chroma_store)

            if CROSS_DEDUP_AUTO_EXECUTE:
                # User mode: execute dedup automatically
                plan = dedup.run(dry_run=False)
                if plan.deletions_executed:
                    logger.info(
                        "[Shutdown] Cross-dedup executed: %d duplicates, %d contradictions, "
                        "%d deletions applied.",
                        plan.duplicates_found,
                        plan.contradictions_found,
                        plan.deletions_executed,
                    )
                else:
                    logger.debug("[Shutdown] Cross-dedup: no cleanup needed")
            else:
                # Dev mode: dry-run preview only
                plan = dedup.run(dry_run=True)
                if plan.duplicates_found or plan.contradictions_found:
                    logger.info(
                        "[Shutdown] Cross-dedup preview: %d duplicates, %d contradictions "
                        "(%d would be deleted). Run from GUI to execute.",
                        plan.duplicates_found,
                        plan.contradictions_found,
                        plan.deletions_executed,
                    )
                else:
                    logger.debug("[Shutdown] Cross-dedup: no duplicates or contradictions found")

            if plan.errors:
                for err in plan.errors:
                    logger.warning("[Shutdown] Cross-dedup error: %s", err)

        except Exception as e:
            logger.warning("[Shutdown] Cross-collection dedup failed (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # run_shutdown_reflection
    # ------------------------------------------------------------------

    async def run_shutdown_reflection(
        self,
        session_conversations: list[dict] | None = None,
        session_summaries: list[dict | str] | None = None,
    ) -> bool:
        """Generate and store an end-of-session reflection."""
        if session_conversations:
            session_conversations = [e for e in session_conversations if not is_test_turn(e)]
        if (
            not REFLECTIONS_ENABLED
            or not hasattr(self, 'model_manager')
            or not hasattr(self.model_manager, 'generate_once')
        ):
            return False

        conv = list(session_conversations or [])
        sums = list(session_summaries or [])

        if not conv:
            try:
                corpus = list(self.corpus_manager.derivable_entries())
                non = [
                    e for e in corpus
                    if (not self._is_summary(e)) and (not self._is_reflection(e))
                    and self._ts(e) >= self.session_start
                ]
                non.sort(key=self._ts, reverse=True)
                if REFLECTION_FALLBACK_RECENT and REFLECTION_FALLBACK_RECENT > 0:
                    conv = non[:REFLECTION_FALLBACK_RECENT]
                else:
                    conv = non
            except (AttributeError, TypeError) as e:
                logger.debug(f"[Shutdown] Reflection context gathering failed: {e}")
                conv = []

        if len([c for c in conv if (c.get("query") or c.get("response"))]) < REFLECTION_MIN_EXCHANGES and not sums:
            return False

        def _slice(e):
            q = (e.get("query") or "").strip()
            a = (e.get("response") or "").strip()
            segs = []
            if q:
                segs.append(f"User: {q[:240]}")
            if a:
                segs.append(f"Assistant: {a[:300]}")
            return "\n".join(segs)

        conv_blocks = [_slice(e) for e in conv if (e.get("query") or e.get("response"))]
        sum_texts = []
        for s in sums:
            if isinstance(s, dict):
                sum_texts.append(str(s.get("content") or s.get("text") or "").strip())
            elif isinstance(s, str):
                sum_texts.append(s.strip())

        prompt = (
            "You are a session reviewer for an AI assistant.\n"
            "Produce a reflection with these sections:\n\n"
            "1) SESSION TOPIC: One line stating the primary topic/domain discussed "
            "(e.g., \"statistics homework on regression\", \"Daemon retrieval benchmarks\", "
            "\"anxiety about work deadlines\").\n\n"
            "2) KEY ENTITIES: List specific names, projects, tools, people, places, "
            "or concepts mentioned (e.g., \"FAISS, Maria, Stanford, ChromaDB\"). "
            "Do NOT list generic words like \"the user\" or \"the assistant\".\n\n"
            "3) WHAT HAPPENED: 3-5 bullets describing specific events, decisions, "
            "breakthroughs, or problems from this session. Use concrete nouns and "
            "names. Do NOT write generic assistant-evaluation language like "
            "\"the assistant effectively\" or \"demonstrated understanding\".\n\n"
            "4) PATTERNS & INSIGHTS: 2-3 bullets on durable observations to carry "
            "forward. Be specific about what and why.\n\n"
            "Keep it concise and fact-dense. Every bullet should contain at least "
            "one specific noun, name, or concrete detail.\n\n"
        )
        if sum_texts:
            sum_use = sum_texts if REFLECTION_MAX_SUMMARIES <= 0 else sum_texts[:REFLECTION_MAX_SUMMARIES]
            prompt += "SESSION SUMMARIES:\n" + "\n\n".join(f"- {t}" for t in sum_use) + "\n\n"
        conv_use = conv_blocks if REFLECTION_MAX_EXCERPTS <= 0 else conv_blocks[-REFLECTION_MAX_EXCERPTS:]
        prompt += "CONVERSATION EXCERPTS:\n" + "\n\n".join(conv_use) + "\n\n"
        prompt += "Produce the three sections now."

        # Pass the reflection model explicitly rather than switch_model() — the
        # latter mutates the shared active_model_name, which races when shutdown
        # LLM calls run concurrently (reflection ‖ process_shutdown_memory).
        # generate_once(model_name=...) routes without touching global state.
        api_models = getattr(self.model_manager, "api_models", {}) or {}
        reflection_model = REFLECTION_MODEL_ALIAS if REFLECTION_MODEL_ALIAS in api_models else None
        text = await self.model_manager.generate_once(
            prompt,
            model_name=reflection_model,
            max_tokens=REFLECTION_MAX_TOKENS,
        )

        text = (text or "").strip()
        if not text:
            return False

        return await self._storage.add_reflection(text, tags=["session:end"], source="shutdown")
