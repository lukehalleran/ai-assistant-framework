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
  - run_shutdown_reflection(session_conversations, topic, session_context) -> Optional[str]
    Generates and persists session-end reflections.
- Internal phases (called by process_shutdown_memory):
  - _generate_block_summaries(entries, conversations, topic) — fixed-size block summaries
  - _store_summary(summary_text, N, b, start, end, block) — persists with source_doc_ids backlinks,
    claim extraction, staleness_ratio, and temporal anchors
  - _extract_session_facts(session_conversations) — rule-based fact extraction with fact verification gate
  - _extract_llm_facts(session_conversations) — LLM-assisted triple extraction with fact verification gate
  - _extract_procedural_skills(session_conversations) — adaptive workflow extraction
  - _generate_proposals(session_conversations) — goal-directed code change proposals
  - _check_implementation_tracking() — lightweight proposal implementation detection
  - _process_open_threads(session_conversations) — resolution detection → extraction → cap enforcement
  - _run_synthesis_dreaming() — cross-domain candidate generation + filter pipeline [async]
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
import logging
from datetime import datetime
from typing import Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger("shutdown_processor")

# --- Environment-driven configuration ---
REFLECTIONS_ENABLED = os.getenv("REFLECTIONS_ENABLED", "1").strip() not in ("0", "false", "no", "off")
REFLECTION_MAX_TOKENS = int(os.getenv("REFLECTION_MAX_TOKENS", "300"))
REFLECTION_MODEL_ALIAS = os.getenv("LLM_REFLECTION_ALIAS", os.getenv("LLM_SUMMARY_ALIAS", "gpt-4o-mini"))
REFLECTION_MIN_EXCHANGES = int(os.getenv("REFLECTION_MIN_EXCHANGES", "4"))
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
        try:
            # 1) Prep corpus slices and counts
            corpus = list(self.corpus_manager.corpus)

            non_summ = [
                e for e in corpus
                if (not self._is_summary(e)) and (not self._is_reflection(e))
            ]
            non_summ.sort(key=self._ts)

            # count consolidator-produced summaries
            def _is_consolidator_summary(e: Dict) -> bool:
                tags = [str(t).lower() for t in (e.get('tags') or [])]
                typ = (e.get('type') or '').lower()
                if not (("summary" in typ) or ("@summary" in tags) or ("type:summary" in tags)):
                    return False
                return ("source:consolidator" in tags) or ("summary:consolidated" in tags)

            consolidator_summaries = [e for e in corpus if _is_consolidator_summary(e)]

            def _parse_block_n(tags: list, n: int) -> int | None:
                try:
                    key = f"block_n:{n}:"
                    for t in (tags or []):
                        t = str(t).strip().lower()
                        if t.startswith(key):
                            return int(t.split(':', 2)[2])
                except (ValueError, IndexError):
                    return None
                return None

            existing_block_indices = []
            N = max(1, int(getattr(self.consolidator, 'consolidation_threshold', 10)))
            for s in consolidator_summaries:
                bi = _parse_block_n(s.get('tags') or [], N)
                if isinstance(bi, int):
                    existing_block_indices.append(bi)

            T = len(non_summ)
            total_blocks = T // N

            existing_set = set(int(x) for x in existing_block_indices if isinstance(x, int))
            all_indices = set(range(total_blocks))
            missing = sorted(all_indices - existing_set)

            cap = max(1, SUMMARY_MAX_BLOCKS_PER_SHUTDOWN)
            selected_blocks = missing[-cap:] if missing else []

            if selected_blocks:
                await self._generate_block_summaries(
                    non_summ, selected_blocks, N,
                    existing_block_indices, consolidator_summaries, T
                )
            else:
                prev_blocks = len(existing_block_indices) if existing_block_indices else len(consolidator_summaries)
                pending_blocks = max(0, (T // N) - prev_blocks)
                since_last = 0 if pending_blocks > 0 else (T % N)
                remaining = 0 if pending_blocks > 0 else (N if since_last == 0 else (N - since_last))
                logger.info(
                    "[Shutdown] No new summaries due (N=%s, T=%s, prev=%s); since_last=%s, remaining=%s, t=%s, backlog=%s",
                    N, T, prev_blocks, since_last, remaining, remaining, pending_blocks
                )

            # 3) Extract facts from this session's user turns only
            await self._extract_session_facts(session_conversations)

            # 4) Optional LLM-assisted facts
            await self._extract_llm_facts(session_conversations)

            # 5) Extract procedural skills (adaptive workflows)
            await self._extract_procedural_skills(session_conversations)

            # 6) Generate code proposals (goal-directed)
            await self._generate_proposals(session_conversations)

            # 6.5) Lightweight implementation tracking (file existence only)
            await self._check_implementation_tracking()

            # 6.75) Extract open threads and detect resolutions
            await self._process_open_threads(session_conversations)

            # 6.8) Synthesis dreaming — generate and filter cross-domain candidates
            await self._run_synthesis_dreaming()

            # 7) Save knowledge graph and entity aliases
            self._save_knowledge_graph()

            # 8) Cross-collection deduplication (if enabled)
            await self._run_cross_collection_dedup()

            logger.info("[Shutdown] Memory processing complete")

        except Exception as e:
            logger.error(f"Shutdown processing error: {e}")

    async def _generate_block_summaries(
        self, non_summ, selected_blocks, N,
        existing_block_indices, consolidator_summaries, T
    ):
        """Generate summaries for selected blocks."""
        for b in selected_blocks:
            start = b * N
            end = start + N
            block = non_summ[start:end]
            if not block:
                break
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
            if not summary_text:
                continue

            self._store_summary(summary_text, N, b, start, end, block)

        prev_blocks = len(existing_block_indices) if existing_block_indices else len(consolidator_summaries)
        new_prev = prev_blocks + len(selected_blocks)
        pending_blocks = max(0, (T // N) - new_prev)
        since_last = 0 if pending_blocks > 0 else (T % N)
        remaining = 0 if pending_blocks > 0 else (N if since_last == 0 else (N - since_last))
        logger.info(
            f"[Shutdown] Created {len(selected_blocks)} summary block(s) (N={N}, T={T}, prev={prev_blocks}); "
            f"since_last={since_last}, remaining={remaining}, t={remaining}, backlog={pending_blocks}"
        )

    async def _consolidate_block(self, content_list: list) -> Optional[str]:
        """Try consolidator API, fall back to micro-summary or extractive LLM prompt."""
        summary_text: Optional[str] = None

        if hasattr(self.consolidator, 'consolidate_memories'):
            try:
                summary_node = await self.consolidator.consolidate_memories(content_list)
                if summary_node and getattr(summary_node, 'content', None):
                    summary_text = summary_node.content
            except Exception as e:
                logger.warning(f"[Shutdown] LLM consolidation failed: {e}, falling back to micro-summary")
                summary_text = None

        if summary_text is None:
            try:
                if len(content_list) <= 2:
                    def _clip(s, n=160):
                        s = s or ''
                        return s if len(s) <= n else (s[:n] + '\u2026')
                    lines = []
                    for it in content_list:
                        if it.get('q'):
                            lines.append(f"- User: {_clip(it['q'])}")
                        if it.get('a'):
                            lines.append(f"- Assistant: {_clip(it['a'])}")
                    summary_text = "\n".join(lines).strip()
                else:
                    excerpts = "\n\n".join(x['content'] for x in content_list if x.get('content'))
                    prompt = (
                        "You are an extractive note-taker. Using ONLY the EXCERPTS below, "
                        "write 3\u20135 factual bullets. Do NOT infer or invent anything not present. "
                        "If information is minimal, output 1\u20132 bullets that quote/paraphrase the text.\n\n"
                        f"EXCERPTS:\n{excerpts}\n\nBullets (no headers):"
                    )
                    if hasattr(self, 'model_manager') and hasattr(self.model_manager, 'generate_once'):
                        summary_text = await self.model_manager.generate_once(prompt, max_tokens=220)
            except Exception as e:
                logger.warning(f"[Shutdown] Extractive summary generation failed: {e}")
                summary_text = None

        return summary_text

    def _store_summary(self, summary_text: str, N: int, b: int, start: int, end: int, block: list):
        """Store summary into corpus and Chroma."""
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
                return
            self.corpus_manager.add_summary({
                'content': summary_text,
                'timestamp': datetime.now(),
                'type': 'summary',
                'tags': [
                    'summary:consolidated',
                    'source:consolidator',
                    f'block_n:{N}:{b}',
                    f'block_span_n:{N}:{start}-{end - 1}',
                ]
            })
        except (AttributeError, TypeError) as e:
            logger.debug(f"[Shutdown] Failed to add summary to corpus: {e}")

        try:
            md = {
                'timestamp': datetime.now().isoformat(),
                'type': 'summary',
                'importance_score': 0.7,
                'tags': f'summary:consolidated,source:consolidator,block_n:{N}:{b},block_span_n:{N}:{start}-{end - 1}',
                'memory_count': len(block),
                'staleness_ratio': 0.0,
            }
            # Add temporal anchors from the block's timestamp range
            try:
                block_times = [self._ts(e) for e in block if self._ts(e) != datetime.min]
                if block_times:
                    md['temporal_anchor_start'] = min(block_times).isoformat()
                    md['temporal_anchor_end'] = max(block_times).isoformat()
            except Exception:
                pass

            # Capture source conversation doc IDs for expand_memory drill-down
            try:
                if md.get('temporal_anchor_start') and md.get('temporal_anchor_end'):
                    ts_lo = datetime.fromisoformat(md['temporal_anchor_start'])
                    ts_hi = datetime.fromisoformat(md['temporal_anchor_end'])
                    all_convos = self.chroma_store.list_all('conversations') if hasattr(self.chroma_store, 'list_all') else []
                    source_ids = []
                    for cdoc in all_convos:
                        cmeta = cdoc.get('metadata') or {}
                        cts = cmeta.get('timestamp', '')
                        if not cts:
                            continue
                        try:
                            ct = datetime.fromisoformat(cts)
                        except (ValueError, TypeError):
                            continue
                        if ts_lo <= ct <= ts_hi:
                            cid = cdoc.get('id')
                            if cid:
                                source_ids.append(cid)
                    if source_ids:
                        md['source_doc_ids'] = ','.join(source_ids)
                        logger.debug(f"[Shutdown] Summary linked to {len(source_ids)} source conversations")
            except Exception as se:
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
                except Exception as ce:
                    logger.debug(f"[Staleness] Claim extraction for summary failed: {ce}")

        except (AttributeError, TypeError) as e:
            logger.debug(f"[Shutdown] Failed to add summary to chroma: {e}")

    async def _extract_session_facts(self, session_conversations):
        """Extract facts from this session's user turns."""
        session_recent: list[Dict]
        if isinstance(session_conversations, list) and session_conversations:
            session_recent = list(session_conversations)
        else:
            try:
                corpus = list(self.corpus_manager.corpus)
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
                q = (conv.get('query') or '').strip()
                if not q:
                    continue
                facts = await self.fact_extractor.extract_facts(q, "")
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
                                        except Exception:
                                            pass
                        except Exception as ve:
                            logger.debug(f"[Shutdown] Verification failed, proceeding: {ve}")

                    result = self.chroma_store.add_fact(
                        fact=fact_content,
                        source='shutdown_extraction',
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

        try:
            last_turns = int(os.getenv("LLM_FACTS_LAST_TURNS", "12"))
        except (ValueError, TypeError):
            last_turns = 12
        try:
            max_triples = int(os.getenv("LLM_FACTS_MAX_TRIPLES", "10"))
        except (ValueError, TypeError):
            max_triples = 10
        try:
            max_chars = int(os.getenv("LLM_FACTS_MAX_INPUT_CHARS", "4000"))
        except (ValueError, TypeError):
            max_chars = 4000
        model_alias = os.getenv("LLM_FACTS_MODEL", "gpt-4o-mini")

        if isinstance(session_conversations, list) and session_conversations:
            sess_items = session_conversations
        else:
            try:
                corpus = list(self.corpus_manager.corpus)
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

        users = [(e.get('query') or '').strip() for e in sess_items if (e.get('query') or '').strip()]
        user_tail = users[-max(1, last_turns):]

        if not user_tail:
            return

        from memory.llm_fact_extractor import LLMFactExtractor
        llm_ex = LLMFactExtractor(
            self.model_manager,
            model_alias=model_alias,
            max_input_chars=max_chars,
            max_triples=max_triples,
        )
        triples = await llm_ex.extract_triples(user_tail)

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
            except Exception as ve:
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
                            except Exception:
                                pass

            try:
                result = self.chroma_store.add_fact(
                    fact=fact_text,
                    source='llm_shutdown',
                    confidence=0.75,
                )
                if result is not None:
                    kept += 1
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
                corpus = list(self.corpus_manager.corpus)
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

        prev_model = None
        try:
            if hasattr(self.model_manager, "get_active_model_name"):
                prev_model = self.model_manager.get_active_model_name()
            if hasattr(self.model_manager, "switch_model"):
                api_models = getattr(self.model_manager, "api_models", {}) or {}
                if model_alias in api_models:
                    self.model_manager.switch_model(model_alias)

            raw = await self.model_manager.generate_once(prompt, max_tokens=600)
        except Exception as e:
            logger.warning(f"[Shutdown] Skill extraction LLM call failed: {e}")
            return
        finally:
            try:
                if prev_model and hasattr(self.model_manager, "switch_model"):
                    self.model_manager.switch_model(prev_model)
            except (AttributeError, TypeError):
                pass

        if not raw or not raw.strip():
            return

        # Parse JSON lines
        import json
        import time
        from memory.procedural_skill import ProceduralSkill, SkillCategory

        kept = 0
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

            doc_id = await self._storage.store_skill(skill)
            if doc_id:
                kept += 1

        if kept:
            logger.info(f"[Shutdown] Extracted {kept} procedural skill(s)")

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
                corpus = list(self.corpus_manager.corpus)
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

            generator = GoalDirectedGenerator(
                model_manager=self.model_manager,
                repo_path=".",
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
            for proposal in proposals:
                existing_id = proposal_store.check_similarity(proposal)
                if existing_id:
                    logger.debug(f"[Shutdown] Proposal skipped (duplicate of {existing_id}): {proposal.title}")
                    continue
                doc_id = proposal_store.store_proposal(proposal)
                if doc_id:
                    kept += 1

            if kept:
                logger.info(f"[Shutdown] Generated {kept} proposal(s)")

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
        except Exception as e:
            logger.debug(f"[Shutdown] Summary retrieval failed: {e}")

        git_commits = []
        try:
            git_commits = self.chroma_store.query_collection(
                "procedural", synthetic_query, n_results=5
            )
        except Exception as e:
            logger.debug(f"[Shutdown] Git commit retrieval failed: {e}")

        profile_text = ""
        try:
            if self.user_profile:
                profile_text = self.user_profile.get_context_injection(
                    max_tokens=300, query=synthetic_query
                )
        except Exception as e:
            logger.debug(f"[Shutdown] User profile retrieval failed: {e}")

        # Await async tasks
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
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

            store = ProposalStore(chroma_store=chroma_store)
            proposals = store.get_pending_and_approved()
            if not proposals:
                return

            detector = ImplementationDetector(repo_path=".")
            updated = 0
            for proposal in proposals:
                result = detector.detect_single(proposal, lightweight=True)
                if not result.skipped_reason:
                    if store.update_tracking_metadata(proposal.id, result):
                        updated += 1

            if updated:
                logger.info(f"[Shutdown] Implementation tracking: updated {updated}/{len(proposals)} proposals")

        except Exception as e:
            logger.warning(f"[Shutdown] Implementation tracking failed: {e}")

    # ------------------------------------------------------------------
    # Open thread extraction
    # ------------------------------------------------------------------

    async def _process_open_threads(self, session_conversations):
        """Extract open threads from session and detect resolutions of existing ones.

        Phase 1: Get existing open threads → run detect_resolutions() → mark resolved
        Phase 2: Run extract_new_threads() → store new threads
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
                corpus = list(self.corpus_manager.corpus)
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

        # Phase 1: Detect resolutions of existing threads
        try:
            existing_open = self.thread_store.list_open_threads()
            if existing_open:
                resolutions = await extractor.detect_resolutions(sess_items, existing_open)
                for thread_id, resolution in resolutions:
                    self.thread_store.resolve_thread(thread_id, resolution)
                if resolutions:
                    logger.info(f"[Shutdown] Resolved {len(resolutions)} thread(s)")
        except Exception as e:
            logger.warning(f"[Shutdown] Thread resolution detection failed: {e}")

        # Phase 2: Extract new threads
        try:
            new_threads = await extractor.extract_new_threads(sess_items)
            stored = 0
            for thread in new_threads:
                doc_id = self.thread_store.store_thread(thread)
                if doc_id:
                    stored += 1
            if stored:
                logger.info(f"[Shutdown] Stored {stored} new thread(s)")
        except Exception as e:
            logger.warning(f"[Shutdown] Thread extraction failed: {e}")

        # Phase 3: Enforce cap
        try:
            self.thread_store.enforce_cap()
        except Exception as e:
            logger.debug(f"[Shutdown] Thread cap enforcement failed: {e}")

    # ------------------------------------------------------------------
    # Synthesis dreaming (cross-domain candidate generation)
    # ------------------------------------------------------------------

    async def _run_synthesis_dreaming(self):
        """Step 6.8: Generate and filter cross-domain synthesis candidates."""
        try:
            from config.app_config import (
                SYNTHESIS_GENERATOR_ENABLED,
                SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION,
            )
            if not SYNTHESIS_GENERATOR_ENABLED:
                return

            from knowledge.synthesis_generator import SynthesisGenerator
            from knowledge.synthesis_filter import SynthesisFilter
            from memory.synthesis_memory import SynthesisMemory

            mc = self.memory_coordinator
            graph_memory = getattr(mc, "graph_memory", None) if mc else None
            entity_resolver = getattr(mc, "entity_resolver", None) if mc else None

            generator = SynthesisGenerator(
                chroma_store=self.chroma_store,
                model_manager=self.model_manager,
                graph_memory=graph_memory,
                entity_resolver=entity_resolver,
            )

            candidates = await generator.generate_candidates(
                count=SYNTHESIS_GENERATOR_CANDIDATES_PER_SESSION,
            )

            if not candidates:
                logger.info("[Shutdown] Synthesis dreaming: no candidates generated")
                return

            synthesis_memory = SynthesisMemory(self.chroma_store)
            filter_pipeline = SynthesisFilter(
                chroma_store=self.chroma_store,
                model_manager=self.model_manager,
                synthesis_memory=synthesis_memory,
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

        from config.app_config import CROSS_DEDUP_ENABLED, CROSS_DEDUP_ON_SHUTDOWN

        if not CROSS_DEDUP_ENABLED or not CROSS_DEDUP_ON_SHUTDOWN:
            logger.debug("[Shutdown] Cross-collection dedup disabled, skipping")
            return

        try:
            from memory.cross_deduplicator import CrossCollectionDeduplicator

            dedup = CrossCollectionDeduplicator(self.chroma_store)
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
                corpus = list(self.corpus_manager.corpus)
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
            "You are a neutral QA reviewer for an AI assistant session.\n"
            "Return three sections (3\u20136 bullets each):\n"
            "1) What went well\n"
            "2) What to improve\n"
            "3) High-level insights (durable heuristics to carry forward)\n"
            "Avoid praise; be specific and actionable.\n\n"
        )
        if sum_texts:
            sum_use = sum_texts if REFLECTION_MAX_SUMMARIES <= 0 else sum_texts[:REFLECTION_MAX_SUMMARIES]
            prompt += "SESSION SUMMARIES:\n" + "\n\n".join(f"- {t}" for t in sum_use) + "\n\n"
        conv_use = conv_blocks if REFLECTION_MAX_EXCERPTS <= 0 else conv_blocks[-REFLECTION_MAX_EXCERPTS:]
        prompt += "CONVERSATION EXCERPTS:\n" + "\n\n".join(conv_use) + "\n\n"
        prompt += "Produce the three sections now."

        prev_model = None
        try:
            if hasattr(self.model_manager, "get_active_model_name"):
                prev_model = self.model_manager.get_active_model_name()
            if hasattr(self.model_manager, "switch_model"):
                api_models = getattr(self.model_manager, "api_models", {}) or {}
                if REFLECTION_MODEL_ALIAS in api_models:
                    self.model_manager.switch_model(REFLECTION_MODEL_ALIAS)
            text = await self.model_manager.generate_once(
                prompt,
                max_tokens=REFLECTION_MAX_TOKENS
            )
        finally:
            try:
                if prev_model and hasattr(self.model_manager, "switch_model"):
                    self.model_manager.switch_model(prev_model)
                    try:
                        cur = (
                            self.model_manager.get_active_model_name()
                            if hasattr(self.model_manager, "get_active_model_name")
                            else prev_model
                        )
                        logger.info(f"[Reflections] restored active model after shutdown reflection: {cur}")
                    except (AttributeError, TypeError):
                        pass
            except (AttributeError, TypeError) as e:
                logger.debug(f"[Shutdown] Could not restore model after reflection: {e}")

        text = (text or "").strip()
        if not text:
            return False

        return await self._storage.add_reflection(text, tags=["session:end"], source="shutdown")
