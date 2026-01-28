# memory/shutdown_processor.py
"""
Shutdown memory processing module.

Handles end-of-session consolidation: block summaries, fact extraction,
LLM-assisted fact extraction, user profile updates, procedural skill
extraction, and session reflections.
"""

import os
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
    - Session-end reflection generation
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
    ):
        self.corpus_manager = corpus_manager
        self.chroma_store = chroma_store
        self.consolidator = consolidator
        self.fact_extractor = fact_extractor
        self.model_manager = model_manager
        self.user_profile = user_profile
        self._storage = storage
        self.session_start = session_start

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
            }
            if hasattr(self.chroma_store, 'add_to_collection'):
                self.chroma_store.add_to_collection('summaries', summary_text, md)
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
                    result = self.chroma_store.add_fact(
                        fact=getattr(fact, 'content', str(fact)),
                        source='shutdown_extraction',
                        confidence=0.7
                    )
                    if result is None:
                        logger.debug(f"[Shutdown] Fact skipped as duplicate: {getattr(fact, 'content', str(fact))}")
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
        kept = 0
        for t in triples:
            subj = t.get('subject')
            rel = t.get('relation')
            obj = t.get('object')
            if not subj or not rel or not obj:
                continue
            fact_text = f"{subj} | {rel} | {obj}"
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

        # Update user profile with extracted facts
        if triples and self.user_profile:
            try:
                added = self.user_profile.add_facts_batch(triples)
                logger.info(f"[Shutdown] Added {added} facts to user profile")
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
