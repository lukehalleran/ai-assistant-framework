"""
# memory/user_profile.py

Module Contract
- Purpose: Persistent user profile manager that aggregates facts across sessions. Provides categorized storage, conflict resolution, and profile export in ChatGPT-style format. ENHANCED: Hybrid retrieval (2/3 semantic + 1/3 recent) for query-relevant fact selection.
- Inputs:
  - add_fact(relation, value, confidence, source_excerpt, category?) → bool
  - add_facts_batch(facts: List[Dict]) → int
  - get_category(category: ProfileCategory) → List[Dict]
  - get_relevant_facts(query: str, category: ProfileCategory, limit: int) → List[Dict] [NEW: hybrid retrieval]
  - get_context_injection(max_tokens: int, query: str) → str [UPDATED: uses query for semantic relevance; timestamps formatted as relative labels via format_relative_timestamp(); includes source_excerpt as (said: "...") when available]
- Outputs:
  - Profile data structures (dicts, lists)
  - Markdown exports, context strings for prompt injection
  - Success/failure booleans and counts
  - UPDATED: Query-relevant facts ranked by hybrid score (2/3 semantic + 1/3 recent)
- Key behaviors:
  - JSON persistence with atomic writes (temp file swap)
  - Append-only fact storage: facts are never deleted, only marked is_current=False
  - Conflict resolution: same (relation,value) → confidence boost; same relation, diff value → supersede
  - Automatic categorization via user_profile_schema
  - Quick profile for identity fields (name, location, age)
  - Chronological raw_log like ChatGPT's memory system
  - UPDATED: Hybrid fact retrieval - 2/3 semantic (keyword overlap), 1/3 recent per category
- Side effects:
  - Writes to data/user_profile.json on save()
  - Thread-safe with lock for concurrent access
"""
import sys
import os
import uuid

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

from utils.logging_utils import get_logger
from memory.user_profile_schema import (
    ProfileCategory, ProfileFact, categorize_relation,
    ProfilePreferences, ProfileIdentity, SCHEMA_VERSION
)
import config.app_config as app_config
from memory.truth_scorer import TruthScorer

logger = get_logger("user_profile")


class UserProfile:
    """
    Manages persistent user profile built from extracted facts.

    Structure:
    {
        "user_id": "default",
        "created_at": "ISO timestamp",
        "updated_at": "ISO timestamp",
        "quick_profile": {
            "name": "...",
            "location": "...",
            ...
        },
        "categories": {
            "identity": [ProfileFact, ...],
            "fitness": [ProfileFact, ...],
            ...
        },
        "raw_log": [  # Chronological fact log like ChatGPT's
            {"timestamp": "...", "fact": "...", "category": "..."},
            ...
        ]
    }
    """

    if getattr(sys, 'frozen', False):
    
        DEFAULT_PATH = os.path.join(os.environ.get('APPDATA', ''), 'Daemon', 'user_profile.json')

    else:
        DEFAULT_PATH = "data/user_profile.json"

    def __init__(self, profile_path: str = None):

        print(f"[DEBUG UserProfile] DEFAULT_PATH={self.DEFAULT_PATH}")
        print(f"[DEBUG UserProfile] profile_path arg={profile_path}")
        print(f"[DEBUG UserProfile] frozen={getattr(sys, 'frozen', False)}")
        self.profile_path = profile_path or self.DEFAULT_PATH
        self._lock = threading.Lock()
        self.profile = self._load_or_init()

        # Initialize new fields (will be populated by _load_or_init or migrate_schema)
        self.version = self.profile.get("version", "0.0")
        self.identity = ProfileIdentity()
        self.preferences = ProfilePreferences()

        # Load identity and preferences from profile dict if they exist
        if "identity" in self.profile:
            self.identity = ProfileIdentity.from_dict(self.profile["identity"])
            logger.debug(f"[UserProfile] Loaded identity: name='{self.identity.name}', pronouns='{self.identity.pronouns}'")
        else:
            logger.debug("[UserProfile] No identity found in profile, using defaults")

        if "preferences" in self.profile:
            self.preferences = ProfilePreferences.from_dict(self.profile["preferences"])
            logger.debug(f"[UserProfile] Loaded preferences: style='{self.preferences.style}'")
        else:
            logger.debug("[UserProfile] No preferences found in profile, using defaults")

        # Run schema migration if needed
        self.migrate_schema()

        logger.info(f"[UserProfile] Initialized from {self.profile_path} - identity.name='{self.identity.name}'")

    def _load_or_init(self) -> Dict:
        """Load existing profile or create new one."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"[UserProfile] Loaded {len(data.get('raw_log', []))} facts")
                return data
            except Exception as e:
                logger.error(f"[UserProfile] Failed to load: {e}")

        # Initialize empty profile
        return {
            "user_id": "default",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "quick_profile": {},
            "categories": {cat.value: [] for cat in ProfileCategory},
            "raw_log": []
        }

    def save(self) -> None:
        """Persist profile to disk (atomic write)."""
        with self._lock:
            self.profile["updated_at"] = datetime.now().isoformat()

            # Update profile dict with new fields before saving
            self.profile["version"] = self.version
            self.profile["identity"] = self.identity.to_dict()
            self.profile["preferences"] = self.preferences.to_dict()

            # Ensure directory exists
            Path(self.profile_path).parent.mkdir(parents=True, exist_ok=True)

            # Atomic write via temp file
            temp_path = f"{self.profile_path}.tmp"
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.profile, f, indent=2, default=str)
                os.replace(temp_path, self.profile_path)
                logger.debug(f"[UserProfile] Saved to {self.profile_path}")
            except Exception as e:
                logger.error(f"[UserProfile] Save failed: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def add_fact(self,
                 relation: str,
                 value: str,
                 confidence: float = 0.7,
                 source_excerpt: str = "",
                 category: ProfileCategory = None,
                 timestamp: datetime = None) -> bool:
        """
        Add a fact to the profile using append-only storage.

        Behavior:
        1. Same (relation, value) seen again: boost confidence by +0.05 (capped at 1.0). No duplicate appended.
        2. Same relation, different value: mark existing facts for that relation as is_current=False,
           append new fact with is_current=True and supersedes=old_fact_id.
        3. New relation: append with is_current=True.

        Args:
            timestamp: Optional timestamp to preserve from imports (defaults to now)

        Returns True if fact was added/updated, False if rejected.
        """
        if not relation or not value:
            return False

        relation = relation.strip().lower()
        value = value.strip()

        # Canonicalize relation name to prevent namespace drift
        from memory.user_profile_schema import canonicalize_profile_relation
        original_relation = relation
        relation = canonicalize_profile_relation(relation, value)
        if relation != original_relation:
            logger.debug(f"[UserProfile] Canonicalized relation: {original_relation} → {relation}")

        # Resolve relative temporal references ("tomorrow" → "Thu 2026-03-13")
        from utils.temporal_resolver import resolve_temporal_references, has_temporal_reference
        if has_temporal_reference(value):
            ref_date = timestamp if isinstance(timestamp, datetime) else datetime.now()
            value = resolve_temporal_references(value, reference_date=ref_date)

        # Re-categorize after canonicalization (unless explicitly provided)
        if category is None:
            category = categorize_relation(relation)

        # Use provided timestamp or default to now
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            # Parse ISO format timestamp strings
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, AttributeError):
                timestamp = datetime.now()

        fact = ProfileFact(
            relation=relation,
            value=value,
            category=category,
            confidence=confidence,
            source_excerpt=source_excerpt[:200],
            timestamp=timestamp,
            fact_id=str(uuid.uuid4()),
            is_current=True,
        )

        with self._lock:
            cat_key = category.value
            facts_list = self.profile["categories"][cat_key]

            # Find existing facts with same canonical relation
            exact_match_idx = None  # same canonical relation AND value
            same_relation_current = []  # same canonical relation, is_current=True, different value

            for i, existing in enumerate(facts_list):
                if not isinstance(existing, dict):
                    continue
                existing_rel = existing.get("relation", "")
                # Compare using canonical forms so e.g. pet_name matches pet
                existing_canonical = canonicalize_profile_relation(
                    existing_rel, existing.get("value")
                )
                if existing_canonical != relation:
                    continue

                existing_val = existing.get("value", "")
                if existing_val.lower() == value.lower():
                    exact_match_idx = i
                elif existing.get("is_current", True):
                    same_relation_current.append(i)

            fact_dict = fact.to_dict()

            if exact_match_idx is not None:
                # Case 1: Same (relation, value) — confirmation: boost confidence + truth
                old = facts_list[exact_match_idx]
                old_conf = old.get("confidence", 0.5)
                new_conf = min(1.0, old_conf + 0.05)
                facts_list[exact_match_idx]["confidence"] = new_conf
                facts_list[exact_match_idx]["timestamp"] = timestamp.isoformat()
                # Truth: confirmation boost + reset decay clock
                old_truth = float(old.get("truth_score", TruthScorer.calculate_initial_score("user_stated")))
                facts_list[exact_match_idx]["truth_score"] = TruthScorer.apply_confirmation(old_truth)
                facts_list[exact_match_idx]["last_confirmed_at"] = timestamp.isoformat()
                facts_list[exact_match_idx]["confirmation_count"] = old.get("confirmation_count", 0) + 1
                # Ensure old fact has fact_id (migration compat)
                if not facts_list[exact_match_idx].get("fact_id"):
                    facts_list[exact_match_idx]["fact_id"] = str(uuid.uuid4())
                logger.debug(f"[UserProfile] Confirmed {relation}='{value}', confidence {old_conf:.2f} → {new_conf:.2f}")
            elif same_relation_current:
                # Case 2: Same relation, different value — correction: penalize old, new gets corrected score
                last_old_id = None
                for idx in same_relation_current:
                    facts_list[idx]["is_current"] = False
                    # Truth: apply correction penalty to superseded fact
                    old_truth = float(facts_list[idx].get("truth_score", 0.7))
                    facts_list[idx]["truth_score"] = TruthScorer.apply_correction(old_truth)
                    # Ensure old fact has fact_id
                    if not facts_list[idx].get("fact_id"):
                        facts_list[idx]["fact_id"] = str(uuid.uuid4())
                    last_old_id = facts_list[idx]["fact_id"]
                fact_dict["supersedes"] = last_old_id
                # Truth metadata for new corrected fact
                fact_dict["truth_score"] = TruthScorer.calculate_initial_score("corrected")
                fact_dict["truth_source"] = "corrected"
                fact_dict["last_confirmed_at"] = timestamp.isoformat()
                fact_dict["confirmation_count"] = 0
                facts_list.append(fact_dict)
                logger.info(f"[UserProfile] Superseded {relation}: → '{value}' (marked {len(same_relation_current)} old as historical)")
            else:
                # Case 3: New relation — append with initial truth metadata
                source = "user_stated" if confidence >= 0.7 else "llm_extracted"
                fact_dict["truth_score"] = TruthScorer.calculate_initial_score(source)
                fact_dict["truth_source"] = source
                fact_dict["last_confirmed_at"] = timestamp.isoformat()
                fact_dict["confirmation_count"] = 0
                facts_list.append(fact_dict)
                logger.info(f"[UserProfile] Added {category.value}/{relation}: '{value}'")

            # Add to raw log
            self.profile["raw_log"].append({
                "timestamp": datetime.now().isoformat(),
                "fact": f"user | {relation} | {value}",
                "category": cat_key,
                "confidence": confidence
            })

            # Update quick_profile for key identity fields
            if category == ProfileCategory.IDENTITY:
                self._update_quick_profile(relation, value)

            # Prune ephemeral facts if category exceeds soft cap
            self._prune_category(cat_key)

        return True

    def _update_quick_profile(self, relation: str, value: str) -> None:
        """Update the quick-access profile fields."""
        quick_keys = ["name", "first_name", "location", "lives_in", "age", "birthday", "timezone"]
        if relation in quick_keys:
            # Normalize key name
            key = "location" if relation == "lives_in" else relation
            self.profile["quick_profile"][key] = value

    def _prune_category(self, cat_key: str) -> None:
        """
        Prune is_current=False ephemeral facts when category exceeds soft cap.

        Only removes historical entries for ephemeral relations (current_activity, etc.).
        Stable facts (name, age, etc.) are never pruned.
        """
        facts_list = self.profile["categories"][cat_key]
        if len(facts_list) <= app_config.PROFILE_CATEGORY_SOFT_CAP:
            return

        ephemeral_set = set(app_config.PROFILE_EPHEMERAL_RELATIONS)
        pruned = 0

        # Group historical ephemeral facts by relation
        for rel in ephemeral_set:
            historical = [
                (i, f) for i, f in enumerate(facts_list)
                if isinstance(f, dict)
                and f.get("relation") == rel
                and not f.get("is_current", True)
            ]
            if len(historical) > app_config.PROFILE_EPHEMERAL_MAX_HISTORY:
                # Sort by timestamp ascending, remove oldest beyond limit
                historical.sort(key=lambda x: x[1].get("timestamp", ""))
                to_remove = len(historical) - app_config.PROFILE_EPHEMERAL_MAX_HISTORY
                indices_to_remove = {historical[j][0] for j in range(to_remove)}
                # Remove in reverse index order to preserve indices
                for idx in sorted(indices_to_remove, reverse=True):
                    facts_list.pop(idx)
                    pruned += 1

        if pruned > 0:
            logger.info(f"[UserProfile] Pruned {pruned} historical ephemeral facts from '{cat_key}'")

    def add_facts_batch(self, facts: List[Dict]) -> int:
        """
        Add multiple facts from extractor output.

        Expected format: [{"relation": "...", "value": "...", "confidence": 0.x}, ...]
        Also accepts: [{"subject": "user", "relation": "...", "object": "..."}, ...]

        Returns count of successfully added facts.
        """
        added = 0
        for fact in facts:
            # Handle both formats
            relation = fact.get("relation", "")
            value = fact.get("value") or fact.get("object", "")
            confidence = fact.get("confidence", 0.7)
            source = fact.get("source_excerpt", "")
            timestamp = fact.get("timestamp")  # Preserve original timestamp if provided

            if self.add_fact(relation, value, confidence, source, timestamp=timestamp):
                added += 1

        if added > 0:
            self.save()

        return added

    # Suffix patterns that indicate ephemeral/transient facts
    _EPHEMERAL_SUFFIXES = (
        "_status", "_condition", "_concern", "_intent",
        "_taken", "_left", "_duration", "_activity",
        "_time", "_deadline", "_variant", "_feeling",
        "_experience", "_plans", "_event",
    )
    # Prefix patterns that indicate ephemeral/transient facts
    _EPHEMERAL_PREFIXES = (
        "current_", "recent_", "upcoming_", "last_", "next_",
        "time_", "waiting_", "took_", "woke_",
    )

    def _is_ephemeral_relation(self, relation: str) -> bool:
        """Check if a relation is ephemeral (transient state, not stable identity)."""
        # Explicit list from config
        if relation in set(app_config.PROFILE_EPHEMERAL_RELATIONS):
            return True
        # Pattern matching
        if any(relation.startswith(p) for p in self._EPHEMERAL_PREFIXES):
            return True
        if any(relation.endswith(s) for s in self._EPHEMERAL_SUFFIXES):
            return True
        return False

    def get_category(self, category: ProfileCategory, include_historical: bool = False) -> List[Dict]:
        """Get facts in a category.

        Default returns only is_current=True facts, excluding stale ephemeral facts
        (older than PROFILE_EPHEMERAL_TTL_HOURS).
        """
        facts = self.profile["categories"].get(category.value, [])
        if include_historical:
            return facts

        ttl_hours = app_config.PROFILE_EPHEMERAL_TTL_HOURS
        now = datetime.now()

        result = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            if not f.get("is_current", True):
                continue
            # Drop stale ephemeral facts beyond TTL
            rel = f.get("relation", "")
            if self._is_ephemeral_relation(rel) and ttl_hours > 0:
                ts_str = f.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                    age_hours = (now - ts).total_seconds() / 3600
                    if age_hours > ttl_hours:
                        continue
                except (ValueError, TypeError):
                    pass
            result.append(f)
        return result

    def get_all_facts(self) -> Dict[str, List[Dict]]:
        """Get all categorized facts."""
        return self.profile["categories"]

    def get_quick_profile(self) -> Dict[str, str]:
        """Get quick-access profile fields."""
        return self.profile.get("quick_profile", {})

    def get_fact_count(self) -> int:
        """Total number of facts stored."""
        return len(self.profile.get("raw_log", []))

    def get_current_view(self, category: ProfileCategory = None) -> Dict[str, List[Dict]]:
        """
        Returns only is_current=True facts, grouped by category.

        Args:
            category: Optional single category to filter. If None, returns all categories.
        """
        result = {}
        categories = [category] if category else list(ProfileCategory)
        for cat in categories:
            current = self.get_category(cat, include_historical=False)
            if current:
                result[cat.value] = current
        return result

    def get_profile_at(self, timestamp: datetime) -> Dict[str, List[Dict]]:
        """
        Point-in-time profile snapshot.

        For each (relation, category), returns the latest fact at-or-before the given timestamp.
        """
        result = {}
        for cat in ProfileCategory:
            all_facts = self.get_category(cat, include_historical=True)
            # Group by relation
            by_relation: Dict[str, List[Dict]] = {}
            for f in all_facts:
                if not isinstance(f, dict):
                    continue
                rel = f.get("relation", "")
                ts_str = f.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                except (ValueError, TypeError):
                    continue
                if ts <= timestamp:
                    by_relation.setdefault(rel, []).append(f)

            # For each relation, pick the latest fact at-or-before timestamp
            cat_facts = []
            for rel, facts in by_relation.items():
                facts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                cat_facts.append(facts[0])

            if cat_facts:
                result[cat.value] = cat_facts
        return result

    def get_fact_history(self, relation: str, category: ProfileCategory = None) -> List[Dict]:
        """
        All values a relation has had, sorted chronologically (oldest first).

        Args:
            relation: The relation name to retrieve history for.
            category: Optional category to narrow search. If None, searches all categories.
        """
        relation = relation.strip().lower()
        results = []
        categories = [category] if category else list(ProfileCategory)
        for cat in categories:
            all_facts = self.get_category(cat, include_historical=True)
            for f in all_facts:
                if isinstance(f, dict) and f.get("relation") == relation:
                    results.append(f)
        results.sort(key=lambda x: x.get("timestamp", ""))
        return results

    def export_markdown(self) -> str:
        """
        Generate ChatGPT-style markdown profile export.
        """
        lines = []
        lines.append(f"# User Profile - {self.profile.get('user_id', 'default')}")
        lines.append(f"\n> Last updated: {self.profile.get('updated_at', 'unknown')}")
        lines.append(f"> Total facts: {self.get_fact_count()}")
        lines.append("")

        # Quick Profile section
        quick = self.get_quick_profile()
        if quick:
            lines.append("## Quick Profile")
            for key, value in quick.items():
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            lines.append("")

        # Categories
        for cat in ProfileCategory:
            facts = self.get_category(cat)
            if facts:
                lines.append(f"## {cat.value.replace('_', ' ').title()}")
                for fact in facts:
                    if isinstance(fact, dict):
                        rel = fact.get("relation", "unknown")
                        val = fact.get("value", "")
                        conf = fact.get("confidence", 0)
                        lines.append(f"- {rel}: {val} (conf: {conf:.2f})")
                lines.append("")

        # Raw log (last 50)
        raw_log = self.profile.get("raw_log", [])
        if raw_log:
            lines.append("## Recent Fact Log (last 50)")
            for entry in raw_log[-50:]:
                ts = entry.get("timestamp", "")[:10]  # Just date
                fact = entry.get("fact", "")
                lines.append(f"{ts}: {fact}")

        return "\n".join(lines)

    # Temporal keywords that signal the user wants historical facts
    TEMPORAL_KEYWORDS = {
        "history", "over time", "trend", "progress", "used to",
        "before", "previously", "last month", "last week", "timeline",
        "changed", "evolution", "pattern", "how has", "how have",
    }

    def _is_temporal_query(self, query: str) -> bool:
        """Check if query contains temporal keywords requesting historical data."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS)

    def get_relevant_facts(self, query: str, category: ProfileCategory, limit: int = 3) -> List[Dict]:
        """
        Get most relevant facts for a category using hybrid approach.

        Strategy: 2/3 semantic (query-relevant) + 1/3 recent
        When temporal keywords detected, includes historical timeline facts.

        Args:
            query: Current user query for semantic relevance
            category: ProfileCategory to retrieve from
            limit: Total number of facts to return (default 3)

        Returns:
            List of fact dicts ranked by hybrid score
        """
        facts = self.get_category(category)
        if not facts:
            return []

        # Filter to medium+ confidence facts (lowered from 0.7 to include more imported facts)
        high_conf = [f for f in facts if isinstance(f, dict) and f.get("confidence", 0) >= 0.55]
        if not high_conf:
            return []

        # Calculate split: 2/3 semantic, 1/3 recent
        semantic_count = max(1, int(limit * 0.67))
        recent_count = max(1, limit - semantic_count)

        # Get semantic matches (simple keyword overlap for now)
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for fact in high_conf:
            value = fact.get("value", "").lower()
            relation = fact.get("relation", "").lower()

            # Simple keyword overlap score
            fact_words = set(value.split()) | set(relation.split())
            overlap = len(query_words & fact_words)
            semantic_score = overlap / max(1, len(query_words))

            scored.append((semantic_score, fact))

        # Sort by semantic score, take top 2/3
        scored.sort(key=lambda x: x[0], reverse=True)
        semantic_facts = [f[1] for f in scored[:semantic_count]]

        # Get most recent facts not already in semantic set
        semantic_ids = {id(f) for f in semantic_facts}
        recent_candidates = [f for f in high_conf if id(f) not in semantic_ids]
        recent_candidates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        recent_facts = recent_candidates[:recent_count]

        # Combine and deduplicate
        result = semantic_facts + recent_facts

        # If temporal query, include historical timeline for matched relations
        if self._is_temporal_query(query):
            matched_relations = set()
            for f in result:
                rel = f.get("relation", "")
                # Check if relation overlaps with query words
                rel_words = set(rel.lower().split("_"))
                if rel_words & query_words:
                    matched_relations.add(rel)

            for rel in matched_relations:
                history = self.get_fact_history(rel, category)
                for hf in history:
                    if hf not in result:
                        result.append(hf)

        return result[:limit]

    def get_context_injection(self, max_tokens: int = 500, query: str = "", facts_per_category: int = 3) -> str:
        """
        Generate compact profile summary for prompt injection.
        Uses hybrid retrieval: 2/3 semantic (query-relevant) + 1/3 recent per category.
        When temporal keywords detected, appends mini-timelines for relevant relations.

        Args:
            max_tokens: Approximate token budget for profile context
            query: Current user query for semantic relevance (optional)
            facts_per_category: Number of facts to retrieve per category (default: 3)

        Returns:
            Formatted profile string for prompt injection
        """
        parts = []
        is_temporal = bool(query) and self._is_temporal_query(query)

        # Quick profile (always include)
        quick = self.get_quick_profile()
        if quick:
            profile_updated = self.profile.get("updated_at", "")
            quick_str = ", ".join(f"{k}={v}" for k, v in quick.items())
            if profile_updated:
                parts.append(f"User: {quick_str} [profile_updated: {profile_updated}]")
            else:
                parts.append(f"User: {quick_str}")

        timeline_relations = set()

        for cat in ProfileCategory:
            if query:
                # Use hybrid retrieval when query provided
                relevant_facts = self.get_relevant_facts(query, cat, limit=facts_per_category)
            else:
                # Fallback to most recent high-confidence facts
                facts = self.get_category(cat)
                relevant_facts = sorted(
                    [f for f in facts if isinstance(f, dict) and f.get("confidence", 0) >= 0.55],
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )[:facts_per_category]

            if relevant_facts:
                from utils.time_manager import format_relative_timestamp
                fact_strs = []
                for f in relevant_facts:
                    # Include timestamp with relative day label for temporal reasoning
                    ts = f.get('timestamp', '')
                    if isinstance(ts, str) and ts:
                        try:
                            from datetime import datetime as _dt
                            ts_str = format_relative_timestamp(_dt.fromisoformat(ts))
                        except (ValueError, TypeError):
                            ts_str = ts
                    elif hasattr(ts, 'isoformat'):
                        ts_str = format_relative_timestamp(ts)
                    else:
                        ts_str = str(ts) if ts else ''
                    entry = f"{f['relation']}={f['value']} [{ts_str}]"
                    # Include source excerpt when available for LLM grounding
                    src_exc = (f.get('source_excerpt') or '').strip()
                    if src_exc:
                        clean_exc = src_exc[:80].replace('"', "'")
                        entry += f' (said: "{clean_exc}")'
                    fact_strs.append(entry)

                    # Track relations that overlap with query for timeline
                    if is_temporal:
                        rel = f.get("relation", "")
                        query_words = set(query.lower().split())
                        rel_words = set(rel.lower().split("_"))
                        if rel_words & query_words:
                            timeline_relations.add((rel, cat))

                parts.append(f"{cat.value}: {'; '.join(fact_strs)}")

        # Append mini-timelines for temporal queries
        if is_temporal and timeline_relations:
            for rel, cat in timeline_relations:
                history = self.get_fact_history(rel, cat)
                if len(history) > 1:
                    entries = []
                    for h in history:
                        ts_str = h.get("timestamp", "")[:10]  # Just date
                        entries.append(f"{h.get('value', '?')} [{ts_str}]")
                    parts.append(f"{rel} timeline: {' → '.join(entries)}")

        result = "\n".join(parts)

        # Rough token estimate (4 chars per token)
        if len(result) > max_tokens * 4:
            result = result[:max_tokens * 4] + "..."

        return result

    # ========================================================================
    # Onboarding Wizard Methods
    # ========================================================================

    def is_first_run(self, corpus_manager) -> bool:
        """
        Determine if wizard should run.

        Returns True if:
        - Corpus has fewer than 5 entries AND
        - Profile identity.name is empty/unset

        This handles edge cases:
        - New user with empty corpus: True (run wizard)
        - User who cleared corpus but has profile: False (skip wizard)
        - User with imported corpus but no profile: False (>5 entries means existing user)

        Args:
            corpus_manager: CorpusManager instance to check corpus size

        Returns:
            bool: True if wizard should run, False otherwise
        """
        corpus_count = 0
        try:
            corpus_count = len(corpus_manager.corpus) if hasattr(corpus_manager, 'corpus') else 0
        except Exception as e:
            logger.debug(f"[UserProfile] Failed to get corpus count: {e}")
            corpus_count = 0

        has_identity = bool(self.identity.name and self.identity.name.strip())
        is_first = corpus_count < 5 and not has_identity

        logger.info(f"[UserProfile] First-run check: corpus_count={corpus_count}, identity.name='{self.identity.name}', has_identity={has_identity}, is_first_run={is_first}")

        return is_first

    VALID_STYLES = {"warm", "balanced", "direct"}

    def update_preferences(self, style: str, check_distress: bool = True, brief_responses: bool = False) -> None:
        """
        Update user preference settings.

        Args:
            style: Conversation style (warm, balanced, or direct)
            check_distress: Whether to enable distress detection
            brief_responses: Whether to prefer brief responses

        Raises:
            ValueError: If style is not one of: warm, balanced, direct
        """
        if style not in self.VALID_STYLES:
            raise ValueError(f"Invalid style '{style}'. Must be one of: {self.VALID_STYLES}")

        self.preferences.style = style
        self.preferences.check_distress = check_distress
        self.preferences.brief_responses = brief_responses
        self.save()
        logger.info(f"[UserProfile] Updated preferences: style={style}")

    def update_identity(self, name: str, pronouns: str) -> None:
        """
        Update user identity metadata (name and pronouns).

        Args:
            name: User's name or nickname
            pronouns: User's pronouns (e.g., he/him, she/her, they/them)
        """
        self.identity.name = name.strip() if name else ""
        self.identity.pronouns = pronouns.strip() if pronouns else ""
        self.save()
        logger.info(f"[UserProfile] Updated identity: name={self.identity.name}, pronouns={self.identity.pronouns}")

    # Style modifier strings for orchestrator injection
    STYLE_MODIFIERS = {
        "warm": """
STYLE: WARM & SUPPORTIVE

Lean into empathy and acknowledgment
Longer responses are welcome when emotional content is present
Prioritize connection over efficiency
""",
        "balanced": "",  # No modifier for default/balanced style
        "direct": """
STYLE: DIRECT & CONCISE

Keep responses shorter than default
Lead with the answer, minimize preamble
Skip emotional scaffolding unless crisis detected
1-3 sentences preferred for most exchanges
""",
    }

    def get_style_modifier(self) -> str:
        """
        Get tone instruction modifier based on user's style preference.

        Returns:
            str: Style modifier string for injection into system prompt, or empty string for balanced
        """
        return self.STYLE_MODIFIERS.get(self.preferences.style, "")

    def migrate_schema(self) -> None:
        """
        Handle schema migrations between versions.

        Currently handles:
        - Migration from 0.0 (no version) to 1.0 (adds identity, preferences, version)
        - Migration from 1.0 to 2.0 (adds fact_id, is_current to all facts; builds supersedes chains)
        """
        if not hasattr(self, 'version') or self.version is None:
            self.version = "0.0"

        # Migration: 0.0 → 1.0 (add identity and preferences)
        if self.version == "0.0":
            logger.info("[UserProfile] Migrating schema from 0.0 to 1.0")

            # Initialize new fields if they don't exist
            if not hasattr(self, 'identity'):
                self.identity = ProfileIdentity()
            if not hasattr(self, 'preferences'):
                self.preferences = ProfilePreferences()

            self.version = "1.0"
            logger.info("[UserProfile] Schema migration 0.0 → 1.0 complete")

        # Migration: 1.0 → 2.0 (append-only facts with fact_id + is_current)
        if self.version == "1.0":
            logger.info("[UserProfile] Migrating schema from 1.0 to 2.0")

            for cat_key, facts_list in self.profile.get("categories", {}).items():
                # Group facts by relation to determine is_current
                by_relation: Dict[str, List[int]] = {}
                for i, fact in enumerate(facts_list):
                    if not isinstance(fact, dict):
                        continue
                    # Add fact_id if missing
                    if not fact.get("fact_id"):
                        fact["fact_id"] = str(uuid.uuid4())
                    rel = fact.get("relation", "")
                    by_relation.setdefault(rel, []).append(i)

                # For each relation, mark only the most recent as is_current=True
                for rel, indices in by_relation.items():
                    # Sort by timestamp
                    indices.sort(key=lambda idx: facts_list[idx].get("timestamp", ""))
                    for j, idx in enumerate(indices):
                        if j == len(indices) - 1:
                            # Most recent → current
                            facts_list[idx]["is_current"] = True
                            facts_list[idx].setdefault("supersedes", None)
                        else:
                            facts_list[idx]["is_current"] = False
                            # Build supersedes chain: next fact supersedes this one
                            next_idx = indices[j + 1]
                            facts_list[next_idx]["supersedes"] = facts_list[idx]["fact_id"]

            self.version = SCHEMA_VERSION
            self.save()
            logger.info(f"[UserProfile] Schema migration 1.0 → 2.0 complete")
