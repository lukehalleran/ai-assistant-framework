"""
# memory/user_profile.py

Module Contract
- Purpose: Persistent user profile manager that aggregates facts across sessions. Provides categorized storage, conflict resolution, and profile export in ChatGPT-style format. ENHANCED: Hybrid retrieval (2/3 semantic + 1/3 recent) for query-relevant fact selection.
- Inputs:
  - add_fact(relation, value, confidence, source_excerpt, category?) → bool
  - add_facts_batch(facts: List[Dict]) → int
  - get_category(category: ProfileCategory) → List[Dict]
  - get_relevant_facts(query: str, category: ProfileCategory, limit: int) → List[Dict] [NEW: hybrid retrieval]
  - get_context_injection(max_tokens: int, query: str) → str [UPDATED: now uses query for semantic relevance]
- Outputs:
  - Profile data structures (dicts, lists)
  - Markdown exports, context strings for prompt injection
  - Success/failure booleans and counts
  - UPDATED: Query-relevant facts ranked by hybrid score (2/3 semantic + 1/3 recent)
- Key behaviors:
  - JSON persistence with atomic writes (temp file swap)
  - Conflict resolution: newer + higher confidence wins
  - Automatic categorization via user_profile_schema
  - Quick profile for identity fields (name, location, age)
  - Chronological raw_log like ChatGPT's memory system
  - UPDATED: Hybrid fact retrieval - 2/3 semantic (keyword overlap), 1/3 recent per category
- Side effects:
  - Writes to data/user_profile.json on save()
  - Thread-safe with lock for concurrent access
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

from utils.logging_utils import get_logger
from memory.user_profile_schema import (
    ProfileCategory, ProfileFact, categorize_relation,
    ProfilePreferences, ProfileIdentity, SCHEMA_VERSION
)

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

    DEFAULT_PATH = "data/user_profile.json"

    def __init__(self, profile_path: str = None):
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
        Add a fact to the profile, handling conflicts.

        Args:
            timestamp: Optional timestamp to preserve from imports (defaults to now)

        Returns True if fact was added/updated, False if rejected.
        """
        if not relation or not value:
            return False

        relation = relation.strip().lower()
        value = value.strip()

        # Auto-categorize if not provided
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
            timestamp=timestamp
        )

        with self._lock:
            cat_key = category.value

            # Check for existing fact with same relation AND value (to allow multiple facts with same relation but different values)
            existing_idx = None
            for i, existing in enumerate(self.profile["categories"][cat_key]):
                if isinstance(existing, dict):
                    existing_rel = existing.get("relation", "")
                    existing_val = existing.get("value", "")
                else:
                    existing_rel = existing.relation if hasattr(existing, 'relation') else ""
                    existing_val = existing.value if hasattr(existing, 'value') else ""

                # Match on both relation AND value to allow multiple facts with same relation
                if existing_rel == relation and existing_val == value:
                    existing_idx = i
                    break

            fact_dict = fact.to_dict()

            if existing_idx is not None:
                # Conflict resolution: newer + higher confidence wins
                old = self.profile["categories"][cat_key][existing_idx]
                old_conf = old.get("confidence", 0.5) if isinstance(old, dict) else 0.5

                if confidence >= old_conf:
                    fact_dict["supersedes"] = old.get("value", "") if isinstance(old, dict) else ""
                    self.profile["categories"][cat_key][existing_idx] = fact_dict
                    logger.info(f"[UserProfile] Updated {relation}: '{fact_dict['supersedes']}' → '{value}'")
                else:
                    logger.debug(f"[UserProfile] Rejected lower-confidence update for {relation}")
                    return False
            else:
                self.profile["categories"][cat_key].append(fact_dict)
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

        return True

    def _update_quick_profile(self, relation: str, value: str) -> None:
        """Update the quick-access profile fields."""
        quick_keys = ["name", "first_name", "location", "lives_in", "age", "birthday", "timezone"]
        if relation in quick_keys:
            # Normalize key name
            key = "location" if relation == "lives_in" else relation
            self.profile["quick_profile"][key] = value

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

    def get_category(self, category: ProfileCategory) -> List[Dict]:
        """Get all facts in a category."""
        return self.profile["categories"].get(category.value, [])

    def get_all_facts(self) -> Dict[str, List[Dict]]:
        """Get all categorized facts."""
        return self.profile["categories"]

    def get_quick_profile(self) -> Dict[str, str]:
        """Get quick-access profile fields."""
        return self.profile.get("quick_profile", {})

    def get_fact_count(self) -> int:
        """Total number of facts stored."""
        return len(self.profile.get("raw_log", []))

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

    def get_relevant_facts(self, query: str, category: ProfileCategory, limit: int = 3) -> List[Dict]:
        """
        Get most relevant facts for a category using hybrid approach.

        Strategy: 2/3 semantic (query-relevant) + 1/3 recent

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
        return result[:limit]

    def get_context_injection(self, max_tokens: int = 500, query: str = "", facts_per_category: int = 3) -> str:
        """
        Generate compact profile summary for prompt injection.
        Uses hybrid retrieval: 2/3 semantic (query-relevant) + 1/3 recent per category.

        Args:
            max_tokens: Approximate token budget for profile context
            query: Current user query for semantic relevance (optional)
            facts_per_category: Number of facts to retrieve per category (default: 3)

        Returns:
            Formatted profile string for prompt injection
        """
        parts = []

        # Quick profile (always include)
        # Note: Quick profile doesn't have individual timestamps, but we can add the profile updated_at
        quick = self.get_quick_profile()
        if quick:
            profile_updated = self.profile.get("updated_at", "")
            quick_str = ", ".join(f"{k}={v}" for k, v in quick.items())
            if profile_updated:
                parts.append(f"User: {quick_str} [profile_updated: {profile_updated}]")
            else:
                parts.append(f"User: {quick_str}")

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
                fact_strs = []
                for f in relevant_facts:
                    # Include timestamp for temporal reasoning
                    ts = f.get('timestamp', '')
                    if isinstance(ts, str):
                        # Format: relation=value [timestamp]
                        fact_strs.append(f"{f['relation']}={f['value']} [{ts}]")
                    else:
                        # Fallback if timestamp is datetime object
                        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                        fact_strs.append(f"{f['relation']}={f['value']} [{ts_str}]")
                parts.append(f"{cat.value}: {'; '.join(fact_strs)}")

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

            # Update version
            self.version = SCHEMA_VERSION
            self.save()
            logger.info(f"[UserProfile] Schema migration complete: {self.version}")
