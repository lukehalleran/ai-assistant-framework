#!/usr/bin/env python3
"""
import_chatgpt_memories.py

Imports ChatGPT memories (from exported JSON) into Daemon's fact extraction format.
Preserves original timestamps and normalizes facts into SRO triples with categories.

Usage:
    python import_chatgpt_memories.py --input luke_memories.json --output daemon_facts.json
"""

import json
import re
import uuid
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# ============================================================================
# Category definitions (matching llm_fact_extractor.py)
# ============================================================================

class ProfileCategory(Enum):
    IDENTITY = "identity"
    EDUCATION = "education"
    CAREER = "career"
    PROJECTS = "projects"
    HEALTH = "health"
    FITNESS = "fitness"
    PREFERENCES = "preferences"
    HOBBIES = "hobbies"
    STUDY = "study"
    FINANCE = "finance"
    RELATIONSHIPS = "relationships"
    GOALS = "goals"
    DIRECTIVES = "directives"  # Special category for user instructions to the AI
    UNKNOWN = "unknown"


# Relation -> Category mapping (extensible)
RELATION_CATEGORY_MAP = {
    # Identity
    "name": ProfileCategory.IDENTITY,
    "full_name": ProfileCategory.IDENTITY,
    "first_name": ProfileCategory.IDENTITY,
    "last_name": ProfileCategory.IDENTITY,
    "age": ProfileCategory.IDENTITY,
    "birthday": ProfileCategory.IDENTITY,
    "birth_date": ProfileCategory.IDENTITY,
    "lives_in": ProfileCategory.IDENTITY,
    "location": ProfileCategory.IDENTITY,
    "located_in": ProfileCategory.IDENTITY,
    "timezone": ProfileCategory.IDENTITY,

    # Education
    "studies": ProfileCategory.EDUCATION,
    "studied": ProfileCategory.EDUCATION,
    "degree": ProfileCategory.EDUCATION,
    "major": ProfileCategory.EDUCATION,
    "school": ProfileCategory.EDUCATION,
    "university": ProfileCategory.EDUCATION,
    "gpa": ProfileCategory.EDUCATION,
    "course": ProfileCategory.EDUCATION,
    "certification": ProfileCategory.EDUCATION,
    "took_course": ProfileCategory.EDUCATION,
    "courses_taken": ProfileCategory.EDUCATION,

    # Career
    "works_at": ProfileCategory.CAREER,
    "worked_at": ProfileCategory.CAREER,
    "occupation": ProfileCategory.CAREER,
    "job": ProfileCategory.CAREER,
    "role": ProfileCategory.CAREER,
    "position": ProfileCategory.CAREER,
    "experience": ProfileCategory.CAREER,
    "skills": ProfileCategory.CAREER,
    "used_tool": ProfileCategory.CAREER,
    "experience_years": ProfileCategory.CAREER,

    # Projects
    "project": ProfileCategory.PROJECTS,
    "working_on": ProfileCategory.PROJECTS,
    "building": ProfileCategory.PROJECTS,
    "tech_stack": ProfileCategory.PROJECTS,
    "implemented": ProfileCategory.PROJECTS,
    "created": ProfileCategory.PROJECTS,

    # Health
    "takes_medication": ProfileCategory.HEALTH,
    "medication": ProfileCategory.HEALTH,
    "takes_supplement": ProfileCategory.HEALTH,
    "supplement": ProfileCategory.HEALTH,
    "condition": ProfileCategory.HEALTH,
    "health_issue": ProfileCategory.HEALTH,
    "symptom": ProfileCategory.HEALTH,
    "sleep_pattern": ProfileCategory.HEALTH,
    "diagnosis": ProfileCategory.HEALTH,

    # Fitness
    "squat_max": ProfileCategory.FITNESS,
    "bench_max": ProfileCategory.FITNESS,
    "deadlift_max": ProfileCategory.FITNESS,
    "ohp_max": ProfileCategory.FITNESS,
    "weight": ProfileCategory.FITNESS,
    "height": ProfileCategory.FITNESS,
    "body_weight": ProfileCategory.FITNESS,
    "workout": ProfileCategory.FITNESS,
    "fitness_goal": ProfileCategory.FITNESS,
    "calories": ProfileCategory.FITNESS,
    "bulking_calories": ProfileCategory.FITNESS,

    # Preferences
    "prefers": ProfileCategory.PREFERENCES,
    "likes": ProfileCategory.PREFERENCES,
    "favorite": ProfileCategory.PREFERENCES,
    "favorite_game": ProfileCategory.PREFERENCES,
    "favorite_beer": ProfileCategory.PREFERENCES,
    "favorite_color": ProfileCategory.PREFERENCES,
    "formatting_preference": ProfileCategory.PREFERENCES,
    "communication_preference": ProfileCategory.PREFERENCES,

    # Hobbies
    "plays": ProfileCategory.HOBBIES,
    "hobby": ProfileCategory.HOBBIES,
    "interest": ProfileCategory.HOBBIES,
    "plays_game": ProfileCategory.HOBBIES,
    "collects": ProfileCategory.HOBBIES,
    "pet": ProfileCategory.HOBBIES,
    "pet_name": ProfileCategory.HOBBIES,
    "cat_name": ProfileCategory.HOBBIES,
    "dog_name": ProfileCategory.HOBBIES,

    # Study
    "studying_for": ProfileCategory.STUDY,
    "exam": ProfileCategory.STUDY,
    "study_hours": ProfileCategory.STUDY,
    "study_goal": ProfileCategory.STUDY,
    "preparing_for": ProfileCategory.STUDY,

    # Finance
    "budget": ProfileCategory.FINANCE,
    "income_goal": ProfileCategory.FINANCE,
    "financial_goal": ProfileCategory.FINANCE,
    "insurance": ProfileCategory.FINANCE,

    # Relationships
    "family": ProfileCategory.RELATIONSHIPS,
    "friend": ProfileCategory.RELATIONSHIPS,
    "partner": ProfileCategory.RELATIONSHIPS,
    "manager": ProfileCategory.RELATIONSHIPS,
    "colleague": ProfileCategory.RELATIONSHIPS,
    "mom": ProfileCategory.RELATIONSHIPS,
    "dad": ProfileCategory.RELATIONSHIPS,
    "cat": ProfileCategory.RELATIONSHIPS,

    # Goals
    "goal": ProfileCategory.GOALS,
    "plan": ProfileCategory.GOALS,
    "plans_to": ProfileCategory.GOALS,
    "wants_to": ProfileCategory.GOALS,
    "intends_to": ProfileCategory.GOALS,
    "aspiration": ProfileCategory.GOALS,

    # Directives (instructions to the AI)
    "directive": ProfileCategory.DIRECTIVES,
    "instruction": ProfileCategory.DIRECTIVES,
    "reminder": ProfileCategory.DIRECTIVES,
}


def categorize_relation(relation: str) -> ProfileCategory:
    """Map a relation string to a ProfileCategory."""
    rel_lower = relation.lower().strip()

    # Direct match
    if rel_lower in RELATION_CATEGORY_MAP:
        return RELATION_CATEGORY_MAP[rel_lower]

    # Partial match (e.g., "favorite_video_game" contains "favorite")
    for key, cat in RELATION_CATEGORY_MAP.items():
        if key in rel_lower or rel_lower in key:
            return cat

    # Keyword-based heuristics
    if any(kw in rel_lower for kw in ["exam", "study", "course", "gpa"]):
        return ProfileCategory.STUDY
    if any(kw in rel_lower for kw in ["lift", "squat", "bench", "deadlift", "workout", "calorie"]):
        return ProfileCategory.FITNESS
    if any(kw in rel_lower for kw in ["prefer", "like", "favorite", "format"]):
        return ProfileCategory.PREFERENCES
    if any(kw in rel_lower for kw in ["medication", "supplement", "symptom", "health", "sleep"]):
        return ProfileCategory.HEALTH
    if any(kw in rel_lower for kw in ["work", "job", "role", "company", "skill"]):
        return ProfileCategory.CAREER
    if any(kw in rel_lower for kw in ["project", "building", "daemon", "atlass"]):
        return ProfileCategory.PROJECTS
    if any(kw in rel_lower for kw in ["game", "play", "hobby", "interest", "pet", "cat"]):
        return ProfileCategory.HOBBIES
    if any(kw in rel_lower for kw in ["goal", "plan", "want", "intend", "aim"]):
        return ProfileCategory.GOALS
    if any(kw in rel_lower for kw in ["remind", "directive", "instruct", "always", "never"]):
        return ProfileCategory.DIRECTIVES

    return ProfileCategory.UNKNOWN


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ExtractedFact:
    """A single extracted fact (SRO triple with metadata)."""
    id: str
    subject: str
    relation: str
    object: str
    confidence: float
    category: str
    method: str
    source_text: str
    timestamp: datetime
    original_index: int
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "value": self.object,  # Alias for compatibility
            "confidence": self.confidence,
            "category": self.category,
            "method": self.method,
            "source_text": self.source_text,
            "timestamp": self.timestamp.isoformat(),
            "original_index": self.original_index,
            "tags": self.tags,
            "content": f"{self.subject} | {self.relation} | {self.object}",
            "type": "fact",
        }


# ============================================================================
# Text processing utilities
# ============================================================================

def _snake(s: str) -> str:
    """Convert string to snake_case relation."""
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")


def normalize_subject(subj: str) -> str:
    """Normalize subject, mapping pronouns to 'user'."""
    s = (subj or "").strip().lower()
    if s in {"i", "me", "my", "mine", "we", "our", "ours", "us", "luke", "user"}:
        return "user"
    return s


# ============================================================================
# Pattern-based extraction from memory text
# ============================================================================

class MemoryFactExtractor:
    """Extract SRO facts from ChatGPT memory text entries."""

    def __init__(self):
        # Patterns for extracting facts from ChatGPT memory text
        # These are tuned for the "User <verb> <object>" style ChatGPT uses

        self.patterns = [
            # Identity patterns
            (r"User(?:'s)?\s+name\s+is\s+(.+?)(?:\.|$)", "name", 0.95),
            (r"(\w+)(?:'s)?\s+name\s+is\s+(\w+\s+\w+)(?:\.|$)", "full_name", 0.95),  # "Luke's name is Luke Halleran"
            (r"User\s+is\s+(\d+)\s+(?:years?\s+old)?", "age", 0.90),
            (r"(\w+)\s+is\s+located\s+in\s+([A-Za-z]+\.?\s*[A-Za-z,\s]+?)(?:\.$)", "lives_in", 0.92),  # "Luke is located in X"
            (r"User\s+(?:is\s+)?located\s+in\s+(.+?)(?:\.|$)", "lives_in", 0.90),
            (r"User\s+lives?\s+in\s+(.+?)(?:\.|$)", "lives_in", 0.90),
            (r"User\s+will\s+be\s+(\d+)\s+on\s+(.+?)(?:\.|$)", "birthday", 0.85),
            (r"(\w+)\s+will\s+be\s+(\d+)\s+on\s+(.+?)(?:\.|$)", "birthday_upcoming", 0.85),  # "Luke will be 32 on Feb 9"

            # Education patterns
            (r"User\s+(?:has\s+)?taken?\s+(?:the\s+following\s+)?courses?[:\s]+(.+?)(?:\.|$)", "courses_taken", 0.85),
            (r"User(?:'s)?\s+GPA\s+is\s+(?:approximately\s+)?([0-9.]+)", "gpa", 0.90),
            (r"User\s+graduated\s+from\s+(.+?)(?:\s+in|\.|$)", "graduated_from", 0.90),

            # Career patterns
            (r"User\s+(?:works?|worked)\s+(?:at|for)\s+(.+?)(?:\s+as|\.|$)", "works_at", 0.85),
            (r"User\s+(?:works?|worked)\s+as\s+(?:a\s+|an\s+)?(.+?)(?:\.|$)", "occupation", 0.85),
            (r"User\s+has\s+(\d+)\s+years?\s+(?:of\s+)?experience", "experience_years", 0.85),
            (r"User\s+(?:has\s+)?worked\s+with\s+(.+?)(?:\s+at|\.|$)", "used_tool", 0.80),

            # Project patterns
            (r"User\s+is\s+(?:working\s+on|building|developing)\s+(.+?)(?:\.|$)", "working_on", 0.85),
            (r"User\s+(?:has\s+)?implemented\s+(.+?)(?:\.|$)", "implemented", 0.80),
            (r"User\s+(?:has\s+)?(?:changed|renamed)\s+(.+?)(?:'s)?\s+name\s+to\s+(.+?)(?:\.|$)", "renamed_project", 0.80),

            # Health patterns
            (r"User\s+takes?\s+(.+?)\s+(?:mg|mcg|IU|g|gram)", "takes_supplement", 0.85),
            (r"User\s+(?:takes?|is\s+on)\s+(.+?)\s+for\s+(.+?)(?:\.|$)", "takes_medication_for", 0.85),
            (r"User\s+(?:has|experiences?)\s+(.+?)\s+(?:symptoms?|issues?|condition)", "health_condition", 0.80),
            (r"User\s+(?:sleeps?|slept)\s+(.+?)(?:\.|$)", "sleep_pattern", 0.75),

            # Fitness patterns
            (r"User\s+is\s+(?:working\s+on\s+)?(?:increasing|improving)\s+(?:his\s+)?(\w+)\s+(?:from\s+)?(\d+)\s+to\s+(\d+)\s*(lbs?|pounds?|kg)?", "fitness_goal", 0.85),
            (r"User(?:'s)?\s+(?:current\s+)?weight\s+(?:is\s+)?(\d+)\s*(lbs?|pounds?|kg)?", "body_weight", 0.90),
            (r"User\s+is\s+(\d+)['\"]?\s*(\d+)?['\"]?\s*(?:tall)?", "height", 0.85),
            (r"User\s+(?:eats?|consumes?)\s+(?:around\s+)?([0-9,]+)\s+calories?", "bulking_calories", 0.85),

            # Preference patterns
            (r"User\s+prefers?\s+(.+?)(?:\.|$)", "prefers", 0.85),
            (r"User(?:'s)?\s+favorite\s+(\w+)\s+is\s+(.+?)(?:\.|$)", "favorite", 0.85),
            (r"User\s+(?:likes?|loves?|enjoys?)\s+(.+?)(?:\.|$)", "likes", 0.80),

            # Hobby patterns
            (r"User\s+is\s+(?:playing|learning\s+to\s+play)\s+(.+?)(?:\.|$)", "plays_game", 0.80),
            (r"User\s+is\s+interested\s+in\s+(.+?)(?:\.|$)", "interest", 0.80),
            (r"User(?:'s)?\s+(?:cat|dog|pet)(?:'s)?\s+(?:is\s+)?named?\s+(.+?)(?:\.|$)", "pet_name", 0.90),
            (r"User\s+(?:has|had)\s+a\s+(?:cat|dog|pet)\s+named\s+(.+?)(?:\.|$)", "pet_name", 0.90),
            (r"(\w+)(?:'s)?\s+(?:cat|dog|pet)(?:'s)?\s+(?:is\s+)?named?\s+(\w+)(?:\.|,|$)", "pet_name", 0.90),  # "Luke's cat is named Flapjack"
            (r"(\w+)\s+(?:has|had)\s+a\s+(cat|dog|pet)\s+named\s+(\w+)", "pet_relationship", 0.88),  # "Luke has a cat named Flapjack"

            # Study patterns
            (r"User\s+is\s+(?:preparing|studying)\s+for\s+(.+?)\s+exam", "studying_for", 0.85),
            (r"User\s+(?:has\s+)?logged\s+(\d+)\s+(?:good\s+)?study\s+hours?", "study_hours", 0.85),
            (r"User\s+is\s+(?:looking\s+for|searching\s+for)\s+(.+?)\s+(?:deck|resources?)", "study_resource", 0.75),

            # Finance patterns
            (r"User\s+(?:estimates?|needs?)\s+(?:about\s+)?\$?([0-9,]+K?)\s+for\s+(.+?)(?:\.|$)", "financial_goal", 0.80),
            (r"User(?:'s)?\s+budget\s+(?:is\s+)?(?:ideally\s+)?(?:no\s+more\s+than\s+)?\$?([0-9,]+K?)", "budget", 0.85),

            # Relationship patterns
            (r"User(?:'s)?\s+(mom|dad|mother|father|wife|husband|partner|friend|manager)(?:'s)?\s+(?:name\s+is\s+)?(.+?)(?:\.|$)", "relationship", 0.85),

            # Goal patterns
            (r"User\s+(?:plans?|wants?|intends?|aims?)\s+to\s+(.+?)(?:\.|$)", "plans_to", 0.80),
            (r"User\s+is\s+(?:considering|exploring)\s+(.+?)(?:\.|$)", "considering", 0.75),
            (r"User(?:'s)?\s+goal\s+is\s+to\s+(.+?)(?:\.|$)", "goal", 0.85),

            # Directive patterns (instructions to the AI)
            (r"(?:Remind|Check\s+in\s+with)\s+(?:Luke|User|user)\s+(.+?)(?:\.|$)", "reminder", 0.90),
            (r"User\s+wants?\s+to\s+be\s+(?:informed|notified|reminded)\s+(.+?)(?:\.|$)", "directive", 0.90),
            (r"User\s+(?:has\s+a\s+)?directive\s+(?:that\s+)?(.+?)(?:\.|$)", "directive", 0.95),
        ]

    def extract_from_memory(self, memory_text: str, timestamp: datetime, original_index: int) -> List[ExtractedFact]:
        """Extract facts from a single ChatGPT memory entry."""
        facts = []
        text = memory_text.strip()

        # Try pattern-based extraction
        for pattern, relation_hint, base_confidence in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    fact = self._build_fact_from_match(
                        match, relation_hint, base_confidence,
                        text, timestamp, original_index
                    )
                    if fact:
                        facts.append(fact)

        # If no patterns matched, create a general fact from the whole text
        if not facts:
            fact = self._create_general_fact(text, timestamp, original_index)
            if fact:
                facts.append(fact)

        return facts

    def _build_fact_from_match(
        self,
        match: Tuple | str,
        relation_hint: str,
        confidence: float,
        source_text: str,
        timestamp: datetime,
        original_index: int
    ) -> Optional[ExtractedFact]:
        """Build an ExtractedFact from a regex match."""

        # Handle tuple vs string matches
        if isinstance(match, tuple):
            if len(match) >= 2:
                # Handle special cases like (lift_type, current, target)
                if relation_hint == "fitness_goal":
                    lift_type = match[0].lower()
                    relation = f"{lift_type}_goal"
                    obj = f"{match[1]} to {match[2]}" + (f" {match[3]}" if len(match) > 3 and match[3] else " lb")
                elif relation_hint == "favorite":
                    relation = f"favorite_{_snake(match[0])}"
                    obj = match[1]
                elif relation_hint == "relationship":
                    relation = f"{_snake(match[0])}_name"
                    obj = match[1]
                elif relation_hint == "takes_medication_for":
                    relation = "takes_medication"
                    obj = f"{match[0]} for {match[1]}"
                elif relation_hint == "renamed_project":
                    relation = "project_name"
                    obj = match[1]  # The new name
                elif relation_hint == "lives_in" and len(match) == 2:
                    # (name, location) - e.g., "Luke is located in St. Charles"
                    relation = "lives_in"
                    obj = match[1].strip()
                elif relation_hint == "birthday_upcoming" and len(match) >= 3:
                    # (name, age, date) - e.g., "Luke will be 32 on Feb 9"
                    relation = "birthday"
                    obj = f"turning {match[1]} on {match[2]}"
                elif relation_hint == "full_name" and len(match) == 2:
                    relation = "full_name"
                    obj = match[1]
                elif relation_hint == "pet_relationship" and len(match) >= 3:
                    # (name, pet_type, pet_name) - e.g., "Luke has a cat named Flapjack"
                    pet_type = match[1].lower()
                    relation = f"{pet_type}_name"
                    obj = match[2]
                elif relation_hint == "pet_name" and len(match) == 2:
                    # (owner_name, pet_name) from "Luke's cat is named Flapjack"
                    relation = "pet_name"
                    obj = match[1]
                else:
                    relation = relation_hint
                    obj = " ".join(str(m) for m in match if m).strip()
            else:
                relation = relation_hint
                obj = str(match[0]).strip()
        else:
            relation = relation_hint
            obj = str(match).strip()

        # Clean up object
        obj = obj.strip(" .,;:")
        if not obj or len(obj) < 2:
            return None

        # Truncate very long objects
        if len(obj) > 500:
            obj = obj[:500] + "..."

        # Determine category
        category = categorize_relation(relation)

        # Build tags
        tags = ["extracted_fact", "chatgpt_import", category.value]

        return ExtractedFact(
            id=str(uuid.uuid4()),
            subject="user",
            relation=_snake(relation),
            object=obj,
            confidence=confidence,
            category=category.value,
            method="chatgpt_import",
            source_text=source_text[:500],
            timestamp=timestamp,
            original_index=original_index,
            tags=tags,
        )

    def _create_general_fact(
        self,
        text: str,
        timestamp: datetime,
        original_index: int
    ) -> Optional[ExtractedFact]:
        """Create a general fact when no specific patterns match."""

        # Skip very short or uninformative text
        if len(text) < 10:
            return None

        # Determine relation and category based on keywords
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["prefers", "preference", "format"]):
            relation = "preference"
            category = ProfileCategory.PREFERENCES
        elif any(kw in text_lower for kw in ["remind", "check in", "notify"]):
            relation = "reminder"
            category = ProfileCategory.DIRECTIVES
        elif any(kw in text_lower for kw in ["plan", "want", "going to", "intend"]):
            relation = "intention"
            category = ProfileCategory.GOALS
        elif any(kw in text_lower for kw in ["work", "job", "career", "role"]):
            relation = "career_note"
            category = ProfileCategory.CAREER
        elif any(kw in text_lower for kw in ["study", "exam", "course", "learn"]):
            relation = "study_note"
            category = ProfileCategory.STUDY
        elif any(kw in text_lower for kw in ["health", "symptom", "medication", "sleep"]):
            relation = "health_note"
            category = ProfileCategory.HEALTH
        elif any(kw in text_lower for kw in ["daemon", "atlass", "project", "implement"]):
            relation = "project_note"
            category = ProfileCategory.PROJECTS
        elif any(kw in text_lower for kw in ["game", "play", "hobby"]):
            relation = "hobby_note"
            category = ProfileCategory.HOBBIES
        else:
            relation = "general_note"
            category = ProfileCategory.UNKNOWN

        tags = ["extracted_fact", "chatgpt_import", "general", category.value]

        return ExtractedFact(
            id=str(uuid.uuid4()),
            subject="user",
            relation=relation,
            object=text[:500],
            confidence=0.60,  # Lower confidence for general facts
            category=category.value,
            method="chatgpt_import_general",
            source_text=text[:500],
            timestamp=timestamp,
            original_index=original_index,
            tags=tags,
        )


# ============================================================================
# Main import function
# ============================================================================

def import_chatgpt_memories(input_path: str, output_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Import ChatGPT memories from JSON and convert to Daemon fact format.

    Returns summary statistics.
    """

    # Load input
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    memories = data.get("long_term_memories", [])
    if verbose:
        print(f"Loaded {len(memories)} memories from {input_path}")

    extractor = MemoryFactExtractor()
    all_facts: List[ExtractedFact] = []
    category_counts: Dict[str, int] = {}

    for memory in memories:
        index = memory.get("index", 0)
        timestamp_str = memory.get("timestamp", "")
        text = memory.get("text", "")

        if not text:
            continue

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            # Try simple date format
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                timestamp = datetime.now()

        # Extract facts
        facts = extractor.extract_from_memory(text, timestamp, index)

        for fact in facts:
            all_facts.append(fact)
            cat = fact.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

            if verbose:
                print(f"  [{index}] {fact.relation}: {fact.object[:60]}... ({cat})")

    # Sort by timestamp
    all_facts.sort(key=lambda f: f.timestamp)

    # Build output structure
    output = {
        "metadata": {
            "source": "chatgpt_export",
            "import_timestamp": datetime.now().isoformat(),
            "total_memories": len(memories),
            "total_facts": len(all_facts),
            "category_counts": category_counts,
        },
        "facts": [f.to_dict() for f in all_facts],
    }

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nWrote {len(all_facts)} facts to {output_path}")
        print(f"\nCategory breakdown:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    return output["metadata"]


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Import ChatGPT memories into Daemon fact format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to ChatGPT memories JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="daemon_facts.json",
        help="Output path for Daemon-compatible facts JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    stats = import_chatgpt_memories(args.input, args.output, verbose=args.verbose)

    print(f"\n✓ Imported {stats['total_facts']} facts from {stats['total_memories']} memories")


if __name__ == "__main__":
    main()
