"""
# memory/user_profile_schema.py

Module Contract
- Purpose: Schema definitions for structured user profile extraction. Defines categories, relations, and validation rules for user facts organized by life domains (identity, career, fitness, etc.). Also defines system preferences and identity metadata for onboarding wizard.
- Inputs:
  - categorize_relation(relation: str) → ProfileCategory
  - ProfileFact.to_dict() → dict
  - ProfileFact.from_dict(data: dict) → ProfileFact
  - ProfilePreferences.to_dict() → dict
  - ProfilePreferences.from_dict(data: dict) → ProfilePreferences
  - ProfileIdentity.to_dict() → dict
  - ProfileIdentity.from_dict(data: dict) → ProfileIdentity
- Outputs:
  - ProfileCategory enum values
  - ProfileFact dataclass instances
  - ProfilePreferences dataclass instances (system preferences like conversation style)
  - ProfileIdentity dataclass instances (name, pronouns)
  - Dictionary representations of facts, preferences, and identity
- Key behaviors:
  - Maps relations to 12 profile categories (identity, education, career, projects, health, fitness, preferences, hobbies, study, finance, relationships, goals)
  - Provides heuristic fallbacks for unknown relations (pattern matching)
  - Serializes/deserializes facts with full metadata (confidence, timestamp, supersedes)
  - Manages schema versioning for future migrations
- Side effects:
  - None (pure data structures and mapping logic)
"""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass
from datetime import datetime

# Schema version for migrations
SCHEMA_VERSION = "1.0"

class ProfileCategory(str, Enum):
    """Categories for organizing user facts - matches ChatGPT's structure."""
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

# Mapping of relations to their natural categories
RELATION_CATEGORY_MAP: Dict[str, ProfileCategory] = {
    # Identity
    "name": ProfileCategory.IDENTITY,
    "first_name": ProfileCategory.IDENTITY,
    "middle_name": ProfileCategory.IDENTITY,
    "last_name": ProfileCategory.IDENTITY,
    "age": ProfileCategory.IDENTITY,
    "birthday": ProfileCategory.IDENTITY,
    "location": ProfileCategory.IDENTITY,
    "lives_in": ProfileCategory.IDENTITY,
    "timezone": ProfileCategory.IDENTITY,

    # Education
    "degree": ProfileCategory.EDUCATION,
    "school": ProfileCategory.EDUCATION,
    "graduated_from": ProfileCategory.EDUCATION,
    "gpa": ProfileCategory.EDUCATION,
    "courses_taken": ProfileCategory.EDUCATION,
    "certification": ProfileCategory.EDUCATION,

    # Career
    "works_at": ProfileCategory.CAREER,
    "occupation": ProfileCategory.CAREER,
    "job_title": ProfileCategory.CAREER,
    "worked_at": ProfileCategory.CAREER,
    "skill": ProfileCategory.CAREER,
    "experience_years": ProfileCategory.CAREER,

    # Projects
    "project_name": ProfileCategory.PROJECTS,
    "tech_stack": ProfileCategory.PROJECTS,
    "project_status": ProfileCategory.PROJECTS,
    "building": ProfileCategory.PROJECTS,

    # Health
    "condition": ProfileCategory.HEALTH,
    "medication": ProfileCategory.HEALTH,
    "supplement": ProfileCategory.HEALTH,
    "symptom": ProfileCategory.HEALTH,
    "sleep_pattern": ProfileCategory.HEALTH,

    # Fitness
    "squat_max": ProfileCategory.FITNESS,
    "bench_max": ProfileCategory.FITNESS,
    "deadlift_max": ProfileCategory.FITNESS,
    "ohp_max": ProfileCategory.FITNESS,
    "body_weight": ProfileCategory.FITNESS,
    "fitness_goal": ProfileCategory.FITNESS,

    # Preferences
    "prefers": ProfileCategory.PREFERENCES,
    "likes": ProfileCategory.PREFERENCES,
    "dislikes": ProfileCategory.PREFERENCES,
    "favorite_color": ProfileCategory.PREFERENCES,
    "favorite_beer": ProfileCategory.PREFERENCES,
    "favorite_video_game": ProfileCategory.PREFERENCES,
    "formatting_preference": ProfileCategory.PREFERENCES,

    # Hobbies
    "hobby": ProfileCategory.HOBBIES,
    "plays_game": ProfileCategory.HOBBIES,
    "pet_name": ProfileCategory.HOBBIES,
    "cat_name": ProfileCategory.HOBBIES,
    "dog_name": ProfileCategory.HOBBIES,
    "interest": ProfileCategory.HOBBIES,

    # Study
    "studying_for": ProfileCategory.STUDY,
    "exam_date": ProfileCategory.STUDY,
    "study_hours": ProfileCategory.STUDY,
    "study_strategy": ProfileCategory.STUDY,
    "weak_point": ProfileCategory.STUDY,

    # Finance
    "insurance": ProfileCategory.FINANCE,
    "budget": ProfileCategory.FINANCE,
    "financial_goal": ProfileCategory.FINANCE,

    # Relationships
    "family_member": ProfileCategory.RELATIONSHIPS,
    "friend": ProfileCategory.RELATIONSHIPS,
    "manager": ProfileCategory.RELATIONSHIPS,
    "reports_to": ProfileCategory.RELATIONSHIPS,

    # Goals
    "goal": ProfileCategory.GOALS,
    "plan": ProfileCategory.GOALS,
    "wants_to": ProfileCategory.GOALS,
    "intends_to": ProfileCategory.GOALS,
}

@dataclass
class ProfileFact:
    """A single fact about the user with full metadata."""
    relation: str
    value: str
    category: ProfileCategory
    confidence: float
    source_excerpt: str
    timestamp: datetime
    supersedes: str = None  # ID of fact this replaces

    def to_dict(self) -> dict:
        return {
            "relation": self.relation,
            "value": self.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "source_excerpt": self.source_excerpt,
            "timestamp": self.timestamp.isoformat(),
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileFact":
        return cls(
            relation=data["relation"],
            value=data["value"],
            category=ProfileCategory(data["category"]),
            confidence=data["confidence"],
            source_excerpt=data.get("source_excerpt", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            supersedes=data.get("supersedes"),
        )

@dataclass
class ProfilePreferences:
    """System preferences collected during onboarding wizard."""
    style: str = "balanced"
    check_distress: bool = True
    brief_responses: bool = False

    def to_dict(self) -> dict:
        return {
            "style": self.style,
            "check_distress": self.check_distress,
            "brief_responses": self.brief_responses,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfilePreferences":
        return cls(
            style=data.get("style", "balanced"),
            check_distress=data.get("check_distress", True),
            brief_responses=data.get("brief_responses", False),
        )

@dataclass
class ProfileIdentity:
    """User identity metadata (name, pronouns) collected during onboarding."""
    name: str = ""
    pronouns: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pronouns": self.pronouns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileIdentity":
        return cls(
            name=data.get("name", ""),
            pronouns=data.get("pronouns", ""),
        )

def categorize_relation(relation: str) -> ProfileCategory:
    """Map a relation to its category, with fallback heuristics."""
    relation_lower = relation.lower().strip()

    # Direct lookup
    if relation_lower in RELATION_CATEGORY_MAP:
        return RELATION_CATEGORY_MAP[relation_lower]

    # Heuristic fallbacks based on relation patterns
    if any(x in relation_lower for x in ["max", "lift", "weight", "rep", "rm", "bench", "squat", "deadlift"]):
        return ProfileCategory.FITNESS
    if any(x in relation_lower for x in ["favorite", "likes", "prefers"]):
        return ProfileCategory.PREFERENCES
    if any(x in relation_lower for x in ["goal", "plan", "want", "intend"]):
        return ProfileCategory.GOALS
    if any(x in relation_lower for x in ["name", "called"]):
        return ProfileCategory.IDENTITY
    if any(x in relation_lower for x in ["work", "job", "employ"]):
        return ProfileCategory.CAREER
    if any(x in relation_lower for x in ["study", "exam", "learn"]):
        return ProfileCategory.STUDY
    if any(x in relation_lower for x in ["med", "health", "symptom", "condition"]):
        return ProfileCategory.HEALTH

    # Default to preferences (safest catch-all for personal info)
    return ProfileCategory.PREFERENCES
