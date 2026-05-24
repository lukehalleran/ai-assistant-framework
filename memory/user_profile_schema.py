"""
# memory/user_profile_schema.py

Module Contract
- Purpose: Schema definitions for structured user profile extraction. Defines categories, relations, and validation rules for user facts organized by life domains (identity, career, fitness, etc.). Also defines system preferences and identity metadata for onboarding wizard.
- Inputs:
  - canonicalize_profile_relation(relation: str, value: str | None) → str
  - categorize_relation(relation: str) → ProfileCategory (5-layer cascade: direct→prefix→token→embedding→default)
  - categorize_relation_deep(relation: str, model_manager) → ProfileCategory (async, adds LLM Layer 5)
  - save_category_cache() — persist Layer 4/5 results to data/category_cache.json
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

import uuid
from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Schema version for migrations
SCHEMA_VERSION = "2.0"

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
    "belief": ProfileCategory.IDENTITY,
    "belief_system": ProfileCategory.IDENTITY,
    "religious_transition": ProfileCategory.IDENTITY,
    "self_perception": ProfileCategory.IDENTITY,
    "self_assessment": ProfileCategory.IDENTITY,
    "height": ProfileCategory.IDENTITY,

    # Education
    "degree": ProfileCategory.EDUCATION,
    "school": ProfileCategory.EDUCATION,
    "graduated_from": ProfileCategory.EDUCATION,
    "gpa": ProfileCategory.EDUCATION,
    "courses_taken": ProfileCategory.EDUCATION,
    "certification": ProfileCategory.EDUCATION,
    "university": ProfileCategory.EDUCATION,

    # Career
    "works_at": ProfileCategory.CAREER,
    "occupation": ProfileCategory.CAREER,
    "job_title": ProfileCategory.CAREER,
    "worked_at": ProfileCategory.CAREER,
    "skill": ProfileCategory.CAREER,
    "experience_years": ProfileCategory.CAREER,
    "resume_status": ProfileCategory.CAREER,
    "resume_update_deadline": ProfileCategory.CAREER,
    "linkedin_status": ProfileCategory.CAREER,
    "job_status": ProfileCategory.CAREER,
    "internship_application_status": ProfileCategory.CAREER,
    "internship_priority": ProfileCategory.CAREER,
    "hourly_wage": ProfileCategory.CAREER,
    "work_schedule": ProfileCategory.CAREER,
    "shift_pattern": ProfileCategory.CAREER,
    "day_off": ProfileCategory.CAREER,

    # Education / Schedule
    "class_schedule": ProfileCategory.EDUCATION,

    # Projects
    "project_name": ProfileCategory.PROJECTS,
    "tech_stack": ProfileCategory.PROJECTS,
    "project_status": ProfileCategory.PROJECTS,
    "building": ProfileCategory.PROJECTS,
    "lines_of_code": ProfileCategory.PROJECTS,
    "python_lines_of_code": ProfileCategory.PROJECTS,
    "total_lines_added": ProfileCategory.PROJECTS,
    "article_count": ProfileCategory.PROJECTS,

    # Health
    "condition": ProfileCategory.HEALTH,
    "medication": ProfileCategory.HEALTH,
    "supplement": ProfileCategory.HEALTH,
    "symptom": ProfileCategory.HEALTH,
    "sleep_pattern": ProfileCategory.HEALTH,
    "feeling": ProfileCategory.HEALTH,
    "feelings": ProfileCategory.HEALTH,
    "meal": ProfileCategory.HEALTH,
    "meal_choice": ProfileCategory.HEALTH,
    "meal_situation": ProfileCategory.HEALTH,
    "caloric_intake": ProfileCategory.HEALTH,
    "caffeine_intake": ProfileCategory.HEALTH,
    "drank_alcohol": ProfileCategory.HEALTH,
    "alcohol_consumption": ProfileCategory.HEALTH,
    "energy_level": ProfileCategory.HEALTH,
    "beverage": ProfileCategory.HEALTH,
    "headache": ProfileCategory.HEALTH,
    "blood_pressure": ProfileCategory.HEALTH,
    "kratom_dose": ProfileCategory.HEALTH,
    "kratom_effects": ProfileCategory.HEALTH,

    # Fitness
    "squat_max": ProfileCategory.FITNESS,
    "bench_max": ProfileCategory.FITNESS,
    "deadlift_max": ProfileCategory.FITNESS,
    "ohp_max": ProfileCategory.FITNESS,
    "body_weight": ProfileCategory.FITNESS,
    "fitness_goal": ProfileCategory.FITNESS,
    "foam_roller_use": ProfileCategory.FITNESS,
    "barbell_curl_estimate": ProfileCategory.FITNESS,

    # Preferences
    "prefers": ProfileCategory.PREFERENCES,
    "likes": ProfileCategory.PREFERENCES,
    "dislikes": ProfileCategory.PREFERENCES,
    "favorite_color": ProfileCategory.PREFERENCES,
    "favorite_beer": ProfileCategory.PREFERENCES,
    "favorite_video_game": ProfileCategory.PREFERENCES,
    "formatting_preference": ProfileCategory.PREFERENCES,
    "opinion": ProfileCategory.PREFERENCES,
    "dislike": ProfileCategory.PREFERENCES,
    "drink_preference": ProfileCategory.PREFERENCES,

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
    "spending_on_deepseek": ProfileCategory.FINANCE,
    "fafsa_filed": ProfileCategory.FINANCE,
    "need_to_refile": ProfileCategory.FINANCE,
    "credit_card_debt": ProfileCategory.FINANCE,

    # Relationships
    "family_member": ProfileCategory.RELATIONSHIPS,
    "friend": ProfileCategory.RELATIONSHIPS,
    "manager": ProfileCategory.RELATIONSHIPS,
    "reports_to": ProfileCategory.RELATIONSHIPS,
    "dinner_with": ProfileCategory.RELATIONSHIPS,
    "matches_on_bumble": ProfileCategory.RELATIONSHIPS,
    "matches_on_dating_app": ProfileCategory.RELATIONSHIPS,
    "dating_app_usage": ProfileCategory.RELATIONSHIPS,
    "language_exchange": ProfileCategory.RELATIONSHIPS,

    # Goals
    "goal": ProfileCategory.GOALS,
    "plan": ProfileCategory.GOALS,
    "wants_to": ProfileCategory.GOALS,
    "intends_to": ProfileCategory.GOALS,
    "priority": ProfileCategory.GOALS,
    "main_priority": ProfileCategory.GOALS,
    "aspiration": ProfileCategory.GOALS,
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
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_current: bool = True
    supersedes: str = None  # fact_id of fact this replaces

    def to_dict(self) -> dict:
        return {
            "relation": self.relation,
            "value": self.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "source_excerpt": self.source_excerpt,
            "timestamp": self.timestamp.isoformat(),
            "fact_id": self.fact_id,
            "is_current": self.is_current,
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
            fact_id=data.get("fact_id") or str(uuid.uuid4()),
            is_current=data.get("is_current", True),
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

# ========================================================================
# Relation Canonicalization (namespace cleanup)
# ========================================================================

# Safe aliases: these relation names are semantically identical and can auto-merge.
# Key = variant, Value = canonical form.
SAFE_RELATION_ALIASES: Dict[str, str] = {
    # Pets
    "pet": "pet_name", "has_pet": "pet_name", "owns_pet": "pet_name",
    # Grades
    "homework_score": "homework_grade", "last_homework_score": "homework_grade",
    "grade": "homework_grade",
    # Goals
    "future_plan": "future_goal", "goals_status": "goal_status",
    # Fitness
    "workout_routine": "workout_focus",
    "current_bench_weight": "current_bench",
    # Mood (singular/plural)
    "current_feelings": "current_feeling",
    # Medication
    "medications_taken": "medication_taken",
    # Family
    "family_vacation_experience": "family_vacations",
    "duration_at_dads": "duration_of_stay",
    # Location
    "errand_location": "proximity_to_locations",
    # Misc exact dupes found in audit
    "last_vacation": "last_vacation_date",
    "video_plan": "video_project",
    "upcoming_exam_date": "upcoming_exam_due_date",
    "dosage_taken": "medication_dose",
    "spending_on_project": "project_cost",
    "available_time": "time_available",
    "tested_for_sleep_apnea": "sleep_apnea_test_result",
    "week_quality": "recent_experience",
    "knowledge": "memory_effect",
    "performance_observation": "perceived_speed",
    "needs_to": "needs_to_do",
    "last_taken": "last_supplement",
    "took_supplement": "supplement_used",
    "planned_activity": "task",
    "class_registration_timeline": "class_registration_timing",
    "intends_to_sleep": "sleep_intention",
    "peer_reviews": "needs_to_do",
    "final_due": "exam_date",
    "current_goal": "goal",
}

# Semantic neighbors: related but distinct — shown to LLM as hints, NOT auto-merged.
SEMANTIC_RELATION_NEIGHBORS: Dict[str, List[str]] = {
    "homework_grade": ["preferred_homework_score", "homework_status", "homework_due", "homework_difficulty"],
    "bench_max": ["current_bench", "previous_bench_max", "goal_bench", "bench_reps"],
    "job_status": ["occupation", "previous_employer", "job_perception", "work_type"],
    "pet_name": ["pet_type", "pet_behavior", "pet_relationship", "pet_activity"],
    "weight_loss_goal": ["weight_start", "weight_current", "body_weight"],
    "medication": ["medication_dose", "medication_time", "medication_status"],
    "sleep_pattern": ["sleep_duration", "sleep_quality", "sleep_goal"],
}

# Ephemeral relations: expire aggressively (TTL-based), truly transient state.
EPHEMERAL_RELATIONS: Set[str] = {
    "current_day", "current_time", "current_drink", "current_activity",
    "current_feeling", "current_mood", "current_temperature",
    "current_state", "current_project_status", "current_project_feeling",
    "current_balance", "current_location", "current_version",
    "current_project",  # which project is active changes often
    "condition", "state_of_mind",
    "appointment_time", "appointment", "appointment_date",
    "plans_today", "plans_tonight", "schedule_today",
}

# Snapshot relations: keep until superseded, NOT expired by TTL.
# These are measurements/states that are valid until explicitly changed.
SNAPSHOT_RELATIONS: Set[str] = {
    "current_weight", "current_bench",
    "current_medication", "current_health_status", "current_health_condition",
    "current_goal", "current_grade", "current_week", "current_year",
    # Schedule: recurring schedules are valid until explicitly superseded
    "work_schedule", "class_schedule", "shift_pattern",
}


def canonicalize_profile_relation(rel: str, value: str | None = None) -> str:
    """
    Map a relation name to its canonical form.

    Uses normalize_relation() for basic cleanup (spaces→underscores, known synonyms),
    then applies profile-specific SAFE_RELATION_ALIASES.

    Special case: 'job' is value-aware — 'job=quit' → 'job_status',
    'job=bartender' → 'occupation'.
    """
    from memory.entity_resolver import normalize_relation
    rel = normalize_relation(rel)

    # Value-aware disambiguation for 'job'
    if rel == "job":
        v = (value or "").lower()
        if v in {"quit", "left", "unemployed", "no", "none", "not working", "fired", "laid off"}:
            return "job_status"
        return "occupation"

    # Also handle quit_job → job_status
    if rel in ("quit_job", "employment_status", "current_job_status"):
        return "job_status"

    return SAFE_RELATION_ALIASES.get(rel, rel)


# Prefix map: first token of underscore-split relation → category
# More robust than substring matching — "sleep_apnea_test_result" matches on "sleep" prefix
_PREFIX_CATEGORY_MAP: Dict[str, ProfileCategory] = {
    # Fitness
    "workout": ProfileCategory.FITNESS, "bench": ProfileCategory.FITNESS,
    "squat": ProfileCategory.FITNESS, "deadlift": ProfileCategory.FITNESS,
    "ohp": ProfileCategory.FITNESS, "gym": ProfileCategory.FITNESS,
    # Health
    "sleep": ProfileCategory.HEALTH, "therapy": ProfileCategory.HEALTH,
    "medication": ProfileCategory.HEALTH, "supplement": ProfileCategory.HEALTH,
    "dosage": ProfileCategory.HEALTH, "vitamin": ProfileCategory.HEALTH,
    "anxiety": ProfileCategory.HEALTH, "health": ProfileCategory.HEALTH,
    # Study
    "homework": ProfileCategory.STUDY, "study": ProfileCategory.STUDY,
    "exam": ProfileCategory.STUDY, "midterm": ProfileCategory.STUDY,
    "course": ProfileCategory.STUDY, "lecture": ProfileCategory.STUDY,
    "notes": ProfileCategory.STUDY, "final": ProfileCategory.STUDY,
    # Career
    "work": ProfileCategory.CAREER, "job": ProfileCategory.CAREER,
    "shift": ProfileCategory.CAREER, "coworker": ProfileCategory.CAREER,
    "boss": ProfileCategory.CAREER, "internship": ProfileCategory.CAREER,
    "resume": ProfileCategory.CAREER, "linkedin": ProfileCategory.CAREER,
    # Projects
    "project": ProfileCategory.PROJECTS, "codebase": ProfileCategory.PROJECTS,
    "feature": ProfileCategory.PROJECTS, "code": ProfileCategory.PROJECTS,
    # Goals
    "goal": ProfileCategory.GOALS, "plan": ProfileCategory.GOALS,
    "upcoming": ProfileCategory.GOALS, "future": ProfileCategory.GOALS,
    # Relationships
    "family": ProfileCategory.RELATIONSHIPS, "relationship": ProfileCategory.RELATIONSHIPS,
    "brother": ProfileCategory.RELATIONSHIPS, "sister": ProfileCategory.RELATIONSHIPS,
    "dad": ProfileCategory.RELATIONSHIPS, "mom": ProfileCategory.RELATIONSHIPS,
    "friend": ProfileCategory.RELATIONSHIPS, "partner": ProfileCategory.RELATIONSHIPS,
    "date": ProfileCategory.RELATIONSHIPS, "dating": ProfileCategory.RELATIONSHIPS,
    # Finance
    "financial": ProfileCategory.FINANCE, "budget": ProfileCategory.FINANCE,
    "insurance": ProfileCategory.FINANCE, "paycheck": ProfileCategory.FINANCE,
    "spending": ProfileCategory.FINANCE, "fafsa": ProfileCategory.FINANCE,
    # Health (nutrition / substance)
    "meal": ProfileCategory.HEALTH, "caloric": ProfileCategory.HEALTH,
    "food": ProfileCategory.HEALTH, "kratom": ProfileCategory.HEALTH,
    "energy": ProfileCategory.HEALTH, "caffeine": ProfileCategory.HEALTH,
    # Hobbies
    "pet": ProfileCategory.HOBBIES, "hobby": ProfileCategory.HOBBIES,
    "game": ProfileCategory.HOBBIES, "video": ProfileCategory.HOBBIES,
    # Preferences
    "favorite": ProfileCategory.PREFERENCES, "preference": ProfileCategory.PREFERENCES,
    # Identity
    "nickname": ProfileCategory.IDENTITY, "birthday": ProfileCategory.IDENTITY,
    "timezone": ProfileCategory.IDENTITY, "location": ProfileCategory.IDENTITY,
    "belief": ProfileCategory.IDENTITY, "religious": ProfileCategory.IDENTITY,
    "self": ProfileCategory.IDENTITY,
    # Education
    "school": ProfileCategory.EDUCATION, "degree": ProfileCategory.EDUCATION,
    "program": ProfileCategory.EDUCATION,
}

# Token-level category keywords for semantic overlap scoring.
# Each category has a set of discriminative tokens. A relation's tokens are
# scored against each category — highest overlap wins.
_CATEGORY_TOKENS: Dict[ProfileCategory, Set[str]] = {
    ProfileCategory.FITNESS: {
        "max", "lift", "weight", "rep", "bench", "squat", "deadlift", "ohp",
        "workout", "exercise", "gym", "reps", "set", "sets", "press", "curl",
        "muscle", "cardio", "run", "mile", "body", "calorie", "deficit",
    },
    ProfileCategory.HEALTH: {
        "sleep", "therapy", "medication", "med", "health", "symptom", "condition",
        "supplement", "dosage", "vitamin", "anxiety", "depression", "fatigue",
        "sick", "illness", "pain", "doctor", "diagnosis", "prescription",
        "nap", "insomnia", "apnea", "disorder", "mental", "vyvanse", "vryalar",
        "meal", "food", "drink", "energy", "caffeine", "alcohol", "headache",
        "blood", "pressure", "kratom", "feeling",
    },
    ProfileCategory.STUDY: {
        "homework", "study", "exam", "midterm", "final", "course", "lecture",
        "notes", "learn", "class", "quiz", "grade", "gpa", "assignment",
        "submission", "peer", "review", "test", "retake", "score",
    },
    ProfileCategory.CAREER: {
        "work", "job", "employ", "shift", "occupation", "coworker", "colleague",
        "boss", "internship", "resume", "linkedin", "hire", "salary", "quit",
        "career", "office", "meeting", "schedule",
    },
    ProfileCategory.PROJECTS: {
        "project", "tech", "stack", "codebase", "feature", "code", "build",
        "deploy", "refactor", "bug", "test", "api", "daemon", "repo",
        "lines", "nodes", "edges", "tool",
    },
    ProfileCategory.GOALS: {
        "goal", "plan", "want", "intend", "upcoming", "future", "aspire",
        "hope", "aim", "target", "next", "intention",
    },
    ProfileCategory.RELATIONSHIPS: {
        "family", "relationship", "brother", "sister", "parent", "dad", "mom",
        "friend", "partner", "dating", "boyfriend", "girlfriend", "spouse",
        "fiance", "roommate", "colleague",
    },
    ProfileCategory.FINANCE: {
        "financial", "money", "budget", "insurance", "paycheck", "debt",
        "expense", "cost", "credit", "saving", "income", "rent",
        "fafsa", "loan", "tuition", "spending", "wage",
    },
    ProfileCategory.HOBBIES: {
        "pet", "hobby", "game", "interest", "cat", "dog", "kitten",
        "music", "movie", "book", "sport", "play", "craft",
        "concert", "album", "band", "travel", "trip", "vacation",
    },
    ProfileCategory.PREFERENCES: {
        "favorite", "likes", "prefers", "dislikes", "preference", "opinion",
        "dislike", "prefer",
    },
    ProfileCategory.IDENTITY: {
        "name", "called", "location", "lives", "nickname", "birthday",
        "age", "timezone", "born", "gender", "pronouns",
        "belief", "religious", "personality", "self", "height",
    },
    ProfileCategory.EDUCATION: {
        "school", "degree", "gpa", "program", "university", "college",
        "certification", "graduate", "enrolled",
    },
}


# ========================================================================
# Persistent category cache (avoids re-computing Layer 4/5 for known relations)
# ========================================================================

import json as _json
import os as _os

_CATEGORY_CACHE_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "data", "category_cache.json",
)

class _CategoryCache:
    """Persistent cache for relation → category mappings resolved by Layers 4-5."""

    def __init__(self, path: str = _CATEGORY_CACHE_PATH):
        self._path = path
        self._cache: Dict[str, str] = {}
        self._dirty = False
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if not self._path or not _os.path.exists(self._path):
            return
        try:
            with open(self._path, "r") as f:
                data = _json.load(f)
            self._cache = data.get("relation_categories", {})
        except (ValueError, OSError):
            pass

    def get(self, relation: str) -> Optional[ProfileCategory]:
        self._ensure_loaded()
        cat_str = self._cache.get(relation.lower())
        if cat_str:
            try:
                return ProfileCategory(cat_str)
            except ValueError:
                pass
        return None

    def set(self, relation: str, category: ProfileCategory):
        self._ensure_loaded()
        key = relation.lower()
        if self._cache.get(key) != category.value:
            self._cache[key] = category.value
            self._dirty = True

    def save(self):
        if not self._dirty or not self._path:
            return
        temp_path = f"{self._path}.tmp"
        try:
            with open(temp_path, "w") as f:
                _json.dump({
                    "version": "1.0",
                    "relation_categories": self._cache,
                }, f, indent=2)
            _os.replace(temp_path, self._path)
            self._dirty = False
        except Exception:
            if _os.path.exists(temp_path):
                _os.remove(temp_path)


# Module-level singleton
_category_cache = _CategoryCache()


def save_category_cache():
    """Called from shutdown_processor to persist cache."""
    _category_cache.save()


# ========================================================================
# Layer 4: Embedding similarity (anchor phrases per category)
# ========================================================================

# Exemplar relation names per category — embedded individually, max-similarity wins.
# Short phrases embed more distinctly than long keyword lists.
_CATEGORY_EXEMPLARS: Dict[ProfileCategory, List[str]] = {
    ProfileCategory.CAREER: [
        "job status", "occupation", "employer", "work schedule", "coworker name",
        "boss name", "resume status", "quit job", "salary", "internship",
    ],
    ProfileCategory.HEALTH: [
        "medication dose", "sleep pattern", "therapy appointment", "health condition",
        "supplement taken", "symptoms", "anxiety level", "adhd meds", "prescription",
    ],
    ProfileCategory.STUDY: [
        "homework grade", "exam date", "study hours", "course difficulty",
        "midterm score", "lecture notes", "assignment due", "quiz result",
    ],
    ProfileCategory.FITNESS: [
        "workout routine", "bench press max", "gym session", "exercise plan",
        "body weight", "squat max", "deadlift max", "cardio duration",
    ],
    ProfileCategory.PROJECTS: [
        "project name", "tech stack", "codebase size", "feature status",
        "code quality", "build deploy", "software project", "repository",
    ],
    ProfileCategory.GOALS: [
        "future goal", "plan for next week", "upcoming event", "intention",
        "aspiration", "target date", "objective", "want to achieve",
    ],
    ProfileCategory.RELATIONSHIPS: [
        "family member", "friend name", "partner", "brother name",
        "dating status", "relationship with", "roommate", "spouse",
    ],
    ProfileCategory.FINANCE: [
        "financial situation", "budget amount", "insurance status", "paycheck",
        "debt amount", "expense", "savings", "income",
    ],
    ProfileCategory.HOBBIES: [
        "pet name", "hobby", "video game", "interest", "music taste",
        "book reading", "sport playing", "craft",
    ],
    ProfileCategory.PREFERENCES: [
        "favorite food", "likes", "dislikes", "preference", "opinion",
    ],
    ProfileCategory.IDENTITY: [
        "full name", "age", "birthday", "location", "timezone", "nickname",
    ],
    ProfileCategory.EDUCATION: [
        "school name", "degree program", "university", "gpa", "enrolled in",
        "graduation date", "certification",
    ],
}

# Cached exemplar embeddings: {category: ndarray of shape (N, 384)}
_exemplar_embeddings: Optional[Dict[ProfileCategory, "numpy.ndarray"]] = None


def _get_exemplar_embeddings():
    """Lazily compute and cache exemplar embeddings per category."""
    global _exemplar_embeddings
    if _exemplar_embeddings is not None:
        return _exemplar_embeddings
    try:
        from models.model_manager import ModelManager
        embedder = ModelManager._get_cached_embedder()
        if embedder is None:
            return None
        # Embed all exemplars in one batch for efficiency
        all_phrases = []
        cat_indices = []
        for cat, phrases in _CATEGORY_EXEMPLARS.items():
            for p in phrases:
                all_phrases.append(p)
                cat_indices.append(cat)
        embs = embedder.encode(all_phrases, convert_to_numpy=True, normalize_embeddings=True)
        # Group by category
        import numpy as np
        result: Dict[ProfileCategory, list] = {cat: [] for cat in _CATEGORY_EXEMPLARS}
        for emb, cat in zip(embs, cat_indices):
            result[cat].append(emb)
        _exemplar_embeddings = {cat: np.array(vecs) for cat, vecs in result.items()}
        return _exemplar_embeddings
    except Exception:
        return None


def _categorize_by_embedding(relation: str, threshold: float = 0.30) -> Optional[ProfileCategory]:
    """Layer 4: Embed the relation name and compare to category exemplars (max-similarity)."""
    exemplars = _get_exemplar_embeddings()
    if exemplars is None:
        return None
    try:
        from models.model_manager import ModelManager
        embedder = ModelManager._get_cached_embedder()
        if embedder is None:
            return None
        # Convert snake_case to natural language
        natural = relation.replace("_", " ").strip()
        import numpy as np
        rel_emb = embedder.encode([natural], convert_to_numpy=True, normalize_embeddings=True)[0]
        best_cat = None
        best_sim = 0.0
        for cat, cat_embs in exemplars.items():
            # Max similarity to any exemplar in this category
            sims = cat_embs @ rel_emb  # (N,384) @ (384,) → (N,)
            max_sim = float(np.max(sims))
            if max_sim > best_sim:
                best_sim = max_sim
                best_cat = cat
        if best_cat and best_sim >= threshold:
            return best_cat
    except Exception:
        pass
    return None


# ========================================================================
# Main categorization function (5-layer cascade)
# ========================================================================

def categorize_relation(relation: str) -> ProfileCategory:
    """
    Map a relation to its category using 5-layer cascade resolution:
    1. Direct lookup in RELATION_CATEGORY_MAP (exact match)
    2. Prefix lookup (first token of underscore-split relation)
    3. Token overlap scoring (relation tokens vs category keyword sets, ≥2 overlap required)
    4. Embedding similarity (all-MiniLM-L6-v2 cosine vs category anchor phrases, >0.40)
    5. Default to PREFERENCES if nothing matches

    Results from Layer 4 are cached persistently in data/category_cache.json.
    For Layer 5 (LLM), use categorize_relation_deep() instead.
    """
    relation_lower = relation.lower().strip()

    # Layer 1: Direct lookup (fastest, most reliable — always authoritative)
    if relation_lower in RELATION_CATEGORY_MAP:
        return RELATION_CATEGORY_MAP[relation_lower]

    # Layer 2: Prefix lookup (first token before underscore)
    tokens = relation_lower.split("_")
    prefix = tokens[0] if tokens else ""
    if prefix in _PREFIX_CATEGORY_MAP:
        return _PREFIX_CATEGORY_MAP[prefix]

    # Check persistent cache (covers Layer 3/4/5 results from prior runs)
    cached = _category_cache.get(relation_lower)
    if cached is not None:
        return cached

    # Layer 3: Token overlap scoring
    rel_tokens = {t for t in tokens if len(t) > 2}
    if rel_tokens:
        best_cat = None
        best_score = 0
        for cat, keywords in _CATEGORY_TOKENS.items():
            overlap = len(rel_tokens & keywords)
            if overlap > best_score:
                best_score = overlap
                best_cat = cat
        if best_cat and best_score >= 2:
            _category_cache.set(relation_lower, best_cat)
            return best_cat

    # Layer 4: Embedding similarity
    emb_result = _categorize_by_embedding(relation_lower)
    if emb_result is not None:
        _category_cache.set(relation_lower, emb_result)
        return emb_result

    # Layer 5: Default (LLM available via categorize_relation_deep)
    return ProfileCategory.PREFERENCES


async def categorize_relation_deep(
    relation: str,
    model_manager=None,
) -> ProfileCategory:
    """
    Async variant with LLM Layer 5 fallback for truly unknown relations.

    Use for batch/cleanup processing where latency is acceptable.
    Results are cached persistently so each relation only triggers LLM once.
    """
    # Try sync layers 1-4 first
    result = categorize_relation(relation)
    if result != ProfileCategory.PREFERENCES:
        return result

    # Check if it's in the cache as PREFERENCES (previously resolved, genuinely preferences)
    cached = _category_cache.get(relation.lower())
    if cached is not None:
        return cached

    # Layer 5: LLM micro-call
    if model_manager is None:
        return result

    try:
        natural = relation.replace("_", " ").strip()
        prompt = (
            f"Classify this user profile fact relation into exactly one category.\n\n"
            f"Relation: {natural}\n"
            f"Categories: identity, education, career, projects, health, fitness, "
            f"preferences, hobbies, study, finance, relationships, goals\n\n"
            f"Reply with ONLY the category name, one word."
        )
        response = await model_manager.generate_once(
            prompt=prompt,
            model_name="gpt-4o-mini",
            system_prompt="You classify user profile facts into categories. Reply with one word only.",
            max_tokens=10,
            temperature=0.0,
        )
        cat_str = (response or "").strip().lower().rstrip(".")
        try:
            llm_cat = ProfileCategory(cat_str)
            _category_cache.set(relation.lower(), llm_cat)
            return llm_cat
        except ValueError:
            pass
    except Exception:
        pass

    # LLM failed — cache as PREFERENCES so we don't retry
    _category_cache.set(relation.lower(), ProfileCategory.PREFERENCES)
    return ProfileCategory.PREFERENCES
