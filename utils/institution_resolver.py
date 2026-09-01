"""
User-institution resolution for web search query construction (2026-08-27).

Mirror of utils/location_resolver.py for the user's SCHOOL. Motivation: a
"confirm the drop date" turn produced the search terms "college drop date
August 2026" / "school withdrawal deadline August 2026" — the trigger LLM
first attached the user's CITY ("Springfield, Illinois college drop date",
the wrong-college class; correctly stripped by strip_unjustified_location)
and then had nothing to name the school with, even though the profile knows
`school = Georgia Tech` (confidence 1.0). Generic academic-logistics queries
return generic pages; the institution is the discriminating term.

Resolution order:
  1. `DAEMON_USER_INSTITUTION` env override (settable via config.local.yaml
     tooling like the location override — owner PII never enters source).
  2. User profile — education facts, `school`-family relations first, then
     `university`/`attends`; is_current only, institution-shaped values only,
     highest confidence wins. Falls back to the quick_profile `school` key.

Scope guards (the wrong-college doctrine, inverted):
  - The institution attaches ONLY to academic-logistics queries (drop/
    withdrawal deadlines, registrar, enrollment, tuition, academic calendar,
    transcripts...). Never to generic queries.
  - Never when the query names a DIFFERENT institution — "when is Harvard's
    drop deadline" must stay Harvard's.
  - Values must look like an institution name (short TitleCase phrase), not
    profile sentence junk ("in third best grad program in nation").
"""

import json
import os
import re
import threading
from typing import List, Optional

from utils.logging_utils import get_logger

logger = get_logger("institution_resolver")

INSTITUTION_ENABLED = os.getenv("INSTITUTION_SEARCH_ENABLED", "1") == "1"
INSTITUTION_OVERRIDE = os.getenv("DAEMON_USER_INSTITUTION", "")

_DEFAULT_PROFILE_PATH = os.path.join("data", "user_profile.json")

# Relations that name the user's school, in preference order. `university`
# last: the profile can carry a PAST school under it (a stored
# "University of Wisconsin-Madison" alongside the current "Georgia Tech").
_SCHOOL_RELATIONS = ("school", "attends_school", "attends", "university")

# Relations that name the user's employer, in preference order.
_EMPLOYER_RELATIONS = ("employer", "works_at", "works_for", "company")

# Relations that name organizations the user belongs to.
_ORG_RELATIONS = ("member_of", "belongs_to", "volunteers_at")

# An institution-shaped value: 1-6 tokens, opens uppercase, tokens are
# capitalized words / acronyms / connectors. Rejects sentence-shaped profile
# junk ("in third best grad program in nation", "get into school stuff").
_INSTITUTION_VALUE_RE = re.compile(
    r"^[A-Z][\w.&'\-]*(?:\s+(?:of|the|at|and|for|[A-Z][\w.&'\-]*|[A-Z&-]+)){0,5}$"
)

# Academic-logistics cues — the ONLY query class the institution attaches to.
# Deliberately narrow (logistics, not coursework): "how does SVM work" is
# schoolwork but not a school-logistics lookup.
_ACADEMIC_CUE_RE = re.compile(
    r"\b(?:"
    r"drop\s+(?:date|deadline|period)|add[/\s-]?drop|"
    r"withdraw(?:al|ing|s)?|re-?enroll(?:ment|ing)?|enroll(?:ment|ing)?|"
    r"registrar|registration|academic\s+calendar|semester|term\s+start|"
    r"tuition|bursar|financial\s+aid|refund\s+(?:date|deadline|policy)|"
    r"transcript|census\s+date|course\s+(?:catalog|schedule|registration)|"
    r"incomplete\s+grade|grade\s+portal|final\s+exam\s+schedule"
    r")\b",
    re.IGNORECASE,
)

# Query already names an institution: TitleCase word(s) adjacent to an
# institutional noun, or acronym+institutional-noun. If it isn't the user's
# own school, injecting theirs would misdirect the search.
_NAMED_INSTITUTION_RE = re.compile(
    r"(?:[A-Z][\w.&'\-]*\s+){0,3}(?:University|College|Institute|Polytechnic|Academy)\b"
    r"|\b(?:University|College)\s+of\s+[A-Z]"
)

# Generic school words inside a search term that the institution name should
# REPLACE ("college drop date" → "Georgia Tech drop date"). Longest first.
_GENERIC_SCHOOL_RE = re.compile(
    r"\b(?:my\s+(?:school|college|university|program)|grad\s+school|"
    r"college|university|school)\b",
    re.IGNORECASE,
)


class InstitutionResolver:
    """Profile-backed institution lookup with mtime caching. Never blocks."""

    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path or _DEFAULT_PROFILE_PATH
        self._cached: Optional[str] = None
        self._mtime: Optional[float] = None
        self._cached_anchors: Optional[List[str]] = None
        self._mtime_anchors: Optional[float] = None
        self._lock = threading.Lock()

    def get_institution(self) -> Optional[str]:
        if not INSTITUTION_ENABLED:
            return None
        override = (INSTITUTION_OVERRIDE or "").strip()
        if override:
            return override
        return self._from_profile()

    def get_anchors(self) -> List[str]:
        """Return user's personal anchors (school, employer, orgs) in order.

        Returns a list of unique anchor strings for use in private-sphere
        query filtering and institutional injection. School appears first
        (via existing resolution), then employer, then orgs. Junk-shaped
        values are excluded.
        """
        if not INSTITUTION_ENABLED:
            return []
        try:
            mtime = os.path.getmtime(self.profile_path)
        except OSError:
            return []
        with self._lock:
            if self._mtime_anchors == mtime:
                return self._cached_anchors or []
            anchors = self._extract_anchors()
            self._cached_anchors = anchors
            self._mtime_anchors = mtime
            return anchors

    # ------------------------------------------------------------------

    def _extract_anchors(self) -> List[str]:
        """Extract school, employer, and org anchors from profile."""
        anchors = []
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            school = self._extract_school(profile)
            if school:
                anchors.append(school)
            employer = self._extract_from_relations(profile, _EMPLOYER_RELATIONS)
            if employer:
                anchors.append(employer)
            org = self._extract_from_relations(profile, _ORG_RELATIONS)
            if org:
                anchors.append(org)
        except Exception as e:
            logger.debug(f"[PersonalAnchors] extraction failed: {e}")
        return anchors

    @staticmethod
    def _extract_school(profile: dict) -> Optional[str]:
        """Extract the user's school (existing logic)."""
        return InstitutionResolver._extract_from_relations(profile, _SCHOOL_RELATIONS)

    @staticmethod
    def _extract_from_relations(profile: dict, relations: tuple) -> Optional[str]:
        """Extract a value from the first-found relation in the given tuple."""
        candidates = []
        categories = profile.get("categories", {}) or {}
        for facts in categories.values():
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict) or not fact.get("is_current", False):
                    continue
                rel = str(fact.get("relation", "")).strip().lower()
                if rel not in relations:
                    continue
                val = str(fact.get("value", "")).strip()
                if not _INSTITUTION_VALUE_RE.match(val):
                    continue
                rank = relations.index(rel)
                conf = float(fact.get("confidence", 0.0) or 0.0)
                candidates.append((rank, -conf, val))
        if candidates:
            candidates.sort()
            return candidates[0][2]
        return None

    # ------------------------------------------------------------------

    def _from_profile(self) -> Optional[str]:
        try:
            mtime = os.path.getmtime(self.profile_path)
        except OSError:
            return None
        with self._lock:
            if self._mtime == mtime:
                return self._cached
            value = None
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                value = self._extract(profile)
            except Exception as e:
                logger.debug(f"[Institution] profile read failed: {e}")
            self._cached = value
            self._mtime = mtime
            return value

    @staticmethod
    def _extract(profile: dict) -> Optional[str]:
        school = InstitutionResolver._extract_from_relations(profile, _SCHOOL_RELATIONS)
        if school:
            return school
        quick = str((profile.get("quick_profile", {}) or {}).get("school", "")).strip()
        if quick and _INSTITUTION_VALUE_RE.match(quick):
            return quick
        return None


_resolver: Optional[InstitutionResolver] = None
_resolver_lock = threading.Lock()


def get_user_institution() -> Optional[str]:
    """Best currently-known institution name, or None."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = InstitutionResolver()
    return _resolver.get_institution()


def get_user_anchors() -> List[str]:
    """Return user's personal anchors (school, employer, orgs) in order."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = InstitutionResolver()
    return _resolver.get_anchors()


def query_is_academic_logistics(query: str) -> bool:
    return bool(_ACADEMIC_CUE_RE.search(query or ""))


def apply_institution(
    terms: List[str], query: str, institution: Optional[str]
) -> List[str]:
    """Deterministic backstop behind the LLM prompts: name the user's school
    in academic-logistics search terms that stayed generic.

    Applies only when the QUERY is academic-logistics-shaped and names no
    other institution; within it, only terms that are themselves academic or
    carry a generic school word are touched — a weather sub-query in a mixed
    request stays untouched. Under-fires by design.
    """
    if not terms or not institution or not (institution := institution.strip()):
        return terms
    q = query or ""
    if not _ACADEMIC_CUE_RE.search(q):
        return terms
    named = _NAMED_INSTITUTION_RE.search(q)
    if named and institution.lower() not in q.lower():
        return terms  # the query is about a school the user NAMED — keep it

    inst_lower = institution.lower()
    out, changed = [], []
    for term in terms:
        t = (term or "").strip()
        if not t or inst_lower in t.lower():
            out.append(term)
            continue
        if not (_ACADEMIC_CUE_RE.search(t) or _GENERIC_SCHOOL_RE.search(t)):
            out.append(term)
            continue
        new = _GENERIC_SCHOOL_RE.sub(institution, t, count=1)
        if new == t:
            new = f"{institution} {t}"
        out.append(new)
        changed.append(f"{t!r} -> {new!r}")
    if changed:
        logger.info(
            f"[Institution] Named the user's school in generic academic "
            f"search terms: {'; '.join(changed)}"
        )
    return out
