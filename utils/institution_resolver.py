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

2026-09-12: query_justifies_institution() / strip_unjustified_institution()
are the institution-side mirror of location_resolver's
query_justifies_location()/strip_unjustified_location() — an unrelated query
must not carry the school into search terms just because the LLM attached it
unprompted. scope_identity_terms() is the single call BOTH search-term
producers (the trigger classifier and WebSearchManager.decompose_query) use
for the full location-then-institution scoping policy (BC-58).
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
#
# Two categories (2026-09-12). SCHOOL-LOGISTICS cues name a school process
# outright. CROSS-DOMAIN cues are logistics words other domains share —
# withdrawal (a medication, troops), registration (voters), enrollment
# (Medicare), transcript (a court hearing). Bare, they had attached the
# user's school to "benzodiazepine withdrawal symptoms" and "voter
# registration deadline" searches: the owner's identity sent to a third-party
# search provider on an unrelated query (BC-59). A cross-domain cue now counts
# only beside a SCHOOL-DOMAIN anchor noun, or the user's own school, in the
# same text. Coverage grows by adding a word to one of these tables, never by
# a per-incident alternative (BC-76).
_SCHOOL_LOGISTICS_CUE_RE = re.compile(
    r"\b(?:"
    r"drop\s+(?:date|deadline|period)|add[/\s-]?drop|"
    r"registrar|academic\s+calendar|semester|"
    r"tuition|bursar|financial\s+aid|"
    r"census\s+date|course\s+(?:catalog|schedule|registration)|"
    r"incomplete\s+grade|grade\s+portal|final\s+exam\s+schedule"
    r")\b",
    re.IGNORECASE,
)
_CROSS_DOMAIN_LOGISTICS_CUE_RE = re.compile(
    r"\b(?:"
    r"withdraw(?:al|ing|s)?|re-?enroll(?:ment|ing)?|enroll(?:ment|ing)?|"
    r"registration|term\s+start|refund\s+(?:date|deadline|policy)|transcript"
    r")\b",
    re.IGNORECASE,
)
_SCHOOL_DOMAIN_ANCHOR_RE = re.compile(
    r"\b(?:class(?:es)?|courses?|school|college|university|campus|"
    r"professors?|syllabus|credit\s+hours?|academic|degree\s+program)\b",
    re.IGNORECASE,
)
# Either category. Used on a search TERM once the query itself has been
# established as school logistics ("withdrawal deadline fall 2026" inside a
# drop-date request is academic), never to decide that on its own.
_ACADEMIC_CUE_RE = re.compile(
    rf"{_SCHOOL_LOGISTICS_CUE_RE.pattern}|{_CROSS_DOMAIN_LOGISTICS_CUE_RE.pattern}",
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
    return _academic_logistics_shape(query)


def _names_institution(text: str, institution: Optional[str]) -> bool:
    """The resolved institution named in `text` (word-bounded, possessive ok)."""
    inst = (institution or "").strip()
    if not inst or not text:
        return False
    return bool(re.search(rf"\b{re.escape(inst)}(?:'s)?\b", text, re.IGNORECASE))


def _academic_logistics_shape(text: str, institution: Optional[str] = None) -> bool:
    """A school-logistics request: a school-logistics cue, or a cross-domain
    cue anchored in the same text by a school-domain noun, "my school/...",
    or the user's own school named (see the cue tables above)."""
    t = text or ""
    if not t:
        return False
    if _SCHOOL_LOGISTICS_CUE_RE.search(t):
        return True
    if not _CROSS_DOMAIN_LOGISTICS_CUE_RE.search(t):
        return False
    return bool(
        _SCHOOL_DOMAIN_ANCHOR_RE.search(t)
        or _MY_SCHOOL_RE.search(t)
        or _names_institution(t, institution)
    )


# A generic self-reference to the user's own school ("my school/college/
# university/program") — distinct from _GENERIC_SCHOOL_RE, which also
# matches bare "college"/"university" with no possessive (that broader form
# is for REPLACING a generic word with the resolved name inside a term, not
# for deciding whether the query justifies naming the school at all).
_MY_SCHOOL_RE = re.compile(
    r"\bmy\s+(?:school|college|university|program)\b", re.IGNORECASE
)


def query_justifies_institution(
    query: str, institution: Optional[str], context: Optional[str] = None
) -> bool:
    """Does the user's own query give a reason to name their institution in
    search terms? True when the query is school-logistics-shaped (drop dates,
    registrar, tuition, or a cross-domain cue beside a school anchor), when it
    names the user's OWN school, or when it refers to "my school/college/
    university/program" generically. Otherwise False — an unrelated query
    carrying no school-relevant cue at all must not have the school attached
    (2026-09-12: "I am referring to voting" produced the search term "Georgia
    Tech voting information" with nothing in the query pointing at school
    logistics or the school itself).

    `context` is the bounded prior-turn digest, passed ONLY for a referential
    follow-up (the caller decides — utils.web_search_trigger
    ._identity_scope_context). An elliptical "is it this Friday?" after a
    drop-deadline exchange carries no cue of its own, and stripping the school
    from its terms would send a generic search. Only a school-logistics SHAPE
    in that context counts: the school merely having been NAMED in an earlier
    turn never re-justifies it, or an unrelated follow-up in a long session
    would inherit the school forever."""
    q = query or ""
    if not q:
        return False
    if _academic_logistics_shape(q, institution):
        return True
    if _names_institution(q, institution):
        return True
    if _MY_SCHOOL_RE.search(q):
        return True
    return bool(context and _academic_logistics_shape(context, institution))


def apply_institution(
    terms: List[str], query: str, institution: Optional[str],
    context: Optional[str] = None,
) -> List[str]:
    """Deterministic backstop behind the LLM prompts: name the user's school
    in academic-logistics search terms that stayed generic.

    Applies only when the QUERY is school-logistics-shaped (or a referential
    follow-up whose bounded `context` is — see query_justifies_institution)
    and names no other institution; within it, only terms that are themselves
    academic or carry a generic school word are touched — a weather sub-query
    in a mixed request stays untouched. Under-fires by design.
    """
    if not terms or not institution or not (institution := institution.strip()):
        return terms
    q = query or ""
    if not (_academic_logistics_shape(q, institution)
            or (context and _academic_logistics_shape(context, institution))):
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


def strip_unjustified_institution(
    terms: List[str], query: str, institution: Optional[str],
    context: Optional[str] = None,
) -> List[str]:
    """Backstop mirroring strip_unjustified_location: when the ORIGINAL
    query gives no reason to name the user's institution
    (query_justifies_institution is False), remove the resolved institution
    name — plus a trailing possessive — from every generated search term.
    Terms that were nothing but the institution are dropped. Only ever
    touches the resolved institution STRING itself; a DIFFERENT institution
    the query or terms name is never removed (2026-09-12: "Georgia Tech
    voting information" needed the school stripped from a query with no
    school cue at all; "Harvard University student voting" must stay
    untouched when the user's own school is Georgia Tech). Returns the
    (possibly unchanged) list; logs when it fires."""
    if not terms or not institution or not (institution := institution.strip()):
        return terms
    if query_justifies_institution(query or "", institution, context=context):
        return terms

    pattern = re.compile(rf"\b{re.escape(institution)}(?:'s)?\b", re.IGNORECASE)
    cleaned = []
    changed = False
    for term in terms:
        t = term or ""
        new = pattern.sub("", t)
        if new != t:
            changed = True
            new = re.sub(r"\s+(?:in|at|near|around|for|of)\s*$", "", new, flags=re.IGNORECASE)
            new = re.sub(r"\s{2,}", " ", new).strip(" ,;-")
        if new:
            cleaned.append(new)
    if changed:
        logger.info(
            f"[Institution] Stripped unjustified school name '{institution}' from "
            f"search terms (query gave no academic-logistics/self-reference cue): "
            f"{terms} -> {cleaned}"
        )
    return cleaned if changed else terms


def scope_identity_terms(
    terms: List[str],
    query: str,
    location: Optional[str],
    institution: Optional[str],
    context: Optional[str] = None,
) -> List[str]:
    """Single scoping policy for BOTH search-term producers — the LLM
    trigger classifier (utils.web_search_trigger._classify_with_llm_unified)
    and WebSearchManager.decompose_query. BC-58: the two producers had grown
    independent two-step "strip location, then apply institution" blocks
    that had drifted (neither protected an institution-name span while
    stripping a bare state out of the SAME term), so a fix to the scoping
    policy had to land twice. From 2026-09-12 both call this instead.

    Order: strip an unjustified location — an institution-name span inside
    each term is protected from the state-name removal — then strip an
    unjustified institution, then run the deterministic institution backstop
    (which can re-introduce the institution into a term that stayed
    academic-logistics-generic once its location was removed, e.g. "drop
    deadline" -> "Georgia Tech drop deadline")."""
    from utils.location_resolver import strip_unjustified_location

    if terms and location:
        terms = strip_unjustified_location(terms, query, location, institution=institution)
    if terms and institution:
        terms = strip_unjustified_institution(terms, query, institution, context=context)
    if terms and institution:
        terms = apply_institution(terms, query, institution, context=context)
    return terms
