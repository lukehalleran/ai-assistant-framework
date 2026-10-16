"""Deterministic provenance checks for extracted facts.

The extractor may propose a triple, but a proposal is not evidence.  This
module joins a proposed triple back to a small, USER-authored span and rejects
triples whose only apparent support is Daemon/model output, a block quote, a
pasted transcript, or the user's report of what another system claimed.

Why (2026-09-02 provenance audit):
  - ``user | uses | WHOOP`` came from a user turn QUOTING another model's
    profile comparison ("it says you use WHOOP") — text inside a user turn is
    not the user's testimony.
  - ``user | has_dog | Mochi`` came from "playing with Mochi and Waffles" —
    the relation name asserted a species the turn never stated (older graph
    metadata calls them cats).
  - ``lived_in=Marrowby`` cited the first 200 characters of a long turn
    (song lyrics) instead of the sentence near the end that actually said it.

Doctrine: deliberately conservative.  Missing a marginal fact is safer than
turning quoted or generated text into a durable belief about the user.  A
user-scoped fact needs a user-owned span (explicit first person, or an
implicit-subject chat fragment such as "Started Lorvatin today"), object
grounding, and — where the relation name itself makes a claim (pet species,
residence, employment, preference, relationship) — a matching verb/noun cue.
Entity facts need the named subject and the object in one span.  There is
intentionally NO "last message" fallback.

Leaf module: imports only stdlib, ``utils.temporal_resolver`` (a shared,
dependency-free date helper), ``utils.test_envelope`` (2026-09-27, the
shared [test]...[/test] structural parser — also stdlib-only), and
``memory.user_profile_schema`` (2026-10-02, stdlib-only at import time — only
for ``RESIDENCE_RELATIONS``, the single residence-relation set) — never from
the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from typing import Any, Iterable, Iterator, Mapping

from memory.user_profile_schema import RESIDENCE_RELATIONS
from utils import test_envelope
from utils.status_claims import EMPLOYMENT_RELATIONS, ENROLLMENT_RELATIONS
from utils.temporal_resolver import resolve_date_expression


_NON_USER_ROLE_RE = re.compile(
    r"^\s*(?:assistant|daemon|system|developer|tool|chatgpt|claude|gemini|gpt|codex|fable|model)"
    r"\s*:\s*",
    re.IGNORECASE,
)
_USER_ROLE_RE = re.compile(r"^\s*user\s*:\s*", re.IGNORECASE)
_INLINE_ROLE_SPLIT_RE = re.compile(r"\b(?:Assistant|Daemon|System|Tool)\s*:", re.IGNORECASE)

# The user REPORTING another speaker/system's words ("the profile lists
# WHOOP", "ChatGPT said I…", "it says you use…").  The words are in the user's
# turn, but the user is not asserting them.
_EXTERNAL_ATTRIBUTION_RE = re.compile(
    r"\b(?:according\s+to\s+(?:the\s+)?(?:assistant|daemon|model|response|planner|profile|export)|"
    r"(?:the\s+)?(?:assistant|daemon|model|response|planner|chatgpt|claude|gemini|gpt|codex|fable|"
    r"knowledge\s+graph|profile|prompt|export|debug\s+dump|other\s+model|it)\s+"
    r"(?:said|says|wrote|writes|replied|claimed|claims|listed|lists|showed|shows|thinks|"
    r"has\s+me\s+(?:as|down|listed)))\b",
    re.IGNORECASE,
)

_FIRST_PERSON_SUBJECT_RE = re.compile(
    r"\b(?:i|i'm|i’m|i've|i’ve|i'd|i’d|i'll|i’ll|we|we're|we’re|we've|we’ve)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_ANY_RE = re.compile(
    r"\b(?:i|i'm|i’m|i've|i’ve|i'd|i’d|i'll|i’ll|me|my|mine|myself|we|we're|we’re|we've|we’ve|"
    r"us|our|ours)\b",
    re.IGNORECASE,
)
_OTHER_PERSON_RE = re.compile(
    r"\b(?:you|your|yours|yourself|he|she|they|him|her|his|hers|their|theirs|them|himself|"
    r"herself|themselves)\b",
    re.IGNORECASE,
)
# "Sam's new job", "Mom's cat" — a possessive proper noun marks a third party's
# fact even when first-person words appear elsewhere in the sentence.
_POSSESSIVE_PROPER_RE = re.compile(r"\b[A-Z][a-z]+(?:'s|’s)\b")
# "My brother moved to Boston" — the grammatical subject is the relative.
# Pets are deliberately NOT in this list: "my cat Biscuit" IS the ownership
# evidence the pet relations require.
_RELATIVE_NOUNS = (
    r"brother|sister|mom|mother|dad|father|parents?|friend|buddy|partner|"
    r"girlfriend|boyfriend|wife|husband|ex|boss|coworker|co-worker|roommate|therapist|doctor|"
    r"psychiatrist|advisor|professor|teacher|cousin|aunt|uncle|grandma|grandpa|grandmother|"
    r"grandfather|kid|son|daughter|neighbou?r|manager|landlord|coach|nurse|dentist"
)
_RELATIVE_SUBJECT_RE = re.compile(
    rf"^\s*(?:my|our)\s+(?:{_RELATIVE_NOUNS})\b",
    re.IGNORECASE,
)
# A place POSSESSED by another person: "my dad's", "her mom's place", "Alex's".
# Used for residence relations — the user is not the resident of such a place.
_OTHER_PERSON_PLACE_RE = re.compile(
    rf"\b(?:(?i:(?:my|our|his|her|their)\s+(?:{_RELATIVE_NOUNS}))|(?!(?:It|That|There|Here|What|Let|Who|Where|How|When)(?:'s|\u2019s))[A-Z][a-z]+)(?:'s|\u2019s)(?=\s|$|[.,;!?])",
)
# An explicit first-person resident assertion ("I live", "we're living",
# "I moved", "I'm based") — the user as the one who resides.
_FIRST_PERSON_RESIDENT_RE = re.compile(
    r"\b(?:i|we)(?:\s+(?:now|currently|just|still|recently|actually))?\s*"
    r"(?:live|lived|living|reside|resided|residing|moved|relocated|"
    r"(?:am|'m|\u2019m|are|'re|\u2019re)\s+(?:now\s+)?(?:living|based|located|from))\b",
    re.IGNORECASE,
)
# A TitleCase sentence opener followed by one of these reads as a third-party
# subject ("Biscuit is…", "Sam just got…") rather than an implied "I".
_THIRD_PERSON_CONTINUATION_RE = re.compile(
    r"^(?:is|was|has|had|does|did|'s|’s|will|can|got|goes|went|says|said|lives|works|likes|"
    r"loves|hates|wants|needs|called|texted|emailed|moved|started|plays|uses|owns|drives|"
    r"teaches|studies|thinks|feels|seems|looks|told|asked|keeps|just|also|still|already|"
    r"recently|finally|and)\b",
    re.IGNORECASE,
)
# Capitalised sentence starters that are not names.
_SENTENCE_STARTERS = frozenset({
    "today", "yesterday", "tomorrow", "tonight", "this", "that", "these", "those", "there",
    "here", "it", "everything", "nothing", "something", "things", "life", "work", "school",
    "also", "so", "and", "but", "now", "still", "just", "currently", "lately", "recently",
    "honestly", "basically", "okay", "ok", "well", "yeah", "yes", "no", "the", "a", "an",
    "my", "our", "i", "update", "note", "fyi", "anyway", "then", "after", "before", "since",
    "last", "next", "first", "second", "day", "week", "month", "year",
})
_PROFILE_FIELD_RE = re.compile(
    r"^\s*(?:name|age|location|hometown|occupation|job|school|timezone)\s*[:=]",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "at", "for", "from", "in", "is", "it", "my", "of",
    "on", "or", "the", "to", "user", "with", "years", "old",
    # Function words / discourse filler (2026-09-05): a single one of these
    # shared between an object and a span is not evidence — "this" (in
    # "this week") joined `works_on = this assistant` to an attachment paste,
    # and "drinking" alone joined a friend's accident story to a meds turn.
    "this", "that", "these", "those", "there", "here", "what", "which",
    "when", "where", "who", "how", "some", "any", "all", "also", "just",
    "like", "really", "very", "have", "has", "had", "been", "being", "was",
    "were", "are", "will", "would", "could", "should", "can", "about",
    "into", "than", "then", "them", "they", "you", "your", "but", "not",
    "yet", "out", "get", "got", "one", "two", "still", "even", "only",
    "other", "over", "own", "same", "too", "did", "does", "doing", "done",
    "thing", "things", "stuff", "bit", "lot", "more", "most", "much", "many",
    "yes", "yeah", "okay", "hmm", "idk", "lol", "kinda", "sorta", "maybe",
    "today", "now", "day", "days", "week", "weeks", "time", "way", "due",
    "past", "because", "since", "while", "after", "before", "around",
})

# Relations whose NAME asserts ownership of a pet.  The possessive species
# phrase ("my cat Biscuit") is the anchor AND the relation evidence.
_PET_OWNERSHIP_SPECIES = {
    "has_cat": r"cat|cats|kitten|kittens|kitty|feline",
    "cat_name": r"cat|cats|kitten|kittens|kitty|feline",
    "owns_cat": r"cat|cats|kitten|kittens|kitty|feline",
    "has_dog": r"dog|dogs|puppy|puppies|pup|canine",
    "dog_name": r"dog|dogs|puppy|puppies|pup|canine",
    "owns_dog": r"dog|dogs|puppy|puppies|pup|canine",
    "has_pet": r"pet|pets|cat|cats|kitten|kittens|kitty|dog|dogs|puppy|puppies|pup",
    "pet_name": r"pet|pets|cat|cats|kitten|kittens|kitty|dog|dogs|puppy|puppies|pup",
}

# Relation families whose name makes a claim the span must corroborate with a
# verb/noun cue.  Pronoun anchoring is handled separately (a chat fragment
# like "Moved to Marrowby in June" has an implied "I").
_RESIDENCE_CUES = (
    r"live|lives|lived|living|reside|resides|residing|moved|moving|move|relocated|relocating|"
    r"based|home|apartment|house|from|staying|stay"
)
_LIKES_CUES = (
    r"like|likes|liked|love|loves|loved|prefer|prefers|enjoy|enjoys|enjoyed|into|fan|favorite|"
    r"favourite|obsessed|adore"
)
# Enrollment is a claim in the relation NAME (2026-09-03): "I dropped ABC 1234
# and kept QRS 7310" token-matched a stale enrolled_in="ABC 1234 and QRS 7310"
# object with no enrollment cue at all.
_ENROLLMENT_CUES = (
    r"enrol|enrolled|enrolling|enrollment|enrolment|register|registered|registering|"
    r"registration|taking|take|takes|took|signed\s+up|sign\s+up|signing\s+up|class|classes|"
    r"course|courses|semester|this\s+term|next\s+term|this\s+fall|this\s+spring|this\s+summer|"
    r"studying|study"
)
_OCCUPATION_CUES = (
    r"work|works|working|worked|job|hired|employed|role|position|title|career|"
    r"i'm\s+a|i\s+am\s+a|as\s+an?\b"
)
# Care-team relations name a CLINICAL role in the relation itself (2026-09-05:
# `has_doctor = "Ellery is cautious about drinking …"` — a friend — and
# `doctor_communication = "received email from advisor about project
# timeline"` — an academic advisor — were both stored on the same afternoon).
# The span must name a clinician; "advisor"/"friend" are not cues.
_CARE_TEAM_CUES = (
    r"doctor|doctors|dr|psychiatrist|psychiatrists|psych|therapist|therapists|physician|"
    r"prescriber|provider|pcp|gp|nurse|np|clinic|clinician|counselor|counsellor|psychologist|"
    r"dentist|pharmacist|pharmacy|prescription|prescribed|refill|portal"
)
# Project-work relations (`works_on`) make a claim too: the LLM path minted
# `works_on = this assistant` whose provenance join hit the word "this" in an
# attachment paste (2026-09-05).
_PROJECT_WORK_CUES = (
    r"work|works|working|worked|build|builds|building|built|project|projects|coding|code|"
    r"coded|develop|develops|developing|developed|writing|wrote|maintain|maintains|"
    r"maintaining|refactor|refactoring|fix|fixes|fixed|fixing|ship|shipped|commit|commits|"
    r"committed|repo|repository|feature|features|implement|implemented|implementing"
)
# Fitness-schedule relations name the ACTIVITY in the relation itself
# (2026-09-08: the shutdown LLM extractor turned "add the abc office hours
# sessions to my google calander, fridays 8 to 9 pm central … weekly through
# the end of the semester" into gym_schedule="Fridays 8 to 9 PM Central …",
# joined on the fragment "weekly through the end of the semester" with no
# gym word anywhere). The span must name exercise.
_FITNESS_CUES = (
    r"gym|workout|workouts|work\s+out|working\s+out|worked\s+out|lift|lifts|lifting|lifted|"
    r"exercise|exercising|exercised|training|train|run|runs|running|ran|jog|jogging|swim|"
    r"swimming|swam|cardio|weights|bench|squat|squats|deadlift|planet\s+fitness|fitness|"
    r"yoga|stretch|stretching|walk\s+to\s+the\s+gym|leg\s+day|chest\s+day|back\s+day"
)
_RELATION_CUE_RES = {
    "gym_schedule": _FITNESS_CUES,
    "workout_schedule": _FITNESS_CUES,
    "exercise_schedule": _FITNESS_CUES,
    "training_schedule": _FITNESS_CUES,
    "gym_days": _FITNESS_CUES,
    "workout_days": _FITNESS_CUES,
    "gym_time": _FITNESS_CUES,
    "workout_time": _FITNESS_CUES,
    "gym_habit": _FITNESS_CUES,
    "workout_habit": _FITNESS_CUES,
    "exercise_habit": _FITNESS_CUES,
    "exercise_routine": _FITNESS_CUES,
    "workout_routine": _FITNESS_CUES,
    "gym_routine": _FITNESS_CUES,
    "works_out": _FITNESS_CUES,
    "workout_plan": _FITNESS_CUES,
    "workout_intention": _FITNESS_CUES,
    "enrolled_in": _ENROLLMENT_CUES,
    "enrolled": _ENROLLMENT_CUES,
    "registered_for": _ENROLLMENT_CUES,
    "takes_class": _ENROLLMENT_CUES,
    "taking_class": _ENROLLMENT_CUES,
    "taking_course": _ENROLLMENT_CUES,
    "current_course": _ENROLLMENT_CUES,
    "current_courses": _ENROLLMENT_CUES,
    "lives_in": _RESIDENCE_CUES,
    "lived_in": _RESIDENCE_CUES,
    "location": _RESIDENCE_CUES,
    "hometown": r"grew\s+up|hometown|born|raised|from|originally",
    "works_at": r"work|works|working|worked|job|hired|employed|employer|start(?:ed|ing)?\s+at|"
                r"internship|intern|shift|shifts|office",
    "works_as": _OCCUPATION_CUES,
    "occupation": _OCCUPATION_CUES,
    # 2026-09-03: `role`/`job_title` name a claim too — "Fixed a small bug"
    # had minted role="bug fixer" with no work/job/title cue in the span.
    "role": _OCCUPATION_CUES,
    "job_title": _OCCUPATION_CUES,
    "works_on": _PROJECT_WORK_CUES,
    "working_on": _PROJECT_WORK_CUES,
    "builds": _PROJECT_WORK_CUES,
    "building": _PROJECT_WORK_CUES,
    "has_doctor": _CARE_TEAM_CUES,
    "has_therapist": _CARE_TEAM_CUES,
    "has_psychiatrist": _CARE_TEAM_CUES,
    "has_psychologist": _CARE_TEAM_CUES,
    "has_prescriber": _CARE_TEAM_CUES,
    "has_physician": _CARE_TEAM_CUES,
    "has_pcp": _CARE_TEAM_CUES,
    "has_nurse": _CARE_TEAM_CUES,
    "has_dentist": _CARE_TEAM_CUES,
    "has_counselor": _CARE_TEAM_CUES,
    "has_provider": _CARE_TEAM_CUES,
    "doctor_communication": _CARE_TEAM_CUES,
    "doctor_relationship": _CARE_TEAM_CUES,
    "doctor_status": _CARE_TEAM_CUES,
    "doctor_availability": _CARE_TEAM_CUES,
    "therapist_communication": _CARE_TEAM_CUES,
    "psychiatrist_communication": _CARE_TEAM_CUES,
    "prescriber_communication": _CARE_TEAM_CUES,
    "provider_communication": _CARE_TEAM_CUES,
    "care_team": _CARE_TEAM_CUES,
    "studies_at": r"study|studies|studying|studied|attend|attending|attends|enrolled|class|classes|"
                  r"course|school|program|degree|major|semester|omsa|university|college",
    "attends": r"study|studies|studying|studied|attend|attending|attends|enrolled|class|classes|"
               r"course|school|program|degree|major|semester|university|college",
    "studies": r"study|studies|studying|studied|learning|class|classes|course|degree|major",
    "uses": r"use|uses|using|used|wear|wears|wearing|wore|have|has|own|owns|got|bought|"
            r"track|tracks|tracking|run|runs|running|on\s+(?:a|an|my)",
    "wears": r"wear|wears|wearing|wore|have|has|own|owns|got|bought|on\s+my",
    "plays": r"play|plays|playing|played|campaign|game|games|session|dm|dming|dm'ing",
    "plays_game": r"play|plays|playing|played|campaign|game|games|session|dm|dming",
    "likes": _LIKES_CUES,
    "prefers": _LIKES_CUES,
    "loves": _LIKES_CUES,
    "enjoys": _LIKES_CUES,
    "hobby": _LIKES_CUES + r"|play|plays|playing|played|do|doing|did|started|hobby|hobbies|weekends",
    "interest": _LIKES_CUES + r"|interest|interested|curious|reading|learning",
    "interested_in": _LIKES_CUES + r"|interest|interested|curious|reading|learning",
    "dislikes": r"dislike|dislikes|hate|hates|hated|can't\s+stand|cannot\s+stand|not\s+a\s+fan|"
                r"don't\s+like|do\s+not\s+like|annoy|annoys|annoying",
    "hates": r"dislike|dislikes|hate|hates|hated|can't\s+stand|cannot\s+stand|not\s+a\s+fan",
    "relationship": r"partner|girlfriend|boyfriend|wife|husband|spouse|fianc[ée]e?|dating|married|"
                    r"relationship|together|ex|broke\s+up|breakup|seeing",
    "relationship_status": r"partner|girlfriend|boyfriend|wife|husband|spouse|fianc[ée]e?|dating|"
                           r"married|single|relationship|together|ex|broke\s+up|breakup|divorced",
    "partner": r"partner|girlfriend|boyfriend|wife|husband|spouse|fianc[ée]e?|dating|married|"
               r"relationship|together|seeing",
    "spouse_of": r"wife|husband|spouse|married|wedding|fianc[ée]e?",
    "takes_medication": r"take|takes|taking|took|started|starting|start|restarted|prescribed|"
                        r"dose|mg|medication|meds|med|pill|pills|on\s+(?:it|the)",
    "medication_name": r"take|takes|taking|took|started|starting|start|restarted|prescribed|"
                       r"dose|mg|medication|meds|med|pill|pills",
    "medication": r"take|takes|taking|took|started|starting|start|restarted|prescribed|"
                  r"dose|mg|medication|meds|med|pill|pills",
}
_RELATION_CUE_CACHE: dict[str, re.Pattern[str]] = {}

# Clause structure is grammar, not vocabulary (2026-09-06): a coordinated
# sentence ("Had the 5mg at 11 and went out with a friend and his dad") reads as
# ONE sentence but is several clauses, and a third-party pronoun in one clause
# ("his dad") must not disqualify a DIFFERENT clause's own claim ("Had the
# 5mg at 11"). Split on commas/semicolons and coordinating conjunctions.
_CLAUSE_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+(?:and|but|so|then)\s+", re.IGNORECASE)

# Negation governors are a closed grammatical set — never relation/medication
# vocabulary. Bare "no" is deliberately excluded: "no patient portal" is an
# affirmative claim ABOUT an absence, not a negated claim about an object.
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|without|neither|nor)\b|n't\b",
    re.IGNORECASE,
)


def _split_clauses(sentence: str) -> list[str]:
    parts = _CLAUSE_SPLIT_RE.split(sentence or "")
    return [p.strip() for p in parts if p and p.strip()]


def _object_bearing_clause(
    clauses: list[str], object_val: str, object_tokens: set[str]
) -> str | None:
    """The clause with the strongest grounding signal for the object, or
    ``None`` when no single clause carries it (e.g. the object phrase itself
    straddles a coordinating conjunction) — callers fall back to the whole
    span in that case."""
    obj_low = (object_val or "").lower().strip()
    best: str | None = None
    best_score = 0
    for clause in clauses:
        low = clause.lower()
        exact = bool(obj_low and obj_low in low)
        overlap = len(object_tokens & _tokens(clause))
        score = (100 if exact else 0) + overlap
        if score > best_score:
            best_score, best = score, clause
    return best


def _object_position(clause: str, object_val: str, object_tokens: set[str]) -> int | None:
    low = clause.lower()
    obj_low = (object_val or "").lower().strip()
    if obj_low and obj_low in low:
        return low.find(obj_low)
    best: int | None = None
    for token in object_tokens:
        pos = low.find(token)
        if pos >= 0 and (best is None or pos < best):
            best = pos
    return best


def _clause_is_negated(clause: str, object_val: str, object_tokens: set[str]) -> bool:
    """True when a negation governor precedes the object's position within
    ``clause`` — "I did not take Zelphex today" negates ``take``d Zelphex``
    even though the object token itself (``zelphex``) is untouched."""
    pos = _object_position(clause, object_val, object_tokens)
    if pos is None:
        return False
    return any(m.start() < pos for m in _NEGATION_RE.finditer(clause))


# ---------------------------------------------------------------------------
# B2 (2026-09-06): claim temporal kind + event date.
#
# A stored fact's object text can claim a tense the source span doesn't
# support (an object phrase carrying "today" can still only be grounded by a
# span that actually describes YESTERDAY — see find_supporting_user_span's
# docstring).  This classifies what KIND of claim a span makes — a durable
# state, a recurring habit, a forward-looking plan, or a discrete PAST EVENT
# — and, for a past event, the calendar date it happened on when that is
# resolvable.  Read-side consumers (user_profile.get_category) use this to
# stop projecting a one-time past event into "current state" forever,
# regardless of what the object's own wording says.
#
# Grammar-level cues ONLY — closed word classes, never wording tied to one
# incident.  Deliberately under-fires: an unrecognized shape is "unknown",
# never misclassified into a stronger kind.
# ---------------------------------------------------------------------------

# CLS-1 (2026-10-16, class: BC-04/BC-51): "every"/"each" is a RECURRENCE cue only
# when its head is a TIME unit. "every chapter"/"each question" quantify things,
# they do not recur. ONE module constant: the core periods are also the unit
# list of ``_SIMPLE_PRESENT_FREQ_RE``; the full set heads "every"/"each".
_CORE_PERIOD_UNITS = ("day", "night", "week")
_TIME_UNITS = _CORE_PERIOD_UNITS + (
    "morning", "evening", "afternoon", "month", "year", "weekend", "weekday", "hour",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
_TIME_UNIT_ALT = "|".join(_TIME_UNITS)
_NUMBER_WORDS = "two|three|four|five|six|seven|eight|nine|ten|few|couple"
_HABIT_CUE_RE = re.compile(
    rf"\b(?:every|each)\s+(?:single\s+)?(?:other\s+)?(?:{_TIME_UNIT_ALT})\b|"
    rf"\bevery\s+(?:\d+|{_NUMBER_WORDS})(?:\s+of)?\s+(?:{_TIME_UNIT_ALT})s?\b|"
    r"\bevery\s+time\b|"
    r"\b(?:daily|nightly|weekly|monthly|yearly|biweekly|usually|always|"
    r"typically|routinely|regularly)\b|"
    r"\bmost\s+(?:days?|nights?|mornings?|evenings?|weeks?)\b",
    re.IGNORECASE,
)
# "I <verb> ... a day/week" — simple-present habitual framing without one of
# the adverbial cues above ("I take it twice a day").
_PERIOD_ALT = "|".join(_CORE_PERIOD_UNITS)
_SIMPLE_PRESENT_FREQ_RE = re.compile(
    rf"\bi\s+[a-z]+\b[^.!?\n]{{0,40}}\b(?:a\s+(?:{_PERIOD_ALT})|per\s+(?:day|week)|"
    r"once\s+a\b|twice\s+a\b|three\s+times\s+a\b)\b",
    re.IGNORECASE,
)

# A closed set of common past-tense verbs — deliberately NOT a blanket
# "-ed" suffix match (that would misfire on "prescribed", "interested",
# "based", none of which report a discrete completed event).
_PAST_TENSE_CUE_RE = re.compile(
    r"\b(?:took|had|did|went|was|were|got|skipped|missed|forgot|finished|"
    r"started|stopped|felt|saw|ate|drank|slept|called|texted|emailed|"
    r"visited|attended|completed|paid|cancelled|canceled|scheduled|showed|"
    r"arrived|left|woke|came)\b",
    re.IGNORECASE,
)

# Prospective framing (2026-09-08): "So ready to take melatonin … but I
# think it is early for that" minted took=melatonin — readiness, wanting,
# needing, being about to, are claims about the FUTURE, not a completed act.
# Closed grammatical set (infinitive/modal frames), not a topic vocabulary.
_PROSPECTIVE_RE = re.compile(
    r"\b(?:ready|about|want|wants|wanna|need|needs|going|gonna|hoping|hope|planning|"
    r"plan|plans|trying|tempted|supposed|mean|meaning|intend|intending|thinking|"
    r"looking\s+forward|due|scheduled|set)\s+to\b|\bgonna\b|\bwanna\b|"
    r"\bthinking\s+(?:about|of)\b|\bconsidering\b|\bmight\b|\bmay\b|"
    r"\bshould\b|\bcould\b|\bwould\b|\bwill\b|\b(?:i'?ll|we'?ll)\b",
    re.IGNORECASE,
)

_PLAN_KIND_CUE_RE = re.compile(
    r"\bwill\b|\bgoing\s+to\b|\bplan(?:s|ning)?\s+to\b|\btomorrow\b|"
    r"\bnext\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\b|"
    r"\b(?:ready|about|want|wants|wanna|need|needs|gonna|hoping|hope|trying|tempted|"
    r"supposed|intend|intending|thinking)\s+to\b|\bgonna\b|\bwanna\b|"
    r"\bthinking\s+(?:about|of)\b|\bconsidering\b",
    re.IGNORECASE,
)

# CLS-1.2 (2026-10-16): intention / future framing in the SAME clause beats a
# recurrence cue ("I plan to run every day starting tomorrow" is a plan, not a
# habit).  Reuses ``_PLAN_KIND_CUE_RE`` (will / going to / plan to / want to /
# tomorrow / next <unit>) and adds only the three framings it lacks.
_INTENT_FRAMING_RE = re.compile(
    _PLAN_KIND_CUE_RE.pattern
    + r"|\b\w+['\u2019]ll\b|\bfrom\s+now\s+on\b"
    + rf"|\bstarting\s+(?:today|tonight|tomorrow|now|next|this|{_TIME_UNIT_ALT})\b",
    re.IGNORECASE,
)

# CLS-3 frame check: HISTORICAL framing (closed grammatical class).  Detected
# LOCALLY by the recurrence gate only — it is deliberately NOT a global claim
# kind (profile/Chroma/graph readers have no parity for one; plan v2.1 R2-4).
_HISTORICAL_FRAME_RE = re.compile(
    r"\bused\s+to\b|\bno\s+longer\b|\bany\s*more\b|\bstopped\b",
    re.IGNORECASE,
)

# Relations whose NAME asserts a completed act. A span that only frames the
# act prospectively ("ready to take", "about to send", "want to go") and
# carries no past-tense cue cannot support them; the correct relation for
# that span is a plan/intention, which the extractors name separately.
COMPLETION_RELATIONS = frozenset({
    "took", "take", "taken", "ate", "eaten", "drank", "did", "done", "sent", "emailed",
    "texted", "called", "messaged", "finished", "completed", "submitted", "turned_in",
    "went", "went_to", "attended", "visited", "paid", "bought", "purchased", "ordered",
    "watched", "read", "slept", "showered", "worked_out", "exercised", "lifted", "ran",
    "walked", "cooked", "cleaned", "posted", "pushed", "committed", "shipped", "deployed",
    "cancelled", "canceled", "scheduled", "booked", "started", "stopped", "quit",
    "medication_taken", "dose_taken", "took_medication", "took_dose",
})


def completion_claim_prospective(relation: str, clause: str) -> bool:
    """True when ``relation`` asserts a completed act but ``clause`` only
    frames it prospectively — a prospective cue with NO past-tense cue.
    Never fires for relations outside COMPLETION_RELATIONS."""
    rel = (relation or "").strip().lower()
    if rel not in COMPLETION_RELATIONS:
        return False
    text = clause or ""
    if _PAST_TENSE_CUE_RE.search(text):
        return False
    return bool(_PROSPECTIVE_RE.search(text))

_STATE_CUE_RE = re.compile(
    r"\bi'?m\b|\bi’m\b|\bi\s+am\b|\bi'?ve\s+been\b|\bi’ve\s+been\b|"
    r"\bi\s+have\s+been\b|\bi\s+feel\b",
    re.IGNORECASE,
)

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WEEKDAY_ALT = "|".join(_WEEKDAY_INDEX)

_EVENT_ANCHOR_RE = re.compile(
    rf"\byesterday\b|\blast\s+night\b|\blast\s+week\b|\blast\s+month\b|"
    rf"\blast\s+year\b|\blast\s+(?:{_WEEKDAY_ALT})\b|\bthis\s+morning\b|"
    rf"\b\d+\s+days?\s+ago\b|\bon\s+(?:{_WEEKDAY_ALT})\b",
    re.IGNORECASE,
)
_DAYS_AGO_RE = re.compile(r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE)
_LAST_WEEKDAY_RE = re.compile(rf"\blast\s+({_WEEKDAY_ALT})\b", re.IGNORECASE)
_ON_WEEKDAY_RE = re.compile(rf"\bon\s+({_WEEKDAY_ALT})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ClaimTime:
    """What kind of claim a span makes, and — for a past event — the date."""

    kind: str = "unknown"  # "event" | "state" | "habit" | "plan" | "unknown"
    event_date: date | None = None
    event_date_source: str = "none"  # "explicit" | "relative" | "none"


def _resolve_relative_event_anchor(text: str, observed: datetime) -> date | None:
    """Deterministic day-arithmetic for the RELATIVE anchors in
    ``_EVENT_ANCHOR_RE`` that ``utils.temporal_resolver.resolve_date_expression``
    does not parse (it resolves explicit month+day expressions only, e.g.
    "April 15th"). Ambiguous anchors ("last week", "last month", "last
    year") deliberately resolve to ``None`` — a single day cannot represent
    them; the caller still reports ``kind="event"`` per spec."""
    low = text.lower()
    today = observed.date()
    if re.search(r"\byesterday\b|\blast\s+night\b", low):
        return today - timedelta(days=1)
    if re.search(r"\bthis\s+morning\b", low):
        return today
    m = _DAYS_AGO_RE.search(low)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            return None
        return today - timedelta(days=n)
    m = _LAST_WEEKDAY_RE.search(low)
    if m:
        target = _WEEKDAY_INDEX[m.group(1)]
        delta = (today.weekday() - target) % 7
        delta = delta or 7  # "last Monday" on a Monday means a week ago
        return today - timedelta(days=delta)
    m = _ON_WEEKDAY_RE.search(low)
    if m:
        target = _WEEKDAY_INDEX[m.group(1)]
        delta = (today.weekday() - target) % 7  # 0 == today itself
        return today - timedelta(days=delta)
    return None


# Weekday-sequence episode (2026-10-08, class: BC-52, BC-04): "Saturday Sunday
# Monday awful migraines etc one cracker a day" names a run of days with no
# past-tense verb and no "yesterday"-style anchor, so it classified "unknown"
# and the three-day episode rendered as current state. Two or more weekday
# names and no future cue = a bounded past episode ending on the LAST day named.
_WEEKDAY_NAME_RE = re.compile(
    rf"\b({_WEEKDAY_ALT}|mon|tue|wed|thu|fri|sat|sun)\b", re.IGNORECASE
)
# 3-letter forms that are also ordinary English words ("sat in the sun") count
# only when capitalised.
_AMBIGUOUS_WEEKDAY_ABBREVS = frozenset({"sat", "sun", "wed"})
_WEEKDAY_FUTURE_CUE_RE = re.compile(
    r"\b(?:next|coming|will|gonna|going\s+to|tomorrow|plans?|planning)\b|\b\w+['’]ll\b",
    re.IGNORECASE,
)


def _weekday_indices(text: str) -> list[int]:
    """Weekday indices (Mon=0) named in ``text``, in order of appearance."""
    days: list[int] = []
    for m in _WEEKDAY_NAME_RE.finditer(text):
        word = m.group(1)
        low = word.lower()
        if low in _AMBIGUOUS_WEEKDAY_ABBREVS and not word[0].isupper():
            continue
        idx = _WEEKDAY_INDEX.get(low)
        if idx is None:
            idx = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}[low]
        days.append(idx)
    return days


# 2026-10-08 (class: BC-04): a first-person SIMPLE-PRESENT clause ("I go to the
# gym Monday and Friday") naming a SPREAD weekday list is a standing schedule —
# a habit — even with no "every"/"weekly" cue. A first-person verb ending in
# "ed" ("I worked Monday Wednesday Friday") reports something that happened.
_FIRST_PERSON_PRESENT_RE = re.compile(r"\b(?:i|we)\s+([a-z]+)\b", re.IGNORECASE)


def _is_spread_weekday_schedule(text: str) -> bool:
    days = _weekday_indices(text)
    if len(days) < 2:
        return False
    if not any((b - a) % 7 != 1 for a, b in zip(days, days[1:])):
        return False
    if not any(not m.group(1).lower().endswith("ed") for m in _FIRST_PERSON_PRESENT_RE.finditer(text)):
        return False
    if _PAST_TENSE_CUE_RE.search(text) or _WEEKDAY_FUTURE_CUE_RE.search(text):
        return False
    return True


def _resolve_weekday_sequence_episode(text: str, observed: datetime) -> date | None:
    """Most recent date STRICTLY before ``observed``'s date whose weekday is the
    LAST weekday named, when ``text`` names >= 2 weekdays and no future cue."""
    days = _weekday_indices(text)
    if len(days) < 2 or _WEEKDAY_FUTURE_CUE_RE.search(text):
        return None
    # Referee (2026-10-08): an episode is a CONSECUTIVE run of days ("Sat Sun
    # Mon"); a spread list ("Monday Wednesday Friday") is a schedule. And a
    # first-person present-tense report of a consecutive run ("I work Monday
    # Tuesday Wednesday") is a standing schedule too — only a past-tense cue or
    # a verbless fragment (the live shape) marks it as something that happened.
    if any((b - a) % 7 != 1 for a, b in zip(days, days[1:])):
        return None
    if _FIRST_PERSON_SUBJECT_RE.search(text) and not _PAST_TENSE_CUE_RE.search(text):
        return None
    today = observed.date()
    delta = (today.weekday() - days[-1]) % 7 or 7
    return today - timedelta(days=delta)


def _habit_cue_is_intended(text: str) -> bool:
    """True when EVERY clause carrying a recurrence cue also carries
    intention/future framing (CLS-1.2).  A cue living only in the whole-text
    frequency pattern ("I ... twice a day") is judged on the whole text."""
    clauses = _split_clauses(text) or [text]
    cue_clauses = [
        c for c in clauses
        if _HABIT_CUE_RE.search(c) or _SIMPLE_PRESENT_FREQ_RE.search(c)
    ]
    if not cue_clauses:
        return bool(_INTENT_FRAMING_RE.search(text))
    return all(_INTENT_FRAMING_RE.search(c) for c in cue_clauses)


def classify_claim_time(span: str, *, observed_at: datetime | None = None) -> ClaimTime:
    """Classify what kind of claim ``span`` makes, relative to ``observed_at``
    (defaults to now when the caller has no turn timestamp).

    Order: habit > event (weekday-sequence episode, or past-tense + temporal
    anchor) > plan > state > unknown. A past-tense cue with NO temporal anchor is under-classified as
    "unknown" rather than guessed as an event — an anchor is required by
    spec. An explicit calendar date (e.g. "September 5th") resolves via
    ``utils.temporal_resolver.resolve_date_expression``; every other
    anchor (yesterday, last night/week/weekday, this morning, N days ago,
    on <weekday>) resolves via direct day-arithmetic relative to
    ``observed_at`` (that function does not parse relative expressions).
    """
    text = span or ""
    observed = observed_at or datetime.now()

    if (_HABIT_CUE_RE.search(text) or _SIMPLE_PRESENT_FREQ_RE.search(text)
            or _is_spread_weekday_schedule(text)):
        # CLS-1.2: future/intention framing in the SAME clause as the
        # recurrence cue makes the claim a plan, not a standing habit.
        if _habit_cue_is_intended(text):
            return ClaimTime(kind="plan")
        return ClaimTime(kind="habit")

    seq_date = _resolve_weekday_sequence_episode(text, observed)
    if seq_date is not None:
        return ClaimTime(kind="event", event_date=seq_date, event_date_source="relative")

    if _PAST_TENSE_CUE_RE.search(text):
        iso, _basis, _conf = resolve_date_expression(text, reference_date=observed)
        if iso:
            try:
                ev_date = datetime.strptime(iso, "%Y-%m-%d").date()
            except ValueError:
                ev_date = None
            if ev_date is not None:
                # The shared resolver is FUTURE-biased by design (it serves
                # scheduling callers): a month+day already passed this year
                # rolls to next year. A PAST-TENSE claim cannot describe a
                # future date, so roll back one year when the resolved date
                # lies after the observation (referee fix, 2026-09-06).
                if ev_date > observed.date():
                    try:
                        ev_date = ev_date.replace(year=ev_date.year - 1)
                    except ValueError:  # Feb 29 → Feb 28
                        ev_date = ev_date.replace(year=ev_date.year - 1, day=28)
                return ClaimTime(kind="event", event_date=ev_date, event_date_source="explicit")
        rel_date = _resolve_relative_event_anchor(text, observed)
        if rel_date is not None:
            return ClaimTime(kind="event", event_date=rel_date, event_date_source="relative")
        if _EVENT_ANCHOR_RE.search(text):
            return ClaimTime(kind="event", event_date=None, event_date_source="none")
        # A past-tense cue with NO temporal anchor is under-classified as
        # unknown (docstring contract): do NOT fall through to plan/state — the
        # span reported something that already happened ("Started getting
        # stressed … needed to get up early" is not a plan).
        return ClaimTime(kind="unknown")

    if _PLAN_KIND_CUE_RE.search(text):
        return ClaimTime(kind="plan")

    if _STATE_CUE_RE.search(text):
        return ClaimTime(kind="state")

    return ClaimTime(kind="unknown")


# ---------------------------------------------------------------------------
# CLS-2 (2026-10-16, class: BC-04): classify the clause that CARRIES the object,
# not the whole sentence.  "Yesterday I worked late and today I saw Rowan" dated
# the Rowan event yesterday; a recurrence cue in a neighbouring clause ("…every
# day, and saw Rowan for two hours") made a one-off a habit.
# ---------------------------------------------------------------------------

_CLAUSE_CONJUNCTIONS = frozenset({"and", "but", "so", "then", "or"})
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)


def _clause_has_own_subject(clause: str) -> bool:
    return bool(_FIRST_PERSON_SUBJECT_RE.search(clause) or _OTHER_PERSON_RE.search(clause))


def _elliptical_head_is_past(clause: str) -> bool:
    """Past-tense head of a subject-less clause.  The closed past-tense set
    PLUS an ``-ed`` suffix on the clause's FIRST token only: at the head of an
    elliptical clause that token is a verb position (``walked``), where the
    blanket ``-ed`` rule (prescribed/based) the sentence-level cue avoids does
    not apply.  Local to scope inheritance — never added to
    ``_PAST_TENSE_CUE_RE``."""
    words = _WORD_RE.findall(clause)
    while words and words[0].lower() in _CLAUSE_CONJUNCTIONS:  # ", and walked home"
        words = words[1:]
    if not words:
        return False
    head = words[0].lower()
    if _PAST_TENSE_CUE_RE.fullmatch(head):
        return True
    return len(head) > 3 and head.endswith("ed")


def _own_clause_time(clause: str, observed: datetime) -> ClaimTime:
    """The clause's OWN cues: recurrence, event anchor, plan, state — plus a
    same-clause ``today`` with a past-tense verb (an event on the observed day;
    ``classify_claim_time`` has no ``today`` anchor, and the profile reader
    only hides events dated BEFORE today)."""
    own = classify_claim_time(clause, observed_at=observed)
    if own.kind == "unknown" and _TODAY_RE.search(clause) and _PAST_TENSE_CUE_RE.search(clause):
        return ClaimTime(kind="event", event_date=observed.date(), event_date_source="relative")
    return own


def classify_claim_time_for_object(
    span: str, object_clause: str, *, observed_at: datetime | None = None
) -> ClaimTime:
    """Claim kind/date for the clause ``object_clause`` of ``span``.

    Rules, in order:
      1. The clause's OWN recurrence / anchor / plan / state cue decides.
      2. A clause with no subject of its own (elliptical: "…and swim") inherits
         the governing clause's whole classification — a shared predicate
         frame — EXCEPT when its head verb is past tense while the governing
         clause is not ("I always run and walked home": a separate completed
         act inherits nothing).
      3. A clause with its own subject and no own cue inherits only the
         nearest PRECEDING clause's event anchor ("Yesterday I worked late and
         I saw Rowan" -> yesterday); one with its own anchor never takes
         another clause's anchor.
    """
    observed = observed_at or datetime.now()
    sentence = span or ""
    clause = (object_clause or "").strip()
    if not clause or clause == sentence.strip():
        return _own_clause_time(clause or sentence, observed)

    own = _own_clause_time(clause, observed)
    if own.kind in ("habit", "event", "plan", "state"):
        return own
    # A standing weekday schedule split across clauses at "and"/"," ("I work
    # Monday and Friday"): the object clause holds one weekday of the list.
    if _weekday_indices(clause) and _is_spread_weekday_schedule(sentence):
        return ClaimTime(kind="habit")

    clauses = _split_clauses(sentence)
    try:
        idx = clauses.index(clause)
    except ValueError:
        return own
    preceding = clauses[:idx]
    result = _inherit_from_neighbours(clause, preceding, own, observed)
    if (result.kind == "unknown"
            and not (_EVENT_ANCHOR_RE.search(clause) or _TODAY_RE.search(clause)
                     or _weekday_indices(clause))):
        # Trailing shared modifier ("I went to the gym and did legs
        # yesterday"): the nearest FOLLOWING clause's event anchor.  Only an
        # anchor — recurrence is never inherited from a following clause.
        for later in clauses[idx + 1:]:
            anchor = _EVENT_ANCHOR_RE.search(later)
            if anchor is not None:
                dated = classify_claim_time(f"{clause} {anchor.group(0)}", observed_at=observed)
                return dated if dated.kind == "event" else result
    return result


def _inherit_from_neighbours(
    clause: str, preceding: list[str], own: ClaimTime, observed: datetime
) -> ClaimTime:
    if not _clause_has_own_subject(clause):
        governing = next((c for c in reversed(preceding) if _clause_has_own_subject(c)), None)
        if governing is None:
            return own
        if _elliptical_head_is_past(clause) and not _PAST_TENSE_CUE_RE.search(governing):
            return own
        # A clause with its OWN date specification (event anchor / today /
        # a named weekday) is a dated report, not a shared predicate frame —
        # it never takes the governing clause's recurrence ("…every day, and
        # hung with Tamsin Sat": "hung" is outside the closed past-tense set).
        if (_EVENT_ANCHOR_RE.search(clause) or _TODAY_RE.search(clause)
                or _weekday_indices(clause)):
            return own
        return _own_clause_time(governing, observed)
    for prior in reversed(preceding):
        anchor = _EVENT_ANCHOR_RE.search(prior)
        if anchor is not None:
            return classify_claim_time(f"{clause} {anchor.group(0)}", observed_at=observed)
    return own


def object_clause_for(span: str, object_val: str) -> str:
    """The clause of ``span`` that carries ``object_val`` (whole span when none
    does) — the public form of the clause choice the provenance join makes."""
    clauses = _split_clauses(span)
    return _object_bearing_clause(clauses, object_val, _tokens(object_val)) or (span or "")


# ---------------------------------------------------------------------------
# CLS-3 (2026-10-16, class: BC-51): a relation whose NAME asserts recurrence
# ("eating_habit", "sleep_pattern", "morning_routine") needs recurrence evidence
# in the user's own words — the same fail-closed pattern as
# ``current_status_claim_supported``.  ONE predicate, two call sites (the LLM
# provenance join and the regex producer).
# ---------------------------------------------------------------------------

RECURRENCE_RELATION_TOKENS = frozenset(
    {"habit", "habits", "routine", "routines", "pattern", "patterns"}
)
# The regex SCHEDULE-pattern producer (fact_extractor: found.append(…,
# "schedule_regex")) emits triples whose match is itself validated schedule
# evidence — the only producer exempt from the gate (never by relation name).
SCHEDULE_PRODUCER_METHOD = "schedule_regex"


def is_recurrence_relation(relation: str) -> bool:
    toks = [t for t in re.split(r"[_\s\-]+", (relation or "").strip().lower()) if t]
    return any(t in RECURRENCE_RELATION_TOKENS for t in toks)


def recurrence_claim_supported(
    relation: str, span: str, object_clause: str, *, observed_at: datetime | None = None
) -> bool:
    """False when ``relation`` names a recurrence but the evidence is not a
    present-tense recurrence claim.  True for every other relation.

    Requires (a) the scoped claim kind of the object clause to be exactly
    ``habit`` and (b) a clean SENTENCE frame: no intention/future framing and
    no historical framing (used to / no longer / anymore / stopped) anywhere in
    the sentence holding the winning clause — even if the object clause has its
    own recurrence cue.  Deliberately fail-closed: "I used to run but now I
    swim every day" is a false reject (Phase 2's occurrence ledger recovers
    genuinely recurring activity)."""
    if not is_recurrence_relation(relation):
        return True
    sentence = span or object_clause or ""
    if _INTENT_FRAMING_RE.search(sentence) or _HISTORICAL_FRAME_RE.search(sentence):
        return False
    kind = classify_claim_time_for_object(sentence, object_clause, observed_at=observed_at).kind
    return kind == "habit"


@dataclass(frozen=True)
class EvidenceSpan:
    """A user-authored span supporting one proposed triple."""

    text: str
    turn_index: int
    role: str = "user"
    support: str = "object_match"
    turn_id: str = ""
    anchor: str = ""
    # B2 (2026-09-06): the claim's temporal kind + resolved event date/
    # observation time, forwarded by both extractors as claim_kind/event_date
    # (ISO or "")/observed_at (ISO) — new keys only, existing ones unchanged.
    # compare=False: an out-of-scope test (tests/unit/test_fact_source.py)
    # does exact EvidenceSpan(...) equality against a literal that predates
    # these fields; they carry real data via attribute access but are
    # deliberately excluded from dataclass equality so that pre-existing
    # comparison keeps working untouched.
    claim_kind: str = field(default="", compare=False)
    event_date: str = field(default="", compare=False)
    observed_at: str = field(default="", compare=False)
    # CLS-4 (2026-10-16): the SOURCE message's own timestamp (ISO), "" when the
    # message carries none.  Distinct from ``turn_id`` (identity) and from
    # ``observed_at`` (which falls back to now() for claim-time anchoring):
    # day-keyed consumers (learned-relation recurrence) must never see "now".
    source_time: str = field(default="", compare=False)


def authored_text(message: Any) -> str:
    """The user-authored text of a message dict (the ONE authorship chokepoint).

    ``user_text`` (2026-09-05) is the user's OWN typed text for a turn whose
    ``query`` is the merged user-text + attachment blob (corpus entries store
    the merged form so retrieval renders attachments). When the key is present
    it is authoritative EVEN WHEN EMPTY: an attachment-only turn has no
    authored words, and falling back to ``query`` would hand the attachment
    text (a CV, a transcript) to the extractors as if the user had written it.
    Only a message with no ``user_text`` key falls back to ``query`` / ``user``.
    """
    if not isinstance(message, Mapping):
        return str(message or "")
    ut = message.get("user_text")
    if "user_text" in message and isinstance(ut, str):
        return ut
    return str(message.get("query") or message.get("user") or "")


def is_test_turn(entry: Any) -> bool:
    """True when a turn's AUTHORED text carries a complete [test]...[/test] envelope.

    The ONE predicate every DERIVED-artifact builder (summaries, reflections,
    notes, narrative, threads, skills, facts) asks, built on ``authored_text``
    (the authorship chokepoint) and ``test_envelope`` (the structural parser).
    """
    return test_envelope.has_envelope(authored_text(entry))


def _message_text_and_id(message: Any) -> tuple[str, str] | None:
    """Return only the user-authored portion of a supported message shape."""
    if isinstance(message, Mapping):
        text = authored_text(message).strip()
        turn_id = str(
            message.get("turn_id")
            or message.get("interaction_id")
            or message.get("id")
            or message.get("timestamp")
            or ""
        ).strip()
    else:
        text = str(message or "").strip()
        turn_id = ""

    if not text or _NON_USER_ROLE_RE.match(text):
        return None
    return _USER_ROLE_RE.sub("", text, count=1).strip(), turn_id


def _message_source_time(message: Any) -> str:
    """The message's own ``timestamp`` as an ISO string, or ``""`` when the
    message has none / it is unparseable.  Never falls back to now()."""
    if not isinstance(message, Mapping):
        return ""
    raw = message.get("timestamp")
    if isinstance(raw, datetime):
        return raw.isoformat()
    if isinstance(raw, str) and raw.strip():
        try:
            return datetime.fromisoformat(raw.strip()).isoformat()
        except ValueError:
            return ""
    return ""


def iter_user_messages(messages: Iterable[Any]) -> Iterator[tuple[int, str, str]]:
    """Yield ``(turn_index, user_text, turn_id)`` without assistant responses."""
    for index, message in enumerate(messages or []):
        parsed = _message_text_and_id(message)
        if parsed is None:
            continue
        text, turn_id = parsed
        if text:
            yield index, text, turn_id


# Pasted correspondence (2026-09-03).  A user turn that quotes an email —
# their own outgoing one included — is reported/historical text, not a live
# claim: "I'm … enrolled in two courses (ABC 1234 and QRS 7310)" inside a
# pasted Aug-27 email superseded the curated enrolled_in=QRS 7310 fact on
# Sep 2, days after the drop.  A block runs from a greeting line to a closing
# line plus the contiguous signature run after it; BOTH ends are required so a
# bare chat "Hi," never swallows a message.
_EMAIL_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening))\b[^\n]{0,80}?[,:!]?\s*$",
    re.IGNORECASE,
)
_EMAIL_CLOSING_RE = re.compile(
    r"^\s*(?:thanks|thank\s+you|thanks\s+again|thank\s+you\s+(?:so\s+much|again)|many\s+thanks|"
    r"best|all\s+the\s+best|best\s+regards|regards|kind\s+regards|warm\s+regards|warmly|"
    r"sincerely|cheers|take\s+care|talk\s+soon|respectfully|gratefully|yours(?:\s+truly)?)"
    r"\s*[,.!]?\s*$",
    re.IGNORECASE,
)
_SIGNATURE_MAX_LINES = 8

# [test]...[/test] convention (2026-09-06, B3; inline form 2026-09-27,
# BC-58): an operator-marked synthetic/replay turn — same treatment as
# [relay:...]...[/relay]. Detected structurally by the bracket markers
# only (utils.test_envelope, the shared parser for BOTH the whole-line
# block form and the inline form — an un-marked probe like
# "[test]Navient. Search that[/test]" used to be invisible here); nothing
# infers "test" from wording, repetition, or a medication name.


def quoted_correspondence_lines(text: str) -> set[int]:
    """Indexes of lines that sit inside a pasted email block (greeting →
    closing → signature run), a [relay:...]...[/relay] block, or a
    [test]...[/test] envelope (inline or whole-line).  Empty set when no
    complete block exists."""
    lines = (text or "").splitlines()
    inside: set[int] = set()
    # Workflow relay convention: agent output is quoted evidence even when
    # its first-person sentences have no Markdown blockquote markers.
    # A closing marker lets the user resume speaking after a relayed block.
    in_relay = False
    for index, line in enumerate(lines):
        if re.match(r"^\s*\[relay:\s*[^\]]+\]", line, re.IGNORECASE):
            in_relay = True
        if in_relay:
            inside.add(index)
        if re.match(r"^\s*\[/relay\]\s*$", line, re.IGNORECASE):
            in_relay = False
    # [test] convention: an operator-marked synthetic/replay turn — the
    # content inside is neither claim evidence nor prose commentary.
    inside |= test_envelope.envelope_line_indices(text)
    i = 0
    while i < len(lines):
        if not _EMAIL_GREETING_RE.match(lines[i]):
            i += 1
            continue
        close = None
        for j in range(i + 1, len(lines)):
            if _EMAIL_CLOSING_RE.match(lines[j]):
                close = j
                break
        if close is None:
            i += 1
            continue
        end = close
        k = close + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        run = 0
        while k < len(lines) and lines[k].strip() and run < _SIGNATURE_MAX_LINES:
            if _EMAIL_GREETING_RE.match(lines[k]):
                break
            end = k
            k += 1
            run += 1
        inside.update(range(i, end + 1))
        i = end + 1
    return inside


def strip_quoted_correspondence(text: str) -> str:
    """Return ``text`` with pasted email blocks removed (framing lines kept)."""
    skip = quoted_correspondence_lines(text)
    if not skip:
        return text or ""
    kept = [ln for idx, ln in enumerate((text or "").splitlines()) if idx not in skip]
    return "\n".join(kept)


def contains_test_block(text: str) -> bool:
    """True when ``text`` contains a COMPLETE ``[test]...[/test]`` envelope
    (2026-09-06, B3; inline form recognised 2026-09-27, BC-58) — an
    operator-marked synthetic/replay turn. Storage tags provenance from
    this (``origin="test"``); extraction ignores the envelope's content
    entirely (it is stripped like quoted correspondence, so genuine
    commentary outside the envelope still yields facts). Detected
    structurally by the bracket markers only (``utils.test_envelope``) —
    nothing infers "test" from wording, repetition, or a medication name."""
    return test_envelope.has_envelope(text)


# A period after a title/common abbreviation is not a sentence boundary
# (2026-09-05: "My doctor Dr. Patel called" split into "My doctor Dr." and
# "Patel called", so no span held both the possessive anchor and the name).
_ABBREV_PERIOD_RE = re.compile(
    r"\b(?:dr|mr|mrs|ms|prof|st|mt|vs|etc|e\.g|i\.e|jr|sr|no)\.\s+(?=\S)", re.IGNORECASE
)
_ABBREV_SENTINEL = "\u0000"


def _split_sentences(line: str) -> list[str]:
    protected = _ABBREV_PERIOD_RE.sub(lambda m: m.group(0).replace(".", _ABBREV_SENTINEL), line)
    return [s.replace(_ABBREV_SENTINEL, ".") for s in re.split(r"(?<=[.!?])\s+|\s*[;]\s*", protected)]


# A soft-wrapped paste carries the SPA/terminal continuation bar at the head
# of every wrapped line ("add the … to my google\n  ▎ calander, fridays 8 to
# 9 pm …" — 2026-09-07 live). Those lines are ONE sentence; splitting them
# yields fragments whose token overlap can anchor the wrong relation. The
# bar is not a blockquote marker (">" still excludes).
_CONTINUATION_BAR_RE = re.compile(r"\n[ \t]*[▎▏▍▌▋▊▉│┃]+[ \t]*")


def _claim_spans(text: str) -> Iterator[str]:
    """Yield prose spans, excluding code fences, blockquotes and role dumps."""
    in_fence = False
    text = _CONTINUATION_BAR_RE.sub(" ", text or "")
    quoted = quoted_correspondence_lines(text)
    for line_idx, raw_line in enumerate((text or "").splitlines() or [text]):
        if line_idx in quoted:
            continue
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith(">"):
            continue
        if _NON_USER_ROLE_RE.match(line):
            continue
        # A labelled transcript can occur halfway through a pasted user turn.
        if _INLINE_ROLE_SPLIT_RE.search(line):
            line = _INLINE_ROLE_SPLIT_RE.split(line, maxsplit=1)[0]
        for span in _split_sentences(line):
            span = span.strip(" \t-*•")
            if not span or _EXTERNAL_ATTRIBUTION_RE.search(span):
                continue
            yield span


def _tokens(value: str) -> set[str]:
    return {
        token.lower() for token in _WORD_RE.findall(value or "")
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _span_is_user_owned(span: str, object_clause: str | None = None) -> str | None:
    """Return the anchor kind when the span asserts something about the USER.

    ``"first_person"``: an explicit I/we subject, or a first-person possessive
    with no competing third party.  ``"implicit_first_person"``: a chat
    fragment with an implied subject ("Started Lorvatin today").  ``None``: the
    span is about someone else ("Sam told me he moved", "My brother moved",
    "Biscuit is being a silly kitten", "it says you use WHOOP").

    ``object_clause`` (2026-09-06) narrows the third-party/possessive and
    sentence-opener checks to the CLAUSE that actually carries the object —
    a coordinated sentence's unrelated clause ("...and his dad...") must not
    disqualify a different clause's own claim ("Had the 5mg at 11").  When
    omitted it defaults to ``span`` (whole-sentence behavior, unchanged for
    every direct caller).  The relative-subject check ("My partner/brother
    …") stays deliberately ambiguous on its own — it is rescued ONLY by an
    explicit first-person subject elsewhere in the full sentence, never by a
    different clause's third party (that already fails the check above it).
    """
    clause = object_clause if object_clause is not None else span
    if _FIRST_PERSON_SUBJECT_RE.search(clause):
        return "first_person"
    if _OTHER_PERSON_RE.search(clause) or _POSSESSIVE_PROPER_RE.search(clause):
        return None
    words = clause.split()
    if len(words) > 1:
        first = words[0].strip("\"'“”‘’([")
        if (
            first[:1].isupper()
            and first.lower() not in _SENTENCE_STARTERS
            and _THIRD_PERSON_CONTINUATION_RE.match(words[1])
        ):
            return None
    if _RELATIVE_SUBJECT_RE.match(clause):
        if clause != span and _FIRST_PERSON_SUBJECT_RE.search(span):
            return "first_person"
        return None
    if _FIRST_PERSON_ANY_RE.search(clause) or _PROFILE_FIELD_RE.search(clause):
        return "first_person"
    return "implicit_first_person"


# Care-team relations (2026-09-05): "My psychiatrist has no portal" names the
# clinician as the grammatical subject, which _span_is_user_owned reads as a
# third party ("My brother moved") — but for a care-team relation the
# possessive clinician phrase IS the anchor, exactly like "my cat Biscuit" for
# pet ownership. Otherwise the span must be user-owned AND carry a cue.
_CARE_TEAM_RELATIONS = frozenset(
    rel for rel, cues in _RELATION_CUE_RES.items() if cues is _CARE_TEAM_CUES
)
_CARE_TEAM_POSSESSIVE_RE = re.compile(
    rf"\b(?:my|our)\s+(?:(?:new|old|current|former|regular|primary)\s+)?(?:{_CARE_TEAM_CUES})\b",
    re.IGNORECASE,
)
_CARE_TEAM_CUE_RE = re.compile(rf"\b(?:{_CARE_TEAM_CUES})\b", re.IGNORECASE)


def _care_team_supported(span: str) -> bool:
    if _CARE_TEAM_POSSESSIVE_RE.search(span):
        return True
    return _span_is_user_owned(span) is not None and bool(_CARE_TEAM_CUE_RE.search(span))


def _pet_ownership_supported(span: str, relation: str) -> bool:
    """``my/our <species>`` with no intervening possessive ("my mom's cat")."""
    species = _PET_OWNERSHIP_SPECIES.get(relation)
    if not species:
        return False
    pattern = (
        rf"\b(?:my|our)\s+(?:(?!\w+['’]s\b)[\w-]+\s+){{0,3}}(?:{species})\b"
        rf"|\b(?:{species})\b[^.!?]{{0,40}}\b(?:is|are)\s+(?:mine|ours)\b"
    )
    return bool(re.search(pattern, span, flags=re.IGNORECASE))


def _residence_resident_is_user(clause: str, object_val: str) -> bool:
    """For a residence relation: is the USER the resident of the place?

    A place possessed by another person that precedes the object ("my dad's
    on Saturday, which is on Bartlet", "my mom's place in Dallas") describes
    THEIR residence, never the user's — unless the clause also carries an
    explicit first-person resident assertion before the place ("I live at my dad's
    in Dallas"). Generic: a possessive-other-person head, not a name list.
    """
    pos = clause.lower().find((object_val or "").lower()) if object_val else -1
    head = clause if pos < 0 else clause[:pos]
    possessive = _OTHER_PERSON_PLACE_RE.search(head)
    if not possessive:
        return True
    return bool(_FIRST_PERSON_RESIDENT_RE.search(head))


# Current-status relations (2026-10-10, BC-75): the employment/enrollment family
# utils.status_claims already defines (ONE relation set — `dropped` is a past
# act, not a standing status).  Their name asserts a standing circumstance, so
# a noun cue ("mercer job") is not enough: the object-bearing clause must be a
# present-tense first-person statement.  Live: "ok so we have project, mercer
# job and skill g2g" (resume talk) minted works_at=Mercer — a PAST employer
# stored as current.
CURRENT_STATUS_RELATIONS = frozenset(
    (ENROLLMENT_RELATIONS - {"dropped"}) | EMPLOYMENT_RELATIONS
)
# Grammar-level past/former framing (closed class: tense + "used to"/"former"
# markers), not employer or topic vocabulary.  Present perfect ("I have worked
# at X for 3 years") is a standing status and is neutralized before the test.
_PRESENT_PERFECT_WORKED_RE = re.compile(r"\b(?:have|has|had|i['\u2019]ve|we['\u2019]ve)\s+worked\b", re.IGNORECASE)
_PAST_STATUS_RE = re.compile(
    r"\b(?:used\s+to|formerly|previously|no\s+longer|any\s*more|former|worked|quit|"
    r"resigned|fired|laid\s+off|left|(?:i|we)\s+(?:was|were))\b|\bex-\w+",
    re.IGNORECASE,
)


def current_status_claim_supported(relation: str, clause: str) -> bool:
    """False when ``relation`` is a current-status relation but ``clause`` is
    not a present-tense first-person statement (explicit I/we subject, no
    past/former framing).  True for every other relation."""
    if (relation or "").strip().lower() not in CURRENT_STATUS_RELATIONS:
        return True
    text = clause or ""
    if not _FIRST_PERSON_SUBJECT_RE.search(text):
        return False
    return not _PAST_STATUS_RE.search(_PRESENT_PERFECT_WORKED_RE.sub(" ", text))


# Future-tense auxiliaries as the LEADING word of a relation name ("will_do")
# mark a planned act, not a standing fact (closed grammatical class; `would`
# is excluded — "would_like" is a preference).
_FUTURE_RELATION_RE = re.compile(r"^(?:will|shall|gonna|going[_\s]to)(?:[_\s]|$)", re.IGNORECASE)


def relation_is_future_planned(relation: str) -> bool:
    """True when the relation name itself is a future-tense planned act."""
    return bool(_FUTURE_RELATION_RE.match((relation or "").strip()))


def _relation_cue_supported(span: str, relation: str) -> bool:
    """Where the relation name itself makes a claim, the span needs a cue."""
    cues = _RELATION_CUE_RES.get(relation)
    if not cues:
        return True
    rx = _RELATION_CUE_CACHE.get(relation)
    if rx is None:
        rx = re.compile(rf"\b(?:{cues})\b", re.IGNORECASE)
        _RELATION_CUE_CACHE[relation] = rx
    return bool(rx.search(span))


def _cropped_excerpt(span: str, needle: str, limit: int) -> str:
    span = re.sub(r"\s+", " ", span).strip()
    if len(span) <= limit:
        return span
    pos = span.lower().find((needle or "").lower())
    if pos < 0:
        pos = len(span) // 2
    start = max(0, min(pos - limit // 3, len(span) - limit))
    excerpt = span[start:start + limit].strip()
    if start:
        excerpt = "…" + excerpt[1:]
    if start + limit < len(span):
        excerpt = excerpt[:-1] + "…"
    return excerpt


def supporting_excerpt(text: str, object_val: str, limit: int = 200) -> str:
    """Return the sentence-sized window of ``text`` that mentions ``object_val``.

    Used by the regex extraction path so a stored ``source_excerpt`` shows the
    span that carries the claim rather than the first N characters of a long
    turn (the lived_in=Marrowby "song lyrics as evidence" defect).
    """
    text = text or ""
    obj_low = (object_val or "").lower().strip()
    spans = list(_claim_spans(text)) or [text]
    best = None
    if obj_low:
        for span in spans:
            if obj_low in span.lower():
                best = span
                break
    if best is None:
        obj_tokens = _tokens(object_val)
        best_score = 0
        for span in spans:
            score = len(obj_tokens & _tokens(span))
            if score > best_score:
                best_score, best = score, span
    if best is None:
        best = text
    return _cropped_excerpt(best, object_val, limit)


def _parse_observed_at(turn_id: str) -> datetime:
    """Best-effort turn timestamp for claim-time classification. Many corpus
    shapes use an ISO ``timestamp`` string as the turn_id fallback (see
    ``_message_text_and_id``); when that isn't parseable there is no other
    signal available, so "now" is the only honest default (relative anchors
    like "yesterday" then resolve relative to classification time, not the
    turn's real time — an accepted approximation, under-fire by design)."""
    if turn_id:
        try:
            return datetime.fromisoformat(turn_id)
        except (ValueError, TypeError):
            pass
    return datetime.now()


def find_supporting_user_span(
    triple: Mapping[str, Any],
    messages: Iterable[Any],
    *,
    excerpt_limit: int = 200,
) -> EvidenceSpan | None:
    """Find direct user evidence for a proposed triple, or return ``None``.

    User-scoped facts require a user-owned span (see ``_span_is_user_owned``)
    plus, for claim-bearing relation names, a corroborating cue; pet ownership
    relations require the possessive species phrase.  Subjects re-scoped by
    the stance layer (``"user's …"``) are user-owned by construction.  Entity
    facts require both the named subject and the object in one span.  Every
    fact requires object grounding; there is intentionally no "last message"
    fallback.  A span whose object-bearing CLAUSE is negated ("I did not take
    Zelphex today") never supports the triple, even when an unrelated
    negated sentence is the only other candidate mentioning the object
    (2026-09-06).
    """
    subject = str(triple.get("subject") or "").strip()
    relation = str(triple.get("relation") or triple.get("predicate") or "").strip().lower()
    object_val = str(triple.get("object") or triple.get("value") or "").strip()
    if not subject or not relation or not object_val:
        return None

    subject_low = subject.lower()
    is_user = subject_low == "user"
    is_scoped_referent = subject_low.startswith("user's ") or subject_low.startswith("user’s ")
    object_tokens = _tokens(object_val)
    subject_tokens = _tokens(subject) if not (is_user or is_scoped_referent) else set()
    if not object_tokens and len(object_val.strip()) < 2:
        return None

    # Overlap floor (2026-09-05): a multi-token object joined on ONE shared
    # token is coincidence, not evidence — `has_doctor = "Ellery is cautious
    # about drinking due to past accident"` was anchored to a meds sentence
    # via the single word "drinking" after the real (third-party) source
    # sentence was correctly rejected. Two content tokens are required once
    # the object carries three or more.
    min_overlap = 1 if len(object_tokens) <= 2 else 2

    # Excerpt integrity (2026-09-06): a lone token overlap is only evidence
    # when that token is the object's own head noun (its last content word —
    # "5 mg Zelphex" -> "zelphex") — an incidental shared adjective/modifier
    # is not the claim.
    _ordered_object_tokens = [
        tok.lower() for tok in _WORD_RE.findall(object_val)
        if len(tok) >= 3 and tok.lower() not in _STOPWORDS
    ]
    object_head_token = _ordered_object_tokens[-1] if _ordered_object_tokens else None

    best: tuple[int, int, str, str, str, str, str] | None = None
    messages = list(messages or [])
    for turn_index, text, turn_id in iter_user_messages(messages):
        for span in _claim_spans(text):
            low = span.lower()
            exact_object = bool(object_val and object_val.lower() in low)
            overlap_tokens = object_tokens & _tokens(span)
            overlap = len(overlap_tokens)
            if not exact_object:
                if overlap < min_overlap:
                    continue
                if overlap == 1 and object_head_token and next(iter(overlap_tokens)) != object_head_token:
                    continue

            clauses = _split_clauses(span)
            object_clause = _object_bearing_clause(clauses, object_val, object_tokens) or span
            if _clause_is_negated(object_clause, object_val, object_tokens):
                continue
            # A completion relation cannot ride on a prospective clause
            # (2026-09-08: took=melatonin from "So ready to take melatonin").
            if completion_claim_prospective(relation, object_clause):
                continue

            if is_user:
                if relation in _PET_OWNERSHIP_SPECIES:
                    if not _pet_ownership_supported(object_clause, relation):
                        continue
                    anchor = "pet_ownership"
                elif relation in _CARE_TEAM_RELATIONS:
                    if not _care_team_supported(object_clause):
                        continue
                    anchor = "care_team"
                else:
                    anchor = _span_is_user_owned(span, object_clause)
                    if anchor is None:
                        continue
                    if not _relation_cue_supported(span, relation):
                        continue
                    if not current_status_claim_supported(relation, object_clause):
                        continue
                    # CLS-3: a recurrence-NAMED relation needs recurrence evidence.
                    if not recurrence_claim_supported(
                        relation, span, object_clause,
                        observed_at=_parse_observed_at(turn_id),
                    ):
                        continue
                    if relation in RESIDENCE_RELATIONS and not _residence_resident_is_user(
                        object_clause, object_val
                    ):
                        continue
            elif is_scoped_referent:
                anchor = "scoped_referent"
            else:
                if subject_tokens and not (subject_tokens & _tokens(span)):
                    continue
                anchor = "entity"

            score = (100 if exact_object else 0) + overlap
            support = "exact_object" if exact_object else "object_token_overlap"
            candidate = (score, turn_index, span, turn_id, support, anchor, object_clause)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ):
                best = candidate

    if best is None:
        return None
    _, turn_index, span, turn_id, support, anchor, winner_clause = best
    excerpt_text = _cropped_excerpt(span, object_val, excerpt_limit)
    observed_dt = _parse_observed_at(turn_id)
    # CLS-2: classify the WINNING span's winning clause (never the loop's last).
    claim_time = classify_claim_time_for_object(span, winner_clause, observed_at=observed_dt)
    return EvidenceSpan(
        text=excerpt_text,
        turn_index=turn_index,
        role="user",
        support=support,
        turn_id=turn_id,
        anchor=anchor,
        claim_kind=claim_time.kind,
        event_date=claim_time.event_date.isoformat() if claim_time.event_date else "",
        observed_at=observed_dt.isoformat(),
        source_time=_message_source_time(messages[turn_index]),
    )
