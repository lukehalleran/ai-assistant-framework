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
  - ``lived_in=Atlanta`` cited the first 200 characters of a long turn
    (song lyrics) instead of the sentence near the end that actually said it.

Doctrine: deliberately conservative.  Missing a marginal fact is safer than
turning quoted or generated text into a durable belief about the user.  A
user-scoped fact needs a user-owned span (explicit first person, or an
implicit-subject chat fragment such as "Started Lorvatin today"), object
grounding, and — where the relation name itself makes a claim (pet species,
residence, employment, preference, relationship) — a matching verb/noun cue.
Entity facts need the named subject and the object in one span.  There is
intentionally NO "last message" fallback.

Leaf module: imports nothing from the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Iterator, Mapping


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
_RELATIVE_SUBJECT_RE = re.compile(
    r"^\s*(?:my|our)\s+(?:brother|sister|mom|mother|dad|father|parents?|friend|buddy|partner|"
    r"girlfriend|boyfriend|wife|husband|ex|boss|coworker|co-worker|roommate|therapist|doctor|"
    r"psychiatrist|advisor|professor|teacher|cousin|aunt|uncle|grandma|grandpa|grandmother|"
    r"grandfather|kid|son|daughter|neighbou?r|manager|landlord|coach|nurse|dentist)\b",
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
# like "Moved to Atlanta in June" has an implied "I").
_RESIDENCE_CUES = (
    r"live|lives|lived|living|reside|resides|residing|moved|moving|move|relocated|relocating|"
    r"based|home|apartment|house|from|staying|stay"
)
_LIKES_CUES = (
    r"like|likes|liked|love|loves|loved|prefer|prefers|enjoy|enjoys|enjoyed|into|fan|favorite|"
    r"favourite|obsessed|adore"
)
# Enrollment is a claim in the relation NAME (2026-09-03): "I dropped CSE 6040
# and kept MGT 6203" token-matched a stale enrolled_in="CSE 6040 and MGT 6203"
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
# `has_doctor = "Rowan is cautious about drinking …"` — a friend — and
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
_RELATION_CUE_RES = {
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


@dataclass(frozen=True)
class EvidenceSpan:
    """A user-authored span supporting one proposed triple."""

    text: str
    turn_index: int
    role: str = "user"
    support: str = "object_match"
    turn_id: str = ""
    anchor: str = ""


def _message_text_and_id(message: Any) -> tuple[str, str] | None:
    """Return only the user-authored portion of a supported message shape."""
    if isinstance(message, Mapping):
        # `user_text` (2026-09-05) is the user's OWN typed text for a turn whose
        # `query` is the merged user-text + attachment blob (corpus entries
        # store the merged form so retrieval renders attachments). Attachment
        # content — lecture transcripts, CSV rows, PDFs — is not user-authored
        # evidence, so the provenance join reads the raw text when present.
        authored = message.get("user_text") if "user_text" in message else (
            message.get("query") or message.get("user") or ""
        )
        text = str(authored or "").strip()
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
# claim: "I'm … enrolled in two courses (CSE 6040 and MGT 6203)" inside a
# pasted Aug-27 email superseded the curated enrolled_in=MGT 6203 fact on
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


def quoted_correspondence_lines(text: str) -> set[int]:
    """Indexes of lines that sit inside a pasted email block (greeting →
    closing → signature run).  Empty set when no complete block exists."""
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


def _claim_spans(text: str) -> Iterator[str]:
    """Yield prose spans, excluding code fences, blockquotes and role dumps."""
    in_fence = False
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
    turn (the lived_in=Atlanta "song lyrics as evidence" defect).
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
    # token is coincidence, not evidence — `has_doctor = "Rowan is cautious
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

    best: tuple[int, int, str, str, str, str] | None = None
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
            elif is_scoped_referent:
                anchor = "scoped_referent"
            else:
                if subject_tokens and not (subject_tokens & _tokens(span)):
                    continue
                anchor = "entity"

            score = (100 if exact_object else 0) + overlap
            support = "exact_object" if exact_object else "object_token_overlap"
            candidate = (score, turn_index, span, turn_id, support, anchor)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ):
                best = candidate

    if best is None:
        return None
    _, turn_index, span, turn_id, support, anchor = best
    return EvidenceSpan(
        text=_cropped_excerpt(span, object_val, excerpt_limit),
        turn_index=turn_index,
        role="user",
        support=support,
        turn_id=turn_id,
        anchor=anchor,
    )
