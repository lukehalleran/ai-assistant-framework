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
_RELATION_CUE_RES = {
    "lives_in": _RESIDENCE_CUES,
    "lived_in": _RESIDENCE_CUES,
    "location": _RESIDENCE_CUES,
    "hometown": r"grew\s+up|hometown|born|raised|from|originally",
    "works_at": r"work|works|working|worked|job|hired|employed|employer|start(?:ed|ing)?\s+at|"
                r"internship|intern|shift|shifts|office",
    "works_as": r"work|works|working|worked|job|hired|employed|role|position|title",
    "occupation": r"work|works|working|worked|job|hired|employed|role|position|title|i'm\s+a|i\s+am\s+a",
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
        text = str(message.get("query") or message.get("user") or "").strip()
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


def _claim_spans(text: str) -> Iterator[str]:
    """Yield prose spans, excluding code fences, blockquotes and role dumps."""
    in_fence = False
    for raw_line in (text or "").splitlines() or [text]:
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
        for span in re.split(r"(?<=[.!?])\s+|\s*[;]\s*", line):
            span = span.strip(" \t-*•")
            if not span or _EXTERNAL_ATTRIBUTION_RE.search(span):
                continue
            yield span


def _tokens(value: str) -> set[str]:
    return {
        token.lower() for token in _WORD_RE.findall(value or "")
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _span_is_user_owned(span: str) -> str | None:
    """Return the anchor kind when the span asserts something about the USER.

    ``"first_person"``: an explicit I/we subject, or a first-person possessive
    with no competing third party.  ``"implicit_first_person"``: a chat
    fragment with an implied subject ("Started Lorvatin today").  ``None``: the
    span is about someone else ("Sam told me he moved", "My brother moved",
    "Biscuit is being a silly kitten", "it says you use WHOOP").
    """
    if _FIRST_PERSON_SUBJECT_RE.search(span):
        return "first_person"
    if _RELATIVE_SUBJECT_RE.match(span):
        return None
    if _OTHER_PERSON_RE.search(span) or _POSSESSIVE_PROPER_RE.search(span):
        return None
    if _FIRST_PERSON_ANY_RE.search(span) or _PROFILE_FIELD_RE.search(span):
        return "first_person"
    words = span.split()
    if len(words) > 1:
        first = words[0].strip("\"'“”‘’([")
        if (
            first[:1].isupper()
            and first.lower() not in _SENTENCE_STARTERS
            and _THIRD_PERSON_CONTINUATION_RE.match(words[1])
        ):
            return None
    return "implicit_first_person"


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
    fallback.
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

    best: tuple[int, int, str, str, str, str] | None = None
    for turn_index, text, turn_id in iter_user_messages(messages):
        for span in _claim_spans(text):
            low = span.lower()
            exact_object = bool(object_val and object_val.lower() in low)
            overlap = len(object_tokens & _tokens(span))
            if not exact_object and overlap == 0:
                continue

            if is_user:
                if relation in _PET_OWNERSHIP_SPECIES:
                    if not _pet_ownership_supported(span, relation):
                        continue
                    anchor = "pet_ownership"
                else:
                    anchor = _span_is_user_owned(span)
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
