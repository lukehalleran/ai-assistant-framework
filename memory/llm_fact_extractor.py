"""
# memory/llm_fact_extractor.py

Additive LLM-assisted fact extractor used at shutdown to augment regex/spaCy/REBEL facts.
ENHANCED: Now extracts facts with category classification for user profile building.
ENHANCED (2026-03): Extracts entity facts (non-user subjects) with user_connection metadata.
ENHANCED (2026-04): Accepts existing profile facts so LLM reuses relation names for updates/cancellations.
ENHANCED (2026-07): Prompt explicitly requests entity–entity relations (both subject AND object
are named entities, e.g. "Sam sibling_of Biscuit") — these become lateral knowledge-graph
edges via the shutdown graph-ingestion hook.
ENHANCED (2026-08-05): Core vocabulary gained care-team relations (has_doctor,
doctor_communication) + an explicit capture rule — statements about how the user
reaches (or fails to reach) a doctor/prescriber are durable facts ("contact your
doctor" advice was given assuming a patient portal the prescriber doesn't have).
COVERAGE FIX (2026-08-05): _build_prompt used to include FULL Daemon responses
per entry and iterate oldest→newest with a hard break on the char budget — only
~4-5 pairs of the window ever reached the LLM and the NEWEST turns were dropped
(weeks of "my doctor doesn't respond" never entered a prompt). Selection walks
newest→oldest and defaults rose (LLM_FACTS_LAST_TURNS 12→30, MAX_INPUT_CHARS
6000→9000, MAX_TRIPLES 10→16).
PROVENANCE BOUNDARY (2026-09-02): Daemon responses are EXCLUDED from the prompt
entirely (generated text is not evidence about the user — a response snippet had
let entity facts be sourced from Daemon's own words), and every surviving triple
must join back to a USER-authored span via memory.fact_source
(find_supporting_user_span): quoted/reported text ("it says you use WHOOP"),
pasted transcripts, and relation names the span doesn't corroborate (has_dog
from "playing with Mochi") are dropped; the stored source_excerpt is the
claim-bearing sentence, not the head of the turn, and carries
source_role/source_turn_id/source_support/source_anchor metadata.
LEARNED RELATIONS (2026-08-05): CORE_RELATIONS is a module constant; recurring
invented relations that SURVIVE the gates are tracked by memory.learned_relations
and auto-promoted into the prompt's PREFER list — the vocabulary grows itself
instead of waiting for an owner edit after each observed coverage gap.
ENHANCED (2026-08-02): Prompt carries a RELATION NAMING section — a ~40-relation core
vocabulary plus the rule that specifics belong in the OBJECT, not the relation name
(inflow-side fix for the single-use-relation explosion; pairs with
entity_resolver.normalize_relation canonicalization on the read/ingest side).

Contract
- Inputs: list of recent messages — either plain strings (user-only) or dicts with
  at least {"query": str}. Assistant/Daemon responses are never extraction input.
- Behavior: calls a compact LLM prompt to extract SRO triples as strict JSON with category metadata
  - When existing_facts provided, prompt instructs LLM to reuse relation names for updates;
    explicit-only guard prevents inferring updates from absence of mentions
  - User facts: subject="user" for personal facts (pronouns normalized)
  - Entity facts: subject=entity name for people/places/orgs/pets stated by the user
    Entity facts include user_connection field (e.g., "user's boss", "user's mom's cat")
- Output: list of dict triples (subject, relation, object, value, category, confidence, fact_scope, source_excerpt)
  - fact_scope: "user" or "entity" — indicates whether fact is about the user or a third-party entity
  - source_excerpt: compact user-authored supporting span (200-char cap), with
    source_role/source_turn_index metadata. Unsupported proposals are dropped.
- Junk/polarity guards (2026-08-03): _normalize_triple applies
  fact_extractor._is_junk_object (adverbial/temporal/negation-fragment objects dropped,
  schedule relations exempt), and _attach_source_excerpts applies
  fact_extractor._polarity_conflict against the full matched source message
  (positive-preference triples the source negates are dropped) — the LLM path
  previously had neither guard, so junk/inverted facts entered the profile through it.
- Long-object cap + salvage (2026-08-26): objects over FACT_OBJECT_MAX_CHARS (300)
  either yield their salutation recipient (fact_extractor._salvage_long_object —
  "Hi Morgan, …700 chars…" -> "Morgan") or drop; this path previously had NO length cap.
- Stance tagging (2026-08-23): the prompt asks for a "stance" field
  (objective/appraisal/reported) with an explicit value-judgment rule; the
  DETERMINISTIC classifier (memory/stance_classifier.classify_triple_stance)
  OVERRIDES the LLM's tag on lexicon hits — the LLM only fills lexicon gaps.
  Evaluative pronoun/role subjects ("she is abusive") are re-scoped via
  scope_unresolved_referent to user-owned strings, never fuzzy-bound to a
  named entity. Output triples carry "stance"; gen max_tokens 900→1100.
"""

from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import json
import re
import os

from utils.logging_utils import get_logger
from memory.user_profile_schema import ProfileCategory, categorize_relation
from collections import defaultdict
from utils.ordered_slice import newest_first as _ordered_newest_first

logger = get_logger("llm_facts")


def _snake(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9 _-]", " ", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower().strip("_-")


def _is_ephemeral_relation(rel: str) -> bool:
    """Check if a relation is ephemeral and should not be stored as a durable fact."""
    try:
        from config.app_config import PROFILE_EPHEMERAL_RELATIONS
        ephemeral = frozenset(r.lower().strip() for r in PROFILE_EPHEMERAL_RELATIONS)
    except ImportError:
        ephemeral = frozenset()
    return rel.lower().strip() in ephemeral


def _is_boolean_noise(obj: str) -> bool:
    """Reject facts where the object is just 'true'/'false'/'yes'/'no'."""
    return obj.strip().lower() in {"true", "false", "yes", "no"}


def _normalize_triple(t: Dict[str, Any]) -> Dict[str, str] | None:
    """Normalize and categorize extracted triple."""
    # Coerce to strings defensively; models sometimes emit numbers/bools
    subj = str(t.get("subject") or "").strip()
    rel = str(t.get("relation") or "").strip()
    obj = str(t.get("object") or t.get("value") or "").strip()

    if not subj or not rel or not obj:
        return None

    # Pronouns → user (keep named entities as-is)
    if subj.lower() in {"i", "me", "my", "we", "us", "you", "user"}:
        subj = "user"

    # Relation snake_case
    rel = _snake(rel)

    # Block ephemeral predicates — transient state, not durable facts
    if _is_ephemeral_relation(rel):
        logger.debug(f"[LLM Facts] Blocked ephemeral relation: {subj}|{rel}|{obj}")
        return None

    # Block boolean-only values — no informational content
    if _is_boolean_noise(obj):
        logger.debug(f"[LLM Facts] Blocked boolean noise: {subj}|{rel}|{obj}")
        return None

    # Block adverbial/temporal/negation-fragment objects ("for a bit",
    # "yesterday", "not good") — same guard the regex extractor applies in
    # _clean_triple. Until 2026-08-03 the LLM path had no junk-object check,
    # so junk like `dad_show_up=for a bit` entered the profile through here
    # and later showed up only as "skipped as duplicate" of itself.
    from memory.fact_extractor import (
        _is_junk_object,
        _is_repo_audit_junk,
        _fact_object_max_chars,
        _salvage_long_object,
    )
    if _is_junk_object(obj, rel):
        logger.debug(f"[LLM Facts] Blocked junk object: {subj}|{rel}|{obj}")
        return None

    # Transient git/repo machine-state from an audit query (2026-09-02) — same
    # guard the regex extractor applies in _clean_triple.
    if _is_repo_audit_junk(subj, rel, obj):
        logger.debug(f"[LLM Facts] Blocked repo-audit junk: {subj}|{rel}|{obj}")
        return None

    # Long-object cap + salutation salvage (2026-08-26). This path had NO
    # length cap at all — a pasted email became the OBJECT of
    # `user | email_sent | "Hi Morgan, …700 chars…"` (2026-06-15), burying the
    # recipient where no embedding or entity resolution could see it. Recover
    # the recipient when a salutation names one; otherwise drop, same as the
    # regex path's cap.
    if len(obj) > _fact_object_max_chars():
        salvaged = _salvage_long_object(obj)
        if not salvaged:
            logger.debug(
                f"[LLM Facts] Blocked over-long object ({len(obj)} chars, "
                f"no salvageable recipient): {subj}|{rel}|{obj[:60]}…"
            )
            return None
        logger.info(
            f"[LLM Facts] Salvaged recipient {salvaged!r} from "
            f"{len(obj)}-char {rel} object"
        )
        obj = salvaged

    # Stance (2026-08-23): read the LLM's tolerantly, but the DETERMINISTIC
    # classifier overrides on lexicon hits — a thick evaluative object
    # (casey|is|evil) is an appraisal no matter what the model tagged. The LLM
    # stance only fills the gaps the lexicon can't see. Unresolved evaluative
    # referents ("she is abusive") are re-scoped to a user-owned subject and
    # NEVER fuzzy-bound to a named entity.
    from memory.stance_classifier import (
        VALID_STANCES,
        classify_triple_stance,
        scope_unresolved_referent,
    )
    scoped_subj = scope_unresolved_referent(subj, obj)
    if scoped_subj:
        logger.info(f"[LLM Facts] Scoped unresolved referent: {subj} -> {scoped_subj}")
        subj = scoped_subj
    _det = classify_triple_stance(subj, rel, obj)
    _llm_stance = str(t.get("stance") or "").strip().lower()
    if _det.stance != "objective":
        stance = _det.stance
    elif _llm_stance in VALID_STANCES:
        stance = _llm_stance
    else:
        stance = _det.stance

    # Auto-categorize
    category = categorize_relation(rel)

    # Get confidence if provided
    confidence = float(t.get("confidence", 0.7))

    # Determine fact scope
    is_user = (subj.lower() == "user")
    fact_scope = "user" if is_user else "entity"

    result = {
        "subject": subj.lower(),
        "relation": rel,
        "object": obj.strip().strip(". "),
        "value": obj.strip().strip(". "),  # Alias for compatibility
        "category": category.value,
        "confidence": confidence,
        "fact_scope": fact_scope,
        "stance": stance,
    }

    # Forward user_connection if the LLM provided one
    user_connection = str(t.get("user_connection") or "").strip()
    if user_connection:
        result["user_connection"] = user_connection

    return result


# The fixed core relation vocabulary shown to the LLM (PREFER list). Single
# source of truth — also consumed by memory.learned_relations to decide which
# invented relations are worth tracking for automatic promotion.
CORE_RELATIONS = (
    "name", "age", "lives_in", "works_at", "works_on", "occupation", "studies",
    "studies_at", "completed", "attends", "likes", "dislikes", "wants_to",
    "owns", "has_cat", "has_dog", "drinks", "eats", "plays", "uses",
    "interested_in", "skilled_at", "concerned_about", "talked_about",
    "friend_of", "sibling_of", "parent_of", "child_of", "spouse_of",
    "works_with", "medication_name", "medication_dose", "condition", "symptom",
    "sleep_quality", "gym_schedule", "work_schedule", "class_schedule",
    "exam_date", "goal", "deadline", "has_doctor", "doctor_communication",
)

_PET_RELATION_SOURCE_RE = {
    "has_cat": re.compile(r"\b(?:cat|cats|kitten|kittens|feline|felines)\b", re.I),
    "has_dog": re.compile(r"\b(?:dog|dogs|puppy|puppies|canine|canines)\b", re.I),
}


def _pet_relation_supported_by_source(source: str, relation: str) -> bool:
    """Species-specific pet ownership relations need source support.

    A live extractor output produced user|has_dog|Biscuit and |Daisy even
    though the matched excerpt only talked about cats. The relation name is
    the claim here, so the source must contain the matching species word.
    """
    rel = (relation or "").strip().lower()
    rx = _PET_RELATION_SOURCE_RE.get(rel)
    if not rx:
        return True
    return bool(rx.search(source or ""))


SHARED_CONTENT_TYPES = frozenset({"lyrics", "poem", "quote"})
SHARED_CONTENT_MIN_CONF = float(os.getenv("FACT_EXTRACT_SKIP_CONTENT_MIN_CONF", "0.7"))


def _entry_is_shared_content(entry: str) -> bool:
    """True when a user turn is a lyrics/poem/quote paste per THE deployed
    content_type_detector (confidence >= SHARED_CONTENT_MIN_CONF)."""
    try:
        from core.content_type_detector import detect_content_type
        text = re.sub(r"^user\s*:\s*", "", entry or "", flags=re.I)
        ct = detect_content_type(text)
        return bool(
            ct.content_type in SHARED_CONTENT_TYPES
            and float(ct.confidence or 0.0) >= SHARED_CONTENT_MIN_CONF
        )
    except Exception:
        return False


class LLMFactExtractor:
    def __init__(self, model_manager, *,
                 model_alias: str = None,
                 max_input_chars: int = 9000,
                 max_triples: int = 15):  # Increased from 10
        self.mm = model_manager
        self.model_alias = model_alias or "gpt-4o-mini"
        self.max_input_chars = int(max_input_chars)
        self.max_triples = int(max_triples)

    @staticmethod
    def _build_alias_hints(existing_facts: List[Dict[str, Any]]) -> str:
        """Build compact alias hints from SAFE_RELATION_ALIASES for relations the user has."""
        try:
            from memory.user_profile_schema import SAFE_RELATION_ALIASES
        except ImportError:
            return ""
        # Invert: canonical → [variants]
        canonical_to_variants = defaultdict(set)
        for variant, canonical in SAFE_RELATION_ALIASES.items():
            canonical_to_variants[canonical].add(variant)
        # Only show aliases where user has a matching fact
        user_rels = {f.get("relation", "") for f in (existing_facts or [])}
        lines = []
        for canonical, variants in sorted(canonical_to_variants.items()):
            if canonical in user_rels or variants & user_rels:
                bad = ", ".join(sorted(variants - {canonical}))
                if bad:
                    lines.append(f"- {canonical}, NOT: {bad}")
        if not lines:
            return ""
        return (
            "\nRELATION ALIASES — always use canonical form:\n"
            + "\n".join(lines[:15]) + "\n"  # cap at 15 to limit token cost
        )

    def _build_prompt(self, user_messages: List, existing_facts: List[Dict[str, Any]] = None) -> str:
        # Coverage fix (2026-08-05): the old loop iterated oldest→newest with
        # FULL Daemon responses in each entry and `break`ed on budget — a
        # ~1K-char response per pair meant ~4-5 pairs of the 12-turn window
        # made it in, and the NEWEST turns were the ones dropped. Weeks of
        # care-team facts ("my doctor doesn't care... they aren't even open")
        # never reached the extractor. Selection walks NEWEST→oldest so budget
        # exhaustion drops the oldest turns instead. 2026-09-02: responses are
        # EXCLUDED entirely — generated text is not evidence about the user
        # (memory.fact_source enforces the same boundary on the output side).
        entries = []  # each: (rendered_text, ts_or_None)
        for m in user_messages[-50:]:  # hard cap safety
            ts_val = None
            if isinstance(m, dict):
                # Assistant responses are generated text, not source evidence.
                # `user_text` (2026-09-05) is the user's own typed text on an
                # attachment turn — the merged `query` also carries the
                # attachment content, which is not the user's words.
                q = (m.get("user_text") or m.get("query") or "").strip()
                if not q:
                    continue
                entry = f"User: {q}"
                ts_val = m.get("turn_id") or m.get("timestamp")
            else:
                entry = (m or "").strip()
                if not entry:
                    continue
                if re.match(r"^(?:assistant|daemon|system|tool)\s*:\s*", entry, re.I):
                    continue
                entry = "User: " + re.sub(r"^user\s*:\s*", "", entry, flags=re.I)
            # Shared content (lyrics/poems/quotes) is not evidence about the
            # user even in first person (2026-09-03: lived_in=Atlanta and a
            # partner name were mined from pasted song lyrics on 08-29).
            if _entry_is_shared_content(entry):
                continue
            entries.append((entry, ts_val))

        # 2026-09-04: when every entry carries a per-item timestamp (the
        # shutdown_processor caller stamps conv_pairs with turn_id), sort
        # defensively via utils.ordered_slice — single source of truth for
        # "sort by timestamp before slicing" — instead of trusting the
        # caller's positional order. When timestamps are absent (e.g.
        # gui/wizard.py's bare-string onboarding call), fall back to the
        # ORIGINAL positional contract (entries arrive oldest-first, so
        # reversed() walks newest-first) rather than the generic per-item
        # None-handling, which would treat every entry as tied-"oldest" and
        # silently walk oldest-first here — the exact 2026-08-05 regression
        # (newest turns dropped by the budget break) this code was fixed to
        # avoid; pinned by
        # test_learned_relations.py::TestCoveragePromptBuild.
        if entries and all(ts is not None for _, ts in entries):
            ordered = _ordered_newest_first(entries, lambda pair: pair[1])
        else:
            ordered = list(reversed(entries))

        msgs = []
        total = 0
        for entry, _ts in ordered:  # newest first — oldest drop off
            if total + len(entry) + 10 > self.max_input_chars:
                break
            msgs.append(entry)
            total += len(entry) + 10
        msgs.reverse()  # render chronologically

        joined = "\n".join(f"- {m}" for m in msgs)

        # Build existing facts section if provided
        existing_facts_section = ""
        if existing_facts:
            lines = []
            for f in existing_facts[:60]:  # cap to avoid prompt bloat
                rel = f.get("relation", "")
                val = f.get("value", "")
                cat = f.get("category", "")
                if rel and val:
                    lines.append(f"  - {rel}={val} [{cat}]")
            if lines:
                existing_facts_section = (
                    "\n\nEXISTING PROFILE FACTS (current beliefs about the user):\n"
                    + "\n".join(lines)
                    + "\n\nUPDATE RULES:\n"
                    "- When the user's messages UPDATE, CANCEL, RESCHEDULE, or CHANGE something "
                    "already in the existing facts, you MUST reuse the SAME relation name with "
                    "the new value. This ensures the old fact gets properly superseded.\n"
                    "- Example: existing fact is date_planned=date in Algonquin at noon on Sat. "
                    "User says 'date got cancelled'. Output: "
                    '{{"subject": "user", "relation": "date_planned", "object": "cancelled — was Algonquin Sat, needs reschedule", "category": "goals", "confidence": 0.95}}\n'
                    "- Example: existing fact is gym_schedule=MWF mornings. "
                    "User says 'switching to evenings'. Output: "
                    '{{"subject": "user", "relation": "gym_schedule", "object": "MWF evenings", "category": "fitness", "confidence": 0.9}}\n'
                    "- Do NOT invent a new relation name when an existing one covers the same topic.\n"
                    "- Do NOT infer updates from absence. If the user does not mention a topic, "
                    "leave the existing fact unchanged — do not extract a contradicting fact.\n"
                    "- Only extract an update when the user EXPLICITLY states a change "
                    '(e.g., "I got a new job", "we broke up", "I moved to Boston").\n'
                )
                # Add relation alias hints from SAFE_RELATION_ALIASES
                alias_hints = self._build_alias_hints(existing_facts)
                if alias_hints:
                    existing_facts_section += alias_hints

        # Enhanced extraction prompt with category awareness and few-shot examples
        prompt_template = """You extract factual information about the user from conversation messages.

CATEGORIES to classify facts into:
- identity: name, age, birthday, location, timezone
- education: degrees, schools, courses, GPA, certifications
- career: jobs, roles, skills, companies, experience
- projects: personal projects, tech stack, status
- health: conditions, medications, supplements, symptoms, sleep
- fitness: lifts, body metrics, workout routines, goals
- preferences: likes, dislikes, favorites, formatting preferences
- hobbies: games, interests, pets, social activities
- study: exams, study hours, strategies, weak points
- finance: insurance, budgeting, financial plans
- relationships: family, friends, colleagues
- goals: plans, intentions, aspirations

OUTPUT FORMAT (strict JSON array):
[
  {{"subject": "user", "relation": "snake_case_relation", "object": "value", "category": "category_name", "confidence": 0.0-1.0, "stance": "objective|appraisal|reported"}}
]

STANCE (epistemic tag — how the statement relates to reality):
- "objective": a checkable world-fact ("lives in Chicago", "takes 20mg").
- "appraisal": a VALUE JUDGMENT about a person or thing — INCLUDING the user
  about themselves ("my ex is evil", "I'm a failure", "my boss is toxic").
  An appraisal is the user's take at the time, never a world-fact.
- "reported": the user relaying someone else's claim ("my mom says I
  overreact").
- Example: "Casey was evil to me" →
  {{"subject": "Casey", "relation": "is", "object": "evil", "category": "relationships", "confidence": 0.8, "stance": "appraisal"}}

RELATION NAMING (keeps the knowledge graph queryable — 2026-08-02: 630 of 696
stored relations were single-use inventions):
- PREFER these core relations whenever one fits: {core_relations_text}.{learned_relations_line}
- CARE TEAM: statements about how the user reaches (or fails to reach) a
  doctor/prescriber/therapist are durable, actionable facts — capture them.
  "My psych never answers messages and there's no portal" →
  {{"relation": "doctor_communication", "object": "prescriber has no patient
  portal and rarely responds to messages", "category": "health"}}.
- The SPECIFICS belong in the OBJECT, not the relation name. Write
  {{"relation": "drinks", "object": "diet coke"}} — NOT
  {{"relation": "drink_preference", "object": "diet coke"}}.
- Only invent a new relation name when no core relation and no existing
  profile relation fits, and keep it a short generic verb or noun.

EXAMPLES:
Input: "I'm Sarah, a software developer from Seattle"
Output: [
  {{"subject": "user", "relation": "name", "object": "Sarah", "category": "identity", "confidence": 0.95}},
  {{"subject": "user", "relation": "occupation", "object": "software developer", "category": "career", "confidence": 0.95}},
  {{"subject": "user", "relation": "lives_in", "object": "Seattle", "category": "identity", "confidence": 0.95}}
]

Input: "I'm a data analyst; I built this app and I'm testing it now"
Output: [
  {{"subject": "user", "relation": "occupation", "object": "data analyst", "category": "career", "confidence": 0.95}},
  {{"subject": "user", "relation": "works_on", "object": "this app", "category": "projects", "confidence": 0.8}}
]
(Building or testing something is an ACTIVITY, not a title — no "role" fact.)

Input: "My name is Alex, I created you"
Output: [
  {{"subject": "user", "relation": "name", "object": "Alex", "category": "identity", "confidence": 0.95}},
  {{"subject": "user", "relation": "works_on", "object": "this assistant", "category": "projects", "confidence": 0.9}}
]

ENTITY FACTS (in addition to user facts):
- Extract facts about people, pets, places, and topics explicitly stated in USER messages
- Subject should be the entity name (NOT "user")
- Add "user_connection" field explaining relation to user (if known)
- Only extract when clearly stated, not hypothetical
- Pay special attention to pets, family members, and recurring people — these are high-value entities
- Example: "My boss Jordan moved from London"
  → {{"subject": "Jordan", "relation": "moved_from", "object": "London", "category": "relationships", "confidence": 0.75, "user_connection": "user's boss"}}
- Never treat Daemon/model/assistant text, quoted text, pasted transcripts, or a
  user's report of what another model claimed as evidence about the user.

ENTITY-ENTITY RELATIONS (high value — do not skip these):
- When a statement relates TWO named entities to each other, extract that link as its own
  fact with the entity NAME as the object — not a description.
- Example: "Sam is Biscuit's brother"
  → {{"subject": "Sam", "relation": "sibling_of", "object": "Biscuit", "category": "hobbies", "confidence": 0.9, "user_connection": "user's cat"}}
- Example: "My professor Dr. Smith teaches Bayesian Statistics"
  → {{"subject": "Dr. Smith", "relation": "teaches", "object": "Bayesian Statistics", "category": "education", "confidence": 0.9, "user_connection": "user's professor"}}
- Example: "Jordan works at Deloitte"
  → {{"subject": "Jordan", "relation": "works_at", "object": "Deloitte", "category": "career", "confidence": 0.9, "user_connection": "user's boss"}}
- If a sentence supports BOTH a user fact and an entity-entity fact, output both.

RULES:
- Subject is "user" for personal facts, or the entity name for entity facts
- Relation must be snake_case (e.g., "squat_max", "lives_in", "favorite_beer")
- Extract DURABLE facts — things that are true across sessions, not just right now
- Confidence: 0.9+ for direct statements, 0.7-0.8 for inferred, <0.7 for uncertain
- Do NOT extract questions or hypotheticals
- IMPORTANT: If the user introduces themselves, extract identity facts. A job title the user STATES ("I'm a nurse", "I work as a data analyst") is `occupation`; what they are building or doing is `works_on`. Never derive a role or title from an activity ("testing the app", "monitoring myself").
- TEMPORAL: Today's date is {today}. When the user mentions relative dates ("tomorrow", "next Monday", "the following day"), resolve them to absolute dates in the object field. Example: "I work tomorrow" on 2026-03-12 → object: "work on Thu 2026-03-13"
- Do NOT extract transient/ephemeral state that changes constantly:
  * current_activity, current_mood, current_feeling, feeling, feels
  * woke_at, walked_to, showered, tidied, will_drive_to, greeting
  * Generic "is", "was", "has", "thinks", "plans", "wants", "needs" predicates
  * Boolean-value facts like "showered=true" or "has_energy=true"
  * One-time actions: "asked_about", "tested", "walked_to"
  These pollute the fact store. Only extract facts that remain true over time.
{existing_facts}
USER MESSAGES (newest last):
{messages}

JSON:"""
        today_str = datetime.now().strftime("%A, %Y-%m-%d")
        # Learned relations: recurring non-core relations promoted from this
        # user's own history (memory.learned_relations) join the PREFER list
        # automatically — the vocabulary grows itself instead of waiting for
        # an owner edit after each observed coverage gap (2026-08-05).
        learned_line = ""
        try:
            from memory.learned_relations import get_learned_relation_store
            promoted = get_learned_relation_store().promoted()
            if promoted:
                learned_line = (
                    "\n- ALSO treat these LEARNED relations (recurring for this "
                    "user) as core: " + ", ".join(promoted) + "."
                )
        except Exception as e:
            logger.debug(f"[LLM Facts] learned relations unavailable: {e}")
        prompt = prompt_template.format(
            messages=joined,
            today=today_str,
            existing_facts=existing_facts_section,
            core_relations_text=", ".join(CORE_RELATIONS),
            learned_relations_line=learned_line,
        )

        # Always log the messages being processed (helps debug empty extractions)
        logger.info(f"[LLM Facts] Building prompt with {len(msgs)} messages, total chars={len(prompt)}")
        logger.info(f"[LLM Facts] Input messages: {joined[:300]}{'...' if len(joined) > 300 else ''}")
        if existing_facts:
            logger.info(f"[LLM Facts] Injected {len(existing_facts)} existing profile facts for relation reuse")

        return prompt

    async def extract_triples(self, user_messages: List,
                              existing_facts: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        if not user_messages:
            return []

        prompt = self._build_prompt(user_messages, existing_facts=existing_facts)

        try:
            text = await self.mm.generate_once(
                prompt=prompt,
                model_name=self.model_alias,
                system_prompt="You output only strict JSON arrays. No prose, no explanation.",
                max_tokens=1100,  # 2026-08-05: 600 capped ~10-14 triples; 2026-08-23: +stance field per triple
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as e:
            logger.warning(f"[LLM Facts] generate_once failed: {e}")
            return []

        if not isinstance(text, str) or not text.strip():
            logger.warning("[LLM Facts] generate_once returned empty or non-string response")
            return []

        # Log the raw response for debugging
        logger.info(f"[LLM Facts] Raw LLM response: {text[:500]}")

        # Attempt to parse a JSON array (robust to leading/trailing junk)
        raw = text.strip()

        # Check for stub response indicating API client issue
        if raw.startswith("[API unavailable]"):
            logger.warning(f"[LLM Facts] API client not available - got stub response: {raw[:100]}")
            return []

        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
            else:
                logger.warning(f"[LLM Facts] No JSON array found in response: {text[:200]}")
                return []
            data = json.loads(raw)
            if not isinstance(data, list):
                logger.warning(f"[LLM Facts] JSON parsed but not a list: {type(data)}")
                return []
        except Exception as e:
            logger.warning(f"[LLM Facts] JSON parse failed: {e} - Response was: {text[:200]}")
            return []

        triples: List[Dict[str, str]] = []
        seen = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            norm = _normalize_triple(item)
            if not norm:
                continue
            key = (norm["subject"], norm["relation"], norm["object"])
            if key in seen:
                continue
            seen.add(key)
            triples.append(norm)
            # Track surviving non-core relations for automatic vocabulary
            # promotion — only triples that PASSED normalization/junk gates
            # ever count (junk-relation floodgate guard).
            if norm.get("relation") not in CORE_RELATIONS:
                try:
                    from memory.learned_relations import get_learned_relation_store
                    get_learned_relation_store().record(
                        norm["relation"], norm.get("object", "")
                    )
                except Exception as e:
                    logger.debug(f"[LLM Facts] learned-relation record failed: {e}")
            if len(triples) >= self.max_triples:
                break

        # Attach source_excerpt by matching each triple back to its source message
        if triples:
            self._attach_source_excerpts(triples, user_messages)
            logger.info(f"[LLM Facts] SUCCESS: extracted={len(triples)} facts using model={self.model_alias}")
            for t in triples:
                logger.info(f"[LLM Facts]   -> {t['relation']}: {t['object']} (conf={t['confidence']})")
        else:
            logger.warning(f"[LLM Facts] WARNING: extracted=0 facts from LLM response. Model returned empty array.")

        return triples

    @staticmethod
    def _attach_source_excerpts(triples: List[Dict[str, str]],
                                user_messages: List) -> None:
        """Attach direct user evidence and drop unsupported proposals."""
        if not user_messages:
            triples[:] = []
            return
        from memory.fact_source import find_supporting_user_span
        from memory.fact_extractor import _polarity_conflict
        kept: List[Dict[str, str]] = []
        for triple in triples:
            evidence = find_supporting_user_span(triple, user_messages, excerpt_limit=200)
            if evidence is None:
                logger.info(
                    f"[LLM Facts] Blocked unsupported proposal: "
                    f"{triple.get('subject')}|{triple.get('relation')}|{triple.get('object')}"
                )
                continue
            if _polarity_conflict(evidence.text, triple.get("relation", ""), triple.get("object", "")):
                logger.info(
                    f"[LLM Facts] Blocked polarity conflict: {triple.get('relation')}|"
                    f"{triple.get('object')} vs source: {evidence.text[:80]!r}"
                )
                continue
            if not _pet_relation_supported_by_source(evidence.text, triple.get("relation", "")):
                logger.info(
                    f"[LLM Facts] Blocked unsupported pet species relation: "
                    f"{triple.get('relation')}|{triple.get('object')} vs source: {evidence.text[:80]!r}"
                )
                continue
            triple["source_excerpt"] = evidence.text
            triple["source_role"] = evidence.role
            triple["source_turn_index"] = evidence.turn_index
            triple["source_support"] = evidence.support
            triple["source_anchor"] = evidence.anchor
            if evidence.turn_id:
                triple["source_turn_id"] = evidence.turn_id
            kept.append(triple)
        triples[:] = kept
