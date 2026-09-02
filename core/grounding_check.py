"""
# core/grounding_check.py

Module Contract
- Purpose: Factual-grounding floor (2026-08-28). Catches the class where the
  assistant asserts or endorses, IN ITS OWN VOICE, a confirmably-false factual
  claim — the live failure was endorsing the discredited "refrigerator mother"
  autism theory as "lands closer to truth" during GROUNDING PRESENCE, on a
  tone where the plan/review gate never runs (should_plan skips CONCERN+).
- Two layers:
  1. GROUNDING_ACCURACY_CLAUSE — prompt-side accuracy floor appended to the
     five emotional/brevity instruction blocks (tone_instructions.py +
     escalation_tracker.py import it; this module imports nothing from core/
     so there is no circularity).
  2. has_checkable_claims() (deterministic pre-filter, word-boundary
     discipline, under-fires by design) → verify_grounding() (small guarded
     LLM call, fail-open) → build_grounding_correction() (visible suffix,
     action-guard idiom, gentler wording on elevated tones).
- Key functions:
  - has_checkable_claims(response_text, query="") -> bool
  - verify_grounding(query, response, model_manager, *, model_name, ...)
      -> Optional[GroundingVerdict]   (None = fail-open, take no action)
  - build_grounding_correction(correction, *, elevated=False) -> str
- Dependencies: pydantic, model_manager.generate_once (passed in). Stateless.
- Wiring: gui/handlers._apply_grounding_check (enhanced + agentic paths).
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from utils.logging_utils import get_logger

logger = get_logger("grounding_check")


# ---------------------------------------------------------------------------
# Layer 1: prompt-side accuracy floor (single source for all five blocks)
# ---------------------------------------------------------------------------

GROUNDING_ACCURACY_CLAUSE = (
    "- ACCURACY FLOOR: brevity constrains length, never accuracy. Acknowledge "
    "feelings without endorsing factual or causal claims embedded in them.\n"
    "- If the user's premise contains a factual error (a discredited theory, a "
    "false causal claim), do NOT validate it — omit it, or correct it gently "
    "in one clause.\n"
    "- Never present a discredited or contested theory as true or \"closer to "
    "the truth\". Never present an appraisal of a person as established fact "
    "— render it as their experience (\"you described X as ...\"), never as "
    "\"X is ...\".\n"
    "- When recapping the user's day or accomplishments, include ONLY events "
    "they actually reported — never assume a planned task (a call, an errand) "
    "was completed unless they said so."
)


# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------

class GroundingVerdict(BaseModel):
    """LLM verifier output for one response's factual-grounding check."""

    false_claim_present: bool = False
    claim: str = ""
    why_false: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    correction: str = ""


# ---------------------------------------------------------------------------
# Layer 2a: deterministic claim-shape pre-filter
# ---------------------------------------------------------------------------
# Word-boundary discipline throughout (the 08-28 'solve' ⊂ "resolution"
# lesson): every alternative is anchored with \b — no bare substring matching.
# The filter UNDER-fires by design: it only decides whether the cheap LLM
# verifier is worth calling, so a miss costs coverage while an over-fire
# costs one small API call.

_STANCE_RE = re.compile(
    r"(?:"
    r"\bland(?:s|ed)?\s+closer\s+to\b"
    r"|\bcloser\s+to\s+(?:the\s+)?truth\b"
    r"|\bis(?:n'?t|\s+not)?\s+real\b"
    r"|\bnot\s+real\s+in\b"
    r"|\b(?:discredited|debunked|disproven|disproved|pseudoscien\w+)\b"
    r"|\bsettled\s+(?:fact|science)\b"
    # Up to four intervening words between subject and verb — "research on
    # this stuff consistently shows" missed the adjacent-only form (08-28).
    r"|\b(?:studies|research|the\s+science|the\s+data)(?:\s+\w+){0,4}?\s+"
    r"(?:show|shows|prove|proves|say|says|suggest|suggests|indicate|indicates|demonstrate|demonstrates)\b"
    r"|\bproven\s+(?:that|to|wrong|right|false|true)\b"
    r"|\bturn(?:s|ed)?\s+out\s+(?:to\s+be\s+)?(?:true|false|right|wrong)\b"
    r"|\bmyth\b"
    r"|\btruth\s+to\b"
    r"|\bonto\s+something\b"
    r")",
    re.IGNORECASE,
)

_CAUSAL_RE = re.compile(
    r"(?:"
    r"\bcauses?\b|\bcaused\s+by\b|\bleads?\s+to\b"
    r"|\bresults?\s+(?:from|in)\b|\bstems?\s+from\b"
    r"|\b(?:is|are|was|were)\s+(?:due\s+to|linked\s+to|responsible\s+for)\b"
    r"|\broot\s+cause\b"
    r")",
    re.IGNORECASE,
)

# Subject terms that make a causal sentence CHECKABLE (a named medical /
# scientific object). Deliberately narrow: bare emotion nouns ("anxiety",
# "burnout") are excluded or every support turn would fire.
_SUBJECT_RE = re.compile(
    r"(?:"
    r"\btheor(?:y|ies)\b|\bhypothes[ei]s\b|\bsyndrome\b|\bdisorder\b"
    r"|\bdiagnos\w+\b|\bvaccin\w+\b|\bmedications?\b|\billness(?:es)?\b"
    r"|\bdiseases?\b|\bgene(?:s|tics?)?\b|\bchemical\s+imbalance\b"
    r"|\bautis\w+\b|\brefrigerator\s+mother\b"
    r")",
    re.IGNORECASE,
)

# Years and percentages — rare in support talk, cheap to verify.
_NUMERIC_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}\b|\b\d{1,3}(?:\.\d+)?\s?(?:%|percent\b))",
    re.IGNORECASE,
)

# Agreement shapes in the RESPONSE — combined with claim shapes in the QUERY
# this catches endorsement of a false user premise the response never restates
# ("you're actually right" after "autism is not real in a lot of places").
_AGREEMENT_RE = re.compile(
    r"(?:"
    r"(?:^|\n)\s*(?:yeah|yep|yes|right|true|exactly|honestly|fair)\b"
    r"|\byou'?re\s+(?:right|not\s+wrong)\b"
    r"|\bthat'?s\s+(?:true|right|fair)\b"
    r"|\bgood\s+point\b"
    r")",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")


def _causal_claim_present(text: str) -> bool:
    """A causal pattern AND a checkable subject term in the SAME sentence."""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if _CAUSAL_RE.search(sentence) and _SUBJECT_RE.search(sentence):
            return True
    return False


def has_checkable_claims(response_text: str, query: str = "") -> bool:
    """Deterministic pre-filter: does this response contain claim-shaped
    content worth spending a verifier call on?

    Fires on: (1) truth-stance language in the response; (2) a causal
    assertion about a named medical/scientific subject; (3) years or
    percentages; (4) an agreement opener in the response when the QUERY
    carries a claim shape (premise-endorsement case). Pure-presence text
    ("I hear you. This is really hard.") never fires.
    """
    if not response_text or not response_text.strip():
        return False
    if _STANCE_RE.search(response_text):
        return True
    if _causal_claim_present(response_text):
        return True
    if _NUMERIC_RE.search(response_text):
        return True
    if query and _AGREEMENT_RE.search(response_text):
        if _STANCE_RE.search(query) or _causal_claim_present(query):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 2b: LLM verifier (fail-open)
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM_PROMPT = (
    "You are a strict factual-claims auditor. Output only valid JSON."
)

_RESPONSE_TRUNC = 1200
# Long queries are usually PASTED SOURCE MATERIAL (a syllabus, an article) that
# the response's dates/numbers came from. At the old 500-char cap the verifier
# never saw the source and flagged a correct due date as unverifiable
# (live 2026-08-29 MGT-6203 turn). Head+tail slices keep the tables that tend
# to sit mid/end of a paste.
_QUERY_LONG_HEAD = 2500
_QUERY_LONG_TAIL = 2500


def _truncate_query(query: str) -> str:
    q = query or ""
    if len(q) <= _QUERY_LONG_HEAD + _QUERY_LONG_TAIL:
        return q
    return (
        q[:_QUERY_LONG_HEAD]
        + "\n[... middle of pasted material snipped ...]\n"
        + q[-_QUERY_LONG_TAIL:]
    )


# On agentic turns the response's facts come from RETRIEVED documents (tool
# results), not the query — the verifier never saw them and flagged a correct
# "Fall 2026" semester against its own training-data prior (live 2026-08-29,
# conf 0.9). Retrieved material is passed as a source block and treated
# exactly like user-pasted material: authoritative for document-specific
# facts.
# Sized to the handlers-side 6000-char collect cap (audit F23 2026-08-31:
# 3500 silently discarded the tail of what handlers gathered). Truncation is
# a head-slice, so the [AUTHORITATIVE RUNTIME CLOCK] block handlers prepends
# FIRST always survives.
_SOURCE_MATERIAL_TRUNC = 6000


def _build_verifier_prompt(query: str, response: str,
                           source_material: str = "") -> str:
    q = _truncate_query(query)
    r = (response or "")[:_RESPONSE_TRUNC]
    src = (source_material or "").strip()[:_SOURCE_MATERIAL_TRUNC]
    src_block = (
        "Source material the assistant retrieved while answering (treat as "
        "AUTHORITATIVE for document-specific facts — dates, deadlines, "
        "policies, names, numbers; claims consistent with it are grounded, "
        "even if you cannot independently confirm them):\n"
        f"\"{src}\"\n\n"
    ) if src else ""
    return (
        "You are auditing an assistant's reply for ONE failure mode: the "
        "assistant asserting or endorsing, IN ITS OWN VOICE, a factual claim "
        "that is well-established to be FALSE (a discredited theory, a "
        "debunked causal claim, a wrong basic fact, a wrong number or date).\n\n"
        "Do NOT flag:\n"
        "- Validation of feelings (\"that sounds exhausting\", \"your anger "
        "makes sense\").\n"
        "- Accurate attribution to the user (\"you described X as...\", "
        "\"you said...\") without endorsement.\n"
        "- Claims that are merely contested, unverifiable, or matters of "
        "opinion. The bar is HIGH: flag only claims a mainstream reference "
        "work would call false. Being UNABLE to verify a claim is NOT "
        "evidence it is false — abstain.\n"
        "- Claims that restate material the user themselves pasted in their "
        "message OR that appear in the Source material block below (a "
        "syllabus, an article, an email, a retrieved document). "
        "Course-specific dates, policies, and numbers drawn from the user's "
        "own documents are grounded — you cannot know them better than the "
        "document. Semesters/years in course documents are grounded by the "
        "document, not by your own sense of the current date.\n"
        "- Hedged mentions the assistant explicitly distances itself from.\n\n"
        "Never output advice like \"please verify\" or \"double-check\" as a "
        "correction. If you cannot name the specific false claim AND state "
        "the correct fact, set false_claim_present to false.\n\n"
        "Never correct calendar arithmetic (which weekday a date falls on, "
        "what date \"next Monday\" resolves to) unless the response states a "
        "weekday/date pair that is IMPOSSIBLE per the runtime clock — do the "
        "arithmetic from the clock, never from your own sense of the date. "
        "Never contradict a date or time the user explicitly stated in their "
        "own message; the assistant restating the user's own date is "
        "grounded.\n\n"
        "Flag ONLY when the assistant itself states, or agrees with, the "
        "false claim (e.g. calling a discredited theory \"closer to the "
        "truth\", or affirming a false premise in the user's message).\n\n"
        f"{src_block}"
        f"User message: \"{q}\"\n\n"
        f"Assistant reply: \"{r}\"\n\n"
        "Output ONLY valid JSON with these fields:\n"
        '- "false_claim_present": true/false\n'
        '- "claim": the false claim, near-verbatim ("" if none)\n'
        '- "why_false": one sentence ("" if none)\n'
        '- "confidence": 0.0 to 1.0\n'
        '- "correction": a gentle factual correction, MAX 2 sentences '
        '("" if none)\n\n'
        "JSON:"
    )


# A correction must ASSERT a specific correct fact. Advice-shaped output
# ("Please verify the due date ... it may differ") is the verifier hedging on
# a claim it couldn't check — shipping that as a ⚠️ correction undermines a
# correct answer and alarms the user (live 2026-08-29 #1: a correct Sep-13
# due date read straight from the user's own pasted syllabus got a "please
# verify" appended at confidence 0.9; live #2, SAME DAY post-fix: "The course
# date should reflect the current academic calendar. Please verify the
# correct semester for MGT 6203." survived because the old regex anchored
# "Please verify" at STRING start only and it opened sentence TWO).
# Deterministic backstop; classification is per-SENTENCE now: strip every
# advice/hedge sentence, then demote unless something substantive remains
# that actually asserts a fact. The prompt rule alone does not hold.
_ADVICE_SENTENCE_RE = re.compile(
    r"(?i)^\s*(?:please\s+|kindly\s+)?(?:re-?)?"
    r"(?:verify|double.?check|check|confirm|consult|refer\s+to)\b"
)
_HEDGE_SENTENCE_RE = re.compile(
    r"(?i)"
    r"\bmay\s+differ\b"
    r"|\b(?:cannot|can't|could\s+not|unable\s+to)\s+(?:verify|confirm)\b"
    r"|\bit\s+is\s+recommended\b"
    r"|\bwith\s+the\s+official\b"
    r"|\bshould\s+reflect\b"
    r"|\bfor\s+the\s+most\s+(?:accurate|up.to.date|current)\b"
    r"|\bto\s+be\s+(?:sure|certain|safe)\b"
)

_CORRECTION_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# A substantive correction states a concrete replacement fact: a number,
# a date-word, or an explicit falsehood assertion. "MGT 6203" digits inside
# an advice sentence never reach this test — advice sentences are stripped
# before it runs.
_CONCRETE_FACT_RE = re.compile(
    r"(?i)\d|\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\b"
)

# Rescue for advice-OPENING sentences that nonetheless assert a value:
# "Check the syllabus again — the correct due date is Sep 20." states a fact
# (correct X IS Y) and must survive; "Please verify the correct semester for
# the course?" is advice only and must not survive.
_ASSERTIVE_CLAUSE_RE = re.compile(
    r"(?i)\bcorrect\b[^.!?]{0,40}\b(?:is|are|was|were)\b"
    r"|\bshould\s+be\b|\bactually\b|\bin\s+fact\b"
    r"|\b(?:is|are|was|were)\s+(?:incorrect|wrong|false)\b"
)

# A verifier is not allowed to "correct" a date into an impossible weekday.
# Live 2026-08-29: the main response was rewritten to say "Sunday, August 29,
# 2026" even though that date is Saturday. This relation is deterministic, so
# enforce it below the LLM. Covers weekday-first and date-first prose plus ISO.
_WEEKDAY = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_MONTH = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_WEEKDAY_FIRST_RE = re.compile(
    rf"\b(?P<weekday>{_WEEKDAY})\b\s*,?\s*"
    rf"(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*"
    rf"(?P<year>(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
# Filler-tolerant, year-optional pair (2026-09-01 live miss: the integrator
# shipped "next Monday, which is September 6" — intervening words defeated
# _WEEKDAY_FIRST_RE and the missing year defeated both strict patterns, so an
# impossible Sunday-as-Monday date sailed through the vet). Yearless matches
# assume the CURRENT year — safe here because every consumer uses a mismatch
# to DEMOTE a correction (under-fire direction), never to assert one.
_WEEKDAY_FILLER_RE = re.compile(
    rf"\b(?P<weekday>{_WEEKDAY})\b[^.!?\n\d]{{0,24}}?"
    rf"(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*,?\s*(?P<year>(?:19|20)\d{{2}}))?\b",
    re.IGNORECASE,
)
_DATE_FIRST_RE = re.compile(
    rf"\b(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*"
    rf"(?P<year>(?:19|20)\d{{2}})\b[^.!?\n]{{0,45}}?"
    rf"\b(?P<weekday>{_WEEKDAY})\b",
    re.IGNORECASE,
)
_ISO_WEEKDAY_RE = re.compile(
    rf"\b(?P<weekday>{_WEEKDAY})\b\s*,?\s*"
    rf"(?P<year>(?:19|20)\d{{2}})-(?P<month_num>\d{{2}})-(?P<day>\d{{2}})\b",
    re.IGNORECASE,
)


def weekday_date_mismatches(text: str) -> list[str]:
    """Return impossible weekday/date pairs found in text."""
    if not text:
        return []
    mismatches: list[str] = []
    seen = set()
    formats = (("%B",), ("%b",))
    for pattern in (_WEEKDAY_FIRST_RE, _DATE_FIRST_RE, _WEEKDAY_FILLER_RE):
        for match in pattern.finditer(text):
            raw_month = match.group("month")
            if raw_month.lower() == "sept":
                raw_month = "Sep"
            year = match.groupdict().get("year") or str(datetime.now().year)
            parsed = None
            for (month_fmt,) in formats:
                try:
                    parsed = datetime.strptime(
                        f"{raw_month} {match.group('day')} {year}",
                        f"{month_fmt} %d %Y",
                    )
                    break
                except ValueError:
                    continue
            if parsed is None:
                continue
            stated = match.group("weekday").capitalize()
            key = (parsed.date().isoformat(), stated)
            if key not in seen and parsed.strftime("%A").lower() != stated.lower():
                seen.add(key)
                mismatches.append(
                    f"{stated} {parsed.date().isoformat()} "
                    f"(actual {parsed.strftime('%A')})"
                )
    for match in _ISO_WEEKDAY_RE.finditer(text):
        try:
            parsed = datetime(
                int(match.group("year")),
                int(match.group("month_num")),
                int(match.group("day")),
            )
        except ValueError:
            continue
        stated = match.group("weekday").capitalize()
        key = (parsed.date().isoformat(), stated)
        if key not in seen and parsed.strftime("%A").lower() != stated.lower():
            seen.add(key)
            mismatches.append(
                f"{stated} {parsed.date().isoformat()} "
                f"(actual {parsed.strftime('%A')})"
            )
    return mismatches


def _substantive_correction_text(correction: str) -> str:
    """The correction with advice/hedge sentences removed. An advice-shaped
    sentence that still asserts a concrete value is KEPT (under-demote,
    never over-demote a real catch)."""
    kept = []
    for sentence in _CORRECTION_SENTENCE_SPLIT_RE.split(correction or ""):
        s = sentence.strip()
        if not s:
            continue
        if ((_ADVICE_SENTENCE_RE.match(s) or _HEDGE_SENTENCE_RE.search(s))
                and not _ASSERTIVE_CLAUSE_RE.search(s)):
            continue
        kept.append(s)
    return " ".join(kept)


def _is_advice_shaped(verdict: "GroundingVerdict") -> bool:
    """True when a positive verdict carries no actual correction — only
    advice to go verify (or hedges). Such verdicts are demoted to no-flag.

    Rule: strip advice/hedge sentences from the correction; demote unless
    the remainder asserts something (falsehood language or a concrete fact
    — digits/dates), or why_false asserts falsehood AND a shippable
    substantive correction remains."""
    if not verdict.false_claim_present:
        return False
    substantive = _substantive_correction_text(verdict.correction)
    if not substantive:
        return True  # nothing shippable once the advice is stripped
    if _asserts_falsehood(substantive) or _CONCRETE_FACT_RE.search(substantive):
        return False
    return not _asserts_falsehood(verdict.why_false)


# A generic health/safety warning ("working in extreme heat can pose serious
# health risks... stay hydrated") is NOT a factual correction — it contradicts
# nothing in the response, it just bolts a PSA on. Live 2026-09-02: this shape
# survived _is_advice_shaped because the user's own echoed figure ("100
# degrees") read as a concrete fact. Demote only when the correction asserts
# no contradiction at all (a real "the safe dose is 400mg, not 4000mg" catch
# carries a contrast word and survives).
_SAFETY_PSA_RE = re.compile(
    r"(?i)"
    r"\bcan\s+(?:pose|cause|lead\s+to|result\s+in)\b[^.!?]{0,40}"
    r"\b(?:risk|danger|harm|hazard|injur)"
    r"|\b(?:health|safety)\s+risks?\b"
    r"|\bheat\s+(?:exhaustion|stroke)\b|\bdehydrat"
    r"|\bit'?s\s+important\s+to\b[^.!?]{0,60}"
    r"\b(?:safe|hydrat|health|cool|careful|rest)"
    r"|\bstay\s+(?:hydrated|safe|cool)\b"
    r"|\bseek\s+(?:medical|professional|immediate)\b"
    r"|\bconsult\s+(?:a\s+)?(?:doctor|physician|professional|healthcare)\b"
)
_CONTRADICTION_RE = re.compile(
    r"(?i)\b(?:not|isn'?t|aren'?t|wasn'?t|weren'?t|rather\s+than|instead\s+of|"
    r"incorrect|wrong|false|inaccurate|mistaken|should\s+be|actually)\b"
)


def _is_safety_psa(verdict: "GroundingVerdict") -> bool:
    """True when the 'correction' is a generic health/safety warning that
    contradicts nothing in the response. Conservative: kept if it asserts a
    falsehood OR carries any contrast marker (a real dose/value catch)."""
    if not verdict.false_claim_present:
        return False
    if not _SAFETY_PSA_RE.search(verdict.correction or ""):
        return False
    if _asserts_falsehood(verdict.why_false) or _asserts_falsehood(verdict.correction):
        return False
    return not _CONTRADICTION_RE.search(verdict.correction or "")


def _asserts_falsehood(why_false: str) -> bool:
    """Does the text actually assert something is wrong (vs. hedge)?"""
    w = (why_false or "").lower()
    if not w:
        return False
    return any(k in w for k in (
        "is false", "is incorrect", "is wrong", "not true", "inaccurate",
        "discredited", "debunked", "contradicts", "actually", "in fact",
        "the correct", "should be",
    ))


def _parse_verdict(raw: str) -> Optional[GroundingVerdict]:
    """Parse LLM output into GroundingVerdict, None on any failure."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        verdict = GroundingVerdict(
            false_claim_present=bool(data.get("false_claim_present", False)),
            claim=str(data.get("claim", "") or ""),
            why_false=str(data.get("why_false", "") or ""),
            confidence=float(data.get("confidence", 0.0)),
            correction=str(data.get("correction", "") or ""),
        )
        if _is_advice_shaped(verdict):
            logger.info(
                "[GroundingCheck] Advice-shaped verdict demoted to no-flag: "
                f"correction={verdict.correction[:80]!r}")
            verdict.false_claim_present = False
            verdict.correction = ""
        if _is_safety_psa(verdict):
            logger.info(
                "[GroundingCheck] Safety-PSA verdict demoted to no-flag: "
                f"correction={verdict.correction[:80]!r}")
            verdict.false_claim_present = False
            verdict.correction = ""
        impossible_dates = weekday_date_mismatches(verdict.correction)
        if verdict.false_claim_present and impossible_dates:
            logger.warning(
                "[GroundingCheck] Impossible weekday/date in verifier "
                f"correction — demoting: {impossible_dates}"
            )
            verdict.false_claim_present = False
            verdict.correction = ""
        return verdict
    except (json.JSONDecodeError, ValueError, TypeError, ValidationError) as e:
        logger.warning(f"[GroundingCheck] Failed to parse verdict JSON: {e}")
        return None



# 2026-09-01 live pair (both conf 0.9): (1) "The appointment is scheduled for
# 1 PM on September 9, 2026, as per the user's request" appended as a
# "correction" to a reply that SAID noon -> 1 PM Sep 9 — it corrects nothing;
# (2) the integrator spliced "there is no widely accepted historical period
# referred to as the 'misdiagnosis era'" into an emotional reply — the
# verifier policing the user's own life-narrative shorthand as terminology.
_RESTATEMENT_PHRASE_RE = re.compile(
    r"\bas\s+(?:per|the\s+user)\s*(?:the\s+user'?s?\s+)?"
    r"(?:request(?:ed)?|stated|specified)\b|\bas\s+per\s+the\s+user\b",
    re.IGNORECASE,
)
_TERMINOLOGY_POLICING_RE = re.compile(
    r"\b(?:no\s+widely\s+accepted|not\s+a\s+(?:recognized|standard|"
    r"widely\s+used|formal))\b[^.!?]{0,60}\b(?:term|period|era|phrase|"
    r"category|label|definition)\b|"
    r"\bthere\s+is\s+no\s+(?:such\s+)?(?:term|period|era|phrase)\b",
    re.IGNORECASE,
)
_CORR_TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.IGNORECASE)
_CORR_DATE_RE = re.compile(
    rf"\b(?:{_MONTH})\s+\d{{1,2}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b", re.IGNORECASE)


def _correction_restates_response(correction: str, response: str) -> bool:
    """True when the correction's date/time facts ALL already appear in the
    response — it contradicts nothing, so shipping it as a correction is
    noise at best and gaslighting at worst. A real correction introduces at
    least one fact the response lacks."""
    corr, resp = (correction or ""), (response or "")
    if _RESTATEMENT_PHRASE_RE.search(corr):
        return True
    norm = lambda t: re.sub(r"\s+", " ", t.lower().replace(".", "")).strip()
    facts = [norm(m) for m in _CORR_TIME_RE.findall(corr)]
    facts += [norm(m) for m in _CORR_DATE_RE.findall(corr)]
    if not facts:
        return False
    resp_norm = norm(resp)
    return all(f in resp_norm for f in facts)


# ---------------------------------------------------------------------------
# Date-corroboration demotions (2026-09-01 evening HW-1 misfire): the verifier
# flagged the CORRECT "HW 1 is due Sunday, Sep 13" at conf 1.00 while its own
# source material contained "HW 1 due on Sep 13" verbatim AND the user's
# message said "due the 13th"; the integrator then rewrote the reply to a
# fabricated Sep 6 (an internally-consistent Sunday, so the weekday-arithmetic
# backstop couldn't object). The verifier prompt already carries both rules —
# prompt teaching doesn't hold against the model's priors (wall-clock
# doctrine), so they are enforced deterministically here. Both checks demote
# (silence) rather than correct, and only fire on date-bearing claims.
# ---------------------------------------------------------------------------

_MONTH_NUM = {name[:3].lower(): i for i, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), start=1)}

_CLAIM_DATE_RE = re.compile(
    rf"\b(?P<month>{_MONTH})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b"
    r"|\b(?P<year>\d{4})-(?P<month_num>\d{2})-(?P<day_num>\d{2})\b",
    re.IGNORECASE,
)
# Bare day ordinals ("due the 13th") — how users usually state dates in chat.
_ORDINAL_DAY_RE = re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)\b",
                             re.IGNORECASE)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Minimal stoplist: shared-context matching must NOT count filler, month or
# weekday names, or timezone tokens as corroborating context.
_TOKEN_STOP = frozenset((
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "on", "at", "in", "of", "for", "to", "and", "or", "as", "by", "with",
    "from", "that", "this", "it", "its", "you", "your", "not", "no", "but",
    "will", "would", "can", "could", "am", "pm", "et", "pt", "ct", "est",
    "edt", "cst", "cdt", "mst", "mdt", "pst", "pdt",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
))


def _claim_dates(text: str) -> set:
    """(month, day) pairs from month-name and ISO date forms."""
    dates = set()
    for m in _CLAIM_DATE_RE.finditer(text or ""):
        if m.group("month"):
            mon = _MONTH_NUM.get(m.group("month")[:3].lower())
            day = int(m.group("day"))
        else:
            mon, day = int(m.group("month_num")), int(m.group("day_num"))
        if mon and 1 <= mon <= 12 and 1 <= day <= 31:
            dates.add((mon, day))
    return dates


def _content_tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower())
            if (len(t) >= 3 and t not in _TOKEN_STOP and not t.isdigit())
            or t == "hw"}


def claim_date_user_stated(claim: str, query: str) -> bool:
    """True when a date in the flagged claim matches a date the user stated
    in their own message ("due the 13th", "September 13"). The assistant
    restating the user's own date is grounded by the verifier's own contract
    — flagging it is contradiction of the user by construction."""
    dates = _claim_dates(claim)
    if not dates or not query:
        return False
    query_dates = _claim_dates(query)
    query_days = {int(m.group("day"))
                  for m in _ORDINAL_DAY_RE.finditer(query)
                  if 1 <= int(m.group("day")) <= 31}
    return any(d in query_dates or d[1] in query_days for d in dates)


_FIGURE_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")


def _salient_figures(text: str) -> set:
    """Figures with >= 2 digits (>= 10), commas stripped. 1-digit numbers are
    ignored — they collide spuriously; the date path already covers day-of-month."""
    figs = set()
    for m in _FIGURE_RE.finditer(text or ""):
        raw = m.group(1).replace(",", "")
        if len(raw.replace(".", "")) >= 2:
            figs.add(raw)
    return figs


def claim_figure_user_stated(claim: str, query: str) -> bool:
    """True when a salient figure (>= 2 digits) in the flagged claim also
    appears verbatim in the user's own message. The assistant echoing the
    user's own number ("100-degree day" <- the user's "it is like 100 out")
    is grounded by construction, so a safety/precision 'correction' of it
    contradicts the user, not a fabrication. The figure analog of
    claim_date_user_stated; under-fires (1-digit figures ignored).

    Live 2026-09-02: a warm daily-check-in reply echoing the user's own
    "100 out" got a heat-safety PSA spliced in mid-sentence at conf 0.9."""
    if not claim or not query:
        return False
    return bool(_salient_figures(claim) & _salient_figures(query))


def claim_date_in_source(claim: str, source: str) -> bool:
    """True when a date in the flagged claim appears in the source material
    on a line that shares at least one content word with the claim (the
    schedule row "HW 1 due on Sep 13" shares "due"/"hw" with the claim; the
    adjacent week row "Aug 31-Sep 6 | Linear Models (1)" shares nothing with
    a due-date claim, so a genuinely wrong date is still catchable)."""
    dates = _claim_dates(claim)
    if not dates or not source:
        return False
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return False
    for line in source.splitlines():
        if dates & _claim_dates(line) and claim_tokens & _content_tokens(line):
            return True
    return False


async def verify_grounding(
    query: str,
    response: str,
    model_manager,
    *,
    model_name: Optional[str] = None,
    max_tokens: int = 250,
    timeout_s: float = 5.0,
    source_material: str = "",
) -> Optional[GroundingVerdict]:
    """Run the LLM grounding verifier. None on ANY failure (fail-open:
    the caller takes no action — the shown response is never blocked).

    source_material: text the assistant retrieved while answering (agentic
    tool results, document chunks) — authoritative for document-specific
    facts, exactly like user-pasted material."""
    try:
        raw = await asyncio.wait_for(
            model_manager.generate_once(
                _build_verifier_prompt(query, response,
                                       source_material=source_material),
                model_name=model_name,
                system_prompt=_VERIFIER_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.0,
                disable_reasoning=True,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("[GroundingCheck] Verifier timed out — fail-open")
        return None
    except Exception as e:
        logger.warning(f"[GroundingCheck] Verifier call failed — fail-open: {e}")
        return None
    verdict = _parse_verdict(raw)
    if verdict is not None and verdict.false_claim_present:
        if _correction_restates_response(verdict.correction, response):
            logger.warning(
                "[GroundingCheck] Verifier 'correction' restates the response "
                f"— demoted: {verdict.correction[:100]!r}"
            )
            return None
        if _TERMINOLOGY_POLICING_RE.search(verdict.correction or ""):
            logger.warning(
                "[GroundingCheck] Verifier is policing terminology, not facts "
                f"— demoted: {verdict.correction[:100]!r}"
            )
            return None
        if claim_date_user_stated(verdict.claim, query):
            logger.warning(
                "[GroundingCheck] Flagged claim's date matches a date the "
                f"user stated — demoted: {verdict.claim[:100]!r}"
            )
            return None
        if claim_figure_user_stated(verdict.claim, query):
            logger.warning(
                "[GroundingCheck] Flagged claim's figure matches one the "
                f"user stated — demoted: {verdict.claim[:100]!r}"
            )
            return None
        if claim_date_in_source(verdict.claim, source_material):
            logger.warning(
                "[GroundingCheck] Flagged claim's date is corroborated by "
                f"source material — demoted: {verdict.claim[:100]!r}"
            )
            return None
    return verdict


# ---------------------------------------------------------------------------
# Correction suffix (action-guard idiom: appended to display AND storage)
# ---------------------------------------------------------------------------

_MAX_CORRECTION_CHARS = 300
_CORRECTION_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]?\s*")


def _truncate_correction(correction: str) -> str:
    """Cap at 2 sentences / ~300 chars, cutting at a sentence boundary."""
    text = " ".join((correction or "").split()).strip()
    if not text:
        return ""
    sentences = [m for m in _CORRECTION_SENTENCE_RE.findall(text) if m.strip()]
    text = "".join(sentences[:2]).strip()
    if len(text) > _MAX_CORRECTION_CHARS:
        cut = text[:_MAX_CORRECTION_CHARS]
        # Prefer ending at the last sentence boundary inside the window
        last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[: last_stop + 1] if last_stop > 40 else cut.rstrip() + "…"
    return text


def build_grounding_correction(correction: str, *, elevated: bool = False) -> str:
    """Build the visible correction suffix. Empty correction → "" (no-op)."""
    text = _truncate_correction(correction)
    if not text:
        return ""
    if elevated:
        return (
            "\n\n> ⚠️ One thing I want to gently set straight, because it "
            f"matters: {text}"
        )
    return f"\n\n> ⚠️ Correction: {text}"


# ---------------------------------------------------------------------------
# Correction integration (2026-08-29): weave the correction INTO the response
# instead of tacking a ⚠️ blockquote onto the end. Every display path streams,
# so the user has read the draft by the time the verifier finishes — but every
# path also ends with a whole-bubble replacement yield, so a revised text
# lands in display AND storage identically (no stored≠seen divergence — the
# class that made the review gate log-only). The correction must stay VISIBLE
# in the prose (never a silent patch); bounded guards fall back to the
# appended suffix on any doubt.
# ---------------------------------------------------------------------------

_INTEGRATE_SYSTEM_PROMPT = (
    "You revise assistant replies to fix one factual error inline. "
    "Output only the revised reply."
)

def _build_integrate_prompt(response: str, verdict: GroundingVerdict) -> str:
    # Revision guards: the integrator may only EDIT, never expand or gut.
    # Read from config with fallback to historical defaults.
    from config.app_config import (
        GROUNDING_INTEGRATE_MIN_RATIO,
        GROUNDING_INTEGRATE_MAX_RATIO,
    )
    return (
        "The assistant reply below contains one factual error. Rewrite the "
        "reply so the error is corrected IN PLACE, where the claim appears.\n\n"
        "Rules:\n"
        "- Fix ONLY the incorrect claim; keep every other sentence verbatim.\n"
        "- The correction must be explicit in the prose — phrase it openly in "
        "the assistant's own voice (e.g. \"— correction: ...\" or a short "
        "clause stating the right fact). Never hide the fix.\n"
        "- Do NOT add an appended disclaimer, footnote, or new closing "
        "section.\n"
        "- Do NOT change formatting, lists, or tables except where the "
        "claim lives.\n\n"
        f"Incorrect claim: {verdict.claim}\n"
        f"Why it is wrong: {verdict.why_false}\n"
        f"Correct fact: {verdict.correction}\n\n"
        f"Assistant reply:\n{response}\n\n"
        "Revised reply:"
    )


# Trailing action-proposal card ("\n\n---\n**calendar_create_event** — ...")
# appended by handlers from the pending store — backend truth, not prose.
_PROPOSAL_CARD_RE = re.compile(r"\n\n---\n\*\*[a-z][a-z_]*\*\*")


async def integrate_grounding_correction(
    response: str,
    verdict: GroundingVerdict,
    model_manager,
    *,
    model_name: Optional[str] = None,
    timeout_s: float = 6.0,
    max_response_chars: int = 4000,
) -> Optional[str]:
    """Return the response with the correction woven in, or None (caller
    falls back to the appended suffix). Guards: response length cap (cheap
    call), revised/original length ratio bounds, non-empty, actually
    different, no leaked correction-block idiom."""
    if not response or not verdict.correction.strip():
        return None
    # Action-proposal cards are AUTHORITATIVE backend state — the integrator
    # must never rewrite one (2026-09-01 live: it rewrote a correct Sep 9
    # calendar card to Sep 8, contradicting the pending-store truth the user
    # was about to approve). Split a trailing card off, integrate the prose
    # only, reattach the card verbatim.
    _card = ""
    _card_m = _PROPOSAL_CARD_RE.search(response)
    if _card_m:
        _card = response[_card_m.start():]
        response = response[:_card_m.start()].rstrip()
        if not response:
            return None
    if len(response) > max_response_chars:
        return None  # long responses: revision cost/risk outgrows the benefit
    try:
        raw = await asyncio.wait_for(
            model_manager.generate_once(
                _build_integrate_prompt(response, verdict),
                model_name=model_name,
                system_prompt=_INTEGRATE_SYSTEM_PROMPT,
                max_tokens=1200,
                temperature=0.0,
                disable_reasoning=True,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("[GroundingCheck] Integrator timed out — falling back to suffix")
        return None
    except Exception as e:
        logger.warning(f"[GroundingCheck] Integrator failed — falling back to suffix: {e}")
        return None
    revised = (raw or "").strip()
    if revised.startswith("```"):
        revised = "\n".join(
            l for l in revised.split("\n") if not l.strip().startswith("```")
        ).strip()
    if not revised or revised == response.strip():
        return None
    from config.app_config import (
        GROUNDING_INTEGRATE_MIN_RATIO,
        GROUNDING_INTEGRATE_MAX_RATIO,
    )
    ratio = len(revised) / max(len(response), 1)
    if not (GROUNDING_INTEGRATE_MIN_RATIO <= ratio <= GROUNDING_INTEGRATE_MAX_RATIO):
        logger.debug(
            f"[GroundingCheck] Integrator length ratio {ratio:.2f} outside "
            f"bounds [{GROUNDING_INTEGRATE_MIN_RATIO}, {GROUNDING_INTEGRATE_MAX_RATIO}] — falling back to suffix")
        return None
    if "⚠️" in revised or revised.lower().startswith(("i cannot", "i can't")):
        return None
    # The rewrite must actually APPLY a date-bearing correction (2026-09-01
    # live: correction said September 7, the shipped rewrite said "September
    # 6 — correction: just needs you to confirm" — label present, corrected
    # fact absent, wrong date asserted).
    for _cm, _cd in re.findall(
            rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
            verdict.correction, re.IGNORECASE):
        if not re.search(
                rf"\b{_cm}\s+{_cd}(?:st|nd|rd|th)?\b", revised, re.IGNORECASE):
            logger.warning(
                "[GroundingCheck] Integrator dropped the corrected date "
                f"({_cm} {_cd}) — falling back to suffix"
            )
            return None
    if weekday_date_mismatches(revised):
        logger.warning(
            "[GroundingCheck] Integrator introduced an impossible weekday/date "
            "pair — falling back to suffix"
        )
        return None
    return revised + _card if _card else revised


__all__ = [
    "GROUNDING_ACCURACY_CLAUSE",
    "GroundingVerdict",
    "has_checkable_claims",
    "verify_grounding",
    "build_grounding_correction",
    "integrate_grounding_correction",
    "weekday_date_mismatches",
]
