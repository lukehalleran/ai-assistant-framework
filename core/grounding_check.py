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
context-sensitive term." names no value (no copula after "correct <noun>") and must not.
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
    for pattern in (_WEEKDAY_FIRST_RE, _DATE_FIRST_RE):
        for match in pattern.finditer(text):
            raw_month = match.group("month")
            if raw_month.lower() == "sept":
                raw_month = "Sep"
            parsed = None
            for (month_fmt,) in formats:
                try:
                    parsed = datetime.strptime(
                        f"{raw_month} {match.group('day')} {match.group('year')}",
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
    return _parse_verdict(raw)


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

# Revision guards: the integrator may only EDIT, never expand or gut.
_INTEGRATE_MIN_RATIO = 0.75
_INTEGRATE_MAX_RATIO = 1.30


def _build_integrate_prompt(response: str, verdict: GroundingVerdict) -> str:
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
    ratio = len(revised) / max(len(response), 1)
    if not (_INTEGRATE_MIN_RATIO <= ratio <= _INTEGRATE_MAX_RATIO):
        logger.debug(
            f"[GroundingCheck] Integrator length ratio {ratio:.2f} outside "
            f"bounds — falling back to suffix")
        return None
    if "⚠️" in revised or revised.lower().startswith(("i cannot", "i can't")):
        return None
    if weekday_date_mismatches(revised):
        logger.warning(
            "[GroundingCheck] Integrator introduced an impossible weekday/date "
            "pair — falling back to suffix"
        )
        return None
    return revised


__all__ = [
    "GROUNDING_ACCURACY_CLAUSE",
    "GroundingVerdict",
    "has_checkable_claims",
    "verify_grounding",
    "build_grounding_correction",
    "integrate_grounding_correction",
    "weekday_date_mismatches",
]
