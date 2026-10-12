"""Shared grounding rules for conversation and evidence-backed synthesis.

Two blocks, deliberately domain-neutral (2026-09-06 generalization pass —
the first version named a drug class, a neurotransmitter, and an alcohol
combination from one owner incident; a fresh clone must ship none of that):

- ``UNIVERSAL_GROUNDING`` (~90 tokens) rides in the CACHED system-prompt prefix
  on every turn: provenance ordering and observation-vs-explanation discipline.
- ``DECISION_SUPPORT_GROUNDING`` (~150 tokens) is appended POST-breakpoint only
  when ``include_decision_support`` says the turn is a decision or a weighty
  one — keyed on signals the pipeline already computes (heavy-topic flag, tone
  level, request shape, small-talk flag, self-report shape), never on topic
  vocabulary. Insight synthesis always gets both (it is a decision turn by
  construction).

``CONTEXTUAL_GROUNDING`` remains as the combined text for any caller that
wants both unconditionally.
"""

import re
from typing import Any, Optional

UNIVERSAL_GROUNDING = """
## Contextual grounding
- The user's current statements and corrections outrank earlier assistant
  replies and generated summaries; earlier advice is history, not evidence.
- Separate observations from explanations: never invent a duration, cause,
  mechanism, or outcome from one ambiguous detail or a single day. Apply a
  correction to your conclusion, not just its wording.
- A report of rest, relief, or a choice stands on its own: never justify it by
  tallying what the user produced earlier (not even if your own earlier replies
  did), and never treat it as a request for endorsement.
"""

DECISION_SUPPORT_GROUNDING = """
## Decision support
When the user weighs a change to a treatment, habit, routine, or commitment,
give a balanced, substantive assessment: the case for continuing and for
changing; the specific thing in question, how long the change would last, and
their own reported prior response. One good or bad day sets no schedule. Say
what the record can and cannot establish, and where a qualified professional's
input matters — without replacing the answer with a referral. Cite sources
that are in front of you; otherwise name the uncertainty rather than
manufacture authority.
"""

CONTEXTUAL_GROUNDING = UNIVERSAL_GROUNDING + DECISION_SUPPORT_GROUNDING


# First-person reference as a SUBJECT/possessive/reflexive (I, I'm, my, mine,
# myself, we, our...), plus "me" ONLY when it is not the benefactive of a
# ditransitive request opener ("tell/give/show me about X" is addressed to the
# assistant and names no personal stake; "which is best for me" does).
_FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|i've|i'd|i'll|my|mine|myself|we|we're|we've|our|ours|ourselves)\b")
_ME_RE = re.compile(r"\bme\b")
_DITRANSITIVE_ME_RE = re.compile(
    r"\b(tell|give|show|send|get|teach|bring|write|make|find|read|read out|"
    r"explain|walk)\s+(it\s+to\s+)?me\b")


def has_personal_stake(text: str) -> bool:
    """Structural test: does the message reference the user's own situation
    (first-person subject/possessive/reflexive, or "me" outside a ditransitive
    request opener)? A third-person, historical or explanatory question
    ("tell me about the role of rail in the French state") has none, so it is
    not a decision for the user however request-shaped it is (2026-10-12)."""
    ql = (text or "").lower().replace("\u2019", "'")
    if _FIRST_PERSON_RE.search(ql):
        return True
    return bool(_ME_RE.search(_DITRANSITIVE_ME_RE.sub(" ", ql)))


def include_decision_support(
    user_input: Optional[str],
    *,
    tone_level: Any = None,
    is_heavy_topic: bool = False,
    is_small_talk: bool = False,
) -> bool:
    """Decide whether ``DECISION_SUPPORT_GROUNDING`` joins this turn's tail.

    Signals only — no vocabulary:
    - a heavy-topic turn or an elevated tone always qualifies;
    - otherwise a request-shaped / info-seeking message qualifies unless it is
      small talk, a bare first-person self-report (the self-report keeps the
      universal block only — a status update is not a decision), or carries no
      personal stake at all (``has_personal_stake``: a historical/third-person
      explanatory question is not the user's decision).
    """
    if is_heavy_topic:
        return True
    # lazy import: cycle (core.agentic.gate -> core.insight.detector -> utils.web_search_trigger)
    from core.agentic.gate import _tone_is_elevated
    if _tone_is_elevated(tone_level):
        return True
    text = (user_input or "").strip()
    if not text or is_small_talk:
        return False
    from utils.query_checker import is_request_shaped, is_self_report  # lazy import: cycle
    if is_self_report(text):
        return False
    if not has_personal_stake(text):
        return False
    return is_request_shaped(text)
