"""Shared "ambiguous word counts only near an anchor" regex composition.

Module Contract
- Purpose: one place that builds the two-order proximity pattern used by every
  word-sense rule of the form "<ambiguous verb> is the technical/mathematical
  sense only when a tech/math OBJECT sits nearby" (BC-01/BC-73).
- Pure stdlib; importable from core.intent_classifier and core.agentic.gate
  without a cycle.
- Origin (2026-10-18/19): the intent classifier's "crashed"/"broke(n)" rule
  (2026-09-10) was the only anchored rule; bare "not working" ("its not
  working i keep getting angry") and the gate's bare "differentiate"/
  "integrate" ("the Assign Module to differentiate our entities") fired on the
  everyday sense. Both now route through ``near_pattern``.
"""

from __future__ import annotations

import re

# Technical objects (software, hardware, network, devices). A CATEGORY, not a
# list of incident words: extend by adding a noun of the category.
TECH_NOUN_RE = (
    r"(?:app(?:lication)?|code(?:base)?|script|server|laptop|computer|pc|"
    r"desktop|phone|python|javascript|java|\br\b|build|test|daemon|"
    r"program|software|website|site|\bapi\b|database|\bdb\b|"
    r"repo(?:sitory)?|container|docker|\bvm\b|browser|terminal|shell|"
    r"module|librar(?:y|ies)|package|function|component|service|system|"
    r"device|driver|kernel|"
    r"wi-?fi|internet|network|router|printer|bluetooth|keyboard|monitor|"
    r"charger|plugin|extension|login|button|feature|tool|widget)"
)

# Mathematical objects by SHAPE (not by example): a call `f(x)`, a power
# `x^2`, a differential `dx`, an operator between operands `2x + 3`, a coefficient
# `3x`, or an explicit mathematical noun phrase.
MATH_OBJECT_RE = (
    r"(?:\b[a-z]\s*\(\s*[a-z0-9]+(?:\s*,\s*[a-z0-9]+)*\s*\)"
    r"|[a-z0-9)]\s*\^\s*[-+]?[a-z0-9(]"
    r"|\bd[xyzt]\b"
    r"|\bwith\s+respect\s+to\b|\bw\.?r\.?t\.?\b"
    r"|\bthe\s+(?:function|equation|expression|polynomial|integrand)\b"
    r"|\d\s*[+*/=]\s*[a-z0-9]|[a-z]\s*[+*/=]\s*\d"
    r"|\b\d+[xyzt]\b)"
)


def near_pattern(word_re: str, anchor_re: str, window: int = 40) -> str:
    """Regex source matching ``word_re`` with ``anchor_re`` within ``window``
    characters, in EITHER order. Callers supply their own ``\\b`` boundaries."""
    return (
        rf"{word_re}[\s\S]{{0,{window}}}{anchor_re}"
        rf"|{anchor_re}[\s\S]{{0,{window}}}{word_re}"
    )


_MATH_VERB_NEAR_RE = re.compile(
    near_pattern(r"\b(?:differentiat(?:e|es|ed|ing)|integrat(?:e|es|ed|ing))\b",
                 MATH_OBJECT_RE, window=60),
    re.IGNORECASE,
)


def math_verb_has_math_object(text: str) -> bool:
    """True when differentiate/integrate (any inflection) has a math object
    within 60 characters."""
    return bool(_MATH_VERB_NEAR_RE.search(text or ""))
