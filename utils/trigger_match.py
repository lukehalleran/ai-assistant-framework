"""utils/trigger_match.py

Single chokepoint for deterministic keyword-trigger matching, closing a
recurring bug class: substring-matched keyword/action triggers that ignore
word boundaries or negation.

History: nine separate substring incidents ('issues'⊂"sleep issues",
'what is '⊂"what is said", 'commit'⊂"commit to", 'document'⊂"documented",
'solve'⊂"resolution", 'how'⊂"shower", 'actions'⊂"not taking any ACTIONS",
bare mid-clause 'what') were each fixed one keyword list at a time in
core/agentic/gate.py, and knowledge/document_generator.py separately grew a
negation lookback guard (2026-09-04) after "Do not save a document" fired
document generation anyway. Neither fix generalized — every OTHER
deterministic trigger site (web search, action detection, the visual-memory
gate, file retrieval) still ignored negation entirely: "don't search for
it", "no need to look it up", "don't add that to my calendar", "don't show
me the photos", "I'm not asking you to calculate it" all still fired their
arms before this module existed.

This module is the single place both concerns live from now on:
  (a) compile_keyword_matcher(keywords) — word-boundary-for-bare-words /
      substring-for-phrases matching (moved verbatim from
      core.agentic.gate._compile_keyword_matcher; that name is now a thin
      alias imported from here).
  (b) NEGATION_CUE_RE + is_negated(text, match_start, window_tokens=5) — a
      negation cue within `window_tokens` tokens BEFORE a match disqualifies
      it (moved from knowledge.document_generator, which now imports it back
      and extended with a couple more cue forms: "stop", "not asking (you)
      to").
  (c) find_hits(text, matcher, honor_negation=True) — matcher hits with
      negated ones dropped; has_non_negated_hit() is the boolean convenience
      wrapper most call sites want.

Under-fire doctrine: negation must sit within `window_tokens` tokens BEFORE
the match. A negation cue AFTER a match (scoping something else, not the
trigger itself) never disqualifies it. Bare "not" is not itself a cue — it
only counts inside the listed cue forms ("do not", "not going to", "not
asking (you) to", ...), so "I'd rather not" does NOT register (only the
literal "rather than" does) and ordinary sentences using "not" adjectivally
("that's not accurate") can't blanket-suppress unrelated later matches.
"""

from __future__ import annotations

import re
from typing import Iterable, List, NamedTuple, Sequence


# ---------------------------------------------------------------------------
# (a) Keyword matching — word-boundary for bare words, substring for phrases
# ---------------------------------------------------------------------------

class KeywordHit(NamedTuple):
    """One matched keyword/phrase and its position in the searched text."""
    keyword: str
    start: int
    end: int


class KeywordMatcher:
    """Callable matcher (`matcher(lower_text) -> bool`, matching the original
    core.agentic.gate._compile_keyword_matcher contract exactly) that also
    exposes position-aware hits via `.iter_hits()` for negation-aware callers.

    'solve' must not match "resolution"/"unresolved" — a memory-ingest paste
    titled "crisis resolution" keyword-routed to a 49s computation+tools loop
    (2026-08-28, same substring class as 'document'⊂"documented"). Only the
    LEFT boundary is enforced so 'solve' still matches "solves"/"solving";
    keywords containing spaces, apostrophes, or trailing-space sentinels keep
    their original substring semantics ('go to http' must still match
    "go to https://...").
    """

    __slots__ = ("_word_pats", "_substrings")

    def __init__(self, keywords: Sequence[str]):
        word_pats: List[tuple] = []
        substrings: List[str] = []
        for kw in keywords:
            if re.fullmatch(r"[a-z][a-z0-9_]*", kw):
                word_pats.append((kw, re.compile(rf"\b{re.escape(kw)}")))
            else:
                substrings.append(kw)
        self._word_pats = word_pats
        self._substrings = substrings

    def __call__(self, lower_text: str) -> bool:
        return (
            any(p.search(lower_text) for _, p in self._word_pats)
            or any(k in lower_text for k in self._substrings)
        )

    def iter_hits(self, lower_text: str) -> Iterable[KeywordHit]:
        """Yield every keyword's first match with its position."""
        for kw, pat in self._word_pats:
            m = pat.search(lower_text)
            if m:
                yield KeywordHit(kw, m.start(), m.end())
        for kw in self._substrings:
            idx = lower_text.find(kw)
            if idx != -1:
                yield KeywordHit(kw, idx, idx + len(kw))


def compile_keyword_matcher(keywords: Sequence[str]) -> KeywordMatcher:
    """Compile `keywords` into a KeywordMatcher — callable with boolean
    semantics identical to the original gate implementation this was moved
    from (`matcher(lower_text) -> bool`), plus position-aware `.iter_hits()`
    for negation-aware callers via find_hits()/has_non_negated_hit()."""
    return KeywordMatcher(keywords)


# ---------------------------------------------------------------------------
# (b) Negation
# ---------------------------------------------------------------------------

# Cue forms that scope a following trigger as NOT requested. Word-bounded;
# multi-word cues tolerate the natural whitespace variants ("do not"/"don't"
# /"dont"). Deliberately does NOT include bare "not" — see module docstring.
NEGATION_CUE_RE = re.compile(
    r"\b(?:"
    r"do\s*not|don't|dont|never|no\s+need\s+to|without|rather\s+than|"
    r"instead\s+of|not\s+going\s+to|won't|wont|skip|avoid|stop|"
    r"not\s+asking\s+(?:you\s+)?to|no\s+need"
    r")\b",
    re.IGNORECASE,
)

_NEGATION_LOOKBACK_CHARS = 60   # generous window; trimmed to the last N tokens below
DEFAULT_NEGATION_WINDOW_TOKENS = 5


def is_negated(
    text: str,
    match_start: int,
    window_tokens: int = DEFAULT_NEGATION_WINDOW_TOKENS,
) -> bool:
    """True if a negation/avoidance cue sits within `window_tokens` tokens
    BEFORE position `match_start` in `text`.

    A cue AFTER match_start (scoping something else) never disqualifies a
    match — only a cue that precedes and could plausibly scope the trigger
    does. This is a lookback check applied to a MATCH POSITION, not a rewrite
    of whatever tuned trigger regex/keyword list produced the match.
    """
    window = text[max(0, match_start - _NEGATION_LOOKBACK_CHARS):match_start]
    tail_tokens = window.split()[-window_tokens:]
    return bool(NEGATION_CUE_RE.search(" ".join(tail_tokens)))


# ---------------------------------------------------------------------------
# (c) Position-aware, negation-honoring hit lookup
# ---------------------------------------------------------------------------

def find_hits(
    text: str,
    matcher: KeywordMatcher,
    *,
    honor_negation: bool = True,
    window_tokens: int = DEFAULT_NEGATION_WINDOW_TOKENS,
) -> List[KeywordHit]:
    """Return `matcher`'s hits against `text`, dropping negated ones.

    `text` should be pre-lowercased the same way it was for a plain
    `matcher(text)` boolean call at the same site — hit offsets are computed
    against exactly the string passed in, and is_negated()'s cue regex is
    itself case-insensitive so a lowercased string works either way.
    """
    hits = list(matcher.iter_hits(text))
    if not honor_negation:
        return hits
    return [h for h in hits if not is_negated(text, h.start, window_tokens)]


def has_non_negated_hit(
    text: str,
    matcher: KeywordMatcher,
    *,
    window_tokens: int = DEFAULT_NEGATION_WINDOW_TOKENS,
) -> bool:
    """Boolean convenience wrapper around find_hits() — the common case at
    call sites that only care whether ANY non-negated hit exists."""
    return bool(find_hits(text, matcher, window_tokens=window_tokens))
