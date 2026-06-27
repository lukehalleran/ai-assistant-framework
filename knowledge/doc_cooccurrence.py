"""
# knowledge/doc_cooccurrence.py

Module Contract
- Purpose: Document co-occurrence oracle — decide whether two concepts are already
  discussed TOGETHER in the wiki corpus, INDEPENDENT of embedding distance. This is the
  Test-B "known / rediscovered" signal that cosine cannot be: cos(A,B) equates "known"
  with "topically close", so it misses non-obvious cross-domain known connections (the
  discovery target, e.g. `simulated annealing ↔ metallurgy`, cos 0.09) and is circular
  with cosine-based candidate selection.
- Public API:
    doc_cooccurrence(a, b, depth=40, min_shared=1) -> DocCooccurResult
    is_known(a, b, depth=40, min_shared=1) -> bool
- Signal (over each concept's top-`depth` retrieved wiki chunks): KNOWN iff they share an
  article TITLE, OR one concept's distinctive term appears in the TEXT of the other's
  articles (the crossover analogy lives in the body, not the title).
- Inputs: two concept strings. Output: DocCooccurResult(shared, shared_titles, mention, known).
- Side effects: FAISS wiki retrievals (semantic_search_with_neighbors). No LLM, no writes.
- Validation: docs/SYNTHESIS_VALIDATION.md (2026-06-27) — v2 text-scan, 93% on n=15 labeled,
  catches 5/6 cross-domain known that cosine catches 0/6, 0 FP on the (small) control.
  Caveat: stem-substring matching is crude; FP rate on a larger set is the open question.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple

from knowledge.semantic_search import semantic_search_with_neighbors

# Generic tokens that shouldn't count as a distinctive cross-mention (too common to
# imply A and B are actually discussed together).
_GENERIC_TOK = {
    "theory", "system", "systems", "model", "models", "method", "methods", "process",
    "science", "general", "number", "function", "problem", "problems", "effect",
    "analysis", "structure", "property", "properties", "concept", "approach",
}


class DocCooccurResult(NamedTuple):
    shared: int            # count of shared article titles
    shared_titles: List[str]
    mention: bool          # one concept's distinctive term appears in the other's article text
    known: bool            # shared >= min_shared OR mention


def _norm(t) -> str:
    return (t or "").strip().lower()


def _stems(phrase: str) -> set:
    """Distinctive 6-char stems of a concept's content tokens (len>=6, non-generic).

    Falls back to len>=4 tokens only if nothing qualifies, so short single-word concepts
    still get a signal.
    """
    toks = [t for t in re.findall(r"[a-z]+", phrase.lower())
            if len(t) >= 6 and t not in _GENERIC_TOK]
    if toks:
        return {t[:6] for t in toks}
    return {t[:6] for t in re.findall(r"[a-z]+", phrase.lower())
            if len(t) >= 4 and t not in _GENERIC_TOK}


def doc_cooccurrence(a: str, b: str, depth: int = 40, min_shared: int = 1) -> DocCooccurResult:
    """Are concepts A and B discussed together in the wiki corpus? (cosine-independent)

    Two signals over each concept's top-`depth` retrieved chunks:
      - shared TITLES (co-listed in the same articles), and
      - cross-MENTION: B's distinctive term appears in the TEXT of A's articles, or
        vice versa (the crossover analogy is usually in the body, not the title).
    """
    ra = semantic_search_with_neighbors(a, k=depth)
    rb = semantic_search_with_neighbors(b, k=depth)

    ta = {_norm(r.get("title")) for r in ra if r.get("title")}; ta.discard("")
    tb = {_norm(r.get("title")) for r in rb if r.get("title")}; tb.discard("")
    shared = ta & tb

    a_text = " ".join((r.get("content") or r.get("text") or "") for r in ra).lower()
    b_text = " ".join((r.get("content") or r.get("text") or "") for r in rb).lower()
    a_stems, b_stems = _stems(a), _stems(b)
    mention = any(s in a_text for s in b_stems) or any(s in b_text for s in a_stems)

    known = len(shared) >= min_shared or mention
    return DocCooccurResult(len(shared), sorted(shared)[:4], mention, known)


def is_known(a: str, b: str, depth: int = 40, min_shared: int = 1) -> bool:
    """Convenience boolean: is the (A, B) connection already in literature?"""
    return doc_cooccurrence(a, b, depth=depth, min_shared=min_shared).known
