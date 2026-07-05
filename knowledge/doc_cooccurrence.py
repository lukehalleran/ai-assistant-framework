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
    doc_cooccurrence(a, b, depth=40, min_shared=1, bidirectional=False) -> DocCooccurResult
    is_known(a, b, depth=40, min_shared=1, bidirectional=False) -> bool
- Signal (over each concept's top-`depth` retrieved wiki chunks): KNOWN iff they share an
  article TITLE, OR one concept's distinctive term appears in the TEXT of the other's
  articles (the crossover analogy lives in the body, not the title). With
  bidirectional=True the text-mention signal requires BOTH directions (A's term in B's
  articles AND B's term in A's) — strips one-directional stem collisions (neighbour-
  vocabulary bleed) for perfect precision (0 FP @ n=99). It is a PRECISION-ONLY KNOB,
  NOT a default: re-validation (2026-06-28) showed it craters hard cross-domain recall
  96%→36% (real cross-domain knowledge is asymmetrically documented — specific articles
  cite the general concept, not vice versa). Keep default False (validated 97%/4%); use
  bidirectional=True only where false positives dominate and recall can be sacrificed.
  See docs/SYNTHESIS_VALIDATION.md (bidirectional re-validation, 2026-06-28).
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


def doc_cooccurrence(a: str, b: str, depth: int = 40, min_shared: int = 1,
                     bidirectional: bool = False) -> DocCooccurResult:
    """Are concepts A and B discussed together in the wiki corpus? (cosine-independent)

    Two signals over each concept's top-`depth` retrieved chunks:
      - shared TITLES (co-listed in the same articles), and
      - cross-MENTION: B's distinctive term appears in the TEXT of A's articles, or
        vice versa (the crossover analogy is usually in the body, not the title).

    bidirectional: when True, the cross-MENTION signal requires BOTH directions (A's term
      in B's articles AND B's term in A's), which strips one-directional stem collisions
      (neighbour-vocabulary bleed). Default False keeps the validated either-direction
      behaviour. Title overlap is unaffected by this flag.
    """
    ra = semantic_search_with_neighbors(a, k=depth)
    rb = semantic_search_with_neighbors(b, k=depth)

    ta = {_norm(r.get("title")) for r in ra if r.get("title")}; ta.discard("")
    tb = {_norm(r.get("title")) for r in rb if r.get("title")}; tb.discard("")
    shared = ta & tb

    a_text = " ".join((r.get("content") or r.get("text") or "") for r in ra).lower()
    b_text = " ".join((r.get("content") or r.get("text") or "") for r in rb).lower()
    a_stems, b_stems = _stems(a), _stems(b)
    b_in_a = any(s in a_text for s in b_stems)   # B's distinctive term in A's articles
    a_in_b = any(s in b_text for s in a_stems)   # A's distinctive term in B's articles
    mention = (b_in_a and a_in_b) if bidirectional else (b_in_a or a_in_b)

    known = len(shared) >= min_shared or mention
    return DocCooccurResult(len(shared), sorted(shared)[:4], mention, known)


def is_known(a: str, b: str, depth: int = 40, min_shared: int = 1,
             bidirectional: bool = False) -> bool:
    """Convenience boolean: is the (A, B) connection already in literature?"""
    return doc_cooccurrence(a, b, depth=depth, min_shared=min_shared,
                            bidirectional=bidirectional).known
