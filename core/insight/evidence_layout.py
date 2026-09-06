"""
core/insight/evidence_layout.py

Module Contract
- Purpose: Fair evidence-block layout for insight-mode synthesis prompts.
  ``render_evidence_block`` (core.insight.provenance) is a hard character-cap
  renderer: it stops adding lines the moment the cap is hit, in whatever
  order the input list arrives in. Deliberation runs append the frozen
  scan's phase-comparison events AND the raw external-research sources
  (PubMed/web/Wolfram) onto the FRONT of the evidence list ahead of the
  sweep's personal evidence (gui.handlers._run_insight_mode) — a handful of
  ~800-char external abstracts can fill the entire render cap before a
  single personal item is ever seen (2026-09-06 live incident: 14 PubMed
  abstracts filled 12000 chars; all 36 personal sweep items were omitted,
  and the synthesis honestly reported "you don't yet have personal data").
- Inputs: a list of EvidenceItem (already deduped/labeled/sorted by the
  caller) + the render budget (max_chars, optional max_items) + an external
  evidence share (external_char_share) and hard external item count
  (external_max_items).
- Outputs: a REORDERED PERMUTATION of the input — nothing dropped, nothing
  invented. Computed evidence (deterministic pattern/phase aggregates)
  always renders first; personal sweep evidence is week-interleaved
  (core.insight.sweep.interleave_evidence_for_coverage) and placed next,
  bounded to the remaining budget after a capped external-evidence
  allocation; anything that doesn't fit the render budget is appended at
  the end (still present, so render_evidence_block's own omission
  accounting stays truthful about what didn't make the cut).
- Side effects: none — pure functions over their inputs; no I/O, no config
  reads (callers pass explicit caps, e.g. from core.insight.sweep.default_caps()).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.insight.provenance import format_evidence_line
from core.insight.sweep import _clip, interleave_evidence_for_coverage
from core.insight.types import EvidenceItem

# Facet markers minted by the pattern engine / longitudinal deliberation
# stage for DETERMINISTIC, already-computed evidence (memory/pattern_engine
# rolling exemplars use "pattern:<dimension>"; the frozen phase-comparison
# scan's events use the literal facet below — see
# core.insight.temporal.run_pattern_stage and gui.handlers._run_insight_mode's
# _interleave_phase_events loop). These are never week-interleaved with the
# generic sweep — they restate a specific computed number and must stay in
# the order the engine produced them.
COMPUTED_FACET_PREFIX = "pattern:"
COMPUTED_FACETS = frozenset({"deliberation:phase-evidence"})

# Facet marker for raw external-research sources (PubMed/web/arXiv/
# StackExchange/Wolfram) appended by the deliberation stage.
EXTERNAL_FACET = "deliberation:research"
EXTERNAL_LABELS = frozenset({"external-research", "computed-evidence"})

# An external-evidence run is useful corroboration, not the whole report —
# cap its share of the render budget so personal history is never starved
# behind it, and cap its raw item count independent of length (many short
# abstracts can still crowd out everything else).
DEFAULT_EXTERNAL_CHAR_SHARE = 0.25
DEFAULT_EXTERNAL_MAX_ITEMS = 8


def is_computed_evidence(item: EvidenceItem) -> bool:
    """True for deterministic engine-computed evidence (pattern rollups,
    frozen phase-comparison events) — never reordered relative to itself."""
    facet = item.facet or ""
    return facet in COMPUTED_FACETS or facet.startswith(COMPUTED_FACET_PREFIX)


def is_external_evidence(item: EvidenceItem) -> bool:
    """True for raw external-research sources (PubMed/web/Wolfram/etc.)."""
    if item.facet == EXTERNAL_FACET:
        return True
    return item.stance_label in EXTERNAL_LABELS


def partition_evidence(
    items: list[EvidenceItem],
) -> tuple[list[EvidenceItem], list[EvidenceItem], list[EvidenceItem]]:
    """Split ``items`` into (computed, personal, external), each in stable
    (input) order. ``personal`` is everything that is neither computed nor
    external — the cross-store sweep's own evidence."""
    computed: list[EvidenceItem] = []
    personal: list[EvidenceItem] = []
    external: list[EvidenceItem] = []
    for item in items:
        if is_computed_evidence(item):
            computed.append(item)
        elif is_external_evidence(item):
            external.append(item)
        else:
            personal.append(item)
    return computed, personal, external


def line_cost(item: EvidenceItem) -> int:
    """Approximate rendered cost of one item, matching
    ``provenance.render_evidence_block``'s per-line accounting
    (``len(line) + 1`` for the joining newline). The index is fixed at 1 —
    layout is a permutation pass that runs BEFORE final numbering, so this
    is a budget estimate, not the exact rendered text."""
    return len(format_evidence_line(1, item)) + 1


def external_evidence_item(src: dict, *, snippet_chars: int) -> EvidenceItem:
    """Build the EvidenceItem for one raw external-research source dict
    (PubMed/web/Wolfram row shape — see gui.handlers._run_insight_mode's
    ``_deliberation.external_evidence`` loop, which this factors out of)."""
    raw = str(
        src.get("snippet") or src.get("abstract")
        or src.get("text") or src.get("content")
        or src.get("document") or src.get("title") or ""
    )
    return EvidenceItem(
        doc_id=str(src.get("source_id") or src.get("id") or src.get("pmid") or ""),
        text=_clip(raw, snippet_chars),
        date=src.get("date") or src.get("published_date"),
        collection=str(src.get("source_class") or "research"),
        stance_label=(
            "computed-evidence" if src.get("source_class") == "wolfram"
            else "external-research"
        ),
        facet=EXTERNAL_FACET,
    )


@dataclass
class LayoutReport:
    """Accounting for a layout pass — how many items of each kind actually
    landed inside the render budget vs. spilled to the tail."""

    computed: int
    personal_in_zone: int
    external_in_zone: int
    personal_tail: int
    external_tail: int


def _take_within_budget(
    pool: list[EvidenceItem], *, char_budget: int, item_budget: Optional[int],
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """Walk ``pool`` in the order given (the caller's contract — computed
    first, personal already week-interleaved, external in sweep order),
    taking while both the running char cost and item count stay within
    budget. Returns (taken, rest) — a stable partition, never a drop."""
    taken: list[EvidenceItem] = []
    rest: list[EvidenceItem] = []
    used = 0
    for idx, entry in enumerate(pool):
        if item_budget is not None and len(taken) >= item_budget:
            rest.extend(pool[idx:])
            break
        cost = line_cost(entry)
        if used + cost > char_budget:
            # One oversized entry must not hide everything behind it (live
            # 2026-09-06: 9 of 58 rendered, 49 "omitted for space" while the
            # block was a third full). Skip it to the tail and keep walking.
            rest.append(entry)
            continue
        taken.append(entry)
        used += cost
    return taken, rest


def clip_evidence_texts(items: list[EvidenceItem], *, max_chars: int) -> list[EvidenceItem]:
    """Cap every item's text at ``max_chars`` (sweep's ellipsis clip). Phase
    events and window-scan chunks join the evidence list WITHOUT passing
    ``sweep._finalize``, so a whole note or turn could ride in unclipped and
    starve the render budget. Returns the same list (texts mutated)."""
    # lazy import: cycle (sweep imports provenance; layout imports both)
    from core.insight.sweep import _clip
    for item in items:
        text = item.text or ""
        if len(text) > max_chars:
            item.text = _clip(text, max_chars)
    return items


def layout_evidence_with_report(
    items: list[EvidenceItem],
    *,
    max_chars: int,
    max_items: Optional[int] = None,
    external_char_share: float = DEFAULT_EXTERNAL_CHAR_SHARE,
    external_max_items: int = DEFAULT_EXTERNAL_MAX_ITEMS,
) -> tuple[list[EvidenceItem], LayoutReport]:
    """Return a fair PERMUTATION of ``items`` for rendering, plus a report of
    how the render budget was allocated.

    Algorithm: partition into (computed, personal, external); personal is
    week-interleaved for date-range coverage (computed is never reordered —
    it restates a specific already-computed number in engine order).
    External evidence is capped to a bounded share of the render budget
    (``external_char_share`` of ``max_chars``, and at most
    ``external_max_items``) so a handful of long research abstracts cannot
    crowd personal history out of the render window entirely. Whatever
    remains of the budget after that allocation goes to computed+personal,
    computed first. Result order: [computed+personal that fit the zone]
    + [external that fit its share] + [computed+personal overflow]
    + [external overflow] — a pure permutation (``len(result) ==
    len(items)`` always), so ``render_evidence_block``'s own omission
    accounting over the returned order stays truthful about what didn't
    make the cut.
    """
    computed, personal, external = partition_evidence(items)
    personal = interleave_evidence_for_coverage(personal)

    external_budget_chars = int(max_chars * external_char_share)
    ext_zone, ext_tail = _take_within_budget(
        external, char_budget=external_budget_chars, item_budget=external_max_items,
    )

    zone_chars = max(0, max_chars - sum(line_cost(i) for i in ext_zone))
    zone_items = None if max_items is None else max(0, max_items - len(ext_zone))
    core_zone, core_tail = _take_within_budget(
        computed + personal, char_budget=zone_chars, item_budget=zone_items,
    )
    personal_in_zone = sum(1 for i in core_zone if not is_computed_evidence(i))
    personal_tail = sum(1 for i in core_tail if not is_computed_evidence(i))

    result = core_zone + ext_zone + core_tail + ext_tail
    report = LayoutReport(
        computed=len(computed),
        personal_in_zone=personal_in_zone,
        external_in_zone=len(ext_zone),
        personal_tail=personal_tail,
        external_tail=len(ext_tail),
    )
    return result, report


def layout_evidence(
    items: list[EvidenceItem],
    *,
    max_chars: int,
    max_items: Optional[int] = None,
    external_char_share: float = DEFAULT_EXTERNAL_CHAR_SHARE,
    external_max_items: int = DEFAULT_EXTERNAL_MAX_ITEMS,
) -> list[EvidenceItem]:
    """Thin wrapper over ``layout_evidence_with_report`` for callers that
    only need the reordered list."""
    result, _ = layout_evidence_with_report(
        items, max_chars=max_chars, max_items=max_items,
        external_char_share=external_char_share,
        external_max_items=external_max_items,
    )
    return result
