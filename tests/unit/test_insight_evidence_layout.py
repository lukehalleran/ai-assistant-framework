"""
Evidence-layout fairness (2026-09-06).

Deliberation runs append computed aggregates + raw external-research sources
onto the FRONT of the evidence list ahead of the sweep's personal evidence
(gui.handlers._run_insight_mode). render_evidence_block is a hard
character-cap renderer that stops at whatever order it's handed — a handful
of long external abstracts could fill the entire render cap before a single
personal item was ever seen (2026-09-06 live incident). core.insight.
evidence_layout.layout_evidence fixes this with a fair, pure-permutation
reordering. Every expected value below is DERIVED from the deployed
functions (line_cost / format_evidence_line / render_evidence_block) over
synthetic inputs, never copied from the incident.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from core.insight.evidence_layout import (
    DEFAULT_EXTERNAL_CHAR_SHARE,
    EXTERNAL_FACET,
    LayoutReport,
    external_evidence_item,
    is_computed_evidence,
    is_external_evidence,
    layout_evidence,
    layout_evidence_with_report,
    line_cost,
    partition_evidence,
)
from core.insight.provenance import format_evidence_line, render_evidence_block
from core.insight.sweep import default_caps, week_bucket_key
from core.insight.types import EvidenceItem, FacetPlan, FacetQuery


# ---------------------------------------------------------------------------
# Synthetic-data helpers — costs are DERIVED from the deployed
# format_evidence_line/line_cost, never hand-computed, so every scenario
# below is exact-by-construction rather than tuned to an observed number.
# ---------------------------------------------------------------------------

def _item(doc_id, *, text, date, collection="conversations",
          stance_label="user-stated", facet="", is_appraisal=False,
          speaker=""):
    return EvidenceItem(
        doc_id=doc_id, text=text, date=date, collection=collection,
        stance_label=stance_label, facet=facet, is_appraisal=is_appraisal,
        speaker=speaker,
    )


def _base_cost(*, collection, stance_label, date, is_appraisal=False):
    """Rendered cost of an item of this shape with EMPTY text — the fixed
    per-line overhead (date/source/label/quotes/markers)."""
    return line_cost(_item(
        "base", text="", date=date, collection=collection,
        stance_label=stance_label, is_appraisal=is_appraisal,
    ))


def _item_with_cost(doc_id, cost, *, collection, stance_label, date,
                     facet="", is_appraisal=False):
    """Build an item whose exact rendered ``line_cost`` equals ``cost`` by
    padding the text — makes render-budget arithmetic exact instead of
    approximate."""
    base = _base_cost(collection=collection, stance_label=stance_label,
                       date=date, is_appraisal=is_appraisal)
    assert cost >= base, f"target cost {cost} below fixed overhead {base}"
    return _item(
        doc_id, text="x" * (cost - base), date=date, collection=collection,
        stance_label=stance_label, facet=facet, is_appraisal=is_appraisal,
    )


def _rendered_prefix_len(out: list[EvidenceItem], rendered_text: str) -> int:
    """How many leading items of ``out`` were actually rendered by the
    deployed render_evidence_block, derived by regenerating each expected
    numbered line (format_evidence_line) and matching it against the
    rendered output in order."""
    rendered_lines = rendered_text.splitlines()
    count = 0
    for idx, item in enumerate(out, start=1):
        expected = format_evidence_line(idx, item)
        if count < len(rendered_lines) and rendered_lines[count] == expected:
            count += 1
        else:
            break
    return count


def _share_bounded_scenario():
    """6 external + 3 personal items sized (via _item_with_cost) so the
    layout's two budget zones have ZERO leftover chars — the render cutoff
    is then exact-by-construction: 2 external items and all 3 personal
    items fit; 4 external items spill to the tail."""
    ext_date = "2026-01-05"
    pers_date = "2026-08-10"
    base_ext = _base_cost(collection="research", stance_label="external-research",
                           date=ext_date)
    base_pers = _base_cost(collection="conversations", stance_label="user-stated",
                            date=pers_date)
    ce = max(base_ext, base_pers) + 40
    cp = 2 * ce
    external = [
        _item_with_cost(f"e{i}", ce, collection="research",
                         stance_label="external-research", date=ext_date,
                         facet=EXTERNAL_FACET)
        for i in range(6)
    ]
    personal = [
        _item_with_cost(f"p{i}", cp, collection="conversations",
                         stance_label="user-stated", date=pers_date)
        for i in range(3)
    ]
    items = external + personal  # external first, as the handler appends it
    max_chars = 8 * ce
    return items, max_chars


# ---------------------------------------------------------------------------
# 1. Pure permutation
# ---------------------------------------------------------------------------

def test_pure_permutation_nothing_dropped_or_invented():
    items = []
    for i in range(15):
        date = f"2026-0{(i % 9) + 1}-0{(i % 9) + 1}"
        kind = i % 3
        if kind == 0:
            items.append(_item(f"c{i}", text=f"computed {i}", date=date,
                                facet="pattern:mood"))
        elif kind == 1:
            items.append(_item(f"p{i}", text=f"personal {i}", date=date))
        else:
            items.append(_item(f"e{i}", text=f"external {i}", date=date,
                                collection="research",
                                stance_label="external-research",
                                facet=EXTERNAL_FACET))
    out = layout_evidence(items, max_chars=1_000_000, max_items=None)
    assert sorted(i.doc_id for i in out) == sorted(i.doc_id for i in items)
    assert len(out) == len(items)


# ---------------------------------------------------------------------------
# 2. External share bounded
# ---------------------------------------------------------------------------

def test_external_evidence_share_of_render_budget_is_bounded():
    items, max_chars = _share_bounded_scenario()
    out, _report = layout_evidence_with_report(items, max_chars=max_chars)
    rendered = render_evidence_block(out, max_chars=max_chars)
    rendered_count = _rendered_prefix_len(out, rendered)

    external_ids = {i.doc_id for i in items if i.facet == EXTERNAL_FACET}
    rendered_prefix = out[:rendered_count]
    external_in_prefix = [i for i in rendered_prefix if i.doc_id in external_ids]

    external_cost = sum(line_cost(i) for i in external_in_prefix)
    assert external_cost <= max_chars * DEFAULT_EXTERNAL_CHAR_SHARE
    # Bounded, not eliminated, and not all of them — some external evidence
    # renders, but nowhere near the full 6.
    assert 0 < len(external_in_prefix) < len(external_ids)


# ---------------------------------------------------------------------------
# 3. Personal evidence is not starved by huge external evidence
# ---------------------------------------------------------------------------

def test_personal_evidence_survives_a_dominating_external_item():
    distinct_weeks = ["2026-01-05", "2026-01-19", "2026-02-02"]
    base_pers = _base_cost(collection="conversations", stance_label="user-stated",
                            date=distinct_weeks[0])
    cp = base_pers + 20
    personal = [
        _item_with_cost(f"p{i}", cp, collection="conversations",
                         stance_label="user-stated", date=d)
        for i, d in enumerate(distinct_weeks)
    ]
    max_chars = 3 * cp + 100  # just enough room for the 3 personal items

    # One external item costed to consume the ENTIRE render budget by
    # itself — with no fairness pass this alone would starve every personal
    # item out of the render window.
    external = [_item_with_cost(
        "e0", max_chars, collection="research", stance_label="external-research",
        date="2026-01-01", facet=EXTERNAL_FACET,
    )]
    items = external + personal

    out, report = layout_evidence_with_report(items, max_chars=max_chars)
    assert report.external_in_zone == 0  # too big for its 25% share, excluded entirely

    rendered = render_evidence_block(out, max_chars=max_chars)
    rendered_count = _rendered_prefix_len(out, rendered)
    rendered_weeks = {
        week_bucket_key(i.date) for i in out[:rendered_count]
        if i.collection == "conversations"
    }
    assert len(rendered_weeks) >= len(distinct_weeks)


# ---------------------------------------------------------------------------
# 4. Computed evidence renders first
# ---------------------------------------------------------------------------

def test_computed_evidence_always_leads():
    computed = [
        _item("c0", text="computed rollup A", date="2026-03-01", facet="pattern:mood"),
        _item("c1", text="computed rollup B", date="2026-03-02", facet="pattern:mood"),
    ]
    personal = [
        _item("p0", text="personal note A", date="2026-03-11"),
        _item("p1", text="personal note B", date="2026-03-12"),
    ]
    external = [
        _item("e0", text="ext A", date="2026-03-21", collection="research",
              stance_label="external-research", facet=EXTERNAL_FACET),
        _item("e1", text="ext B", date="2026-03-22", collection="research",
              stance_label="external-research", facet=EXTERNAL_FACET),
    ]
    items = personal + external + computed  # scattered on purpose

    out, report = layout_evidence_with_report(items, max_chars=1_000_000)
    assert report.computed == len(computed)
    assert [i.doc_id for i in out[:2]] == [c.doc_id for c in computed]
    assert all(is_computed_evidence(i) for i in out[:2])


# ---------------------------------------------------------------------------
# 5. rendered_count matches the report's zone accounting
# ---------------------------------------------------------------------------

def test_rendered_count_equals_reported_zone_totals():
    items, max_chars = _share_bounded_scenario()
    out, report = layout_evidence_with_report(items, max_chars=max_chars)
    rendered = render_evidence_block(out, max_chars=max_chars)
    rendered_count = _rendered_prefix_len(out, rendered)
    assert rendered_count == report.computed + report.personal_in_zone + report.external_in_zone


# ---------------------------------------------------------------------------
# 6. A non-empty tail is disclosed honestly (never silently dropped)
# ---------------------------------------------------------------------------

def test_nonempty_tail_is_disclosed_by_week():
    items, max_chars = _share_bounded_scenario()
    out, report = layout_evidence_with_report(items, max_chars=max_chars)
    assert report.external_tail > 0
    rendered = render_evidence_block(out, max_chars=max_chars)
    assert "omitted for space" in rendered
    assert "week of" in rendered


# ---------------------------------------------------------------------------
# 7. max_items bounds the zone without dropping anything
# ---------------------------------------------------------------------------

def test_max_items_bounds_zone_but_keeps_every_item():
    base_pers = _base_cost(collection="conversations", stance_label="user-stated",
                            date="2026-04-01")
    cp = base_pers + 5
    personal = [
        _item_with_cost(f"p{i}", cp, collection="conversations",
                         stance_label="user-stated", date="2026-04-01")
        for i in range(15)
    ]
    max_chars = 10 * cp  # exactly the cost of 10 items

    out, report = layout_evidence_with_report(personal, max_chars=max_chars, max_items=10)
    assert len(out) == len(personal)  # a permutation — nothing dropped
    assert report.personal_in_zone <= 10

    rendered = render_evidence_block(out, max_chars=max_chars)
    rendered_count = _rendered_prefix_len(out, rendered)
    assert rendered_count <= 10


# ---------------------------------------------------------------------------
# 8. external_evidence_item clips to the configured snippet size
# ---------------------------------------------------------------------------

def test_external_evidence_item_clips_to_configured_snippet_chars():
    caps = default_caps()
    snippet_chars = caps["external_snippet_chars"]
    item = external_evidence_item({"abstract": "x" * 5000}, snippet_chars=snippet_chars)
    assert len(item.text) == snippet_chars + 1
    assert item.text.endswith("…")
    assert item.facet == EXTERNAL_FACET
    assert item.stance_label == "external-research"


# ---------------------------------------------------------------------------
# 9. format_evidence_line / render_evidence_block extraction is behavior-preserving
# ---------------------------------------------------------------------------

def test_render_single_item_matches_format_evidence_line():
    item = _item("only", text="a single fact", date="2026-05-01")
    block = render_evidence_block([item])
    assert block.splitlines()[0] == format_evidence_line(1, item)


# ---------------------------------------------------------------------------
# 10. Handler wiring: pattern_temporal run keeps personal sweep evidence in
#     the synthesis prompt despite a large external-research batch.
# ---------------------------------------------------------------------------

class TestHandlerEvidenceLayoutWiring:
    def test_pattern_run_keeps_personal_evidence_visible(self, monkeypatch):
        import core.insight.coordinator as coordinator_mod
        import core.insight.facets as facets_mod
        import core.insight.sweep as sweep_mod
        import core.insight.synthesizer as synth_mod
        import gui.handlers as handlers

        personal_items = [
            EvidenceItem(
                doc_id=f"p{i}", text=f"you mentioned mood note {i} on that day",
                date=f"2026-06-{i + 1:02d}", collection="conversations",
                speaker="user", stance_label="user-stated",
            )
            for i in range(5)
        ]
        monkeypatch.setattr(facets_mod, "decompose", AsyncMock(
            return_value=FacetPlan(
                facets=[FacetQuery(name="f", query_text="mood")], claims=["c"],
            )
        ))
        monkeypatch.setattr(sweep_mod, "run_sweep", AsyncMock(return_value=list(personal_items)))
        monkeypatch.setattr(handlers, "_dispatch_storage", lambda *a, **k: None)

        captured = {}

        async def _fake_synthesize_stream(intent, evidence, assessment, **kw):
            captured["evidence"] = evidence
            yield "report"

        monkeypatch.setattr(synth_mod, "synthesize_stream", _fake_synthesize_stream)

        class _FakeCoordinator:
            def __init__(self, **_kw):
                pass

            async def run(self, query):
                return SimpleNamespace(
                    freeze=SimpleNamespace(status="failed", spec=None),
                    scan=None,
                    internal_events=[],
                    external_evidence=[
                        {"abstract": "x" * 3000, "date": "2026-01-01",
                         "source_class": "pubmed", "id": f"ext{i}"}
                        for i in range(20)
                    ],
                    channels=[], claim_chain=[], manifest={},
                )

        monkeypatch.setattr(coordinator_mod, "LongitudinalDeliberationCoordinator", _FakeCoordinator)

        orchestrator = MagicMock()
        orchestrator.model_manager.get_active_model_name = MagicMock(return_value="kimi-3")
        orchestrator.memory_system.chroma_store = MagicMock()
        ctx = handlers.SubmitContext(
            user_text="how has my mood trended", files=None, history=[],
            use_raw_gpt=False, orchestrator=orchestrator, personality=None,
            fast_mode=False, conversation_logger=None, file_names=[],
            merged_input="how has my mood trended", files_result=None,
        )
        ctx.raw_context = {}
        ctx.gate_decision = MagicMock()
        ctx.gate_decision.insight_intent = {
            "kind": "pattern_temporal", "theme": "mood",
            "wants_document": False, "raw_query": "how has my mood trended",
        }

        async def _run():
            chunks = []
            async for c in handlers._run_insight_mode(ctx):
                chunks.append(c)
            return chunks

        asyncio.run(_run())

        evidence = captured["evidence"]
        rendered = render_evidence_block(evidence)
        assert all(p.text in rendered for p in personal_items)
