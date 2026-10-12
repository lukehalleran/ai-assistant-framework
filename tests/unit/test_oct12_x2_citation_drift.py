"""X2 (2026-10-12): log-only citation-drift check + per-turn citation numbering.

class: BC-46, BC-75
"""
import types
from types import SimpleNamespace
import pytest

from core.agentic.tools import ToolExecutor
from core.citation_drift import find_citation_drift, normalize_number, parse_web_blocks
from core.citation_extractor import extract_web_citation_ids
from knowledge.web_search_manager import WebPage, WebSearchResult
import gui.handlers as handlers
from tests.unit.test_evidence_transport import _fake_decision_factory, _run, controller  # noqa: F401

IMF = "IMF cuts growth forecast to 2.6% for 2026, Reuters reports. Oil rose 4 percent."
HEADLINES = "21 troops have died in Iran war, officials said."


def test_drift_detected_for_number_from_other_source():
    resp = "21 US troops have reportedly died in the conflict [WEB_5]."
    items = find_citation_drift(resp, {"WEB_5": IMF, "WEB_17": HEADLINES})
    assert len(items) == 1
    assert items[0]["missing_numbers"] == ["21"]
    assert "US" in items[0]["missing_entities"]
    assert items[0]["cited"] == ["WEB_5"]


def test_correct_citation_not_flagged():
    resp = "21 troops have reportedly died [WEB_17]. Growth falls to 2.6% [WEB_5]."
    assert find_citation_drift(resp, {"WEB_5": IMF, "WEB_17": HEADLINES}) == []


def test_number_formats_normalised():
    assert normalize_number("1,200") == "1200"
    assert normalize_number("2.60") == "2.6"
    resp = "Shipments hit 1,200 units [WEB_1]."
    assert find_citation_drift(resp, {"WEB_1": "Shipments reached 1200 units"}) == []


def test_range_and_list_citations():
    assert extract_web_citation_ids("x [WEB_1-3] y [WEB_5]") == ["WEB_1", "WEB_2", "WEB_3", "WEB_5"]
    assert extract_web_citation_ids("x [WEB_2, WEB_4]") == ["WEB_2", "WEB_4"]
    srcs = {"WEB_1": "a", "WEB_2": "b", "WEB_3": "total 77 cases"}
    # number present in ONE of the range's sources -> supported
    assert find_citation_drift("There were 77 cases [WEB_1-3].", srcs) == []
    # absent from every cited source in the range -> drift
    assert len(find_citation_drift("There were 99 cases [WEB_1-3].", srcs)) == 1


def test_marker_after_period_attaches_to_previous_sentence():
    items = find_citation_drift("Deaths reached 99. [WEB_5]", {"WEB_5": IMF})
    assert len(items) == 1


def test_unknown_source_and_uncited_sentences_ignored():
    assert find_citation_drift("There are 42 cats [WEB_9].", {"WEB_1": "x"}) == []
    assert find_citation_drift("There are 42 cats.", {"WEB_1": "x"}) == []


def test_parse_web_blocks():
    text = "[WEB_1] **A** (u)\nfoo 1\n\n[WEB_2] **B** (u)\nbar 2"
    b = parse_web_blocks(text)
    assert set(b) == {"WEB_1", "WEB_2"} and "bar 2" in b["WEB_2"]


def test_handler_records_receipt_agentic_and_base_paths():
    resp = "21 US troops have reportedly died [WEB_5]."
    ctx = SimpleNamespace(telemetry={}, raw_context={}, citation_sources_text=(
        "[WEB_5] **IMF** (u)\n" + IMF + "\n\n[WEB_17] **H** (u)\n" + HEADLINES))
    handlers._record_citation_drift(ctx, resp)
    assert ctx.telemetry["citation_drift"][0]["missing_numbers"] == ["21"]

    page = WebPage(url="https://a.example/x", title="IMF", content=IMF, snippet=IMF, score=0.9, source="t")
    base = WebSearchResult(query="q", pages=[page])
    ctx2 = SimpleNamespace(telemetry={}, raw_context={"web_search_results": base})
    handlers._record_citation_drift(ctx2, "21 troops died [WEB_1].")
    assert len(ctx2.telemetry["citation_drift"]) == 1
    ctx3 = SimpleNamespace(telemetry={}, raw_context={"web_search_results": base})
    handlers._record_citation_drift(ctx3, "Growth is 2.6% [WEB_1].")
    assert ctx3.telemetry["citation_drift"] == []


# --- item 2: per-turn numbering ------------------------------------------------

def _exec():
    class _E:
        pass
    ex = _E()
    ex._current_web_source_map = {}
    ex._current_wiki_source_map = {}
    ex._merge_web_ids = types.MethodType(ToolExecutor._merge_web_ids, ex)
    return ex


def _pages(n, tag):
    return [WebPage(url=f"https://{tag}{i}.example/p", title=f"{tag}{i}", content=f"{tag} content {i}",
                    snippet="s", score=1.0 - i / 100, source="t") for i in range(n)]


@pytest.mark.asyncio
async def test_executor_map_does_not_leak_numbering_across_turns(controller):  # noqa: F811
    """Base prompt numbers WEB_1..5; the seeded loop must reuse those ids even
    after earlier turns filled the executor's session-wide map (live: WEB_21..25)."""
    controller._tool_executor._merge_web_ids(_pages(20, "old"))
    assert len(controller._tool_executor._current_web_source_map) == 20
    captured = {}
    controller._get_model_decision = _fake_decision_factory(capture=captured)
    await _run(
        controller, query="weather?", system_prompt="s", model_name="test-model",
        initial_search_terms=[], skip_initial_search=True,
        initial_context={"web_search_results": WebSearchResult(query="q", pages=_pages(5, "new"))},
    )
    ids = sorted(controller._tool_executor._current_web_source_map, key=lambda k: int(k.split("_")[1]))
    assert ids == [f"WEB_{i}" for i in range(1, 6)], ids
    assert "[WEB_1]" in captured["prompt"] and "[WEB_21]" not in captured["prompt"]
