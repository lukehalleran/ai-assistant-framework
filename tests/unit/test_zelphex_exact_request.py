"""The originating medication query is a regression fixture, not a recognizer."""
import json

import pytest

from core.agentic.gate import evaluate_agentic_gate
from core.agentic.protocols import XMLMarkerHandler
from core.insight.deliberation import deterministic_fallback_spec, plan_deliberation
from core.insight.detector import detect_insight_request

QUERY = (
    "It probably does. Therapist raised point that needs to be both verified and "
    "considered/weighed. She said Zelphex has some research suggesting it helps "
    "with irritability in people on spectrum. I stopped a bit more than 7 weeks "
    "ago after a one month taper. Is it likely I will need another medication, or "
    "more likely this will be positive long term? Please use the pattern tool and "
    "look at data before and after total cessation. Utilize wiki, web, PubMed etc."
)

# The GUI sends this wrapped form. Keep the line breaks: the original bug was
# invisible to the single-line surrogate above.
GUI_QUERY = (
    "It probably does. Therapist raised point that needs\n"
    "  to be both verified and considered/weighed. She said Zelphex\n"
    "  has some research suggesting it helps with\n"
    "  irritability in people on the autism spectrum. I\n"
    "  stopped it a bit more than 7 weeks ago after a one-\n"
    "  month taper. Is it more likely that I’ll need\n"
    "  another medication, or more likely that stopping\n"
    "  will be positive long term? Please use the pattern\n"
    "  tool to compare data before and after total\n"
    "  cessation. Search broadly and deeply using PubMed,\n"
    "  web, and Wikipedia, with both keyword and semantic\n"
    "  matching. For PubMed, use multiple query formulations\n"
    "  and aim to retrieve at least 10 relevant papers when available.\n"
    "  Clearly separate direct evidence, related evidence, and irrelevant/\n"
    "  no-result searches."
)

PROPOSAL = {
    "analysis_kind": "event_phased",
    "claims": [
        {"claim_id": "A", "proposition": "Irritability differed across cessation phases", "dependencies": [], "authority": "assessment"},
        {"claim_id": "B", "proposition": "Published research supports the reported population-level effect", "dependencies": [], "authority": "assessment"},
        {"claim_id": "C", "proposition": "The personal timeline is consistent with Zelphex having affected irritability", "dependencies": ["A", "B"], "authority": "assessment"},
        {"claim_id": "D", "proposition": "A particular additional medication is needed", "dependencies": ["C"], "authority": "outside_authority"},
    ],
    "outcome_terms": ["irritability", "irritable"],
    "behavioral_indicators": ["short fuse", "snapped at someone"],
    "directional_indicators": {"increase": ["more irritable", "short fuse worsened"], "decrease": ["less irritable", "short fuse improved"]},
    "phases": [
        {"label": "stable-on", "metadata": {"start_offset_days": -120, "end_offset_days": -31}},
        {"label": "taper", "metadata": {"start_offset_days": -30, "end_offset_days": -1}},
        {"label": "off", "metadata": {"start_offset_days": 0, "end_offset_days": 60}},
    ],
    "anchor": {"label": "total cessation", "relative_value": 7, "relative_unit": "weeks", "direction": "past", "uncertainty_days": 7, "date_basis": "user said a bit more than seven weeks ago"},
    "admissible_source_classes": ["user", "users-own-note"],
    "supporting_facets": ["irritability while on Zelphex", "irritability after cessation"],
    "refuting_facets": ["stable irritability across cessation", "other irritability changes"],
    "rival_explanations": ["sleep", "other medication changes", "life events"],
    "requested_channels": ["pattern", "corpus", "notes", "pubmed", "web", "wiki"],
    "research_queries": {"pubmed": "cariprazine irritability autism spectrum clinical study", "web": "cariprazine irritability autism evidence review", "wiki": "cariprazine"},
    "assumptions": ["phase windows are descriptive, not causal"],
}


@pytest.mark.asyncio
async def test_exact_query_routes_to_generic_longitudinal_mode_not_file_access():
    decision = await evaluate_agentic_gate(QUERY, model_manager=None)
    assert decision.insight_intent["kind"] == "pattern_temporal"
    assert "file" not in decision.reason.lower()
    assert detect_insight_request(QUERY).kind == "pattern_temporal"


@pytest.mark.asyncio
async def test_gui_wrapped_query_routes_to_insight_and_not_file_access():
    decision = await evaluate_agentic_gate(GUI_QUERY, model_manager=None)
    assert detect_insight_request(GUI_QUERY).kind == "pattern_temporal"
    assert decision.modes == ["insight"]
    assert decision.veto_exempt is True


def test_quoted_prompt_mention_does_not_become_new_pattern_command():
    quoted = 'Here is a prompt I am debugging: > please use the pattern tool'
    assert detect_insight_request(quoted) is None


@pytest.mark.asyncio
async def test_exact_query_freezes_through_same_generic_planner_contract():
    class Planner:
        async def generate_once(self, *_args, **_kwargs):
            return json.dumps(PROPOSAL)

    frozen = await plan_deliberation(QUERY, model_manager=Planner())
    assert frozen.status == "ready"
    assert frozen.spec.analysis_kind == "event_phased"
    assert frozen.spec.outcome_terms[:2] == ["irritability", "irritable"]
    assert [phase.label for phase in frozen.spec.phases] == ["stable-on", "taper", "off"]
    assert {"pattern", "pubmed", "web", "wiki"} <= set(frozen.spec.requested_channels)
    assert frozen.spec.claims[-1].authority == "outside_authority"


@pytest.mark.asyncio
async def test_invalid_planner_json_gets_one_bounded_repair_attempt():
    class Planner:
        def __init__(self):
            self.calls = 0

        async def generate_once(self, *_args, **_kwargs):
            self.calls += 1
            return "not json" if self.calls == 1 else json.dumps(PROPOSAL)

    planner = Planner()
    frozen = await plan_deliberation(QUERY, model_manager=planner)
    assert frozen.status == "ready"
    assert frozen.planner_provenance == "llm-repaired"
    assert planner.calls == 2


def test_timeout_fallback_freezes_explicit_cessation_contract():
    spec = deterministic_fallback_spec(GUI_QUERY)
    assert spec is not None
    assert spec.analysis_kind == "event_phased"
    assert spec.outcome_terms == ["irritability"]
    assert spec.anchor.relative_value == 7
    assert [phase.label for phase in spec.phases] == ["before", "after"]
    assert {"corpus", "notes", "pattern"} <= set(spec.requested_channels)


def test_timeout_fallback_also_freezes_plain_before_after_sleep_request():
    query = (
        "Compare my sleep and daytime functioning before and after stopping Zelphex. "
        "Separate direct user statements, notes, behavioral proxies, summaries, hypotheses, and confounders."
    )
    spec = deterministic_fallback_spec(query)
    assert spec is not None
    assert spec.analysis_kind == "event_phased"
    assert spec.outcome_terms == ["sleep and daytime functioning"]
    assert spec.behavioral_indicators == ["sleep", "daytime functioning"]
    assert {"corpus", "notes", "pattern"} <= set(spec.requested_channels)


def test_timeout_fallback_freezes_established_three_phase_rerun_wording():
    spec = deterministic_fallback_spec(RERUN_QUERY)
    assert spec is not None
    assert spec.analysis_kind == "event_phased"
    assert spec.outcome_terms == ["irritability, sleep, and daytime functioning"]
    assert {"corpus", "notes", "pattern"} <= set(spec.requested_channels)


def test_timeout_fallback_freezes_stable_treatment_taper_after_stopping_wording():
    query = (
        "Re-run the established Zelphex cessation timeline, comparing "
        "irritability, sleep quality, and daytime functioning during stable "
        "treatment, taper, and after stopping. Separate my direct statements "
        "from notes, proxies, assistant summaries, hypotheses, confounders, "
        "and external research. Include phase counts, denominators, "
        "post-cessation quotes, channel status, and causal limitations."
    )
    spec = deterministic_fallback_spec(query)
    assert spec is not None
    assert spec.analysis_kind == "event_phased"
    assert spec.outcome_terms == ["irritability, sleep, and daytime functioning"]
    assert [phase.label for phase in spec.phases] == ["before", "after"]
    assert {"corpus", "notes", "pattern"} <= set(spec.requested_channels)


def test_fallback_research_queries_do_not_leak_zelphex_domain_terms():
    query = (
        "Compare my sleep and daytime functioning before and after stopping "
        "the medication. Use PubMed, web, and Wikipedia."
    )
    spec = deterministic_fallback_spec(query)
    assert spec is not None
    joined = " ".join(spec.research_queries.values()).lower()
    assert "autism" not in joined
    assert "irritability" not in joined
    assert "sleep" in joined


def test_xml_pubmed_and_pattern_tools_remain_executable_and_nonleaking():
    out = XMLMarkerHandler().parse_response("<pubmed>cariprazine autism irritability</pubmed>")
    assert out[0].wants_pubmed and not out[0].wants_answer
    spec = '{"outcome_terms":["irritability"],"phases":[]}'
    out = XMLMarkerHandler().parse_response(f"<pattern_scan spec='{spec}'></pattern_scan>")
    assert out[0].wants_pattern_scan


# The 2026-08-31 16:11 re-run wording (verbatim, GUI hard-wraps preserved).
# It dropped EVERY possessive-record cue the detector knew ("Compare
# irritability…" not "Compare my sleep…"; "the established timeline" not "my
# timeline"; bare channel word "notes" not "my notes") — the only matching
# deliberation arm was notes→PubMed at char 225, which the long-message
# incidental guard correctly suppressed, and the turn misrouted to agentic
# knowledge mode. The head-anchored "my … analysis" possessive is the cue
# that must carry it.
RERUN_QUERY = (
    "Re-run my Zelphex cessation analysis using the\n"
    "  established timeline. Compare irritability, sleep,\n"
    "  and daytime functioning before, during taper, and\n"
    "  after full cessation. Separate direct user\n"
    "  statements, user-authored notes, behavioral proxies,\n"
    "  assistant-generated summaries, hypotheses, and\n"
    "  confounders. Use pattern, corpus, notes, PubMed,\n"
    "  web, and Wikipedia. For PubMed, use multiple query\n"
    "  formulations and retrieve at least 10 relevant\n"
    "  papers when available. Report every channel's\n"
    "  status, the deterministic phase counts and\n"
    "  denominators, assessor status and confidence, direct\n"
    "  versus related versus irrelevant research, and\n"
    "  clearly separate descriptive findings from causal\n"
    "  conclusions."
)


class TestRerunWordingRoutesToInsight:
    def test_rerun_wording_detects_pattern_temporal(self):
        intent = detect_insight_request(RERUN_QUERY)
        assert intent is not None, "re-run wording must route to insight mode"
        assert intent.kind == "pattern_temporal"
        assert intent.raw_query == RERUN_QUERY

    def test_possessive_analysis_cue_is_tight(self):
        # Third-party possessives and operation-free mentions stay out.
        assert detect_insight_request(
            "I read my friend's analysis of the election and compared it "
            "to before"
        ) is None
        assert detect_insight_request(
            "The teacher said my analysis was wrong"
        ) is None
