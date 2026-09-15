"""Search-term identity scoping (2026-09-12): a location or institution the
resolvers know about must only ride into generated web-search terms when the
user's OWN query gives a reason for it.

Live incident: during a weighted-voting thought experiment the user wrote
"I am referring to voting". The trigger LLM invented the search terms
"current voting issues Illinois" and "Vermont Wrenfield voting information" — a
BARE state name (no city to anchor the existing location backstop on) and an
unprompted school name, neither justified by anything in the query.

strip_unjustified_location() already stripped a city; it never stripped the
bare state. apply_institution() only ever ADDS a school name; nothing
removed one the LLM invented. This file drives BOTH search-term producers —
utils.web_search_trigger._classify_with_llm_unified (the LLM trigger) and
knowledge.web_search_manager.WebSearchManager.decompose_query — through
their real, deployed code with fake model outputs, exactly like the live
turn, and checks the terms and the prompts they built.

Class tags: BC-59 (owner identity leaking into a query), BC-61 (state name
surviving a city-only backstop), BC-58 (two producers, one policy).
"""
from __future__ import annotations

import asyncio
import json
import weakref

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures / drivers
# ---------------------------------------------------------------------------

class _RecordingTriggerMM:
    """Fake ModelManager for utils.web_search_trigger._classify_with_llm_unified
    — async generate_once, returns a canned raw JSON string, records the
    prompt it was given."""

    def __init__(self, raw: str):
        self.raw = raw
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.raw


def _trigger_raw(search_terms, should_search=True):
    return json.dumps({
        "should_search": should_search,
        "confidence": 0.8,
        "search_terms": search_terms,
        "search_depth": "standard",
        "num_searches": max(1, len(search_terms)),
        "reasoning": "test fixture",
    })


def classify(monkeypatch, query, search_terms, location=None, institution=None,
             should_search=True, conversation_context=None):
    """Drive the deployed trigger classifier end to end. Returns
    (parsed_response, recorded_prompts)."""
    import utils.institution_resolver as ir
    import utils.location_resolver as lr
    import utils.web_search_trigger as wst

    monkeypatch.setattr(ir, "get_user_institution", lambda: institution)
    monkeypatch.setattr(ir, "get_user_anchors", lambda: [institution] if institution else [])
    monkeypatch.setattr(lr, "get_user_location", lambda: location)

    mm = _RecordingTriggerMM(_trigger_raw(search_terms, should_search=should_search))
    parsed = asyncio.run(wst._classify_with_llm_unified(
        query, mm, conversation_context=conversation_context))
    return parsed, mm.prompts


def _decompose_raw(sub_queries):
    lines = ["SHOULD_SPLIT: yes", "CONFIDENCE: 0.9", "REASON: multiple facets",
              "SUB_QUERIES:"]
    lines += [f"- {q}" for q in sub_queries]
    return "\n".join(lines)


def decompose(monkeypatch, query, sub_queries, location=None, institution=None):
    """Drive the deployed WebSearchManager.decompose_query end to end.
    Returns (QueryDecomposition, recorded_prompts)."""
    import knowledge.web_search_manager as wsm
    import utils.institution_resolver as ir
    import utils.location_resolver as lr

    monkeypatch.setattr(ir, "get_user_institution", lambda: institution)
    monkeypatch.setattr(lr, "get_user_location", lambda: location)
    monkeypatch.setattr(wsm, "_LIVE_RATE_LIMITERS", weakref.WeakSet())

    limiter = wsm.WebSearchRateLimiter(daily_limit=100, state_file="/dev/null")
    monkeypatch.setattr(limiter, "_save_state", lambda: None)
    manager = wsm.WebSearchManager(api_key="synthetic", rate_limiter=limiter)

    prompts = []

    class _FakeDecomposeMM:
        def __init__(self, *a, **kw):
            pass

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return _decompose_raw(sub_queries)

    monkeypatch.setattr("models.model_manager.ModelManager", _FakeDecomposeMM)

    decomposition = asyncio.run(manager.decompose_query(query))
    return decomposition, prompts


def _joined(terms):
    return " ".join(terms or [])


WRAPPED_VOTING_QUERY = "I am referring\n  to voting"
CLEAN_VOTING_QUERY = "I am referring to voting"


# ---------------------------------------------------------------------------
# The live incident, both producers, clean + wrapped query text
# ---------------------------------------------------------------------------

class TestLiveVotingIncidentTrigger:

    @pytest.mark.parametrize("query", [CLEAN_VOTING_QUERY, WRAPPED_VOTING_QUERY])
    def test_neither_state_nor_school_survives(self, monkeypatch, query):
        parsed, prompts = classify(
            monkeypatch, query,
            ["current voting issues Illinois", "Vermont Wrenfield voting information"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert parsed is not None
        joined_lower = _joined(parsed.search_terms).lower()
        assert "illinois" not in joined_lower
        assert "vermont" not in joined_lower
        assert "wrenfield" not in joined_lower

    @pytest.mark.parametrize("query", [CLEAN_VOTING_QUERY, WRAPPED_VOTING_QUERY])
    def test_prompt_carries_no_school_line(self, monkeypatch, query):
        _, prompts = classify(
            monkeypatch, query,
            ["current voting issues Illinois", "Vermont Wrenfield voting information"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert prompts, "the trigger must have been called"
        assert "User's school:" not in prompts[0]


class TestLiveVotingIncidentDecompose:

    @pytest.mark.parametrize("query", [CLEAN_VOTING_QUERY, WRAPPED_VOTING_QUERY])
    def test_neither_state_nor_school_survives(self, monkeypatch, query):
        decomposition, prompts = decompose(
            monkeypatch, query,
            ["current voting issues Illinois", "Vermont Wrenfield voting information"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        joined_lower = _joined(decomposition.sub_queries).lower()
        assert "illinois" not in joined_lower
        assert "vermont" not in joined_lower
        assert "wrenfield" not in joined_lower

    @pytest.mark.parametrize("query", [CLEAN_VOTING_QUERY, WRAPPED_VOTING_QUERY])
    def test_prompt_carries_no_school_line(self, monkeypatch, query):
        _, prompts = decompose(
            monkeypatch, query,
            ["current voting issues Illinois", "Vermont Wrenfield voting information"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert prompts, "decompose_query must have called the model"
        assert "User's school:" not in prompts[0]


# ---------------------------------------------------------------------------
# State-name justification (kept vs stripped), both producers
# ---------------------------------------------------------------------------

class TestStateJustificationTrigger:

    def test_full_state_name_in_query_keeps_it(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "What are Illinois voting issues?",
            ["Illinois voting issues 2026"],
            location="Springfield, Illinois",
        )
        assert "illinois" in _joined(parsed.search_terms).lower()

    def test_uppercase_abbreviation_in_query_keeps_it(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "What are IL voting deadlines?",
            ["IL voting deadlines 2026"],
            location="Springfield, IL",
        )
        assert "IL" in _joined(parsed.search_terms)

    def test_lowercase_in_is_not_a_state_code(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "what should I do in the morning",
            ["IL morning routine ideas"],
            location="Springfield, IL",
        )
        assert "IL" not in _joined(parsed.search_terms)

    def test_lowercase_in_is_not_indiana_either(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "what should I do in the morning",
            ["IN morning routine ideas"],
            location="Springfield, IN",
        )
        assert "IN" not in _joined(parsed.search_terms)


class TestStateJustificationDecompose:

    def test_full_state_name_in_query_keeps_it(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "What are Illinois voting issues?",
            ["Illinois voting issues 2026"],
            location="Springfield, Illinois",
        )
        assert "illinois" in _joined(decomposition.sub_queries).lower()

    def test_lowercase_in_is_not_a_state_code(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "what should I do in the morning",
            ["IL morning routine ideas"],
            location="Springfield, IL",
        )
        assert "IL" not in _joined(decomposition.sub_queries)


# ---------------------------------------------------------------------------
# Institution justification (kept/added vs stripped), both producers
# ---------------------------------------------------------------------------

class TestInstitutionJustificationTrigger:

    def test_own_school_named_non_logistics_kept_and_prompted(self, monkeypatch):
        parsed, prompts = classify(
            monkeypatch, "What is Vermont Wrenfield's policy on student voting?",
            ["Vermont Wrenfield policy on student voting 2026"],
            institution="Vermont Wrenfield",
        )
        assert "vermont wrenfield" in _joined(parsed.search_terms).lower()
        assert "User's school: Vermont Wrenfield" in prompts[0]

    def test_different_named_school_kept_users_own_removed(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "How does Quellmoor University handle student voting?",
            ["Quellmoor University student voting", "Vermont Wrenfield student voting"],
            institution="Vermont Wrenfield",
        )
        joined_lower = _joined(parsed.search_terms).lower()
        assert "quellmoor university" in joined_lower
        assert "vermont wrenfield" not in joined_lower

    def test_remote_school_deadline_gains_school_loses_location(self, monkeypatch):
        parsed, prompts = classify(
            monkeypatch, "when is the drop deadline",
            ["drop deadline Springfield Illinois"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert parsed.search_terms == ["Vermont Wrenfield drop deadline"]
        assert "User's school: Vermont Wrenfield" in prompts[0]

    def test_overlap_location_equals_institution_city(self, monkeypatch):
        """location='Marrowby, Vermont' + institution='Vermont Wrenfield': the
        bare-state strip must not amputate 'Vermont Wrenfield' into 'Wrenfield'."""
        parsed, _ = classify(
            monkeypatch, "I am referring to voting",
            ["Vermont Wrenfield voting information", "Vermont voting issues"],
            location="Marrowby, Vermont", institution="Vermont Wrenfield",
        )
        assert parsed.search_terms == ["voting information", "voting issues"]

    def test_overlap_own_school_deadline_drops_trailing_city_state(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "when is the Vermont Wrenfield drop deadline",
            ["Vermont Wrenfield drop deadline Marrowby, Vermont"],
            location="Marrowby, Vermont", institution="Vermont Wrenfield",
        )
        assert parsed.search_terms == ["Vermont Wrenfield drop deadline"]

    def test_weather_still_localizes(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "what's the weather tomorrow",
            ["weather tomorrow Springfield, Illinois"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert parsed.search_terms == ["weather tomorrow Springfield, Illinois"]


class TestInstitutionJustificationDecompose:

    def test_own_school_named_non_logistics_kept_and_prompted(self, monkeypatch):
        decomposition, prompts = decompose(
            monkeypatch, "What is Vermont Wrenfield's policy on student voting?",
            ["Vermont Wrenfield policy on student voting 2026"],
            institution="Vermont Wrenfield",
        )
        assert "vermont wrenfield" in _joined(decomposition.sub_queries).lower()
        assert "User's school: Vermont Wrenfield" in prompts[0]

    def test_different_named_school_kept_users_own_removed(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "How does Quellmoor University handle student voting?",
            ["Quellmoor University student voting", "Vermont Wrenfield student voting"],
            institution="Vermont Wrenfield",
        )
        joined_lower = _joined(decomposition.sub_queries).lower()
        assert "quellmoor university" in joined_lower
        assert "vermont wrenfield" not in joined_lower

    def test_remote_school_deadline_gains_school_loses_location(self, monkeypatch):
        decomposition, prompts = decompose(
            monkeypatch, "when is the drop deadline",
            ["drop deadline Springfield Illinois"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert decomposition.sub_queries == ["Vermont Wrenfield drop deadline"]
        assert "User's school: Vermont Wrenfield" in prompts[0]

    def test_overlap_location_equals_institution_city(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "I am referring to voting",
            ["Vermont Wrenfield voting information", "Vermont voting issues"],
            location="Marrowby, Vermont", institution="Vermont Wrenfield",
        )
        assert decomposition.sub_queries == ["voting information", "voting issues"]

    def test_weather_still_localizes(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "what's the weather tomorrow",
            ["weather tomorrow Springfield, Illinois"],
            location="Springfield, Illinois", institution="Vermont Wrenfield",
        )
        assert decomposition.sub_queries == ["weather tomorrow Springfield, Illinois"]


# ---------------------------------------------------------------------------
# Helper-level extras (optional; direct, local imports so a pre-fix run of
# this file only errors these specific tests, not the whole module)
# ---------------------------------------------------------------------------

class TestHelperLevelExtras:

    def test_query_justifies_institution_academic_cue(self):
        from utils.institution_resolver import query_justifies_institution
        assert query_justifies_institution(
            "when is the class withdrawal deadline", "Vermont Wrenfield")

    def test_query_justifies_institution_own_school_possessive(self):
        from utils.institution_resolver import query_justifies_institution
        assert query_justifies_institution(
            "What is Vermont Wrenfield's policy on voting?", "Vermont Wrenfield")

    def test_query_justifies_institution_my_school_generic(self):
        from utils.institution_resolver import query_justifies_institution
        assert query_justifies_institution(
            "can you look up my school's policy on this", "Vermont Wrenfield")

    def test_query_justifies_institution_false_for_unrelated_query(self):
        from utils.institution_resolver import query_justifies_institution
        assert not query_justifies_institution(
            "I am referring to voting", "Vermont Wrenfield")

    def test_query_justifies_institution_false_for_different_school(self):
        from utils.institution_resolver import query_justifies_institution
        assert not query_justifies_institution(
            "How does Quellmoor University handle student voting?", "Vermont Wrenfield")

    def test_strip_unjustified_institution_removes_unprompted_school(self):
        from utils.institution_resolver import strip_unjustified_institution
        out = strip_unjustified_institution(
            ["Vermont Wrenfield voting information"], "I am referring to voting",
            "Vermont Wrenfield")
        assert out == ["voting information"]

    def test_strip_unjustified_institution_noop_when_justified(self):
        from utils.institution_resolver import strip_unjustified_institution
        terms = ["Vermont Wrenfield withdrawal deadline"]
        out = strip_unjustified_institution(
            terms, "when is the class withdrawal deadline", "Vermont Wrenfield")
        assert out == terms

    def test_strip_unjustified_institution_never_touches_other_school(self):
        from utils.institution_resolver import strip_unjustified_institution
        terms = ["Quellmoor University student voting"]
        out = strip_unjustified_institution(
            terms, "I am referring to voting", "Vermont Wrenfield")
        assert out == terms

    def test_query_justifies_location_bare_state_full_name(self):
        from utils.location_resolver import query_justifies_location
        assert query_justifies_location(
            "What are Illinois voting issues?", "Springfield, Illinois")

    def test_query_justifies_location_bare_state_abbrev_case_sensitive(self):
        from utils.location_resolver import query_justifies_location
        assert query_justifies_location(
            "What are IL voting deadlines?", "Springfield, IL")
        assert not query_justifies_location(
            "what should I do in the morning", "Springfield, IL")

    def test_query_justifies_location_state_inside_institution_span_excluded(self):
        from utils.location_resolver import query_justifies_location
        assert not query_justifies_location(
            "when is the Vermont Wrenfield drop deadline", "Marrowby, Vermont",
            institution="Vermont Wrenfield")

    def test_strip_unjustified_location_bare_state_no_city(self):
        from utils.location_resolver import strip_unjustified_location
        out = strip_unjustified_location(
            ["current voting issues Illinois"], "I am referring to voting",
            "Springfield, Illinois")
        assert out == ["current voting issues"]

    def test_strip_unjustified_location_protects_institution_span(self):
        from utils.location_resolver import strip_unjustified_location
        out = strip_unjustified_location(
            ["Vermont Wrenfield drop deadline Marrowby, Vermont"],
            "when is the Vermont Wrenfield drop deadline",
            "Marrowby, Vermont", institution="Vermont Wrenfield")
        assert out == ["Vermont Wrenfield drop deadline"]

    def test_scope_identity_terms_full_policy(self):
        from utils.institution_resolver import scope_identity_terms
        out = scope_identity_terms(
            ["current voting issues Illinois", "Vermont Wrenfield voting information"],
            "I am referring to voting", "Springfield, Illinois", "Vermont Wrenfield")
        joined_lower = " ".join(out).lower()
        assert "illinois" not in joined_lower
        assert "vermont" not in joined_lower


# ---------------------------------------------------------------------------
# Referee follow-up (frontier, 2026-09-12): cross-domain logistics words
# ---------------------------------------------------------------------------
#
# Found while refereeing this batch: a bare "withdrawal" / "registration" /
# "enrollment" / "transcript" made ANY query "academic logistics". Probed
# through the deployed scope_identity_terms before the fix, every one of these
# came back prefixed with the user's school ("Vermont Wrenfield benzodiazepine
# withdrawal symptoms") and the trigger prompt carried the school line — the
# owner's identity attached to a medical, civic or legal query sent to a
# third-party search provider. Present since 2026-08-27 (apply_institution);
# the batch's new prompt gate made the same vocabulary load-bearing.

CROSS_DOMAIN_CASES = [
    ("what are benzodiazepine withdrawal symptoms", "benzodiazepine withdrawal symptoms"),
    ("when is the voter registration deadline", "voter registration deadline 2026"),
    ("Medicare enrollment period 2026 changes", "Medicare enrollment period 2026"),
    ("did they release the court transcript yet", "court transcript release"),
]


class TestCrossDomainCuesNeverCarryTheSchool:

    @pytest.mark.parametrize("query,term", CROSS_DOMAIN_CASES)
    def test_trigger_producer(self, monkeypatch, query, term):
        parsed, prompts = classify(
            monkeypatch, query, [term], institution="Vermont Wrenfield")
        assert parsed.search_terms == [term]
        assert "User's school:" not in prompts[0]

    @pytest.mark.parametrize("query,term", CROSS_DOMAIN_CASES)
    def test_decompose_producer(self, monkeypatch, query, term):
        decomposition, prompts = decompose(
            monkeypatch, query, [term, f"{term} explained"],
            institution="Vermont Wrenfield")
        assert decomposition.sub_queries == [term, f"{term} explained"]
        assert "User's school:" not in prompts[0]

    def test_llm_slip_on_a_civic_query_is_stripped(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "when is the voter registration deadline",
            ["Vermont Wrenfield voter registration deadline"],
            institution="Vermont Wrenfield")
        assert parsed.search_terms == ["voter registration deadline"]

    @pytest.mark.parametrize("query", [
        "when is the class withdrawal deadline",
        "when is the class\n  withdrawal deadline",
    ])
    def test_anchored_cross_domain_cue_still_names_the_school(self, monkeypatch, query):
        parsed, prompts = classify(
            monkeypatch, query, ["class withdrawal deadline fall 2026"],
            institution="Vermont Wrenfield")
        assert parsed.search_terms == ["Vermont Wrenfield class withdrawal deadline fall 2026"]
        assert "User's school: Vermont Wrenfield" in prompts[0]


SCHOOL_EXCHANGE = (
    "User: when is the drop deadline for my class this term?\n"
    "Assistant: The registrar's academic calendar lists the drop deadline."
)
UNRELATED_EXCHANGE = (
    "User: I keep thinking about weighted voting as an idea\n"
    "Assistant: A weighted franchise raises questions about fairness."
)


class TestReferentialFollowUpContext:
    """The prompt gate and the strip must not turn an elliptical follow-up
    inside a school-logistics exchange into a generic search — but prior
    context re-justifies the school ONLY for a referential follow-up, and
    ONLY through a school-logistics shape."""

    @pytest.mark.parametrize("query", ["is it this Friday?", "is it\n  this Friday?"])
    def test_referential_follow_up_keeps_the_school(self, monkeypatch, query):
        parsed, prompts = classify(
            monkeypatch, query, ["Vermont Wrenfield drop deadline Friday"],
            institution="Vermont Wrenfield", conversation_context=SCHOOL_EXCHANGE)
        assert parsed.search_terms == ["Vermont Wrenfield drop deadline Friday"]
        assert "User's school: Vermont Wrenfield" in prompts[0]

    def test_referential_follow_up_gains_the_school_for_generic_terms(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "is it this Friday?", ["drop deadline Friday 2026"],
            institution="Vermont Wrenfield", conversation_context=SCHOOL_EXCHANGE)
        assert parsed.search_terms == ["Vermont Wrenfield drop deadline Friday 2026"]

    def test_non_referential_query_does_not_inherit_the_school(self, monkeypatch):
        parsed, prompts = classify(
            monkeypatch, "what is the latest election news",
            ["Vermont Wrenfield election news"],
            institution="Vermont Wrenfield", conversation_context=SCHOOL_EXCHANGE)
        assert parsed.search_terms == ["election news"]
        assert "User's school:" not in prompts[0]

    def test_referential_follow_up_without_school_context_strips(self, monkeypatch):
        parsed, prompts = classify(
            monkeypatch, "is that fair?", ["Vermont Wrenfield weighted voting fairness"],
            institution="Vermont Wrenfield", conversation_context=UNRELATED_EXCHANGE)
        assert parsed.search_terms == ["weighted voting fairness"]
        assert "User's school:" not in prompts[0]

    def test_school_merely_named_earlier_does_not_rejustify(self, monkeypatch):
        named_only = ("User: my Vermont Wrenfield homework is done\n"
                      "Assistant: Nice work finishing it.")
        parsed, _ = classify(
            monkeypatch, "is that fair?", ["Vermont Wrenfield voting fairness"],
            institution="Vermont Wrenfield", conversation_context=named_only)
        assert parsed.search_terms == ["voting fairness"]
