"""2026-09-12 follow-up review F4: stripping the owner's state must not cut a
proper name in half.

The adversarial-review batch taught the location backstop to remove a bare
state name the trigger LLM injected ("current voting issues Illinois"). It
removed EVERY occurrence of the state, including the first word of a named
source: for a user in Washington,

    scope_identity_terms(["Washington Post election coverage"],
                         "summarize the election story",
                         "Seattle, Washington", "Vermont Wrenfield")

returned ``["Post election coverage"]`` — a different search target. A US
state name is also the leading word of newspapers, companies, characters and
schools, so this is a shape, not a Washington Post exception.

The rule is structural. Capitalization is the only local signal, so it is
read only when the term also has lowercase words (an all-TitleCase headline
carries no proper-noun information). Inside a run of two or more capitalized
words, a location word counts as part of a proper name when a non-location
word follows it (a place used as a modifier: "Washington Post", "Texas
Instruments", "New York Times"), or when the user typed that exact run
themselves ("George Washington"). A run that is only the location ("Seattle
Washington", "Washington State") or that contains the full "City State"
sequence is still an injected location and is still stripped; so is a state
appended after a proper name ("Chicago Tribune Illinois").

Every case drives the deployed functions: the shared policy
(``scope_identity_terms``, ``strip_unjustified_location``,
``query_justifies_location``) and both term producers
(``utils.web_search_trigger._classify_with_llm_unified`` and
``WebSearchManager.decompose_query``) with fake model output.

class: BC-59, BC-58
"""
from __future__ import annotations

import pytest

from tests.unit.test_sep12_search_identity_scope import classify, decompose
from utils.institution_resolver import scope_identity_terms
from utils.location_resolver import query_justifies_location, strip_unjustified_location

SCHOOL = "Vermont Wrenfield"

# (term, user query, resolved location) — each term names something whose
# first word is the owner's state or city; the query gives no local cue.
PROPER_NAME_TERMS = [
    ("Washington Post election coverage", "summarize the election story", "Seattle, Washington"),
    ("The Washington Post review of the film", "what did critics say about it", "Seattle, Washington"),
    ("Texas Instruments earnings report", "how did chip stocks do this quarter", "Austin, Texas"),
    ("Indiana Jones trailer reactions", "what did people think of the new trailer", "Bloomington, Indiana"),
    ("New York Times coverage of the strike", "summarize the strike coverage", "Albany, New York"),
    ("Virginia Woolf essays on writing", "recommend some essays about writing", "Richmond, Virginia"),
]

# (term, user query, resolved location, expected term after scoping) — the
# location is injected, not part of a name, and must still be removed.
INJECTED_LOCATION_TERMS = [
    ("election news Washington", "summarize the election story", "Seattle, Washington", "election news"),
    ("Washington election news", "summarize the election story", "Seattle, Washington", "election news"),
    ("Washington State voter guide", "how do I register to vote", "Seattle, Washington", "voter guide"),
    ("Seattle Washington transit delays", "why is the train late again", "Seattle, Washington", "transit delays"),
    ("Seattle Washington Transit Authority delays", "why is the train late again",
     "Seattle, Washington", "Transit Authority delays"),
    ("Chicago Tribune Illinois politics", "summarize the politics story", "Springfield, Illinois",
     "Chicago Tribune politics"),
    ("current voting issues Illinois", "I am referring to voting", "Springfield, Illinois",
     "current voting issues"),
]


def _wrapped(query: str) -> str:
    """A client-style soft line wrap in the middle of the user's query."""
    words = query.split(" ")
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]) + "\n  " + " ".join(words[mid:])


class TestSharedPolicyKeepsProperNames:

    @pytest.mark.parametrize("term,query,location", PROPER_NAME_TERMS)
    def test_scope_identity_terms_keeps_the_name(self, term, query, location):
        assert scope_identity_terms([term], query, location, SCHOOL) == [term]

    @pytest.mark.parametrize("term,query,location", PROPER_NAME_TERMS)
    def test_wrapped_query_keeps_the_name(self, term, query, location):
        assert scope_identity_terms([term], _wrapped(query), location, SCHOOL) == [term]

    def test_a_name_the_user_typed_is_kept_even_with_the_state_last(self):
        term = "George Washington farewell address"
        query = "What did George Washington say in his farewell address"
        assert strip_unjustified_location([term], query, "Seattle, Washington") == [term]


class TestSharedPolicyStillStripsInjectedLocation:

    @pytest.mark.parametrize("term,query,location,expected", INJECTED_LOCATION_TERMS)
    def test_scope_identity_terms_strips_it(self, term, query, location, expected):
        assert scope_identity_terms([term], query, location, SCHOOL) == [expected]

    def test_headline_case_term_carries_no_name_signal(self):
        # No lowercase word anywhere: capitalization says nothing, so the
        # pre-existing strip applies (documented limit, not a proper-name read).
        assert strip_unjustified_location(
            ["Washington Election Coverage"], "summarize the election story",
            "Seattle, Washington") == ["Election Coverage"]


class TestQueryJustification:

    def test_a_state_inside_a_named_source_is_not_the_user_naming_their_state(self):
        assert not query_justifies_location(
            "What did the Washington Post report about the election", "Seattle, Washington")

    def test_a_state_named_on_its_own_still_justifies(self):
        assert query_justifies_location(
            "What are the Washington voter registration deadlines", "Seattle, Washington")

    def test_school_span_still_excluded(self):
        assert not query_justifies_location(
            "when is the Vermont Wrenfield drop deadline", "Marrowby, Vermont", institution=SCHOOL)


class TestBothProducers:

    def test_trigger_classifier(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "summarize the election story",
            ["Washington Post election coverage", "election results Washington"],
            location="Seattle, Washington", institution=SCHOOL,
        )
        assert parsed is not None
        assert parsed.search_terms == ["Washington Post election coverage", "election results"]

    def test_decompose_query(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "summarize the election story and the polling",
            ["Washington Post election coverage", "polling averages Washington"],
            location="Seattle, Washington", institution=SCHOOL,
        )
        assert decomposition.sub_queries == ["Washington Post election coverage", "polling averages"]

    def test_trigger_classifier_other_state(self, monkeypatch):
        parsed, _ = classify(
            monkeypatch, "how did chip stocks do this quarter",
            ["Texas Instruments earnings report", "semiconductor stocks Texas"],
            location="Austin, Texas", institution=SCHOOL,
        )
        assert parsed.search_terms == ["Texas Instruments earnings report", "semiconductor stocks"]

    def test_decompose_query_other_state(self, monkeypatch):
        decomposition, _ = decompose(
            monkeypatch, "how did chip stocks and memory makers do this quarter",
            ["Texas Instruments earnings report", "memory chip makers Texas"],
            location="Austin, Texas", institution=SCHOOL,
        )
        assert decomposition.sub_queries == ["Texas Instruments earnings report", "memory chip makers"]
