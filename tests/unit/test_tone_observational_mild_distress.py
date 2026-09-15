"""tests/unit/test_tone_observational_mild_distress.py

T02 (2026-09-13, docs/execution/generalization/T02_design.md): both sides of
T01's own documented limitations under observational framing. (1)
Under-escalation: mild first-person distress ("...I feel hopeless") stayed
CONVERSATIONAL on T01 alone — T02's Tier B returns CONCERN, trigger
`observational_first_person_distress`. (2) Over-escalation: T01 alone raised
a negated HIGH phrase and someone else's distress ("my students want to
die") to MEDIUM — T02 refines Tier A's rule (b) to a SUBJECT-form partition
of `_HISTORY_FIRST_PERSON_RE` (BC-76) and scopes negation to after the last
first-person token; a qualifying HIGH hit that IS negated is Tier B instead,
trigger `observational_negated_crisis` (CONCERN that turn only; T03 owns
carry-over).

Fixtures are the T02_design.md probe rows (probes/t02_probe.py /
t02_probe_output.txt), parent-verified 30/30 PASS; re-proved here against
the DEPLOYED `detect_crisis_level`, plus qualifier-edge/BC-29 tests against
the classifier/helper. `model_manager=None` throughout.
"""

import pytest

import utils.tone_detector as tone_detector
from utils.tone_detector import CrisisLevel, detect_crisis_level


def _wrap(text: str, lead_words: int = 3) -> str:
    """Line-wrap a fixture the way a real chat client sends it (BC-64): a
    newline+indent after the first `lead_words` words, BEFORE the
    subject-to-hit window under test (T02_draft.md: never at a midpoint
    that could split subject and hit). A word-midpoint wrap lands inside
    the multi-word marker 'people are suffering' for several fixtures here
    (it and 'people are' are two DISTINCT markers over the same span,
    CGR-20260913-003); splitting them drops Stage 0's marker count from 2
    to 1 and it stops firing, silently skipping the row. `lead_words=3`
    clears every fixture but one, verified empirically (T02.md); see
    _WRAP_OVERRIDES.
    """
    words = text.split(" ")
    lead = min(lead_words, max(len(words) - 1, 0))
    return " ".join(words[:lead]) + "\n  " + " ".join(words[lead:])


_WRAP_OVERRIDES = {"third_person_they": 5}


def _wrapped(case_id: str, message: str) -> str:
    return _wrap(message, _WRAP_OVERRIDES.get(case_id, 3))


# CRISIS: Tier A. Expect >= MEDIUM, trigger != "observational_language".
CRISIS_CASES = [
    ("want_to_die_b", "people are suffering and I want to die"),
    ("cant_go_on_b", "According to the news, officials say millions are struggling, and honestly I can't go on"),
    ("dont_want_to_be_here_b", "People are dying in the news and I don't want to be here anymore"),
    ("ready_to_die_b", "the news is awful, people are suffering, and I'm ready to die"),
    ("want_to_die_article_b", "According to the article people are struggling and I really just want to die"),
    ("kill_myself_a", "According to the news people are suffering and I want to kill myself"),
    # Inversion guard: negation is scoped AFTER the last "I", not back across it to "stop".
    ("inversion_guard_b", "people are suffering and I can't stop thinking I want to die"),
    ("cue_inside_phrase_b", "According to the news people are struggling and I can't stop crying"),
    ("we_subject_b", "According to officials people are suffering and we want to die"),  # "we" qualifies the crisis tier
]


@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("case_id,message", CRISIS_CASES,
                          ids=[c[0] for c in CRISIS_CASES])
async def test_crisis_rows(case_id, message, wrapped):
    text = _wrapped(case_id, message) if wrapped else message
    result = await detect_crisis_level(text, model_manager=None)
    assert result.level in (CrisisLevel.MEDIUM, CrisisLevel.HIGH)
    assert result.trigger != "observational_language"


# CONCERN: Tier B. Expect CONCERN; "observational_negated_crisis" for the two
# negated-only rows, "observational_first_person_distress" for the rest.
_DISTRESS = "observational_first_person_distress"
_NEGATED_CRISIS = "observational_negated_crisis"

CONCERN_CASES = [
    ("hopeless_ab", "people are suffering and I feel hopeless", _DISTRESS),
    ("alone_b", "According to the news, officials say millions are struggling and I feel so alone", _DISTRESS),
    ("overwhelmed_b", "According to the news people are suffering and I'm overwhelmed", _DISTRESS),
    ("worthless_ab", "According to officials people are struggling and honestly I feel worthless", _DISTRESS),
    ("losing_my_mind_a", "people are suffering and I'm losing my mind", _DISTRESS),
    ("cant_cope_b", "the news says people are suffering and I can't cope", _DISTRESS),
    ("negated_want_to_die", "people are suffering but I don't want to die", _NEGATED_CRISIS),
    ("negated_kill_myself", "According to the news people are suffering, I don't want to kill myself though", _NEGATED_CRISIS),
    ("lost_job_b", "According to the news people are suffering and I lost my job", _DISTRESS),
]


@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("case_id,message,trigger", CONCERN_CASES,
                          ids=[c[0] for c in CONCERN_CASES])
async def test_concern_rows(case_id, message, trigger, wrapped):
    text = _wrapped(case_id, message) if wrapped else message
    result = await detect_crisis_level(text, model_manager=None)
    assert result.level == CrisisLevel.CONCERN
    assert result.trigger == trigger


def test_trigger_constants_exported():
    """T03 imports OBSERVATIONAL_NEGATED_CRISIS_TRIGGER to exclude that
    turn from tone carry-over (T03_design.md) — both must be real module
    exports, not just literals repeated at call sites."""
    from utils.tone_detector import (
        OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER,
        OBSERVATIONAL_NEGATED_CRISIS_TRIGGER,
    )
    assert OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER == _DISTRESS
    assert OBSERVATIONAL_NEGATED_CRISIS_TRIGGER == _NEGATED_CRISIS


# STAY: no qualifying evidence. Expect CONVERSATIONAL / "observational_language".
STAY_CASES = [
    ("possessive_students", "According to the news, my students want to die"),
    ("possessive_our_kids", "According to the news, our kids want to die"),
    ("suicide_rates", "According to officials, suicide rates rose among young people"),
    ("suicidal_after_policies", "The news article says people are suicidal after the policies"),
    ("subject_outside_window_coverage", "I think the news coverage of suicide is irresponsible, people are struggling"),
    ("third_person_they", "They say people are suffering and they can't go on"),
    ("millions_want_to_die_war", "According to the news, millions of people want to die in that war"),
    ("struggling_policies_officials", "People are struggling with the new policies, according to officials"),
    ("we_excluded_mild_tier", "I think we are failing as a country, according to the news"),  # "we" excluded from mild tier
    ("event_vocabulary_excluded", "According to the news I can't believe this government"),
    ("negated_mild_hit", "According to officials, people are struggling, but I don't feel alone"),
    ("subject_outside_window_read", "I read that people want to die, according to the news"),
]


@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("case_id,message", STAY_CASES,
                          ids=[c[0] for c in STAY_CASES])
async def test_stay_rows(case_id, message, wrapped):
    text = _wrapped(case_id, message) if wrapped else message
    result = await detect_crisis_level(text, model_manager=None)
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert result.trigger == "observational_language"


# Inversion rows (T02_design.md "Known limitations"): "stop" falls inside
# the post-pronoun span, so these still misread as negated. Floor only
# (>= CONCERN), not an exact level.
INVERSION_CASES = [
    ("cant_stop_cutting_myself", "According to the news people are suffering and I can't stop cutting myself"),
    ("cant_stop_wanting_to_kill_myself", "According to the news people are suffering and I can't stop wanting to kill myself"),
]


@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("case_id,message", INVERSION_CASES,
                          ids=[c[0] for c in INVERSION_CASES])
async def test_inversion_rows_at_least_concern(case_id, message, wrapped):
    text = _wrapped(case_id, message) if wrapped else message
    result = await detect_crisis_level(text, model_manager=None)
    assert result.level in (CrisisLevel.CONCERN, CrisisLevel.MEDIUM, CrisisLevel.HIGH)


# Qualifier edges, against the classifier/helper directly (precise token/
# subject-form arithmetic, same rationale as T01's helper-direct tests).
_TIER_A_EDGE_CASES = [
    ("possessive_not_tier_a", "According to the news, my students want to die", False),
    ("we_qualifies_tier_a", "According to officials people are suffering and we want to die", True),
    ("negation_span_scoped_after_pronoun", "people are suffering and I can't stop thinking I want to die", True),
]


@pytest.mark.parametrize("case_id,message,expected", _TIER_A_EDGE_CASES,
                          ids=[c[0] for c in _TIER_A_EDGE_CASES])
def test_tier_a_edges(case_id, message, expected):
    from utils.tone_detector import _has_first_person_high_crisis_hit
    assert _has_first_person_high_crisis_hit(message) is expected


# "we" is excluded from the mild subject partition; EVENT_DISTRESS
# vocabulary is excluded from Tier B by construction — both would
# otherwise qualify.
_TIER_B_EDGE_MESSAGES = [
    ("we_excluded_from_tier_b", "According to the news, we feel hopeless"),
    ("event_vocabulary_excluded", "According to the news I can't believe this government"),
]


@pytest.mark.parametrize("case_id,message", _TIER_B_EDGE_MESSAGES,
                          ids=[c[0] for c in _TIER_B_EDGE_MESSAGES])
def test_tier_b_edges_return_none(case_id, message):
    from utils.tone_detector import _classify_observational_evidence
    assert _classify_observational_evidence(message) is None


# BC-29: no self-teaching from Tier A or either Tier B trigger; control
# shows a non-observational first-person crisis message still teaches.
_NO_TEACH_CASES = [
    ("tier_a", CRISIS_CASES[0][1], False),
    ("tier_b_distress", "people are suffering and I feel hopeless", True),
    ("tier_b_negated_crisis", "people are suffering but I don't want to die", True),
]


@pytest.mark.parametrize("case_id,message,is_concern", _NO_TEACH_CASES,
                          ids=[c[0] for c in _NO_TEACH_CASES])
async def test_no_self_teaching(monkeypatch, case_id, message, is_concern):
    calls = []
    monkeypatch.setattr(
        tone_detector, "_learn_tone_exemplar",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = await detect_crisis_level(message, model_manager=None)
    if is_concern:
        assert result.level == CrisisLevel.CONCERN
    else:
        assert result.level != CrisisLevel.CONVERSATIONAL
    assert calls == []


async def test_control_non_observational_still_teaches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tone_detector, "_learn_tone_exemplar",
        lambda message, level_key, source, model_manager=None: calls.append((level_key, source)),
    )
    control = "I want to die and I can't go on"
    assert tone_detector._check_observational_language(control) is False
    result = await detect_crisis_level(control, model_manager=None)
    assert result.level != CrisisLevel.CONVERSATIONAL
    assert calls, "expected _learn_tone_exemplar to be called for a clean first-person crisis message"
