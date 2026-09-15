"""tests/unit/test_tone_domain_strain.py

H02b (2026-09-14, H02_design.md "Domain-strain rule", contract items 1-9): a
domain anchor + first-person strain evidence in ONE clause floors a
CONVERSATIONAL `detect_crisis_level` verdict to CONCERN, trigger
`domain_strain`. Fixtures: the owner-reviewed round-3 probe's STRESS/
CONTROLS rows (probes/h02_candidate_probe_round3.py, 22/23 stress, 0/41
controls), re-proved against DEPLOYED `detect_crisis_level(model_manager=
None)`; pre-edit levels pinned from probes/h02a_noop_round3_output.txt (sha
a7bf3e56...) and this file's own failing-first run. D-H02-1..3: no new
seeds; "us" only as a composed THREAT_LEADS x DISPLACEMENT_SHAPES object;
business stress stays a miss. `_wrap`/`_bare_pipeline` conventions follow
test_tone_observational_mild_distress.py / test_tone_negated_news_no_carryover.py.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import utils.tone_detector as td
from utils.tone_detector import (
    CrisisLevel,
    OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER,
    ToneAnalysis,
    detect_crisis_level,
    format_tone_log,
)
from core.context_pipeline import ContextPipeline, ToneLevel

# New names don't exist pre-edit; `td.` access fails one TEST, not collection.
DOMAIN_STRAIN_TRIGGER = getattr(td, "DOMAIN_STRAIN_TRIGGER", "domain_strain")

def _domain_strain_hit(message):
    return td._domain_strain_hit(message)

def _wrap(text: str, lead_words: int = 1) -> str:
    """BC-64 wrap, newline BEFORE the anchor-strain window (`\\n` is a
    clause boundary, contract item 4). 0 for two short rows in _WRAP_OVERRIDES."""
    words = text.split(" ")
    lead = min(lead_words, max(len(words) - 1, 0))
    return " ".join(words[:lead]) + "\n  " + " ".join(words[lead:])

_WRAP_OVERRIDES = {"we_might_lose_apartment": 0, "cant_keep_up_classes": 0}

def _wrapped(case_id: str, message: str, default: int = 1) -> str:
    return _wrap(message, _WRAP_OVERRIDES.get(case_id, default))

# STRESS: round-3 probe rows (23); expect_trigger=True iff the rule alone raises a pre-edit CONVERSATIONAL row.
STRESS_ROWS = [
    ("running_on_nothing", "my kids haven't slept in days and I'm running on nothing", False),
    ("wearing_me_down", "my mom and I keep fighting and it's wearing me down", False),
    ("partner_falling_apart", "things with my partner are falling apart and I don't know what to do", False),
    ("wearing_me_out", "taking care of my dad after his stroke is wearing me out", True),
    ("cant_keep_up_grandmother", "I'm the only one caring for my grandmother and I can't keep up", True),
    ("exhausting_me", "juggling my sister's medical appointments is exhausting me", True),
    ("cant_do_anything", "my back pain flared up again and I can't do anything", True),
    ("eating_me_alive", "waiting on the biopsy results is eating me alive", True),
    ("migraines_scared", "my migraines keep getting worse and I'm scared", False),
    ("dont_know_how_well_rent", "I don't know how we'll make rent this month", True),
    ("wiped_out_our_savings", "the car repair wiped out our savings", True),
    ("landlord_trying_to_evict_us", "our landlord is trying to evict us", True),
    ("we_might_lose_apartment", "we might lose the apartment", True),
    ("black_mold_cant_afford", "the house has black mold and we can't afford to move", True),
    ("barely_keeping_boss", "my boss keeps piling on more and I'm barely keeping my head up", True),
    ("about_to_get_laid_off", "I'm about to get laid off and I can't stop thinking about it", False),
    ("novel_crushing_me", "I've been blocked on my novel for weeks and it's crushing me", False),
    ("failing_chemistry", "I'm failing chemistry and my parents are going to be furious", False),
    ("thesis_panicking", "my thesis defense is next week and I'm panicking", True),
    ("cant_keep_up_classes", "I can't keep up with my classes anymore", True),
    ("kick_us_out_apartment", "the landlord says he's going to kick us out of the apartment next month", True),
    ("foreclose_on_us", "the bank is threatening to foreclose on us and the house is all we have", True),
    # H02_design.md's "we" example: only 1 marker, floors via the general path not Stage 0.
    ("prices_up_we_cant_afford_rent", "prices keep going up in the news and we can't afford rent", True),
]
# D-H02-3 (expect_trigger=None): impersonal business stress, no subject, stays a miss.
CASH_FLOW_MISS = ("cash_flow_business", "cash flow is tight and payroll is due friday", None)

@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("case_id,message,expect_trigger", STRESS_ROWS + [CASH_FLOW_MISS],
                          ids=[c[0] for c in STRESS_ROWS] + [CASH_FLOW_MISS[0]])
async def test_stress_rows_at_least_concern(case_id, message, expect_trigger, wrapped):
    text = _wrapped(case_id, message) if wrapped else message
    result = await detect_crisis_level(text, model_manager=None)
    if expect_trigger is None:
        assert result.level == CrisisLevel.CONVERSATIONAL
        assert _domain_strain_hit(message) is None
        return
    assert result.level != CrisisLevel.CONVERSATIONAL
    if expect_trigger:
        assert result.trigger == DOMAIN_STRAIN_TRIGGER
        assert result.explanation == td._DOMAIN_STRAIN_EXPLANATION

# CONTROLS: round-3 probe rows (41), verbatim, all pre-edit CONVERSATIONAL.
CONTROL_ROWS = [
    ("kids_soccer", "my kids have soccer on saturday"),
    ("mom_visiting", "my mom is visiting next week"),
    ("neighbors_cat", "I'm taking care of my neighbor's cat this weekend"),
    ("dads_pt", "my dad's physical therapy starts monday"),
    ("dentist_appt", "I have a dentist appointment tomorrow"),
    ("new_vitamin", "I started taking a new vitamin"),
    ("paid_rent", "I paid my rent today"),
    ("budgeting_vacation", "I'm budgeting for a vacation"),
    ("looking_apartments", "we're looking at apartments downtown"),
    ("landlord_fixed_sink", "the landlord fixed the sink"),
    ("meeting_boss", "I have a meeting with my boss at 3"),
    ("work_fine", "work was fine today"),
    ("chemistry_exam_friday", "my chemistry exam is on friday"),
    ("finished_homework", "I finished my homework"),
    ("kids_wore_me_out_park", "the kids wore me out at the park but it was a great day"),
    ("not_stressed_rent", "I'm not stressed about rent anymore"),
    ("coworker_evicted", "my coworker is getting evicted and I'm helping her move"),
    ("cant_keep_up_shows", "I can't keep up with all the good shows right now"),
    ("dads_surgery_well", "my dad's surgery went really well"),
    ("thesis_almost_done", "my thesis is almost done and I'm excited"),
    ("rent_up_fine", "rent went up a little but we're fine"),
    ("kids_cant_keep_up_hikes", "the kids can't keep up with me on hikes anymore haha"),
    ("kids_wore_me_out_zoo", "my kids wore me out at the zoo but it was so fun"),
    ("sister_exhausted_job", "my sister is exhausted from her new job"),
    ("landlord_fixed_heater", "our landlord finally fixed the heater"),
    ("partner_panicking_exam", "my partner is panicking about his exam, I'm trying to help"),
    ("paid_off_loan_relieved", "we paid off the loan and I feel so relieved"),
    ("kids_energy", "I don't know how the kids have so much energy"),
    ("paid_bills_morning", "I paid the bills this morning"),
    ("work_busy_good", "work was busy but good"),
    ("parents_dinner", "my parents are coming over for dinner"),
    ("moved_new_place", "we moved into our new place last week"),
    ("business_great_month", "the business had a great month"),
    ("looking_after_nephew", "I'm looking after my nephew this afternoon"),
    ("landlord_promised_wont_evict", "our landlord promised he won't evict us"),
    ("tried_evict_years_ago", "they tried to evict us years ago but we're doing fine now"),
    ("worried_kick_out_renewed", "we were worried they'd kick us out but the lease got renewed"),
    ("bouncer_kick_us_out_bar", "the bouncer is going to kick us out of the bar lol"),
    ("game_kick_us_out_lobby", "the game keeps trying to kick us out of the lobby haha"),
    ("evict_tenants_not_us", "the landlord is trying to evict the tenants downstairs, not us"),
    ("landlord_never_evict_family", "my landlord would never evict us, he's like family"),
]

@pytest.mark.parametrize("wrapped", [False, True], ids=["clean", "wrapped"])
@pytest.mark.parametrize("case_id,message", CONTROL_ROWS, ids=[c[0] for c in CONTROL_ROWS])
async def test_control_rows_unchanged_conversational(case_id, message, wrapped):
    text = _wrapped(case_id, message, default=3) if wrapped else message
    result = await detect_crisis_level(text, model_manager=None)
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert result.trigger != DOMAIN_STRAIN_TRIGGER

def test_trigger_constant_value_and_visibility():
    # Strict access (no fallback): must fail pre-edit.
    assert td.DOMAIN_STRAIN_TRIGGER == "domain_strain"
    analysis = ToneAnalysis(
        level=CrisisLevel.CONCERN, confidence=0.5, trigger=td.DOMAIN_STRAIN_TRIGGER,
        raw_scores={}, explanation=td._DOMAIN_STRAIN_EXPLANATION,
    )
    assert td.DOMAIN_STRAIN_TRIGGER in format_tone_log(analysis, "our landlord is trying to evict us")

# ---- Floor-path coverage: one test per CONVERSATIONAL return path. ----

ANCHOR_STRAIN_MSG = "taking care of my dad after his stroke is wearing me out"

async def _assert_floors(message, inner_trigger, model_manager=None):
    """Shared assertion: the inner (pre-floor) impl returns CONVERSATIONAL
    via `inner_trigger`, and the public wrapper floors it to CONCERN."""
    inner = await td._detect_crisis_level_impl(message, model_manager=model_manager)
    assert inner.level == CrisisLevel.CONVERSATIONAL and inner.trigger == inner_trigger
    result = await detect_crisis_level(message, model_manager=model_manager)
    assert result.level == CrisisLevel.CONCERN and result.trigger == DOMAIN_STRAIN_TRIGGER

async def test_floor_path_semantic():
    await _assert_floors(ANCHOR_STRAIN_MSG, "semantic")

async def test_floor_path_short_casual(monkeypatch):
    # No real fixture fits (anchor+strain never <=2 words); proves path-agnostic coverage.
    monkeypatch.setattr(td, "_domain_strain_hit", lambda message: "first_person_strain")
    await _assert_floors("thanks!", "short_casual")

async def test_floor_path_positive_state_report(monkeypatch):
    def _fake_semantic(message, conversation_history=None, model_manager=None, force_escalation=False):
        return (CrisisLevel.CONCERN, 0.5, {"concern": 0.5, "conversational": 0.2})
    monkeypatch.setattr(td, "_semantic_crisis_detection", _fake_semantic)
    monkeypatch.setattr(td, "_is_positive_state_report", lambda message: True)
    await _assert_floors(ANCHOR_STRAIN_MSG, "positive_state_report")

async def test_floor_path_arbiter_llm_fallback(monkeypatch):
    def _fake_semantic(message, conversation_history=None, model_manager=None, force_escalation=False):
        return (CrisisLevel.CONVERSATIONAL, 0.2, {"conversational": 0.5, "concern": 0.35})

    async def _fake_arbiter(message, model_manager):
        return (CrisisLevel.CONVERSATIONAL, 0.5)

    monkeypatch.setattr(td, "_semantic_crisis_detection", _fake_semantic)
    monkeypatch.setattr(td, "_llm_crisis_fallback", _fake_arbiter)
    await _assert_floors(ANCHOR_STRAIN_MSG, "llm_fallback", model_manager=object())

async def test_floor_never_alters_concern_plus_result():
    # Already >= CONCERN pre-edit; must stay untouched despite the rule hit.
    message = "my mom and I keep fighting and it's wearing me down"
    assert _domain_strain_hit(message) is not None
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level in (CrisisLevel.MEDIUM, CrisisLevel.HIGH, CrisisLevel.CONCERN)
    assert result.trigger != DOMAIN_STRAIN_TRIGGER

# ---- Stage 0 inheritance (BC-58, design item 5, contract item 6). ----

# "we" qualifies via the producer hook, unlike the mild-hit loop's _MILD_SUBJECT_RE.
_STAGE0_TIER_B_CASES = [
    ("i_subject", "the news says people are struggling and I can't keep up with my rent"),
    ("we_subject", "according to officials people are struggling and we can't afford rent"),
]

@pytest.mark.parametrize("case_id,message", _STAGE0_TIER_B_CASES, ids=[c[0] for c in _STAGE0_TIER_B_CASES])
async def test_stage0_tier_b_domain_strain(case_id, message):
    assert td._check_observational_language(message) is True
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level == CrisisLevel.CONCERN
    assert result.trigger == OBSERVATIONAL_FIRST_PERSON_DISTRESS_TRIGGER

async def test_stage0_collective_we_no_anchor_unchanged():
    # No anchor -> rule never fires; pinned pre-edit CONCERN is unrelated (borderline backstop).
    message = "the news is awful and we are all scared"
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level == CrisisLevel.CONCERN
    assert result.trigger == "borderline_backstop"
    assert _domain_strain_hit(message) is None

async def test_stage0_negated_strain_not_raised():
    message = "according to the news people are suffering, and I don't feel worried about my rent"
    assert td._check_observational_language(message) is True
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level == CrisisLevel.CONVERSATIONAL
    assert result.trigger == "observational_language"
    assert _domain_strain_hit(message) is None

# ---- No learning (BC-29): rule outcomes never teach; a plain keyword CONCERN still does. ----

_LEARNING_CASES = [
    ("domain_strain_clean", ANCHOR_STRAIN_MSG, False),
    ("domain_strain_wrapped", _wrap(ANCHOR_STRAIN_MSG, 1), False),
    ("stage0_i_subject", "the news says people are struggling and I can't keep up with my rent", False),
    ("stage0_we_subject", "according to officials people are struggling and we can't afford rent", False),
    ("keyword_concern_control", "I feel hopeless and worthless", True),
]

@pytest.mark.parametrize("case_id,message,teaches", _LEARNING_CASES, ids=[c[0] for c in _LEARNING_CASES])
async def test_learning_gate(monkeypatch, case_id, message, teaches):
    calls = []
    monkeypatch.setattr(td, "_learn_tone_exemplar", lambda *a, **kw: calls.append((a, kw)))
    result = await detect_crisis_level(message, model_manager=None)
    assert result.level != CrisisLevel.CONVERSATIONAL
    assert bool(calls) == teaches

# ---- Carry-over (BC-28), through the real two-turn ContextPipeline. ----

def _bare_pipeline(tmp_path):
    pipe = object.__new__(ContextPipeline)
    pipe._TONE_STATE_PATH = str(tmp_path / "tone_state.json")
    pipe._last_tone_level = None
    pipe._floor_chain = 0
    pipe.memory_system = None
    pipe.model_manager = None
    return pipe

def _row(text, heavy, minutes_ago=0):
    return {"query": text, "is_heavy_topic": heavy,
            "timestamp": (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()}

def _state(pipe):
    p = Path(pipe._TONE_STATE_PATH)
    return json.loads(p.read_text()) if p.exists() else None

DOMAIN_STRAIN_TURN = "our landlord is trying to evict us"
NEUTRAL_FOLLOWUP = "the weather was grey this morning"

async def test_domain_strain_carries_over_unlike_negated_crisis(tmp_path):
    pipe = _bare_pipeline(tmp_path)
    tone1, ctx1 = await pipe._detect_tone(DOMAIN_STRAIN_TURN, None)
    assert tone1 == ToneLevel.CONCERN
    assert ctx1.tone_trigger == DOMAIN_STRAIN_TRIGGER
    assert _state(pipe)["trigger"] == DOMAIN_STRAIN_TRIGGER
    history = [_row(DOMAIN_STRAIN_TURN, heavy=False)]
    tone2, ctx2 = await pipe._detect_tone(NEUTRAL_FOLLOWUP, history)
    assert tone2 == ToneLevel.CONCERN
    assert ctx2.tone_trigger == "distress_sticky_floor"
