"""probe_decision_model_eval.select_policy tie-break + the harder borderline cases.

Incident 2026-10-09 (Jev), fixed 2026-10-27: many policies tie at cost 0 and the
grid order put the most extreme thresholds first (picked taus [0.65, 0.1, 0.1]).
class: BC-58 (selection decided by an arbitrary enumeration order, not a stated
preference). Drives the DEPLOYED select_policy and tone_verdict.
"""
import importlib.util
import re
from pathlib import Path

from eval import decision_model_cases as cases
from utils import tone_detector as tone

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("probe_decision_model_eval", REPO / "scripts" / "probe_decision_model_eval.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

_SEV = {"id": "s", "group": "high", "expected": "HIGH", "llm": {"level": "HIGH"},
        "jev": {"status": "ok", "probs": [0.6, 0.0, 0.0, 0.4]}}
_CALM = {"id": "c", "group": "borderline_conv", "expected": "CONVERSATIONAL", "llm": {"level": "CONVERSATIONAL"},
         "jev": {"status": "ok", "probs": [0.9, 0.05, 0.0, 0.05]}}


def _distance(params, key="taus"):
    return sum(abs(v - n) for v, n in zip(params[key], (0.5, 0.5, 0.5)))


def test_cost_zero_ties_go_to_the_params_nearest_neutral():
    choice = h.select_policy([_SEV, _CALM])
    assert choice["policy"] == "cumulative" and choice["cost"] == 0.0
    # The old grid-order pick was the first cost-0 grid point, taus [0.15, 0.1, 0.1] (distance 1.1).
    assert choice["params"]["taus"] == [0.5, 0.5, 0.4]
    assert _distance(choice["params"]) < _distance({"taus": [0.15, 0.1, 0.1]})
    # the pick still satisfies both cases through THE deployed verdict function
    assert tone.tone_verdict(_SEV["jev"]["probs"], "cumulative", choice["params"]).name == "HIGH"
    assert tone.tone_verdict(_CALM["jev"]["probs"], "cumulative", choice["params"]).name == "CONVERSATIONAL"


def test_argmax_still_wins_a_cost_tie_against_parameterised_policies():
    easy = {"id": "e", "group": "high", "expected": "HIGH", "llm": {"level": "HIGH"},
            "jev": {"status": "ok", "probs": [0.0, 0.0, 0.1, 0.9]}}
    assert h.select_policy([easy, _CALM])["policy"] == "argmax"


def test_neutral_params_are_the_deployed_defaults():
    assert h._neutrality("weighted", {"cuts": [0.5, 1.5, 2.5]}) == (1, 0.0)
    assert h._neutrality("cumulative", {"taus": [0.5, 0.5, 0.5]}) == (1, 0.0)
    assert h._neutrality("argmax", {}) < h._neutrality("cumulative", {"taus": [0.5, 0.5, 0.5]})


def test_selection_fixture_has_the_hyperbole_and_sarcasm_borderlines():
    texts = [c["text"] for c in cases.CASES if c["group"] == "borderline_conv"]
    assert any("kill me" in t.lower() for t in texts)
    assert any("living the dream" in t.lower() for t in texts)
    new = [c for c in cases.CASES if c["group"] == "borderline_conv" and c["id"] >= "borderline_conv_25"]
    assert len(new) >= 6 and all(c["expected"] == "CONVERSATIONAL" and len(c["text"]) < 400 for c in new)
    assert not any(re.search(r"\d{3,4}\s*mg", c["text"]) for c in cases.CASES)
