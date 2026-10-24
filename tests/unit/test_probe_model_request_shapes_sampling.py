"""
Probe plan for the third request shape, `sampling` (2026-10-23, dry-run only).

scripts/probe_model_request_shapes.py now plans a temperature+top_p request per
model: a row declaring `sampling_params: False` must be REFUSED by the route and
the deployed resolve_sampling() output must be ACCEPTED; any other row must be
ACCEPTED. The existing off-switch / forced-tool shapes are unchanged. Nothing
here sends a request.

class: BC-82, BC-83.
"""
import importlib.util
from pathlib import Path

import pytest

from models.model_manager import DEFAULT_TEMPERATURE, MODEL_CAPABILITIES

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_model_request_shapes.py"
KIMI31 = "moonshotai/kimi-k3.1-pro"
KIMI3 = "moonshotai/kimi-k3"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("probe_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _by_label(plan):
    return {label: (body, accept) for label, body, accept in plan}


def test_unconstrained_row_plans_one_accepted_sampling_request(probe):
    sampling = {k: v for k, v in _by_label(probe._requests_for(KIMI31)).items() if k.startswith("sampling")}
    assert list(sampling) == ["sampling temperature+top_p"]
    body, accept = sampling["sampling temperature+top_p"]
    assert accept is True
    assert body["temperature"] == DEFAULT_TEMPERATURE and "top_p" in body


def test_forced_top_p_row_probes_with_its_forced_value(probe):
    body, accept = _by_label(probe._requests_for(KIMI3))["sampling temperature+top_p"]
    assert body["top_p"] == 0.95 and accept is True


def test_refusing_row_expects_refusal_then_accepts_the_deployed_substitute(probe, monkeypatch):
    monkeypatch.setitem(MODEL_CAPABILITIES, KIMI31, {**MODEL_CAPABILITIES[KIMI31], "sampling_params": False})
    plan = _by_label(probe._requests_for(KIMI31))
    raw_body, raw_accept = plan["sampling temperature+top_p"]
    sub_body, sub_accept = plan["deployed sampling substitute"]
    assert raw_accept is False and "temperature" in raw_body
    assert sub_accept is True
    assert not any(k in sub_body for k in ("temperature", "top_p", "top_k"))


def test_existing_shapes_unchanged(probe):
    labels = set(_by_label(probe._requests_for(KIMI31)))
    assert {"off-switch enabled=false", "forced tool_choice"} <= labels


def test_dry_run_prints_plan_and_sends_nothing(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe, "_post", lambda *a, **k: pytest.fail("dry run must not send"))
    monkeypatch.setattr("sys.argv", ["probe", "--models", "kimi-k3.1-pro"])
    assert probe.main() == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "sampling temperature+top_p" in out and "expect ACCEPT" in out
