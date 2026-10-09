"""Decision-model tone receipts reach turn_records (plan 2026-10-08, batch B5; D13, BC-72/47/58).

Path under test (every hop is the deployed code, nothing re-derived):
  ToneAnalysis.arbiter_receipt -> analyze_emotional_context -> EmotionalContext.arbiter_receipt
  -> ContextResult.emotional_context -> DaemonOrchestrator.build_full_prompt (_last_turn_signals,
  the ONE producer both answer routes share) -> _hook_turn_telemetry -> record_turn.

Evidence for the flattening choice: utils.turn_telemetry._sanitize_value DOES keep small nested
dicts of floats (pinned below), so the flat tone_dm_p0..p3 scalars are a deliberate choice that
matches the plan's wording and keeps every field greppable, not a sanitizer limitation.
Fixtures are synthetic. No network. Every test that reaches record_turn writes to tmp_path.
"""
import ast
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import utils.emotional_context as ec_mod
import utils.turn_telemetry as tt
from config import app_config
from core.context_pipeline import ContextResult, ToneLevel
from core.escalation_tracker import EscalationTracker
from core.orchestrator import DaemonOrchestrator, PostResponseHookContext, _hook_turn_telemetry
from utils.emotional_context import EmotionalContext, analyze_emotional_context
from utils.tone_detector import CrisisLevel, ToneAnalysis

REPO = Path(app_config.__file__).resolve().parents[1]   # the code under test (clone or base)
MESSAGE = "my sister Marguerite quietly mentioned the zephyrine ledger again yesterday"

ACTIVE_JEV = {
    "mode": "active", "backend": "jev", "status": "ok", "reason": None, "policy": "argmax",
    "retried": False, "latency_ms": 412, "served_model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe", "cost_usd": 2.0e-05, "probs": [0.1234, 0.5, 0.3, 0.0766],
    "decision_confidence": 0.8123, "level": "CONCERN",
}
SHADOW_AGREE = dict(ACTIVE_JEV, mode="shadow", backend="llm", llm_level="CONCERN", agrees=True)
SHADOW_UNAVAILABLE = {
    "mode": "shadow", "backend": "llm", "status": "unavailable", "reason": "timeout",
    "policy": "argmax", "retried": True, "latency_ms": 1500, "served_model": None,
    "provider": None, "cost_usd": None, "llm_level": "MEDIUM", "agrees": None,
}


def _ec(receipt, **kw):
    base = dict(
        crisis_level=CrisisLevel.CONCERN, need_type=None, tone_confidence=0.7,
        need_confidence=0.5, tone_trigger="llm_fallback", need_trigger="x", explanation="e",
        arbiter_receipt=receipt,
    )
    base.update(kw)
    return EmotionalContext(**base)


def _set_mode(monkeypatch, mode):
    monkeypatch.setattr(
        app_config, "DECISION_MODEL_CFG",
        {"enabled": mode != "off", "roles": {"tone_arbiter": mode, "heavy_topic": "off"},
         "tone_policy": "argmax"},
    )
    monkeypatch.setattr(app_config, "DECISION_MODEL_TONE_POLICY", "argmax")


def _telemetry_path(monkeypatch, tmp_path):
    monkeypatch.setattr(app_config, "TURN_TELEMETRY_ENABLED", True)
    path = tmp_path / "turns.jsonl"
    monkeypatch.setattr(app_config, "TURN_TELEMETRY_PATH", str(path))
    return path


# 1. The assembly function --------------------------------------------------------------------

class TestAssembly:
    def test_active_jev_receipt_flattened(self):
        out = ec_mod.tone_receipt_fields(_ec(ACTIVE_JEV))
        assert out == {
            "tone_dm_mode": "active", "tone_dm_status": "ok", "tone_dm_reason": None,
            "tone_dm_level": "CONCERN", "tone_dm_decision_confidence": 0.8123,
            "tone_dm_policy": "argmax", "tone_dm_retried": False, "tone_dm_latency_ms": 412,
            "tone_dm_served_model": "typesafe/jev-1.13-20260917", "tone_dm_provider": "TypeSafe",
            "tone_dm_cost_usd": 2.0e-05, "tone_dm_p0": 0.123, "tone_dm_p1": 0.5,
            "tone_dm_p2": 0.3, "tone_dm_p3": 0.077, "tone_dm_deciding_level": "CONCERN",
            "tone_arbiter_backend": "jev",
        }

    def test_shadow_receipt_records_agreement_and_llm_deciding_level(self):
        out = ec_mod.tone_receipt_fields(_ec(dict(SHADOW_AGREE, llm_level="MEDIUM", agrees=False)))
        assert out["tone_dm_mode"] == "shadow" and out["tone_arbiter_backend"] == "llm"
        assert out["tone_dm_agrees"] is False
        assert out["tone_dm_deciding_level"] == "MEDIUM"     # the LLM decided in shadow
        assert out["tone_dm_level"] == "CONCERN"             # what Jev would have said

    def test_unavailable_shadow_keeps_status_reason_and_no_probs(self):
        out = ec_mod.tone_receipt_fields(_ec(SHADOW_UNAVAILABLE))
        assert out["tone_dm_status"] == "unavailable" and out["tone_dm_reason"] == "timeout"
        assert out["tone_dm_retried"] is True
        assert not any(f"tone_dm_p{i}" in out for i in range(4))

    def test_off_mode_receipt_has_a_status_never_empty(self):
        out = ec_mod.tone_receipt_fields(_ec({"mode": "off", "backend": "llm"}))
        assert out["tone_dm_mode"] == "off"
        assert out["tone_dm_status"] == "not_run"
        assert out["tone_arbiter_backend"] == "llm"

    @pytest.mark.parametrize("mode", ["off", "shadow", "active"])
    @pytest.mark.parametrize("ctx", [None, "no_receipt", "mock"])
    def test_arbiter_never_ran_records_resolved_mode_and_none_backend(self, monkeypatch, mode, ctx):
        _set_mode(monkeypatch, mode)
        arg = {
            None: None, "no_receipt": _ec(None),
            "mock": SimpleNamespace(arbiter_receipt=object()),   # non-dict receipt
        }[ctx]
        out = ec_mod.tone_receipt_fields(arg)
        assert out["tone_dm_mode"] == mode
        assert out["tone_dm_status"] == "not_run"
        assert out["tone_arbiter_backend"] == "none"
        assert "tone_dm_level" not in out and "tone_dm_p0" not in out

    def test_non_finite_and_non_numeric_probs_are_dropped(self):
        out = ec_mod.tone_receipt_fields(_ec(dict(
            ACTIVE_JEV, probs=[float("nan"), True, float("inf"), 0.25],
            latency_ms=float("inf"), decision_confidence="high")))
        assert [k for k in out if k.startswith("tone_dm_p") and k[-1].isdigit()] == ["tone_dm_p3"]
        assert out["tone_dm_latency_ms"] is None and out["tone_dm_decision_confidence"] is None
        assert all(not (isinstance(v, float) and not math.isfinite(v)) for v in out.values())

    def test_wrong_length_probs_are_ignored(self):
        out = ec_mod.tone_receipt_fields(_ec(dict(ACTIVE_JEV, probs=[0.5, 0.5])))
        assert not any(k.startswith("tone_dm_p") and k[-1].isdigit() for k in out)

    def test_free_text_in_a_label_field_cannot_ride_the_record(self):
        out = ec_mod.tone_receipt_fields(_ec(dict(ACTIVE_JEV, reason=MESSAGE, provider=MESSAGE)))
        assert out["tone_dm_reason"] == "invalid_label" and out["tone_dm_provider"] == "invalid_label"
        assert "zephyrine" not in json.dumps(out)

    def test_never_raises(self):
        class Boom:
            @property
            def arbiter_receipt(self):
                raise RuntimeError("boom")
        assert ec_mod.tone_receipt_fields(Boom()) == {}

    def test_every_emitted_field_is_a_scalar_and_documented(self):
        out = ec_mod.tone_receipt_fields(_ec(dict(SHADOW_AGREE)))
        doc = tt.__doc__
        for name, value in out.items():
            assert value is None or isinstance(value, (str, int, float, bool)), name
            if name.startswith("tone_dm_p") and name[-1].isdigit():
                continue                      # documented as the range tone_dm_p0..tone_dm_p3
            assert name in doc, f"{name} missing from the turn_telemetry docstring field list"
        for name in ("tone_dm_p0..tone_dm_p3", "tone_trigger", "tone_confidence", "tone_arbiter_backend"):
            assert name in doc


# 2. analyze_emotional_context carries the receipt ----------------------------------------------

def _tone(receipt, level=CrisisLevel.CONCERN):
    return ToneAnalysis(level=level, confidence=0.7, trigger="llm_fallback", raw_scores={},
                        explanation="x", arbiter_receipt=receipt)


class TestAnalyzeCarriesReceipt:
    async def test_receipt_copied_from_tone_analysis(self, monkeypatch):
        monkeypatch.setattr(ec_mod, "detect_crisis_level", AsyncMock(return_value=_tone(ACTIVE_JEV)))
        ctx = await analyze_emotional_context(MESSAGE)
        assert ctx.arbiter_receipt == ACTIVE_JEV

    async def test_absent_receipt_stays_none(self, monkeypatch):
        monkeypatch.setattr(ec_mod, "detect_crisis_level", AsyncMock(return_value=_tone(None)))
        assert (await analyze_emotional_context(MESSAGE)).arbiter_receipt is None


# 3. End to end: detect_crisis_level result -> build_full_prompt -> hook -> record ---------------

class _DoneTask:
    def add_done_callback(self, cb):
        cb(self)


class _Builder:
    async def build_prompt_from_context(self, context):
        return {"_section_outcomes": {}}

    def _assemble_prompt(self, context, user_input, system_prompt):
        return "PROMPT"


def _orchestrator(monkeypatch):
    orch = object.__new__(DaemonOrchestrator)       # same technique as test_section_outcome_receipts
    orch.escalation_tracker = EscalationTracker()
    orch.safety_canary = None
    orch.response_planner = None
    orch.logger = logging.getLogger("test_decision_model_receipts")
    monkeypatch.setattr(orch, "_build_system_prompt", lambda c, r: "SYSTEM")
    orch.prompt_builder = _Builder()
    return orch


async def _turn_record(monkeypatch, tmp_path, tone, telemetry_task=None):
    path = _telemetry_path(monkeypatch, tmp_path)
    monkeypatch.setattr(ec_mod, "detect_crisis_level", AsyncMock(return_value=tone))
    emotional = await analyze_emotional_context(MESSAGE)
    context = ContextResult(
        processed_query=MESSAGE, original_query=MESSAGE, tone_level=ToneLevel.CONCERN,
        tone_instructions="", emotional_context=emotional,
    )
    orch = _orchestrator(monkeypatch)
    await orch.build_full_prompt(context, use_raw_mode=False)
    _hook_turn_telemetry(PostResponseHookContext(
        orchestrator=orch, user_input=MESSAGE, response_text=None, mode="enhanced",
        session_id="s", model_name="m", response_len=0, telemetry={}, t_prepare_elapsed=0.0,
        telemetry_task=telemetry_task,
    ))
    return json.loads(path.read_text())


class TestEndToEnd:
    @pytest.mark.parametrize("task", [None, _DoneTask()], ids=["sync", "deferred"])
    async def test_receipt_reaches_the_record(self, monkeypatch, tmp_path, task):
        row = await _turn_record(monkeypatch, tmp_path, _tone(dict(SHADOW_AGREE)), task)
        assert row["tone_trigger"] == "llm_fallback"
        assert row["tone_dm_mode"] == "shadow" and row["tone_dm_status"] == "ok"
        assert row["tone_arbiter_backend"] == "llm" and row["tone_dm_agrees"] is True
        assert [row[f"tone_dm_p{i}"] for i in range(4)] == [0.123, 0.5, 0.3, 0.077]
        assert row["tone_dm_served_model"] == "typesafe/jev-1.13-20260917"
        assert row["tone_dm_latency_ms"] == 412

    async def test_turn_without_a_receipt_still_records_mode_status_backend(self, monkeypatch, tmp_path):
        _set_mode(monkeypatch, "shadow")
        row = await _turn_record(monkeypatch, tmp_path, _tone(None))
        assert row["tone_dm_mode"] == "shadow"
        assert row["tone_dm_status"] == "not_run"
        assert row["tone_arbiter_backend"] == "none"

    async def test_record_is_content_free(self, monkeypatch, tmp_path):
        poisoned = dict(SHADOW_AGREE, reason=MESSAGE, status=MESSAGE, policy=MESSAGE, level=MESSAGE)
        row = await _turn_record(monkeypatch, tmp_path, _tone(poisoned))
        # `query` is the pre-existing, by-design field; every other field must be free of the text.
        rest = json.dumps({k: v for k, v in row.items() if k != "query"}).lower()
        lowered = MESSAGE.lower()
        runs = {lowered[i:i + 6] for i in range(len(lowered) - 5) if " " not in lowered[i:i + 6]}
        assert runs and not [r for r in runs if r in rest], [r for r in runs if r in rest]
        assert all(
            v is None or isinstance(v, (str, int, float, bool))
            for k, v in row.items() if k.startswith("tone_dm_") or k == "tone_arbiter_backend"
        )


# 4. Producer/consumer structure -----------------------------------------------------------------

class TestSharedProducer:
    def test_record_turn_is_called_only_from_the_shared_hook(self):
        """Both answer routes (GUI generator and process_user_query) reach record_turn through the
        POST_RESPONSE_HOOKS registry, so wiring the producer covers both (BC-58)."""
        sites = []
        for root in ("gui", "api", "core"):
            for path in sorted((REPO / root).rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for fn in ast.walk(tree):
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for node in ast.walk(fn):
                            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                    and node.func.id == "record_turn"):
                                sites.append((str(path.relative_to(REPO)), fn.name))
        assert set(sites) == {("core/orchestrator.py", "_hook_turn_telemetry"),
                              ("core/orchestrator.py", "write_completed")}, sites

    def test_producer_dict_includes_the_receipt_fields(self):
        src = (REPO / "core" / "orchestrator.py").read_text(encoding="utf-8")
        assert src.count("**tone_receipt_fields(") == 1

    def test_sanitizer_keeps_small_float_dicts_but_receipts_are_flat_by_design(self):
        assert tt._sanitize_value({"p": {"0": 0.1, "1": 0.9}}) == {"p": {"0": 0.1, "1": 0.9}}
