"""The shutdown behavioral-pattern step is gone and must stay gone.

Incident 2026-10-15/16 (class: BC-75, BC-46): ``eating_habit = orders food
after completing homework`` was stored from ONE evening's messages by the
single-session behavioral-pattern step. The step was deleted (plan
PLAN_20261016 section 3.2) instead of patched a third time; this file is the
regression: the deployed Phase A schedules no pattern step, no pattern prompt
is ever sent, the profile receives no pattern facts, and nothing in the source
tree reads the ``behavioral_patterns`` config key.
"""
import ast
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory import shutdown_processor as sp
from memory.shutdown_processor import ShutdownProcessor
from memory.user_profile import UserProfile

ROOT = Path(__file__).resolve().parents[2]
PHASE_A = ("_extract_session_facts", "_extract_llm_facts", "_extract_procedural_skills")
OTHERS = ("summarize_uncovered_blocks", "_generate_proposals", "_check_implementation_tracking",
          "_process_open_threads", "_run_wiki_enrichment", "_run_wikidata_enrichment",
          "_run_cross_collection_dedup")


def _drive(tmp_path, reply):
    profile = UserProfile(profile_path=str(tmp_path / "profile.json"))
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value=reply)
    proc = ShutdownProcessor(
        corpus_manager=MagicMock(), chroma_store=MagicMock(),
        consolidator=MagicMock(), fact_extractor=MagicMock(),
        model_manager=mm, user_profile=profile, storage=MagicMock(),
        session_start=datetime.now(),
    )
    scheduled = []
    for name in PHASE_A + OTHERS:
        async def _rec(*a, _n=name, **k):
            scheduled.append(_n)
        setattr(proc, name, _rec)
    proc._save_knowledge_graph = lambda: None
    proc._maybe_spawn_agent_branch = lambda: None
    convs = [{"query": q, "response": "ok", "timestamp": datetime.now().isoformat()}
             for q in ("I ordered burgers and had a beer", "worked on hw 4 for a while", "finished hw 4")]
    asyncio.run(proc.process_shutdown_memory(convs))
    facts = [f for cat in profile.profile["categories"].values() for f in cat if isinstance(f, dict)]
    return proc, mm, scheduled, facts


@pytest.mark.parametrize("reply", [
    '[{"relation": "eating_habit", "object": "orders food after completing homework", "confidence": 0.75}]',
    "Sure! Here are some patterns you might like.",
    '[{"relation": "x_habit", "object": ',
])
def test_phase_a_schedules_no_pattern_step(tmp_path, reply):
    proc, mm, scheduled, facts = _drive(tmp_path, reply)
    assert not hasattr(proc, "_extract_behavioral_patterns")
    assert set(PHASE_A) <= set(scheduled)
    assert not [n for n in scheduled if "behavioral" in n or "pattern" in n]
    for call in mm.generate_once.call_args_list:
        blob = str(call).lower()
        assert "behavioral pattern" not in blob and "recurring behavioral" not in blob
    assert facts == []


def test_config_yaml_still_carrying_the_key_validates():
    from config.schema import validate_config
    cfg = {"behavioral_patterns": {"enabled": True, "min_turns": 3, "max_patterns": 3}}
    out = validate_config(cfg)
    assert out["behavioral_patterns"]["enabled"] is True


def test_no_code_reads_behavioral_patterns():
    hits = []
    for d in ("memory", "core", "gui", "config", "api", "utils", "scripts"):
        for p in (ROOT / d).rglob("*.py"):
            try:
                tree = ast.parse(p.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for n in ast.walk(tree):
                s = None
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    s = n.value
                elif isinstance(n, ast.Name):
                    s = n.id
                elif isinstance(n, ast.Attribute):
                    s = n.attr
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    s = n.name
                if s and ("behavioral_pattern" in s.lower() or "behavioralpattern" in s.lower()):
                    hits.append(f"{p.relative_to(ROOT)}:{getattr(n, 'lineno', 0)}")
    assert hits == []
    assert "behavioral_patterns:" not in (ROOT / "config/config.yaml").read_text()
    assert not hasattr(sp.ShutdownProcessor, "_extract_behavioral_patterns")
