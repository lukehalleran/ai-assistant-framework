"""app_config fallback drift ratchet + config-honesty contracts.

Incident 2026-10-24 (SETTINGS_AUDIT F4/F5/F7/F9): deliberate OFF/tuned decisions lived only
in config.yaml while ``app_config`` carried the OLD value as its ``.get("key", default)``
fallback, so a lost key or a YAML that failed to parse silently re-enabled them; web-trigger
YAML keys were dead (env-only reads); builder section caps read keys no YAML carried.

class: BC-82 (live config), BC-88 (env reads outside config), BC-58 (guard covered one
path), BC-15 (dead wiring).

The scan is pure ``ast`` + ``yaml`` over the working tree (no app import, no git state).
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Count of app_config ``.get("key", literal)`` fallbacks whose literal differs from the
# committed config.yaml value (2026-10-25). Deliberately-fixed set (below) is excluded
# from it by test_decided_off_fallbacks_match_yaml. Lower this when you align another.
FALLBACK_DRIFT_CEILING = 27  # 27 with the oct25 PIPE lane (removes a drifting features key); 28 on CFG alone

# Fallbacks the owner decided deliberately; each must equal the YAML value.
DECIDED = {
    ("token_budget", "default"),
    ("uncertainty_fallback", "enabled"),
    ("code_proposals", "enabled"),
    ("graph_walk", "enabled"),
    ("synthesis_generator", "enabled"),
    ("synthesis_retrieval", "enabled"),
    ("personal_claim_check", "timeout_s"),
    ("personal_claim_check", "max_tokens"),
}


def _scan(root: Path):
    cfg = yaml.safe_load((root / "config/config.yaml").read_text())
    tree = ast.parse((root / "config/app_config.py").read_text())
    def lit(n):
        try: return True, ast.literal_eval(n)
        except Exception: return False, None

    def sec_of_call(n):
        # config.get("sec", {}) or config.get("sec") or {}
        if isinstance(n, ast.BoolOp): n = n.values[0]
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr=='get'
                and isinstance(n.func.value, ast.Name) and n.func.value.id=='config'
                and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value,str)):
            return n.args[0].value
        return None
    varsec = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            val = node.value
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            s = sec_of_call(val) if val is not None else None
            if s:
                for t in tgts:
                    if isinstance(t, ast.Name): varsec[t.id]=s
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr=='get' and len(n.args)>=2:
            recv = n.func.value
            sec = None
            if isinstance(recv, ast.Name) and recv.id in varsec: sec = varsec[recv.id]
            else: sec = sec_of_call(recv)
            if not sec or not isinstance(n.args[0], ast.Constant) or not isinstance(n.args[0].value,str): continue
            key = n.args[0].value
            ok, d = lit(n.args[1])
            if not ok or isinstance(d,(dict,list,tuple,set)) or d is None: continue
            sv = cfg.get(sec) or {}
            if not isinstance(sv, dict) or key not in sv: continue
            y = sv[key]
            if isinstance(y,(dict,list)) : continue
            if isinstance(d,bool) or isinstance(y,bool):
                same = (d is y) if isinstance(d,bool) and isinstance(y,bool) else False
            elif isinstance(d,(int,float)) and isinstance(y,(int,float)): same = float(d)==float(y)
            else: same = (d==y)
            if not same: out.append((sec,key,d,y,n.lineno))
    return out


def test_fallback_drift_does_not_grow():
    drift = _scan(REPO_ROOT)
    assert len(drift) <= FALLBACK_DRIFT_CEILING, (
        f"{len(drift)} app_config fallbacks differ from config.yaml (ceiling "
        f"{FALLBACK_DRIFT_CEILING}). Align the new fallback with the YAML value:\n"
        + "\n".join(map(str, drift))
    )


def test_fallback_drift_ceiling_not_slack():
    drift = _scan(REPO_ROOT)
    assert len(drift) == FALLBACK_DRIFT_CEILING, (
        f"drift is now {len(drift)}; lower FALLBACK_DRIFT_CEILING to match"
    )


def test_decided_off_fallbacks_match_yaml():
    drifted = {(s, k) for s, k, *_ in _scan(REPO_ROOT)}
    assert not (drifted & DECIDED), sorted(drifted & DECIDED)


def test_scan_detects_a_planted_drift(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/config.yaml").write_text("sec:\n  flag: false\n")
    (tmp_path / "config/app_config.py").write_text(
        "SEC_CFG = config.get('sec', {})\nFLAG = SEC_CFG.get('flag', True)\n"
        "OK = config.get('sec', {}).get('flag', False)\n"
    )
    assert [(s, k) for s, k, *_ in _scan(tmp_path)] == [("sec", "flag")]


# ---- parse failure aborts startup (F9) -----------------------------------------------

def test_broken_config_yaml_aborts(tmp_path):
    from config.app_config import load_yaml_config
    from config.config_errors import ConfigParseError
    p = tmp_path / "broken.yaml"
    p.write_text("a: [unclosed\n  b: : :\n")
    with pytest.raises(ConfigParseError) as ei:
        load_yaml_config(str(p))
    assert str(p) in str(ei.value) and "Refusing to start" in str(ei.value)


def test_non_mapping_config_yaml_aborts(tmp_path):
    from config.app_config import load_yaml_config
    from config.config_errors import ConfigParseError
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigParseError):
        load_yaml_config(str(p))


def test_broken_local_overrides_abort_missing_is_fine(tmp_path, monkeypatch):
    import config.app_config as ac
    from config.config_errors import ConfigParseError
    monkeypatch.delenv("DAEMON_IGNORE_LOCAL_CONFIG", raising=False)
    broken = tmp_path / "config.local.yaml"
    broken.write_text("daemon: {mode: dev\n")
    monkeypatch.setattr(ac, "_candidate_config_paths", lambda name: [broken])
    with pytest.raises(ConfigParseError):
        ac.load_local_overrides()
    broken.unlink()
    assert ac.load_local_overrides() == {}  # missing local file stays optional
    good = tmp_path / "config.local.yaml"
    good.write_text("daemon:\n  mode: dev\n")
    assert ac.load_local_overrides()["daemon"]["mode"] == "dev"


def test_main_guard_precedes_other_daemon_imports():
    tree = ast.parse((REPO_ROOT / "main.py").read_text())
    guard = logging_import = None
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Try) and any(
            isinstance(h.type, ast.Name) and h.type.id == "ConfigParseError" for h in node.handlers
        ):
            guard = i
        if isinstance(node, ast.ImportFrom) and node.module == "utils.logging_utils":
            logging_import = i
    assert guard is not None, "main.py must catch ConfigParseError at import time"
    assert logging_import is not None and guard < logging_import
    src = ast.get_source_segment((REPO_ROOT / "main.py").read_text(), tree.body[guard])
    assert "sys.exit(1)" in src


# ---- web trigger reads web_search.* YAML (F5) ----------------------------------------

def test_web_trigger_constants_follow_yaml():
    import config.app_config as ac
    import utils.web_search_trigger as wst
    ws = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text())["web_search"]
    assert ws["trigger_max_tokens"] == 400  # the 2026-10-08 value, not the stale 150
    assert wst.SEARCH_TRIGGER_MAX_TOKENS == ac.WEB_SEARCH_TRIGGER_MAX_TOKENS
    assert wst.SEARCH_TRIGGER_MODEL == ac.WEB_SEARCH_TRIGGER_MODEL
    assert wst.SEARCH_TRIGGER_TIMEOUT == ac.WEB_SEARCH_TRIGGER_TIMEOUT
    assert wst.LLM_HEURISTIC_BLEND_WEIGHT == ac.WEB_SEARCH_LLM_HEURISTIC_WEIGHT
    assert wst.SEARCH_CONFIDENCE_THRESHOLD == ac.WEB_SEARCH_CONFIDENCE_THRESHOLD
    assert wst.LLM_FIRST_ENABLED == ac.WEB_SEARCH_LLM_FIRST_ENABLED


def test_web_trigger_env_still_overrides_in_a_fresh_process():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(DAEMON_IGNORE_LOCAL_CONFIG="1", DAEMON_TEST_MODE="1",
               WEB_SEARCH_TRIGGER_MAX_TOKENS="777", WEB_SEARCH_LLM_FIRST_ENABLED="false",
               WEB_SEARCH_TRIGGER_MODEL="env-model")
    code = ("import utils.web_search_trigger as w;"
            "print(w.SEARCH_TRIGGER_MAX_TOKENS, w.LLM_FIRST_ENABLED, w.SEARCH_TRIGGER_MODEL)")
    out = subprocess.run([sys.executable, "-s", "-c", code], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=180)
    assert out.stdout.strip().splitlines()[-1] == "777 False env-model", out.stdout + out.stderr


# ---- builder section caps read the keys the YAML carries (F7) ------------------------

def test_builder_caps_read_their_yaml_section_keys():
    import core.prompt.builder as b
    y = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text())
    assert b.PROMPT_MAX_GRAPH_SENTENCES == y["knowledge_graph"]["max_prompt_sentences"]
    assert b.PROMPT_MAX_PROACTIVE_INSIGHTS == y["proactive_surfacing"]["max_prompt_insights"]
    assert b.PROMPT_MAX_SKILLS == y["procedural_skills"]["prompt_max_skills"]
    assert b.PROMPT_MAX_SURFACED_THREADS == y["thread_surfacing"]["max_surfaced"]
    assert b.PROMPT_MAX_VISUAL_MEMORIES == y["visual_memory"]["max_images_prompt"]


def test_force_min_memories_key_is_the_one_the_code_reads():
    y = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text())["memory"]
    assert "force_min_memories" not in y and y["semantic_force_min_memories"] is True


def test_dead_gating_keys_are_gone():
    y = yaml.safe_load((REPO_ROOT / "config/config.yaml").read_text())["gating"]
    from config.schema import GatingSection
    for k in ("cosine_threshold", "cosine_similarity_threshold", "use_reranking", "rerank_use_llm",
              "cross_encoder_weight", "confidence_threshold", "topic_confidence_threshold",
              "summary_cosine_threshold", "reflection_cosine_threshold"):
        assert k not in y and k not in GatingSection.model_fields, k
