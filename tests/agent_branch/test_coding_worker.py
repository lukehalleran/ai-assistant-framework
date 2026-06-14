# tests/agent_branch/test_coding_worker.py
"""Pure-helper tests for the general coding worker + the objective wiring.

The worker runs inside a container, but its goals/context/prompt assembly are pure
functions importable on the host. Also covers worker_env_for — the supervisor
ALWAYS hands the worker the manifest's objective (a worker can't be pointed at a
goal the supervisor didn't authorize)."""

from agent_branch.manifest import BranchManifest
from agent_branch.supervisor import worker_env_for
from agent_branch.workers import coding_worker as cw


def _make_repo(tmp_path):
    (tmp_path / "agent_branch" / "goals").mkdir(parents=True)
    (tmp_path / "agent_branch" / "goals" / "_shared.md").write_text("SHARED RULES", encoding="utf-8")
    (tmp_path / "agent_branch" / "goals" / "reliability.md").write_text("BE RELIABLE", encoding="utf-8")
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "sandbox" / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "sandbox" / "util.py").write_text("HELPER = 1\n", encoding="utf-8")
    return str(tmp_path)


# -- goals loading (shared + per-agent lens) ---------------------------------

def test_load_goals_concatenates_shared_then_lens(tmp_path):
    repo = _make_repo(tmp_path)
    g = cw.load_goals(repo, "agent_branch/goals/reliability.md", "agent_branch/goals/_shared.md")
    assert "SHARED RULES" in g and "BE RELIABLE" in g
    assert g.index("SHARED RULES") < g.index("BE RELIABLE")  # shared first, lens layered on top


def test_load_goals_missing_lens_keeps_shared(tmp_path):
    repo = _make_repo(tmp_path)
    g = cw.load_goals(repo, "agent_branch/goals/nope.md", "agent_branch/goals/_shared.md")
    assert "SHARED RULES" in g


def test_load_goals_all_missing_is_empty(tmp_path):
    assert cw.load_goals(str(tmp_path), "x.md", "y.md") == ""


# -- bounded repo map --------------------------------------------------------

def test_repo_map_lists_files_and_skips_heavy_dirs(tmp_path):
    repo = _make_repo(tmp_path)
    (tmp_path / "data").mkdir(); (tmp_path / "data" / "huge.bin").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir(); (tmp_path / ".git" / "config").write_text("x", encoding="utf-8")
    m = cw.build_repo_map(repo)
    assert "sandbox/calc.py" in m
    assert "data/huge.bin" not in m and ".git/config" not in m  # heavy/irrelevant skipped


def test_repo_map_is_capped(tmp_path):
    (tmp_path / "pkg").mkdir()
    for i in range(50):
        (tmp_path / "pkg" / f"f{i}.py").write_text("x", encoding="utf-8")
    assert "truncated" in cw.build_repo_map(str(tmp_path), cap=10)


# -- context files -----------------------------------------------------------

def test_read_context_files_skips_missing_and_caps(tmp_path):
    repo = _make_repo(tmp_path)
    ctx = cw.read_context_files(repo, ["sandbox/util.py", "nope.py"], cap_chars=4)
    assert "sandbox/util.py" in ctx and "nope.py" not in ctx
    assert "truncated" in ctx["sandbox/util.py"]


# -- prompt assembly ---------------------------------------------------------

def test_build_user_prompt_has_objective_scope_goals_context_target(tmp_path):
    repo = _make_repo(tmp_path)
    goals = cw.load_goals(repo, "agent_branch/goals/reliability.md", "agent_branch/goals/_shared.md")
    prompt = cw.build_user_prompt(
        objective="implement subtract", goals=goals, target="sandbox/calc.py",
        target_content="def add(a, b):\n    return a + b\n",
        context_files={"sandbox/util.py": "HELPER = 1\n"},
        repo_map="sandbox/calc.py\nsandbox/util.py",
    )
    assert "implement subtract" in prompt
    assert "SHARED RULES" in prompt and "BE RELIABLE" in prompt
    assert "ONLY this file" in prompt          # scope rule present
    assert "sandbox/calc.py" in prompt
    assert "HELPER = 1" in prompt              # context file included


def test_strip_fences():
    assert cw.strip_fences("```python\nx = 1\n```") == "x = 1\n"
    assert cw.strip_fences("x = 1") == "x = 1\n"


# -- objective wiring (supervisor) -------------------------------------------

def test_worker_env_injects_manifest_objective():
    m = BranchManifest.issue(branch_id="b", objective="implement subtract",
                             allowed_paths=["sandbox/**"])
    env = worker_env_for(m, {"WORKER_STRATEGY": "surgical"})
    assert env["WORKER_OBJECTIVE"] == "implement subtract"
    assert env["WORKER_STRATEGY"] == "surgical"  # explicit env preserved


def test_explicit_worker_objective_overrides_manifest():
    m = BranchManifest.issue(branch_id="b", objective="from manifest",
                             allowed_paths=["sandbox/**"])
    env = worker_env_for(m, {"WORKER_OBJECTIVE": "override"})
    assert env["WORKER_OBJECTIVE"] == "override"
