# tests/agent_branch/test_goal_runner.py
"""Goal-driven proposal mode: target safety, the auto regression proof, and the
per-lens objective derivation (mocked — no LLM/podman). The full derive→implement→
gate→ingest path is a live run; these pin the cheap, safety-critical logic:
a derived objective can only target safe code, and the regression proof imports it."""

import pytest

from agent_branch import goal_runner as gr


# -- target safety -----------------------------------------------------------

def test_module_path():
    assert gr.module_path("utils/foo.py") == "utils.foo"
    assert gr.module_path("a/b/c.py") == "a.b.c"
    assert gr.module_path("README.md") == ""


def test_acceptable_target_allows_safe_code_areas():
    assert gr.acceptable_target("utils/foo.py")
    assert gr.acceptable_target("memory/bar.py")
    assert gr.acceptable_target("core/prompt/baz.py")


def test_acceptable_target_rejects_unsafe_or_nonpy():
    for bad in ["tests/test_x.py", "config/app.py", "agent_branch/x.py",
                "scripts/y.py", "docs/z.md", "utils/foo.txt", "random/x.py", ""]:
        assert not gr.acceptable_target(bad), bad


def test_autoproof_imports_the_target_module():
    rel, content = gr.autoproof_for("utils/foo.py")
    assert rel.startswith("agent_branch/proofs/_autoproof_") and rel.endswith(".py")
    assert "import_module('utils.foo')" in content


# -- derivation (self-proposer reused, lens as mandate) ----------------------

class _FakeProposal:
    def __init__(self, title, desc, files):
        self.title, self.description, self.affected_files = title, desc, files


def _fake_gen(proposals):
    class _Gen:
        def __init__(self, **kw):
            pass

        async def generate_proposals(self, **kw):
            return proposals
    return _Gen


@pytest.mark.asyncio
async def test_derive_objective_picks_first_code_target(monkeypatch):
    import knowledge.proposal_generator as pg
    monkeypatch.setattr(pg, "GoalDirectedGenerator", _fake_gen([
        _FakeProposal("Write docs", "stuff", ["docs/x.md"]),        # skipped (not code)
        _FakeProposal("Add a helper", "a small helper", ["utils/helper.py"]),  # picked
    ]))
    obj = await gr.derive_objective("be reliable", repo=".", model_manager=object())
    assert obj["target"] == "utils/helper.py"
    assert obj["allowed"] == ["utils/helper.py"]
    assert "Add a helper" in obj["objective"]


@pytest.mark.asyncio
async def test_derive_objective_none_when_no_safe_code_target(monkeypatch):
    import knowledge.proposal_generator as pg
    monkeypatch.setattr(pg, "GoalDirectedGenerator", _fake_gen([
        _FakeProposal("Docs", "d", ["docs/x.md"]),
        _FakeProposal("Edit tests", "t", ["tests/test_y.py"]),      # forbidden
        _FakeProposal("Touch config", "c", ["config/app_config.py"]),  # forbidden
    ]))
    obj = await gr.derive_objective("lens", repo=".", model_manager=object())
    assert obj is None


@pytest.mark.asyncio
async def test_goal_run_refuses_while_daemon_running(monkeypatch):
    # The running-Daemon guard must short-circuit before any heavy work (OOM safety).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(gr, "wait_for_daemon_idle", lambda *a, **k: False)
    out = await gr.run_goal_driven(["reliability"], source=".")
    assert out == []
