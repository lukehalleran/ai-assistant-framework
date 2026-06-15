# tests/agent_branch/test_objective_queue.py
"""The shutdown-triggered objective queue: store semantics, the drainer's
fail-safe gates, and that the shutdown spawn is OFF by default.

The drain/spawn paths that actually run podman+LLM aren't exercised here (that's a
live run); these pin the cheap, safety-critical edges: a corrupt queue can't crash
shutdown, no-key / empty-queue do nothing, and nothing spawns unless explicitly
enabled."""

from agent_branch import run, run_queue
from agent_branch.queue import (
    ObjectiveQueue,
    enqueue,
    enqueue_failing_test,
)


# -- queue store -------------------------------------------------------------

def test_enqueue_explicit(tmp_path):
    p = tmp_path / "q.json"
    e = enqueue("do x", target="utils/x.py", allowed=["utils/x.py"], proof=["t.py"], path=p)
    assert e.status == "pending" and e.target == "utils/x.py" and e.proof == ["t.py"]
    assert ObjectiveQueue.load(p).pending()[0].id == e.id


def test_enqueue_failing_test_uses_test_as_proof(tmp_path):
    p = tmp_path / "q.json"
    e = enqueue_failing_test("tests/proof_foo.py", target="utils/foo.py",
                             allowed=["utils/foo.py"], path=p)
    assert e.proof == ["tests/proof_foo.py"]
    assert "tests/proof_foo.py" in e.objective and "Do NOT modify" in e.objective


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "q.json"
    enqueue("a", target="x", allowed=["x"], proof=["t"], path=p)
    enqueue("b", target="y", allowed=["y"], proof=["t"], path=p)
    assert [e.objective for e in ObjectiveQueue.load(p).entries] == ["a", "b"]


def test_load_missing_is_empty(tmp_path):
    assert ObjectiveQueue.load(tmp_path / "nope.json").entries == []


def test_load_corrupt_is_empty(tmp_path):
    p = tmp_path / "q.json"
    p.write_text("{not json", encoding="utf-8")
    assert ObjectiveQueue.load(p).entries == []   # must not crash shutdown


def test_pending_excludes_done(tmp_path):
    p = tmp_path / "q.json"
    enqueue("a", target="x", allowed=["x"], proof=["t"], path=p)
    q = ObjectiveQueue.load(p)
    q.entries[0].status = "done"
    q.save(p)
    assert ObjectiveQueue.load(p).pending() == []


# -- drainer fail-safe gates (no podman/LLM touched) -------------------------

def test_drain_no_key_returns_zero(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    qp = tmp_path / "q.json"
    enqueue("x", target="t", allowed=["t"], proof=["p"], path=qp)
    assert run_queue.drain(queue_path=qp, max_items=1) == 0  # no key -> nothing runs


def test_drain_empty_queue_returns_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert run_queue.drain(queue_path=tmp_path / "empty.json", max_items=1) == 0


def test_drain_refuses_while_daemon_running(monkeypatch, tmp_path):
    # Don't overlap main.py's memory (OOM hazard) — refuse before any heavy work.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(run_queue, "wait_for_daemon_idle", lambda *a, **k: False)
    qp = tmp_path / "q.json"
    enqueue("x", target="t", allowed=["t"], proof=["p"], path=qp)
    assert run_queue.drain(queue_path=qp, max_items=1) == 0


def test_run_exposes_reusable_helpers():
    assert callable(run.run_objective) and callable(run.live_proposal_store)


# -- shutdown spawn is OFF by default ----------------------------------------

def test_shutdown_spawn_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_BRANCH_SHUTDOWN_ENABLED", raising=False)
    import subprocess
    called = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.append(a))
    from memory.shutdown_processor import ShutdownProcessor
    sp = object.__new__(ShutdownProcessor)   # no heavy __init__ needed
    sp._maybe_spawn_agent_branch()
    assert called == []   # default off -> never spawns


def test_shutdown_goal_driven_needs_its_own_flag(monkeypatch):
    # enabled + podman + key + empty queue, but no GOALS flag -> nothing spawns
    monkeypatch.setenv("AGENT_BRANCH_SHUTDOWN_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AGENT_BRANCH_SHUTDOWN_GOALS", raising=False)
    import shutil, subprocess
    import agent_branch.queue as q
    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/podman")
    monkeypatch.setattr(q.ObjectiveQueue, "load",
                        classmethod(lambda cls, *a, **k: q.ObjectiveQueue()))  # empty
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **k: spawned.append(argv))
    from memory.shutdown_processor import ShutdownProcessor
    object.__new__(ShutdownProcessor)._maybe_spawn_agent_branch()
    assert spawned == []   # empty queue + no GOALS flag -> nothing runs


def test_shutdown_goal_driven_spawns_goal_runner_when_enabled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)   # the spawn opens logs/ — keep it out of the repo
    monkeypatch.setenv("AGENT_BRANCH_SHUTDOWN_ENABLED", "1")
    monkeypatch.setenv("AGENT_BRANCH_SHUTDOWN_GOALS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import shutil, subprocess
    import agent_branch.queue as q
    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/podman")
    monkeypatch.setattr(q.ObjectiveQueue, "load",
                        classmethod(lambda cls, *a, **k: q.ObjectiveQueue()))  # empty queue
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **k: spawned.append(argv))
    from memory.shutdown_processor import ShutdownProcessor
    object.__new__(ShutdownProcessor)._maybe_spawn_agent_branch()
    flat = " ".join(" ".join(a) for a in spawned)
    assert "agent_branch.goal_runner" in flat        # goal-driven fired
    assert "agent_branch.run_queue" not in flat       # queue empty -> not fired
