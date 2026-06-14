# tests/agent_branch/test_run.py
"""Tests for the real-run CLI's pure wiring (no podman, no key, no network).

Covers the assembly the runner does before it ever spawns a container: per-lens
worker specs, the manifest template, per-branch proxy construction, and the
fail-fast when no LLM key is present."""

from pathlib import Path

from agent_branch import run
from agent_branch.provisioning import NetworkMode


def test_build_specs_one_coding_worker_per_lens():
    specs = run.build_specs(["reliability", "coverage"], target="utils/x.py", model="m")
    assert [s.branch_id for s in specs] == ["reliability", "coverage"]
    for s in specs:
        assert s.worker_script.endswith("coding_worker.py")
        assert s.network == NetworkMode.LLM_UDS
        assert s.worker_env["WORKER_TARGET"] == "utils/x.py"
        assert s.worker_env["WORKER_MODEL"] == "m"
    # each lens points at its own goals file
    assert specs[0].worker_env["WORKER_GOALS"] == "agent_branch/goals/reliability.md"
    assert specs[1].worker_env["WORKER_GOALS"] == "agent_branch/goals/coverage.md"


def test_build_template_maps_args():
    t = run.build_template(objective="do x", allowed=["utils/x.py"],
                           proofs=["tests/proof_x.py"], max_diff_lines=120,
                           wallclock=200, token_budget=50_000)
    assert t["objective"] == "do x"
    assert t["allowed_paths"] == ["utils/x.py"]
    assert t["required_tests"] == ["tests/proof_x.py"]   # proof gates correctness
    assert t["max_diff_lines"] == 120 and t["wallclock_seconds"] == 200
    assert t["token_budget"] == 50_000


def test_build_proxies_one_per_branch_with_host_and_budget(tmp_path):
    specs = run.build_specs(["reliability", "capability"], target="utils/x.py", model="m")
    proxies = run.build_proxies(specs, upstream="https://openrouter.ai/api",
                                key="sk-test", runs_root=tmp_path, token_budget=42)
    assert set(proxies) == {"reliability", "capability"}
    px = proxies["reliability"]
    assert px.upstream == "https://openrouter.ai/api"
    assert px.api_key == "sk-test"
    assert "openrouter.ai" in px.allowed_hosts      # host derived from upstream
    assert px.token_budget == 42
    assert px.uds_path == str(tmp_path / "llm-reliability.sock")
    # distinct sockets per branch -> independent metering
    assert proxies["capability"].uds_path != px.uds_path


def test_api_key_reads_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run.api_key() is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert run.api_key() == "sk-openai"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    assert run.api_key() == "sk-or"                 # OpenRouter preferred


def test_main_without_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = run.main(["--objective", "o", "--target", "utils/x.py",
                   "--allowed", "utils/x.py", "--proof", "tests/proof_x.py"])
    assert rc == 2                                   # no podman / proxy touched
