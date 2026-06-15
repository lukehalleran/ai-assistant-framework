# tests/agent_branch/test_redteam_hardening.py
"""Red-team hardening tests for the static gate and the LLM proxy.

Each test corresponds to a way an adversarial branch could (a) hide work from
the diff parser to bypass size/scope checks or deflate its own score, (b) forge
a passing proof by editing a pytest-config file the gate didn't watch, or (c)
burn the LLM budget unmetered. All pure — no podman.
"""

import shlex

from agent_branch.eval_sandboxed import (
    _branch_test_cmd,
    default_test_cmd,
    evaluate_in_sandbox,
)
from agent_branch.eval_static import GateFlag, parse_diff, run_static_gate
from agent_branch.llm_proxy import LLMProxy
from agent_branch.manifest import BranchManifest


def _m(**over):
    base = dict(branch_id="g", objective="o", allowed_paths=["sandbox/**"],
                max_diff_lines=50, required_tests=[])
    base.update(over)
    return BranchManifest.issue(**base)


# -- parse_diff: a content line must never be mistaken for a file header -----

def test_added_content_line_looking_like_a_header_is_counted_not_a_path():
    # The worker adds a line whose CONTENT is '++ memory/evil'; in the unified
    # diff that renders as '+++ memory/evil'. The old parser read it as a +++
    # file header: it was NOT counted and 'memory/evil' was recorded as touched.
    d = ("diff --git a/sandbox/c.py b/sandbox/c.py\n"
         "--- a/sandbox/c.py\n+++ b/sandbox/c.py\n"
         "@@ -0,0 +1,3 @@\n+real = 1\n+++ memory/evil\n+another = 2\n")
    touched, added, removed, is_empty, ws = parse_diff(d)
    assert touched == ["sandbox/c.py"]      # NOT 'memory/evil'
    assert added == 3                        # the masquerading line is real work
    assert not is_empty


def test_removed_content_line_looking_like_a_header_is_counted():
    # A removed line whose CONTENT is '-- old/path' renders as '--- old/path',
    # which the old parser mistook for a '---' file header.
    d = ("diff --git a/sandbox/c.py b/sandbox/c.py\n"
         "--- a/sandbox/c.py\n+++ b/sandbox/c.py\n"
         "@@ -1,2 +1,1 @@\n-keep = 1\n--- old/path comment\n")
    touched, added, removed, is_empty, ws = parse_diff(d)
    assert touched == ["sandbox/c.py"]
    assert removed == 2


def test_header_injection_cannot_bypass_diff_size_limit():
    # 60 added lines, each disguised as a '+++' header. They must still count
    # toward max_diff_lines and must not appear as out-of-scope touched paths.
    body = "\n".join(f"+++ memory/x{i}" for i in range(60))
    d = ("diff --git a/sandbox/c.py b/sandbox/c.py\n"
         "--- a/sandbox/c.py\n+++ b/sandbox/c.py\n"
         f"@@ -0,0 +1,60 @@\n{body}\n")
    r = run_static_gate(d, _m(max_diff_lines=10))
    assert r.killed and GateFlag.DIFF_TOO_LARGE in r.flags
    assert r.touched_paths == ["sandbox/c.py"]
    assert all("memory/" not in p for p in r.touched_paths)


def test_legit_rename_still_parses_both_sides():
    # regression guard: the state machine must not break header parsing
    d = ("diff --git a/sandbox/a.py b/sandbox/b.py\nrename from sandbox/a.py\n"
         "rename to sandbox/b.py\n")
    touched, *_ = parse_diff(d)
    assert "sandbox/a.py" in touched and "sandbox/b.py" in touched


# -- self-redirect: a worker can't rewrite its own mandate --------------------

def test_editing_agent_goals_file_is_killed_high_risk():
    # Goals files live under agent_branch/ (forbidden + high-risk), so any diff
    # touching one is killed at the static gate, high-risk, before evaluation —
    # a worker can never rewrite the mandate it's being held to.
    for path in ["agent_branch/goals/reliability.md", "agent_branch/goals/_shared.md"]:
        d = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
             "@@ -1 +1 @@\n-old\n+evil\n")
        r = run_static_gate(d, _m())
        assert r.killed and r.high_risk and GateFlag.SAFETY_TOUCHED in r.flags, path


# -- pytest-config tamper: every file pytest reads config from is watched -----

def test_pytest_config_files_are_high_risk_config_tamper():
    # pytest reads config from all of these; editing one can forge a passing proof
    # (e.g. addopts=--collect-only). .gitattributes can also hide diff content.
    for path in ["pyproject.toml", "setup.cfg", "tox.ini", ".gitattributes",
                 "deep/nested/pyproject.toml"]:
        d = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
             "@@ -0,0 +1 @@\n+x = 1\n")
        r = run_static_gate(d, _m())
        assert r.killed and r.high_risk and GateFlag.CONFIG_TAMPER in r.flags, path


def test_pytest_config_files_in_default_forbidden_list():
    fb = BranchManifest.issue(branch_id="b", objective="o").forbidden_paths
    for path in ["pyproject.toml", "setup.cfg", "tox.ini", ".gitattributes"]:
        assert path in fb, path


# -- LLM proxy metering: the prompt is always metered ------------------------

def _proxy(**over):
    over.setdefault("token_budget", 1_000_000)
    return LLMProxy("/tmp/agentbranch-test-never.sock", "http://127.0.0.1:1", **over)


def test_metering_counts_prompt_when_upstream_omits_usage():
    # JSON response with NO usage field — old code metered this as 0 tokens.
    p = _proxy()
    p.record_usage(b"x" * 4000, b'{"choices":[{"message":{"content":"ok"}}]}', ok=True)
    assert p.tokens_spent >= 1000  # ~4000 bytes of prompt / 4
    assert p.calls == 1


def test_metering_counts_prompt_for_streaming_response():
    # SSE stream is not a single JSON doc — old code metered only response bytes,
    # so a huge prompt with a tiny stream slipped through nearly free.
    p = _proxy()
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    p.record_usage(b"q" * 8000, sse, ok=True)
    assert p.tokens_spent >= 2000  # the 8000-byte prompt was metered


def test_metering_counts_failed_calls():
    # Upstream error — old code never called record_usage, so failures were free.
    p = _proxy()
    p.record_usage(b"z" * 4000, b"", ok=False)
    assert p.tokens_spent >= 1000
    assert p.calls == 1


def test_metering_trusts_upstream_total_tokens_when_present():
    # When upstream reports an authoritative total, use it as-is (it already
    # includes the prompt) — keeps the existing socket tests' numbers exact.
    p = _proxy()
    p.record_usage(b"x" * 4000, b'{"usage":{"total_tokens":123}}', ok=True)
    assert p.tokens_spent == 123


def test_budget_overshoot_metering_accumulates():
    p = _proxy(token_budget=500)
    for _ in range(3):
        p.record_usage(b"x" * 4000, b"no-usage", ok=True)
    assert p.tokens_spent >= 3000
    assert p.budget_exceeded() is True


def test_proxy_path_allowlist_rejects_nonchat_endpoints():
    # A worker must not reach arbitrary provider endpoints (key mgmt, uploads,
    # differently-priced routes) with the supervisor's injected key.
    p = _proxy()
    assert p.path_allowed("/v1/chat/completions")
    assert p.path_allowed("/chat/completions")
    assert not p.path_allowed("/v1/admin/api-keys")
    assert not p.path_allowed("/v1/files")
    assert not p.path_allowed("/v1/chat/completions-evil")  # no prefix smuggling


# -- sandbox eval: no free passes, no pytest-in-a-bare-image ------------------

def test_no_proof_tests_is_killed_not_a_free_pass():
    # A diff that cleared the static gate but has NO supervisor proof test is
    # UNPROVABLE — it must fail closed, not trivially pass. Worker-added tests are
    # evidence (D3), never proof, so they can't rescue it. Pure: kills before podman.
    res = evaluate_in_sandbox("/nonexistent/repo", "diff", [],
                              branch_test_files=["sandbox/test_mine.py"])
    assert res.killed and not res.passed
    assert "unprovable" in res.reason.lower()


def test_default_test_cmd_is_stdlib_not_pytest():
    # The bare worker image has no pytest; the default proof command must run the
    # supervisor checks as plain stdlib scripts so a default manifest isn't killed
    # for "No module named pytest" instead of on its actual merits.
    cmd = default_test_cmd(["check_calc.py", "check_m2.py"])
    joined = " ".join(cmd)
    assert "pytest" not in joined
    assert "python check_calc.py" in joined and "python check_m2.py" in joined


def test_no_proof_killed_even_with_explicit_test_cmd():
    # Supplying a custom test_cmd must NOT rescue an empty proof set — the
    # fail-closed check fires first (before podman), so an unprovable branch can't
    # be certified by a worker-chosen command.
    res = evaluate_in_sandbox("/nonexistent/repo", "diff", [],
                              test_cmd=["python", "-c", "print('ok')"])
    assert res.killed and not res.passed
    assert "unprovable" in res.reason.lower()


def test_default_test_cmd_shell_quotes_filenames():
    # required_tests names flow into an `sh -c` string; a name with shell
    # metacharacters must be shlex-quoted so it can't break out of the command.
    nasty = "sandbox/x.py; touch /tmp/pwn"
    cmd = default_test_cmd([nasty])
    assert cmd[:2] == ["sh", "-c"]
    assert cmd[2] == f"python {shlex.quote(nasty)}"
    # the metacharacter survives only INSIDE the quoted arg, never as a bare command
    assert "; touch" not in cmd[2].replace(shlex.quote(nasty), "")


def test_branch_test_cmd_shell_quotes_filenames():
    # Worker-added test filenames are even less trusted — same quoting guarantee.
    files = ["sandbox/a b.py", "sandbox/$(evil).py"]
    cmd = _branch_test_cmd(files)
    assert cmd[:2] == ["sh", "-c"]
    assert cmd[2] == " && ".join(f"python {shlex.quote(f)}" for f in files)


# -- parse_diff / static gate: more masking + smuggling vectors --------------

def test_rename_into_forbidden_path_is_caught():
    # A rename whose destination is a config/threshold file must be flagged — the
    # forbidden side can't hide behind an allowed source.
    d = ("diff --git a/sandbox/x.py b/config/evil.yaml\n"
         "rename from sandbox/x.py\nrename to config/evil.yaml\n")
    r = run_static_gate(d, _m())
    assert r.killed and r.high_risk and GateFlag.CONFIG_TAMPER in r.flags
    assert "config/evil.yaml" in r.touched_paths


def test_new_file_devnull_header_is_not_recorded_as_a_path():
    # A new-file diff has a `--- /dev/null` header; /dev/null must never appear as
    # a touched path, and the added lines must count.
    d = ("diff --git a/sandbox/new.py b/sandbox/new.py\n"
         "new file mode 100644\n--- /dev/null\n+++ b/sandbox/new.py\n"
         "@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n")
    touched, added, removed, is_empty, ws = parse_diff(d)
    assert touched == ["sandbox/new.py"]
    assert "/dev/null" not in touched
    assert added == 2 and not is_empty


def test_all_violations_collected_not_short_circuited():
    # A diff touching BOTH a safety path and a config file must report BOTH flags —
    # the gate collects every violation so the Trial Report is complete.
    d = ("diff --git a/utils/python_fs_guard.py b/utils/python_fs_guard.py\n"
         "--- a/utils/python_fs_guard.py\n+++ b/utils/python_fs_guard.py\n"
         "@@ -1 +1 @@\n-x\n+y\n"
         "diff --git a/config/app_config.py b/config/app_config.py\n"
         "--- a/config/app_config.py\n+++ b/config/app_config.py\n"
         "@@ -1 +1 @@\n-a\n+b\n")
    r = run_static_gate(d, _m())
    assert r.killed and r.high_risk
    assert GateFlag.SAFETY_TOUCHED in r.flags   # python_fs_guard.py
    assert GateFlag.CONFIG_TAMPER in r.flags    # config/app_config.py
    assert {"utils/python_fs_guard.py", "config/app_config.py"} <= set(r.touched_paths)


def test_fake_hunk_header_in_content_does_not_hide_following_lines():
    # A worker writes content lines that look like a hunk header ('+@@ ...') to try
    # to reset the parser and uncount the lines after it. Content lines carry a '+'
    # prefix so they never match the @@ hunk-header regex — all stay counted.
    body = "+@@ -0,0 +1,99 @@\n" + "\n".join(f"+evil_{i} = {i}" for i in range(40))
    d = ("diff --git a/sandbox/c.py b/sandbox/c.py\n"
         "--- a/sandbox/c.py\n+++ b/sandbox/c.py\n"
         f"@@ -0,0 +1,41 @@\n{body}\n")
    touched, added, removed, is_empty, ws = parse_diff(d)
    assert touched == ["sandbox/c.py"]
    assert added == 41   # the fake '@@' line + all 40 lines after it counted
    r = run_static_gate(d, _m(max_diff_lines=10))
    assert r.killed and GateFlag.DIFF_TOO_LARGE in r.flags
