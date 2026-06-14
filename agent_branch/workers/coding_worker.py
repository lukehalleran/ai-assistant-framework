#!/usr/bin/env python3
# agent_branch/workers/coding_worker.py
"""
General LLM coding worker — produces a real, objective-driven change.

Unlike the scripted demo workers (which bake in "implement subtract"), this worker
takes an ARBITRARY objective from the manifest (WORKER_OBJECTIVE) and a per-agent
LENS (a goals file), gathers code context (the target file + extra context files +
a bounded repo map), prompts the LLM over the single mediated UDS proxy, and writes
the result. v1 is one-shot, single target file (option a): rich-context in, one
file out. Multi-file editing is a later step.

The agent's mandate is an INPUT it cannot change: the objective is the manifest's
(supervisor-authorized), and the goals files live under agent_branch/ — a worker
diff that touches them is killed at the static gate (high-risk), so an agent can
never rewrite the goal it is being held to.

No API key in the worker (the proxy injects it); no IP egress (only the UDS).
Stdlib only — runs in the bare python:3.11-slim worker image.

Env:
- WORKER_REPO (/work/repo), WORKER_ARTIFACTS (/work/artifacts), WORKER_LLM_SOCK (/work/llm.sock)
- WORKER_OBJECTIVE  — what to build (wired from the manifest by the supervisor)
- WORKER_TARGET     — the file to write (default sandbox/calc.py)
- WORKER_GOALS      — per-agent goals file, repo-relative (e.g. agent_branch/goals/reliability.md)
- WORKER_SHARED_GOALS — shared principles file (default agent_branch/goals/_shared.md)
- WORKER_CONTEXT    — comma-separated extra files to READ for context (optional)
- WORKER_MODEL      — model id for the proxy
"""

import http.client
import json
import os
import socket
import sys
from pathlib import Path

REPO = os.environ.get("WORKER_REPO", "/work/repo")
ARTIFACTS = os.environ.get("WORKER_ARTIFACTS", "/work/artifacts")
LLM_SOCK = os.environ.get("WORKER_LLM_SOCK", "/work/llm.sock")
OBJECTIVE = os.environ.get("WORKER_OBJECTIVE", "")
TARGET = os.environ.get("WORKER_TARGET", "sandbox/calc.py")
GOALS = os.environ.get("WORKER_GOALS", "")
SHARED_GOALS = os.environ.get("WORKER_SHARED_GOALS", "agent_branch/goals/_shared.md")
CONTEXT = os.environ.get("WORKER_CONTEXT", "")
MODEL = os.environ.get("WORKER_MODEL", "anthropic/claude-haiku-4.5")

# bounded repo map — never traverse heavy/irrelevant trees, cap the listing
_SKIP_DIRS = {".git", "__pycache__", "data", "node_modules", "venv", ".venv",
              "dist", "build", ".egg-info", ".mypy_cache", ".pytest_cache"}
_REPO_MAP_CAP = 200


# --- pure helpers (host-testable) -------------------------------------------

def load_goals(repo: str, goals_path: str, shared_path: str) -> str:
    """Concatenate the shared principles + this agent's lens. Missing files are
    skipped (so a worker with no lens still gets the shared principles)."""
    parts = []
    for label, rel in (("SHARED PRINCIPLES", shared_path), ("YOUR LENS", goals_path)):
        if not rel:
            continue
        p = Path(repo, rel)
        try:
            text = p.read_text(encoding="utf-8").strip()
        except (OSError, IOError):
            continue
        if text:
            parts.append(f"## {label}\n{text}")
    return "\n\n".join(parts)


def build_repo_map(repo: str, cap: int = _REPO_MAP_CAP) -> str:
    """A compact, bounded listing of the repo's files (skipping heavy/irrelevant
    dirs) so the model has structural context without being flooded."""
    root = Path(repo)
    rels = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".egg-info")]
        for f in filenames:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            rels.append(rel)
            if len(rels) >= cap:
                rels.sort()
                return "\n".join(rels) + f"\n... (truncated at {cap} files)"
    rels.sort()
    return "\n".join(rels)


def read_context_files(repo: str, paths, cap_chars: int = 6000) -> dict:
    """Read each context file (capped), skipping any that don't exist."""
    out = {}
    for rel in paths:
        rel = rel.strip()
        if not rel:
            continue
        p = Path(repo, rel)
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, IOError):
            continue
        out[rel] = text if len(text) <= cap_chars else text[:cap_chars] + "\n# ... (truncated)"
    return out


def system_prompt() -> str:
    return ("You are a careful senior engineer working on ONE bounded change inside an "
            "isolated sandbox. Follow the shared principles and your lens exactly. "
            "Reply with ONLY the complete new content of the target file — no prose, "
            "no explanation, no markdown fences.")


def build_user_prompt(*, objective: str, goals: str, target: str, target_content: str,
                      context_files: dict, repo_map: str) -> str:
    sections = []
    if goals:
        sections.append(goals)
    sections.append(f"## OBJECTIVE\n{objective}")
    sections.append("## SCOPE\nYou may write ONLY this file: "
                    f"`{target}`. Do not emit changes to any other path.")
    if repo_map:
        sections.append(f"## REPO MAP (for orientation)\n{repo_map}")
    if context_files:
        ctx = "\n\n".join(f"### {path}\n```\n{body}\n```" for path, body in context_files.items())
        sections.append(f"## CONTEXT FILES (read-only, for reference)\n{ctx}")
    sections.append(f"## TARGET FILE — `{target}` (current content)\n```\n{target_content}\n```")
    sections.append("Now output the COMPLETE new content of the target file.")
    return "\n\n".join(sections)


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip() + "\n"


# --- LLM call over the UDS proxy --------------------------------------------

class _UDSConnection(http.client.HTTPConnection):
    def __init__(self, uds_path: str, timeout: float = 120.0):
        super().__init__("localhost", timeout=timeout)
        self._uds_path = uds_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._uds_path)
        self.sock = sock


def ask_llm(system: str, user: str, *, model: str = MODEL, sock: str = LLM_SOCK) -> str:
    conn = _UDSConnection(sock)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 4000,
        "temperature": 0,
    })
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={"Content-Type": "application/json", "Host": "llm"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"proxy returned {resp.status}: {data.get('error')}")
    return data["choices"][0]["message"]["content"]


def main() -> int:
    Path(ARTIFACTS).mkdir(parents=True, exist_ok=True)
    report = {"objective": OBJECTIVE, "target": TARGET, "goals": GOALS}
    try:
        if not OBJECTIVE:
            raise RuntimeError("no WORKER_OBJECTIVE provided")
        target = Path(REPO, TARGET)
        target_content = target.read_text(encoding="utf-8") if target.exists() else ""
        goals = load_goals(REPO, GOALS, SHARED_GOALS)
        context_files = read_context_files(REPO, CONTEXT.split(",") if CONTEXT else [])
        repo_map = build_repo_map(REPO)

        user = build_user_prompt(
            objective=OBJECTIVE, goals=goals, target=TARGET,
            target_content=target_content, context_files=context_files, repo_map=repo_map,
        )
        new_content = strip_fences(ask_llm(system_prompt(), user))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")

        report["status"] = "ok"
        Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
        print(f"coding worker applied change to {TARGET} for objective: {OBJECTIVE[:80]}")
        return 0
    except Exception as e:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {e}"
        Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
        print(f"coding worker failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
