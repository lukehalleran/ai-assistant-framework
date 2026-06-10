#!/usr/bin/env python3
# agent_branch/workers/llm_strategy_worker.py
"""
Real-LLM strategy worker — the MANUAL demo driver (not used in CI).

Same bounded objective as the scripted strategy worker, but the edit is produced
by a real LLM over the single mediated UDS proxy, seeded with a strategy
directive (surgical / refactor / test-first). Use it for the convincing
end-to-end portfolio demo against a real small module; the automated reaper proof
uses the deterministic scripted workers instead.

No API key in the worker (the proxy injects it); no IP egress (only the UDS).
Stdlib only.
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
TARGET = os.environ.get("WORKER_TARGET", "sandbox/calc.py")
STRATEGY = os.environ.get("WORKER_STRATEGY", "surgical")
OBJECTIVE = os.environ.get("WORKER_OBJECTIVE", "implement subtract(a, b) returning a - b")
MODEL = os.environ.get("WORKER_MODEL", "anthropic/claude-haiku-4.5")

_STRATEGY_HINT = {
    "surgical": "Make the smallest possible change — append the new function only.",
    "refactor": "Restructure the module cleanly (e.g. a dispatch table) before adding the feature.",
    "testfirst": "First add a stdlib self-test, then implement, so the change is test-driven.",
}


class _UDSConnection(http.client.HTTPConnection):
    def __init__(self, uds_path: str, timeout: float = 90.0):
        super().__init__("localhost", timeout=timeout)
        self._uds_path = uds_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._uds_path)
        self.sock = sock


def ask_llm(prompt: str) -> str:
    conn = _UDSConnection(LLM_SOCK)
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful Python engineer. Reply with ONLY the complete new file content, no prose, no markdown fences."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 600,
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


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip() + "\n"


def main() -> int:
    Path(ARTIFACTS).mkdir(parents=True, exist_ok=True)
    target = Path(REPO, TARGET)
    report = {"strategy": STRATEGY, "objective": OBJECTIVE, "target": TARGET}
    try:
        original = target.read_text(encoding="utf-8")
        hint = _STRATEGY_HINT.get(STRATEGY, "")
        new_content = ask_llm(
            f"Objective: {OBJECTIVE}. Strategy: {STRATEGY}. {hint}\n"
            f"Keep all existing behavior working.\n\nCurrent {TARGET}:\n{original}"
        )
        target.write_text(_strip_fences(new_content), encoding="utf-8")
        report["status"] = "ok"
        Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
        print(f"llm strategy worker '{STRATEGY}' applied edit")
        return 0
    except Exception as e:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {e}"
        Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
        print(f"llm strategy worker failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
