#!/usr/bin/env python3
# agent_branch/workers/benign_llm.py
"""
Minimal LLM-driven worker — runs INSIDE the isolated container (happy path).

It makes ONE bounded, allowed edit: ask the LLM (via the supervisor's UDS proxy)
for a one-line docstring and prepend it to the target file. This proves the
end-to-end happy path — a real LLM call over the single mediated channel, an
in-scope edit, a supervisor-computed diff that passes both gates — WITHOUT
reusing the 20-tool agentic controller (deferred to M2).

The worker never holds an API key (the proxy injects it) and has no IP egress;
its only channel out is the Unix-domain socket at ``$WORKER_LLM_SOCK``.

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
MODEL = os.environ.get("WORKER_MODEL", "anthropic/claude-haiku-4.5")


class _UDSConnection(http.client.HTTPConnection):
    """HTTP/1.1 over a Unix-domain socket (the worker's only egress)."""

    def __init__(self, uds_path: str, timeout: float = 60.0):
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
            {"role": "system", "content": "You write a single short Python docstring. Reply with only the docstring line."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": 0,
    })
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={"Content-Type": "application/json", "Host": "llm"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"proxy returned {resp.status}: {data.get('error')}")
    return data["choices"][0]["message"]["content"].strip()


def main() -> int:
    Path(ARTIFACTS).mkdir(parents=True, exist_ok=True)
    target = Path(REPO, TARGET)
    report = {"objective": "prepend a docstring via the LLM", "target": TARGET}
    try:
        original = target.read_text(encoding="utf-8")
        docstring = ask_llm(f"Write a one-line docstring for this code:\n{original}")
        if not (docstring.startswith('"""') or docstring.startswith("'''")):
            docstring = f'"""{docstring}"""'
        target.write_text(docstring + "\n" + original, encoding="utf-8")
        report["status"] = "ok"
        report["docstring"] = docstring
        print("benign LLM worker applied edit")
        Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
        return 0
    except Exception as e:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {e}"
        Path(ARTIFACTS, "report.json").write_text(json.dumps(report), encoding="utf-8")
        print(f"benign LLM worker failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
