# agent_branch/llm_proxy.py
"""
LLMProxy — the single supervisor-mediated channel out of an LLM worker.

Module Contract
- Purpose: give the LLM worker EXACTLY ONE egress path — a Unix-domain socket to
  this proxy — while the container keeps ``--network=none`` (no IP egress). The
  proxy is the boundary's one deliberate hole, so it is deliberately narrow.
- Guarantees:
  - Reverse proxy to ONE fixed upstream. The worker cannot choose the host; any
    request whose ``Host:``/absolute-URI names a non-allowlisted host → 403
    (default-deny, so it is NOT an open relay — red-team #10). The PATH is pinned
    too (``allowed_request_paths``, default chat-completions/messages only) so the
    worker can't reach key-management/upload/differently-priced endpoints with the
    injected key.
  - The real API key lives ONLY here. The proxy injects ``Authorization`` on the
    way out; the worker never holds a credential. The socket's directory is 0700
    (supervisor-only) so no other host process can reach the UDS, while the socket
    stays world-connectable for the container's unmapped userns uid.
  - Token metering: the PROMPT (request body) is metered on every call — even
    failed, streaming, or usage-less ones — and upstream ``usage.total_tokens``
    is used when reported (it already covers the prompt). Cumulative spend is
    summed; once it reaches ``token_budget`` further calls are refused (429) and
    ``budget_exceeded`` flips so the supervisor can kill the worker. The worker
    can't under-report spend it never sees, and can't dodge the meter by asking
    for a stream or triggering an upstream error.
  - Request bodies are size-capped (``max_body_bytes``) so a worker can't OOM the
    supervisor with a multi-GB ``Content-Length``.
  - Every call is logged to JSONL (untrusted bodies are not executed).
- Side effects: binds a UDS, opens upstream TCP connections, appends to a log.
- Stdlib-only (runs supervisor-side); upstream is configurable so tests point it
  at a local stub instead of a real API.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn, UnixStreamServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from utils.logging_utils import get_logger

logger = get_logger("agent_branch.llm_proxy")

_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authorization", "te", "trailer",
               "transfer-encoding", "upgrade", "host", "content-length"}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence default stderr logging
        return

    def _reject(self, code: int, reason: str) -> None:
        self._respond(code, json.dumps({"error": reason}).encode())

    def _respond(self, code: int, body: bytes) -> None:
        # One request per connection: workers make one-shot calls, and closing
        # avoids a keep-alive read on a hung-up socket (broken-pipe noise).
        self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        proxy: "LLMProxy" = self.server.proxy  # type: ignore[attr-defined]

        # Default-deny only the dangerous case: a FORWARD-proxy request (absolute
        # request URI) aimed at a non-allowlisted host. A normal reverse-proxy
        # request (origin-form path) can't choose the upstream — we always forward
        # to the one fixed LLM endpoint — so its Host header is irrelevant.
        target_host = self._forward_proxy_host()
        if target_host is not None and target_host not in proxy.allowed_hosts:
            proxy._log({"event": "rejected_host", "host": target_host})
            return self._reject(403, f"host {target_host!r} not allowlisted")

        # Pin the path too: the worker may hit only the chat endpoint, never other
        # (e.g. key-management, upload, differently-priced) routes on the upstream
        # with the supervisor's injected key.
        req_path = self._request_path()
        if not proxy.path_allowed(req_path):
            proxy._log({"event": "rejected_path", "path": req_path})
            return self._reject(403, f"path {req_path!r} not allowlisted")

        if proxy.budget_exceeded():
            proxy._log({"event": "rejected_budget", "tokens": proxy.tokens_spent})
            return self._reject(429, "token budget exceeded")

        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > proxy.max_body_bytes:
            proxy._log({"event": "rejected_body_too_large", "length": length})
            return self._reject(413, f"request body exceeds {proxy.max_body_bytes} bytes")
        body = self.rfile.read(length) if length else b""
        try:
            status, resp = proxy.forward(self.path, dict(self.headers), body)
        except urllib.error.HTTPError as e:
            # Meter the prompt even on upstream failure — a worker must not get
            # free attempts by spamming requests that error out.
            proxy.record_usage(body, b"", ok=False)
            return self._reject(e.code, f"upstream http {e.code}")
        except Exception as e:  # noqa: BLE001 — surface as 502, never crash the proxy
            proxy.record_usage(body, b"", ok=False)
            return self._reject(502, f"upstream error: {e}")

        proxy.record_usage(body, resp, ok=True)
        self._respond(status, resp)

    def _forward_proxy_host(self) -> Optional[str]:
        """The host of an absolute (forward-proxy) request URI, else None.
        Origin-form requests return None — they go to the fixed upstream."""
        if self.path.startswith(("http://", "https://")):
            return urlparse(self.path).hostname
        return None

    def _request_path(self) -> str:
        """The path component of the request (absolute URI or origin-form),
        sans query string — what gets matched against the path allowlist."""
        p = self.path
        if p.startswith(("http://", "https://")):
            p = urlparse(p).path
        return p.split("?", 1)[0]


class _ProxyServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class LLMProxy:
    def __init__(
        self,
        uds_path: str | Path,
        upstream: str,
        *,
        api_key: Optional[str] = None,
        allowed_hosts: Optional[List[str]] = None,
        allowed_request_paths: Optional[List[str]] = None,
        token_budget: int = 200_000,
        log_path: Optional[str | Path] = None,
        upstream_timeout: float = 60.0,
        max_body_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.uds_path = str(uds_path)
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.allowed_hosts = set(allowed_hosts or [urlparse(self.upstream).hostname or ""])
        # Least-privilege egress: only chat-completions style endpoints by default
        # (the workers all POST /v1/chat/completions; Anthropic uses /v1/messages).
        self.allowed_request_paths = list(allowed_request_paths or [
            "/v1/chat/completions", "/chat/completions", "/v1/messages",
        ])
        self.token_budget = token_budget
        self.log_path = Path(log_path) if log_path else None
        self.upstream_timeout = upstream_timeout
        self.max_body_bytes = max_body_bytes

        self._tokens = 0
        self._calls = 0
        self._lock = threading.Lock()
        self._server: Optional[_ProxyServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> "LLMProxy":
        parent = Path(self.uds_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        # Lock the socket's directory to the supervisor (0700) so no OTHER host
        # process can reach the UDS path and spend the budget / use the key. The
        # socket itself stays world-connectable below (the container connects as an
        # unmapped userns uid); podman bind-mounts the socket inode into the
        # container, so container access is unaffected by the directory mode.
        os.chmod(parent, 0o700)
        if os.path.exists(self.uds_path):
            os.unlink(self.uds_path)
        self._server = _ProxyServer(self.uds_path, _Handler)
        self._server.proxy = self  # type: ignore[attr-defined]
        os.chmod(self.uds_path, 0o777)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("LLM proxy listening on %s -> %s (budget=%d)",
                    self.uds_path, self.upstream, self.token_budget)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if os.path.exists(self.uds_path):
            try:
                os.unlink(self.uds_path)
            except OSError:
                pass

    def __enter__(self) -> "LLMProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- egress allowlist ----------------------------------------------------

    def path_allowed(self, path: str) -> bool:
        """True iff ``path`` exactly matches an allowed endpoint or is a child of
        one (``/p/x``), so ``/v1/chat/completions-evil`` can't smuggle past
        ``/v1/chat/completions``."""
        return any(path == p or path.startswith(p.rstrip("/") + "/")
                   for p in self.allowed_request_paths)

    # -- metering ------------------------------------------------------------

    @property
    def tokens_spent(self) -> int:
        with self._lock:
            return self._tokens

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def budget_exceeded(self) -> bool:
        return bool(self.token_budget) and self.tokens_spent >= self.token_budget

    def record_usage(self, req_bytes: bytes, resp_bytes: bytes = b"", *,
                     ok: bool = True) -> None:
        """Meter one call. The prompt (``req_bytes``) is ALWAYS counted — on
        failures, streams, and usage-less responses alike — so a worker can't get
        unmetered spend by streaming, by triggering an upstream error, or because
        the response carried no ``usage`` block. When upstream reports an
        authoritative ``usage.total_tokens`` we trust it (it already includes the
        prompt); otherwise we fall back to a ``bytes // 4`` heuristic over both
        sides."""
        req_tokens = max(1, len(req_bytes) // 4)
        used = req_tokens
        if ok and resp_bytes:
            total = 0
            try:
                data = json.loads(resp_bytes.decode("utf-8", errors="replace"))
                if isinstance(data, dict) and isinstance(data.get("usage"), dict):
                    total = int(data["usage"].get("total_tokens") or 0)
            except (json.JSONDecodeError, ValueError, AttributeError):
                total = 0
            used = total if total > 0 else req_tokens + max(1, len(resp_bytes) // 4)
        with self._lock:
            self._tokens += used
            self._calls += 1
        self._log({"event": "call", "ok": ok, "tokens": used, "cumulative": self.tokens_spent})

    # -- forwarding ----------------------------------------------------------

    def forward(self, path: str, headers: Dict[str, str], body: bytes) -> Tuple[int, bytes]:
        # Worker can't pick the host: we forward only to the fixed upstream.
        rel = path
        if rel.startswith(("http://", "https://")):
            rel = urlparse(rel).path
        url = self.upstream + (rel if rel.startswith("/") else "/" + rel)

        fwd_headers = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
        fwd_headers["Content-Type"] = "application/json"
        if self.api_key:  # key injected here — the worker never holds it
            fwd_headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=fwd_headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.upstream_timeout) as resp:
            return resp.status, resp.read()

    # -- logging -------------------------------------------------------------

    def _log(self, entry: dict) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass
