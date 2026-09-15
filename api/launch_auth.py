"""Loopback Host / same-origin / launch-token authorization (F01, G06-T02).

One pure-ASGI middleware wraps the whole app (api/app.py adds it last, so it
wraps outside CORSMiddleware, before any router). It rejects without awaiting
`receive()` (body never read) and never wraps `receive`/`send`, so an
accepted request — including the /api/chat SSE stream — is unbuffered.

Host must be a loopback literal (127.0.0.1/localhost/::1) or an
owner-configured trusted hostname/IP (A01b: `api.host` + `api.allowed_hosts`,
exact-match only — no suffix/wildcard/pattern matching; see
`normalize_trusted_hostnames`/`is_allowed_host`); its port, when present, is
checked against `scope["server"][1]` — the port this connection actually
landed on (uvicorn: the bound socket; tests: from the request URL) — never a
hardcoded config value. Origin is required and must be an allowed
loopback-or-trusted authority for every state-changing method: the app's own
origin plus configured API_CORS_ORIGINS (the Vite dev-server origin).

Classification is computed fresh per request against a LIVE route table
(`route_source.routes`, read at request time — mount_admin_and_frontend adds
routes after create_app returns), not just a path prefix, so a new route
defaults to protected:
  1. "/admin" or "/admin/*" -> admin policy (unchanged).
  2. Explicit public allowlist ("/health", "/") on GET/HEAD -> Host only.
  3. "/api/*", OR any path Match'd (FULL or PARTIAL — a route exists there
     even if this method is wrong) by a non-Mount route -> protected: token
     required for every method including OPTIONS (so a CORS preflight
     cannot get this far), Origin required for a mutating method. Also
     protects FastAPI's own /docs, /redoc, /openapi.json — intended.
  4. Everything else (the StaticFiles catch-all, or a genuine 404): GET/HEAD
     public (Host only); a mutating method still needs token + Origin.
"""

import hmac
import ipaddress
import json
import re
import secrets
from urllib.parse import urlsplit

from starlette.routing import Match, Mount

LAUNCH_TOKEN_HEADER = "X-Daemon-Launch-Token"
_LAUNCH_TOKEN_HEADER_ASGI = LAUNCH_TOKEN_HEADER.lower()
LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_TOKEN_ENTROPY_BYTES = 32  # >= 256 bits
_PUBLIC_GET_PATHS = {"/health", "/"}


def generate_launch_secret() -> str:
    """A fresh high-entropy per-launch secret; never persisted or logged."""
    return secrets.token_urlsafe(_TOKEN_ENTROPY_BYTES)


def _split_host_port(value: str):
    # str.isdigit() accepts non-ASCII digits ("²") that int() then rejects;
    # require isascii() too so a hostile Host header can never raise here.
    value = (value or "").strip()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return value.lower(), None
        host, rest = value[1:end].lower(), value[end + 1:]
        port_s = rest[1:] if rest[:1] == ":" else ""
        return host, int(port_s) if port_s.isascii() and port_s.isdecimal() else None
    if ":" in value:
        host, _, port_s = value.rpartition(":")
        if port_s.isascii() and port_s.isdecimal():
            return host.lower(), int(port_s)
        return value.lower(), None
    return value.lower(), None


_HOSTNAME_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def _ip_literal_or_none(host: str):
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def _valid_hostname(value: str) -> bool:
    if not value or len(value) > 253:
        return False
    labels = value.split(".")
    return all(1 <= len(lbl) <= 63 and _HOSTNAME_LABEL_RE.match(lbl) for lbl in labels)


def _normalize_trusted_entry(value):
    """One api_host/allowed_hosts entry -> a trusted hostname string, or None
    to reject it (see normalize_trusted_hostnames for the full contract)."""
    v = ("" if value is None else str(value)).strip()
    if v.endswith("."):
        v = v[:-1]
    v = v.lower()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    if not v or any(c in v for c in ("*", "/", "@")) or any(c.isspace() for c in v) or "://" in v:
        return None
    ip = _ip_literal_or_none(v)
    if ip is not None:
        return None if ipaddress.ip_address(ip).is_unspecified else ip
    if ":" in v:  # host:port shape that is not an IPv6 literal — ambiguous, reject
        return None
    return v if _valid_hostname(v) else None


def normalize_trusted_hostnames(api_host, allowed_hosts) -> tuple:
    """Pure normalizer for the owner-configured trusted-Host allowance
    (A01b: `api.host` + `api.allowed_hosts`). Exact-match only — no suffix,
    wildcard, or pattern matching. Loopback names need no entry here (they
    are always allowed by is_allowed_host). Returns (trusted, rejected_count);
    a rejected entry is skipped and counted, never trusted."""
    trusted = set()
    rejected = 0
    for raw in (api_host, *(allowed_hosts or ())):
        if raw is None:
            continue
        norm = _normalize_trusted_entry(raw)
        if norm is None:
            rejected += 1
        else:
            trusted.add(norm)
    return frozenset(trusted), rejected


def is_allowed_host(host_header, server_port, trusted_hostnames: frozenset = frozenset()) -> bool:
    if not host_header:
        return False
    host, port = _split_host_port(host_header)
    host = _ip_literal_or_none(host) or host
    if host not in LOOPBACK_HOSTNAMES and host not in trusted_hostnames:
        return False
    return port is None or server_port is None or port == server_port


def is_loopback_host(host_header, server_port) -> bool:
    """Unchanged behaviour: loopback only, no trusted-host allowance."""
    return is_allowed_host(host_header, server_port, frozenset())


def _origin_authority(origin: str):
    try:
        parts = urlsplit(origin)
        if parts.scheme != "http" or not parts.hostname:
            return None
        port = parts.port  # raises ValueError for an out-of-range port (e.g. 99999)
    except ValueError:
        return None
    host = parts.hostname.lower()
    return f"http://{host}:{port}" if port else f"http://{host}"


def build_allowed_origins(cors_origins, server_port, trusted_hostnames=()) -> set:
    allowed = {_origin_authority(o) for o in (cors_origins or ())}
    allowed.discard(None)
    if server_port is not None:
        allowed.update(f"http://{h}:{server_port}" for h in LOOPBACK_HOSTNAMES)
        allowed.update(f"http://{h}:{server_port}" for h in (trusted_hostnames or ()))
    return allowed


def is_allowed_origin(origin_header, allowed_origins: set) -> bool:
    if not origin_header or origin_header == "null":
        return False
    norm = _origin_authority(origin_header)
    return norm is not None and norm in allowed_origins


def _is_admin_path(path: str) -> bool:
    return path == "/admin" or path.startswith("/admin/")


def classify_request(scope, route_source) -> str:
    """See the module docstring for the 4-step order. `route_source` is the
    live FastAPI app (or None, e.g. the isolated websocket/pure-unit tests);
    its `.routes` is read fresh on every call."""
    path = scope.get("path", "") or ""
    method = scope.get("method", "GET").upper()

    if _is_admin_path(path):
        return "admin"
    if method in ("GET", "HEAD") and path in _PUBLIC_GET_PATHS:
        return "public"
    if path.startswith("/api/"):
        return "protected"
    for route in getattr(route_source, "routes", ()) or ():
        if isinstance(route, Mount):
            continue  # the StaticFiles catch-all: never itself "a route that exists"
        match, _child_scope = route.matches(scope)
        if match != Match.NONE:
            return "protected"
    return "default"


async def _reject(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": body})


class LaunchAuthMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware); covers "http"/"websocket", passes
    "lifespan" straight through."""

    def __init__(self, app, *, secret: str, cors_origins=(), admin_packaged: bool = False,
                 route_source=None, trusted_hostnames=frozenset()):
        self.app = app
        # Compared as bytes: compare_digest raises TypeError on non-ASCII str,
        # and header values decode as latin-1, so a hostile byte would 500.
        self._secret = secret.encode("utf-8")
        self._cors_origins = tuple(cors_origins or ())
        self._admin_packaged = admin_packaged
        self._route_source = route_source  # the FastAPI app; NOT `app` above (the wrapped ASGI callable)
        # A01b: owner-configured trusted-Host allowance (api.host + api.allowed_hosts),
        # exact-match only. Loopback is always allowed regardless of this set.
        self._trusted_hostnames = frozenset(trusted_hostnames or ())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""
        if _is_admin_path(path) and self._admin_packaged:
            await _reject(send, 404, "not found")  # never reveal the unauthenticated Gradio surface
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        server_port = (scope.get("server") or (None, None))[1]
        if not is_allowed_host(headers.get("host"), server_port, self._trusted_hostnames):
            await _reject(send, 400, "invalid host")
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4403})  # no ws route; fail closed
            return

        method = scope.get("method", "GET").upper()
        route_class = classify_request(scope, self._route_source)

        if route_class == "protected" or (route_class == "default" and method in _MUTATING_METHODS):
            token = headers.get(_LAUNCH_TOKEN_HEADER_ASGI)
            if not (token and hmac.compare_digest(token.encode("latin-1"), self._secret)):
                await _reject(send, 401, "unauthorized")
                return

        if route_class in ("protected", "default", "admin") and method in _MUTATING_METHODS:
            allowed = build_allowed_origins(self._cors_origins, server_port, self._trusted_hostnames)
            if not is_allowed_origin(headers.get("origin"), allowed):
                await _reject(send, 403, "origin not allowed")
                return

        await self.app(scope, receive, send)
