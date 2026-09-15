"""A01b: owner-configured trusted-Host allowance for the launch-auth
middleware (Tailscale access). `normalize_trusted_hostnames`/`is_allowed_host`
add an exact-match `api.host` + `api.allowed_hosts` allowlist on top of A01's
loopback-only Host check; A01's own tests (unmodified) prove the loopback
path is unaffected. Synthetic values only: IPv4 100.64.0.10 (RFC 6598), IPv6
2001:db8::10 (RFC 3849), hostnames under .example. Drives the real ASGI app
via httpx.ASGITransport (never a live socket)."""

import pytest

from api.launch_auth import (
    LAUNCH_TOKEN_HEADER,
    is_allowed_host,
    is_loopback_host,
    normalize_trusted_hostnames,
)
from tests.unit.api_launch_auth_client import TEST_LAUNCH_SECRET
from tests.unit.api_launch_auth_client import make_test_app as create_app_with_secret
from tests.unit.helpers_orchestrator import _make_orchestrator
from tests.unit.test_api_origin_security import _full_app, _raw_client

TRUSTED_IP = "100.64.0.10"
TRUSTED_IPV6 = "2001:db8::10"
TRUSTED_HOST = "daemon-host.tailnet.example"
FOREIGN_HOST = "evil.example"


class TestNormalizeTrustedHostnames:  # pure normalizer

    def test_ip_literal_api_host_is_trusted(self):
        trusted, rejected = normalize_trusted_hostnames(TRUSTED_IP, [])
        assert trusted == frozenset({TRUSTED_IP})
        assert rejected == 0

    def test_bracketed_ipv6_normalizes_to_compressed_lowercase(self):
        trusted, rejected = normalize_trusted_hostnames("127.0.0.1", ["[2001:DB8:0:0:0:0:0:10]"])
        assert trusted == frozenset({"127.0.0.1", TRUSTED_IPV6})
        assert rejected == 0

    def test_hostname_normalizes_case_and_trailing_dot(self):
        trusted, rejected = normalize_trusted_hostnames("127.0.0.1", ["Daemon-Host.Tailnet.Example."])
        assert TRUSTED_HOST in trusted
        assert rejected == 0

    def test_rejected_entries_are_skipped_and_counted(self):
        bad = ["", "0.0.0.0", "::", "*", "*.example", "daemon-host.example:8000",
               "http://daemon-host.example", "a b", "bad_host!.example"]
        trusted, rejected = normalize_trusted_hostnames("127.0.0.1", bad)
        assert trusted == frozenset({"127.0.0.1"})  # only the loopback api_host survives
        assert rejected == len(bad)

    def test_loopback_api_host_accepted_without_error(self):
        trusted, rejected = normalize_trusted_hostnames("127.0.0.1", [])
        assert rejected == 0


class TestIsAllowedHost:  # pure Host-header check

    def _trusted(self):
        return normalize_trusted_hostnames("127.0.0.1", [TRUSTED_IP, f"[{TRUSTED_IPV6}]"])[0]

    def test_trusted_ip_matching_port_or_no_port_true(self):
        trusted = self._trusted()
        assert is_allowed_host(f"{TRUSTED_IP}:8000", 8000, trusted) is True
        assert is_allowed_host(TRUSTED_IP, None, trusted) is True

    def test_trusted_ip_wrong_port_false(self):
        trusted = self._trusted()
        assert is_allowed_host(f"{TRUSTED_IP}:9999", 8000, trusted) is False

    def test_no_suffix_or_pattern_match(self):
        trusted = self._trusted()
        assert is_allowed_host(f"{TRUSTED_IP}.evil.example", 8000, trusted) is False
        assert is_allowed_host(FOREIGN_HOST, 8000, trusted) is False

    def test_bracketed_ipv6_host_header_matches(self):
        trusted = self._trusted()
        assert is_allowed_host(f"[{TRUSTED_IPV6}]:8000", 8000, trusted) is True

    def test_loopback_still_true_with_empty_set(self):
        assert is_allowed_host("127.0.0.1", 8000, frozenset()) is True
        assert is_loopback_host("127.0.0.1", 8000) is True  # unchanged delegate, F3 case included
        assert is_loopback_host("127.0.0.1:²", 8000) is False


class TestDeployedTrustedHost:
    def _app(self, **kwargs):
        kwargs.setdefault("trusted_hosts", [TRUSTED_IP])
        return create_app_with_secret(_make_orchestrator(), start_background=False, **kwargs)

    @pytest.mark.asyncio
    async def test_health_via_trusted_host(self):
        app = self._app()
        async with _raw_client(app, base_url=f"http://{TRUSTED_IP}:8000") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_full_app_shell_via_trusted_host(self, tmp_path):
        app = _full_app(tmp_path, trusted_hosts=[TRUSTED_IP])
        async with _raw_client(app, base_url=f"http://{TRUSTED_IP}:8000") as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert 'name="daemon-launch-token"' in resp.text

    @pytest.mark.asyncio
    async def test_protected_route_needs_token_via_trusted_host(self):
        app = self._app()
        base = f"http://{TRUSTED_IP}:8000"
        async with _raw_client(app, base_url=base) as client:
            unauth = await client.get("/api/session")
        assert unauth.status_code == 401
        async with _raw_client(app, base_url=base, headers={LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET}) as client:
            auth = await client.get("/api/session")
        assert auth.status_code == 200

    @pytest.mark.asyncio
    async def test_mutating_route_origin_enforcement_via_trusted_host(self):
        app = self._app()
        base = f"http://{TRUSTED_IP}:8000"
        headers_ok = {LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET, "Origin": base}
        async with _raw_client(app, base_url=base, headers=headers_ok) as client:
            ok = await client.delete("/api/session")
        assert 200 <= ok.status_code < 300

        headers_bad = {LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET, "Origin": f"http://{FOREIGN_HOST}"}
        async with _raw_client(app, base_url=base, headers=headers_bad) as client:
            bad = await client.delete("/api/session")
        assert bad.status_code == 403

        headers_missing = {LAUNCH_TOKEN_HEADER: TEST_LAUNCH_SECRET}
        async with _raw_client(app, base_url=base, headers=headers_missing) as client:
            missing = await client.delete("/api/session")
        assert missing.status_code == 403

    @pytest.mark.asyncio
    async def test_wrong_port_and_foreign_host_still_rejected(self):
        """Controls: base_url stays on the real port (8000, so scope["server"]
        is 8000 under ASGITransport) while the Host header is overridden to a
        mismatched port, and separately to a foreign Host — both still 400
        even though the IP itself is trusted."""
        app = self._app()
        base = f"http://{TRUSTED_IP}:8000"
        async with _raw_client(app, base_url=base, headers={"Host": f"{TRUSTED_IP}:9999"}) as client:
            wrong_port = await client.get("/health")
        assert wrong_port.status_code == 400
        async with _raw_client(app, base_url=f"http://{FOREIGN_HOST}:8000") as client:
            wrong_host = await client.get("/health")
        assert wrong_host.status_code == 400


class TestConfigDrivenTrustedHost:  # trusted_hosts=None -> reads config.app_config
    """One parametrized deployed check per config scenario — same
    build-app/request/assert shape, only the config + expectation differ."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("api_host,allowed,host,expected", [
        (TRUSTED_IP, [], f"{TRUSTED_IP}:8000", 200),                 # api.host itself trusted
        ("0.0.0.0", [], "0.0.0.0:8000", 400),                        # wildcard bind never trusted...
        ("0.0.0.0", [], f"{TRUSTED_IP}:8000", 400),                  # ...nor does it trust anything else
        ("127.0.0.1", ["*"], f"{FOREIGN_HOST}:8000", 400),           # "*" entry rejected
        ("127.0.0.1", [TRUSTED_HOST], f"{TRUSTED_HOST}:8000", 200),  # trusted hostname entry
        ("127.0.0.1", [], f"{TRUSTED_IP}:8000", 400),                # default control: A01 preserved
    ])
    async def test_config_driven_host_check(self, monkeypatch, api_host, allowed, host, expected):
        monkeypatch.setattr("config.app_config.API_HOST", api_host)
        monkeypatch.setattr("config.app_config.API_ALLOWED_HOSTS", allowed)
        app = create_app_with_secret(_make_orchestrator(), start_background=False, trusted_hosts=None)
        async with _raw_client(app, base_url=f"http://{host}") as client:
            resp = await client.get("/health")
        assert resp.status_code == expected, (api_host, allowed, host)


class TestTrustedHostLoggingPrivacy:

    @pytest.mark.asyncio
    async def test_startup_log_has_counts_not_values(self, monkeypatch, caplog):
        monkeypatch.setattr("config.app_config.API_HOST", TRUSTED_IP)
        monkeypatch.setattr("config.app_config.API_ALLOWED_HOSTS", [TRUSTED_HOST, "bad*host"])
        with caplog.at_level("INFO"):
            create_app_with_secret(_make_orchestrator(), start_background=False, trusted_hosts=None)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert TRUSTED_IP not in text
        assert TRUSTED_HOST not in text
        assert "bad*host" not in text
        assert "Trusted non-loopback hostnames: 2" in text
        assert "Rejected trusted-host entries: 1" in text

    @pytest.mark.asyncio
    async def test_startup_log_count_excludes_loopback_api_host(self, monkeypatch, caplog):
        """Parent review: the default api.host (127.0.0.1) stays in the
        normalized set but is not a non-loopback hostname, so the count is 0."""
        monkeypatch.setattr("config.app_config.API_HOST", "127.0.0.1")
        monkeypatch.setattr("config.app_config.API_ALLOWED_HOSTS", [])
        with caplog.at_level("INFO"):
            create_app_with_secret(_make_orchestrator(), start_background=False, trusted_hosts=None)
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Trusted non-loopback hostnames: 0" in text
