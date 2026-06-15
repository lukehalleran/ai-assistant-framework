# tests/agent_branch/test_llm_proxy.py
"""Unit tests for the UDS LLM proxy (no podman — proxy runs supervisor-side).

Covers: forwarding + token metering, the key-injection property (the worker
never holds a credential), default-deny of forward-proxy attempts to a
non-allowlisted host (red-team #10), and the budget kill switch.
"""

import json
import os
import tempfile
import time

import pytest

from agent_branch.llm_proxy import LLMProxy


@pytest.fixture
def proxy_factory(stub_upstream):
    """Yields a function that starts a proxy against a fresh stub upstream."""
    started = []

    def _make(*, content='"""doc"""', tokens=42, token_budget=100_000, api_key=None):
        cm = stub_upstream(content=content, tokens=tokens)
        port, handler = cm.__enter__()
        uds = os.path.join(tempfile.mkdtemp(), "llm.sock")
        proxy = LLMProxy(uds, f"http://127.0.0.1:{port}", allowed_hosts=["127.0.0.1"],
                         token_budget=token_budget, api_key=api_key,
                         log_path=os.path.join(os.path.dirname(uds), "proxy.jsonl")).start()
        time.sleep(0.1)
        started.append((proxy, cm))
        return proxy, handler

    yield _make
    for proxy, cm in started:
        proxy.stop()
        cm.__exit__(None, None, None)


def test_forwards_and_meters_tokens(proxy_factory, uds_post):
    proxy, _ = proxy_factory(content='"""hi"""', tokens=123)
    status, data = uds_post(proxy.uds_path, "/v1/chat/completions", {"q": "x"})
    assert status == 200
    assert json.loads(data)["choices"][0]["message"]["content"] == '"""hi"""'
    assert proxy.tokens_spent == 123
    assert proxy.calls == 1


def test_api_key_injected_by_proxy_not_worker(proxy_factory, uds_post):
    """The real key lives in the proxy; the worker sends none, upstream sees it."""
    proxy, handler = proxy_factory(api_key="SECRET-supervisor-key")
    uds_post(proxy.uds_path, "/v1/chat/completions", {"q": "x"})  # worker sends no auth
    assert handler.captured_headers.get("Authorization") == "Bearer SECRET-supervisor-key"


def test_rejects_forward_proxy_to_nonallowlisted_host(proxy_factory, uds_post):
    """Red-team #10: a forward-proxy attempt to another host is denied (not an
    open relay)."""
    proxy, _ = proxy_factory()
    status, data = uds_post(proxy.uds_path, "http://evil.example.com/v1/chat/completions", {"q": "x"})
    assert status == 403
    assert "not allowlisted" in json.loads(data)["error"]
    assert proxy.calls == 0  # nothing was forwarded


def test_rejects_request_to_nonallowlisted_path(proxy_factory, uds_post):
    """The worker may only hit the chat endpoint — not key-management, uploads,
    or other (differently-priced) routes — with the supervisor's injected key."""
    proxy, _ = proxy_factory()
    status, data = uds_post(proxy.uds_path, "/v1/admin/api-keys", {"q": "x"})
    assert status == 403
    assert "not allowlisted" in json.loads(data)["error"]
    assert proxy.calls == 0  # nothing was forwarded upstream


def test_socket_directory_is_private_to_supervisor(tmp_path):
    """The socket dir is locked to 0700 so no OTHER host process can reach the
    UDS (and indirectly spend the budget / use the key). The socket itself stays
    world-connectable — the container connects as an unmapped userns uid — but
    podman bind-mounts the inode, so container access is unaffected by dir mode."""
    sock_dir = tmp_path / "open_dir"
    sock_dir.mkdir(mode=0o755)
    os.chmod(sock_dir, 0o755)  # world-readable before start()
    proxy = LLMProxy(str(sock_dir / "llm.sock"), "http://127.0.0.1:1").start()
    try:
        assert (os.stat(sock_dir).st_mode & 0o777) == 0o700
    finally:
        proxy.stop()


def test_origin_form_request_is_allowed_regardless_of_host_header(proxy_factory, uds_post):
    proxy, _ = proxy_factory()
    status, _ = uds_post(proxy.uds_path, "/v1/chat/completions", {"q": "x"},
                         headers={"Host": "anything.example"})
    assert status == 200


def test_budget_exceeded_refuses_further_calls(proxy_factory, uds_post):
    # one call spends the whole budget; the budget check at the start of the next
    # call refuses it.
    proxy, _ = proxy_factory(tokens=2000, token_budget=2000)
    s1, _ = uds_post(proxy.uds_path, "/v1/chat/completions", {"q": "1"})
    assert s1 == 200
    assert proxy.tokens_spent == 2000
    assert proxy.budget_exceeded() is True
    s2, _ = uds_post(proxy.uds_path, "/v1/chat/completions", {"q": "2"})
    assert s2 == 429
    assert proxy.calls == 1  # the refused call was never forwarded
