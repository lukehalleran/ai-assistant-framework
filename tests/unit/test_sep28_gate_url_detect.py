"""2026-09-28 (BC-01): gate URL check goes through utils.url_detect."""
import pytest

from utils.url_detect import contains_url
import core.agentic.gate as gate


@pytest.mark.parametrize("text", [
    "https://x.com/y", "check http://a.b", "see (https://x.com) ok", "HTTPS://X.COM/A",
])
def test_real_urls_detected(text):
    assert contains_url(text)


@pytest.mark.parametrize("text", [
    "the https protocol", "http:", "https://", "example.com", "www.example.com",
    "xhttp://a.b", "ftp://a.b/http://", "", None,
])
def test_non_urls_not_detected(text):
    assert not contains_url(text)


def test_gate_uses_shared_helper():
    # Misbehaving-input probe: the old substring test treated a glued
    # 'xhttp://a.b' as a URL; the gate must now route through contains_url.
    assert gate.contains_url is contains_url
    import inspect
    src = inspect.getsource(gate.evaluate_agentic_gate)
    assert "'http://' in" not in src
    assert "contains_url(" in src
