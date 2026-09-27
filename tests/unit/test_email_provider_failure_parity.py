"""Wiring parity guard for BC-47 (failure collapsed into a valid empty result).

2026-09-27: a revoked Gmail token made every search return ``[]`` and the tool
reported "No emails found". The closure is that every provider can say WHY it
did not search (``unavailable_reason()``), and the registry routes that into
``provider_coverage()['failed']``. This test fails loudly when a provider is
added to ``core.email.registry.PROVIDERS`` without that hook, so the next adapter
cannot silently reintroduce the "failed = empty" shape.
"""

import pytest

from core.email import registry


@pytest.mark.parametrize("name", sorted(registry.PROVIDERS))
def test_every_registered_provider_reports_unavailable_reason(name):
    row = registry.PROVIDERS[name]
    provider = row["factory"]()
    reason_fn = getattr(provider, "unavailable_reason", None)
    assert callable(reason_fn), (
        f"email provider {name!r} has no unavailable_reason(); a failed search "
        "would be indistinguishable from an empty inbox (BC-47)"
    )
    reason = reason_fn()
    assert reason is None or isinstance(reason, str)


def test_coverage_routes_a_failing_provider_to_failed(monkeypatch):
    class _Failing:
        name = "failing"

        def is_configured(self):
            return True

        def unavailable_reason(self):
            return "authorization revoked"

    monkeypatch.setattr(
        registry, "PROVIDERS",
        {"failing": {"factory": _Failing, "enabled": lambda: True}},
    )
    cov = registry.provider_coverage()
    assert cov["searched"] == []
    assert cov.get("failed") == {"failing": "authorization revoked"}
    assert "FAILED" in registry.coverage_note()
