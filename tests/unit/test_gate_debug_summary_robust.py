"""`_gate_debug_summary` (2026-09-02 gate-reason surfacing) must never raise.

It runs inside both the agentic and enhanced handler paths; a non-string
`reason` (a MagicMock in tests, but any object in prod) raised inside its join
and took down the agentic turn AND the enhanced fallback.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gui.handlers import _gate_debug_summary


def test_string_fields_render():
    d = SimpleNamespace(reason="tier2_entity_recall", modes=["memory", "web"],
                        veto_exempt=True, deferred_request=None)
    assert _gate_debug_summary(d) == "tier2_entity_recall | modes=['memory', 'web'] | veto_exempt"


def test_non_string_fields_never_raise():
    d = SimpleNamespace(reason=MagicMock(), modes=[MagicMock(), "web"], veto_exempt=False,
                        deferred_request=None)
    out = _gate_debug_summary(d)
    assert isinstance(out, str) and "web" in out
    assert isinstance(_gate_debug_summary(MagicMock()), str)
    assert _gate_debug_summary(None) == ""


def test_broken_attribute_access_fails_soft():
    class Exploding:
        @property
        def reason(self):
            raise RuntimeError("boom")
    assert _gate_debug_summary(Exploding()) == ""
