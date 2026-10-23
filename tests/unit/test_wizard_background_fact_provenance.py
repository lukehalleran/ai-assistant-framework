"""First-run wizard background facts must join back to the user's own text.

Incident 2026-10-23 (finding first logged 2026-10-15 by lane HABIT).
Class: BC-46 (LLM output stored as a user fact) with BC-58 parity (claim-time
metadata the normal LLM path stamps) -- the wizard stored every extracted
triple with the literal excerpt "Initial wizard background".

Drives the DEPLOYED wizard path (_handle_background -> _finalize_wizard) with
the real LLMFactExtractor and a misbehaving fake model.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gui.wizard import WizardState, WizardStep, _handle_background

CLEAN = (
    "I work as a bike mechanic and I live in Lisbon. "
    "Last year I broke my wrist skiing in March."
)
WRAPPED = "\n".join("    " + ln for ln in [
    "I work as a bike mechanic and",
    "I live in Lisbon. Last year I broke my",
    "wrist skiing in March.",
]) + "\n"


class FakeModel:
    """Misbehaving model: proposes an unsupported employer, a supported
    occupation, and a past-dated event."""

    def __init__(self, text):
        self.text = text

    async def generate_once(self, **kw):
        return json.dumps([
            {"subject": "user", "relation": "works_at", "object": "Acme Corp",
             "confidence": 0.9},
            {"subject": "user", "relation": "occupation",
             "object": "bike mechanic", "confidence": 0.9},
            {"subject": "user", "relation": "injury",
             "object": "broke my wrist skiing in March", "confidence": 0.8},
        ])


class FakeProfile:
    def __init__(self):
        self.calls = []
        self.identity = SimpleNamespace(name="", pronouns="")
        self.preferences = SimpleNamespace(style="balanced")

    def update_identity(self, **kw):
        pass

    def update_preferences(self, **kw):
        pass

    def add_fact(self, relation, value, confidence=0.7, source_excerpt="",
                 category=None, timestamp=None, stance="", claim_kind="",
                 event_date=""):
        self.calls.append(dict(relation=relation, value=value,
                               source_excerpt=source_excerpt,
                               claim_kind=claim_kind, event_date=event_date))
        return True

    def save(self):
        pass


class FakeChroma:
    def __init__(self):
        self.calls = []

    def add_fact(self, fact, source=None, **kw):
        self.calls.append((fact, dict(source or {})))
        return "id"


def _run(text):
    profile, chroma = FakeProfile(), FakeChroma()
    orch = SimpleNamespace(
        model_manager=FakeModel(text), user_profile=profile,
        memory_system=SimpleNamespace(chroma_store=chroma))
    state = WizardState(step=WizardStep.BACKGROUND,
                        collected_data={"style": "balanced"})
    asyncio.run(_handle_background(text, state, orch))
    return profile, chroma


@pytest.mark.parametrize("text", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
def test_unsupported_triple_never_reaches_profile_or_chroma(text):
    profile, chroma = _run(text)
    assert all(c["value"] != "Acme Corp" for c in profile.calls)
    assert all("Acme Corp" not in f for f, _ in chroma.calls)


@pytest.mark.parametrize("text", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
def test_supported_triple_carries_real_excerpt(text):
    profile, chroma = _run(text)
    occ = [c for c in profile.calls if c["relation"] == "occupation"]
    assert len(occ) == 1
    assert occ[0]["source_excerpt"] != "Initial wizard background"
    assert "bike mechanic" in occ[0]["source_excerpt"]
    src = [s for f, s in chroma.calls if "bike mechanic" in f]
    assert src and "bike mechanic" in src[0].get("source_excerpt", "")
    assert src[0].get("source_role") == "user"


@pytest.mark.parametrize("text", [CLEAN, WRAPPED], ids=["clean", "wrapped"])
def test_past_event_carries_claim_time_metadata(text):
    profile, chroma = _run(text)
    inj = [c for c in profile.calls if c["relation"] == "injury"]
    assert len(inj) == 1
    # claim-time metadata is stamped (any non-empty kind); the hard-wrapped
    # form classifies "unknown" (never planned -- the safe under-fire), the
    # clean form "event".
    assert inj[0]["claim_kind"] in ("event", "unknown")
    if text is CLEAN:
        assert inj[0]["claim_kind"] == "event"
    src = [s for f, s in chroma.calls if "wrist" in f]
    assert src and src[0].get("claim_kind") == inj[0]["claim_kind"]
