"""
Shutdown regex fact path obeys the fact-admission chokepoint.

Incident 2026-10-18 (class: BC-58, guard covers only the first-observed path):
at the 19:56 shutdown ``_extract_session_facts`` re-mined the session's
homework pastes and stored course-material sentences as facts, while the
same text on the per-turn path was dropped by the paste guard. Drives THE
deployed ``ShutdownProcessor._extract_session_facts`` with the deployed
regex ``FactExtractor`` and a recording fake store.
"""
import asyncio
from datetime import datetime

import pytest

from memory import shutdown_processor as sp
from memory.fact_extractor import FactExtractor


class _Store:
    def __init__(self):
        self.facts = []

    def add_fact(self, fact, source=None, confidence=None):
        self.facts.append(fact)
        return "id-%d" % len(self.facts)

    def update_metadata(self, *a, **k):
        return None


_LESSON = (
    "Zones are groups of distinct belts. "
    "A controller is unavailable after a time we define. "
    "Buffers are widely used for parcel staging. "
    "The sorter is released from the schedule. "
    "Jam is a confusing name. "
    "Conveyor zones are the basic unit of the line and every lesson in this "
    "unit builds on that idea before moving to the next topic. "
)
PASTE = (_LESSON * 9).strip()
USER_LINE = " I live in Tampa."


def _run(text, wrap=False):
    if wrap:
        text = "\n".join("  " + ln for ln in text.replace(". ", ".\n").splitlines())
    store = _Store()
    proc = sp.ShutdownProcessor(
        None, store, None, FactExtractor(use_rebel=False), None, None, None,
        datetime.now())
    conv = {"query": text, "user_text": text, "response": "ok",
            "timestamp": datetime.now().isoformat()}
    asyncio.run(proc._extract_session_facts([conv]))
    return store.facts


@pytest.mark.parametrize("wrap", [False, True])
def test_paste_triples_are_not_stored_but_user_triple_is(wrap):
    assert len(PASTE) > 2000
    facts = _run(PASTE + USER_LINE, wrap=wrap)
    joined = "\n".join(facts)
    assert not [f for f in facts if not f.lower().startswith("user |")], joined
    assert any(f.lower().startswith("user |") for f in facts), joined


def test_short_message_non_user_triples_still_stored_control():
    """Control: the guard is the paste size, not a blanket drop."""
    facts = _run("Zones are groups of distinct belts.")
    assert facts, "short declaratives must still yield facts"
