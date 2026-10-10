"""Lane 1B (2026-10-10): shutdown budget + shutdown-path paste guard.

L20 (BC-69): `_check_implementation_tracking` ran a SYNC chroma/file loop on the
event loop inside the phase-B gather (233 proposals -> ~40 s), so the shared
60 s wait_for cancelled the open-thread pass. It now runs on a worker thread
and is capped per shutdown (oldest-checked first).

L21/L39 (BC-75, BC-46): the per-turn `_paste_guard_filter` never ran on the
shutdown LLM-extraction path; pasted third-party text minted
"genius | is_a | the ultimate source of music knowledge". The SAME guard
helpers (memory.memory_storage.is_paste_sized / subject_is_user_anchored) now
gate triples in `LLMFactExtractor._attach_source_excerpts`.

Both are driven through the deployed methods.

class: BC-69, BC-75, BC-46
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.llm_fact_extractor import LLMFactExtractor
from memory.memory_storage import _FACT_EXTRACT_PASTE_CHARS


# ---------------------------------------------------------------------------
# L20: implementation tracking off the event loop, capped
# ---------------------------------------------------------------------------

def _processor():
    from memory.shutdown_processor import ShutdownProcessor
    mm = MagicMock()
    mm.generate_once = AsyncMock(return_value="")
    return ShutdownProcessor(
        corpus_manager=MagicMock(), chroma_store=MagicMock(),
        consolidator=MagicMock(), fact_extractor=MagicMock(),
        model_manager=mm, user_profile=MagicMock(), storage=MagicMock(),
        session_start=datetime.now(), memory_coordinator=None,
    )


def _proposal(i, last):
    p = MagicMock()
    p.id = f"p{i}"
    p.last_tracked_at = last
    return p


class _SlowStore:
    """Fake ProposalStore whose per-item update blocks (like chroma list_all)."""
    def __init__(self, proposals, delay):
        self._proposals = proposals
        self._delay = delay
        self.updated = []

    def get_pending_and_approved(self):
        return list(self._proposals)

    def update_tracking_metadata(self, pid, result):
        time.sleep(self._delay)
        self.updated.append(pid)
        return True


def _detector():
    det = MagicMock()
    res = MagicMock()
    res.skipped_reason = None
    det.detect_single.return_value = res
    return det


@pytest.fixture
def tracking_on(monkeypatch):
    monkeypatch.setattr("config.app_config.IMPL_TRACKING_ENABLED", True)
    monkeypatch.setattr("config.app_config.IMPL_TRACKING_AT_SHUTDOWN", True)


@pytest.mark.asyncio
async def test_tracking_does_not_block_sibling_phase_b_task(tracking_on):
    proc = _processor()
    store = _SlowStore([_proposal(i, None) for i in range(10)], delay=0.1)
    done_at = {}
    t0 = time.monotonic()  # gather start: a loop-blocking sibling delays us

    async def thread_pass():  # stands in for _process_open_threads
        await asyncio.sleep(0.05)
        done_at["thread"] = time.monotonic() - t0

    with patch("memory.proposal_store.ProposalStore", return_value=store), \
         patch("knowledge.implementation_detector.ImplementationDetector",
               return_value=_detector()):
        await asyncio.gather(proc._check_implementation_tracking(), thread_pass())

    # tracking takes ~1.0 s of blocking sleeps; the sibling must finish in
    # about its own 50 ms, not wait behind the loop-blocking sync loop.
    assert done_at["thread"] < 0.5, done_at
    assert len(store.updated) == 10


@pytest.mark.asyncio
async def test_tracking_cap_honoured_oldest_checked_first(tracking_on, monkeypatch):
    import memory.shutdown_processor as sp
    monkeypatch.setattr(sp, "IMPL_TRACKING_MAX_PER_SHUTDOWN", 3)
    proc = _processor()
    props = [_proposal(0, 500.0), _proposal(1, None), _proposal(2, 100.0),
             _proposal(3, 900.0), _proposal(4, 300.0)]
    store = _SlowStore(props, delay=0)
    with patch("memory.proposal_store.ProposalStore", return_value=store), \
         patch("knowledge.implementation_detector.ImplementationDetector",
               return_value=_detector()):
        await proc._check_implementation_tracking()
    assert store.updated == ["p1", "p2", "p4"]  # never-tracked, then oldest


# ---------------------------------------------------------------------------
# L21/L39: paste guard on the shutdown LLM-extraction path
# ---------------------------------------------------------------------------

class _FakeMM:
    def __init__(self, triples):
        self._payload = json.dumps(triples)

    async def generate_once(self, **kwargs):
        return self._payload


async def _extract(triples, messages):
    ex = LLMFactExtractor(_FakeMM(triples))
    return await ex.extract_triples(messages)


_GENIUS = {"subject": "Genius", "relation": "is_a",
           "object": "the ultimate source of music knowledge", "confidence": 0.9}
_USER_STUDY = {"subject": "user", "relation": "studies",
               "object": "actuarial science", "confidence": 0.9}


def _long_paste(extra=""):
    filler = ("Lyrics sites vary widely in quality and coverage across genres. " * 40)
    return (filler + "Genius is the ultimate source of music knowledge. "
            + extra + filler)


@pytest.mark.asyncio
async def test_pasted_third_party_text_yields_no_non_user_triples():
    msg = _long_paste()
    assert len(msg) > _FACT_EXTRACT_PASTE_CHARS
    out = await _extract([_GENIUS], [msg])
    assert out == []


@pytest.mark.asyncio
async def test_short_message_third_party_triple_unaffected():
    out = await _extract(
        [_GENIUS], ["Genius is the ultimate source of music knowledge."])
    assert [t["subject"].lower() for t in out] == ["genius"]


@pytest.mark.asyncio
async def test_short_first_person_unaffected():
    out = await _extract([_USER_STUDY], ["I study actuarial science."])
    assert [(t["subject"], t["object"]) for t in out] == [("user", "actuarial science")]


@pytest.mark.asyncio
async def test_long_message_keeps_the_user_anchored_triple():
    msg = _long_paste(extra="I study actuarial science. ")
    out = await _extract([_GENIUS, _USER_STUDY], [msg])
    assert [(t["subject"], t["object"]) for t in out] == [("user", "actuarial science")]
