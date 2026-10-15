"""2026-10-14 (class: BC-58, BC-06): Tier-1 keyword arms are scoped like the file arm.

A vent plus an inline pasted progress report ('the integral of the pdf is 1')
routed a 36 s agentic computation loop because 'integral' matched mid-paste.
Every Tier-1 keyword arm now goes through `_tier1_keyword_hit`
(code-stripped, head/tail window on long messages). Drives the DEPLOYED
`evaluate_agentic_gate`; synthetic course content.
"""
import pytest

from core.agentic.gate import evaluate_agentic_gate

VENT = ("I feel terrible. I worked for about 5 hours. but I feel like i wasted my "
        "time. I didnt even get more problems done just fixing and finishing old ones:")

_FILLER = (
    "Section {n}. Finished the STAT 5100 simulation writeup and revised the plots "
    "for the second part. Fixed the labels on the axes, reran the sampler with a "
    "larger sample, and cleaned the notes for the group. "
)


def _paste(middle: str, tail: str = "") -> str:
    body = "".join(_FILLER.format(n=i) for i in range(6))
    half = len(body) // 2
    return body[:half] + middle + " " + body[half:] + tail


INCIDENT = VENT + "\n\n" + _paste(
    " I checked that the integral of the pdf is 1 and I need to compute the "
    "normalizing constant again later.")


@pytest.mark.asyncio
async def test_pasted_report_with_midbody_keywords_is_not_computation():
    assert len(INCIDENT.split()) > 150
    d = await evaluate_agentic_gate(INCIDENT)
    assert "computation" not in d.modes
    assert not d.should_trigger or d.reason != "triggered: computation"


@pytest.mark.asyncio
async def test_short_compute_request_still_computation():
    d = await evaluate_agentic_gate("compute the integral of x^2 from 0 to 1")
    assert d.should_trigger and "computation" in d.modes


@pytest.mark.asyncio
async def test_long_paste_ending_in_request_uses_tail_window():
    d = await evaluate_agentic_gate(_paste("") + " Can you calculate the variance of this?")
    assert "computation" in d.modes


@pytest.mark.asyncio
async def test_short_web_search_still_web():
    d = await evaluate_agentic_gate("search the web for python tutorials please")
    assert "web_search" in d.modes
