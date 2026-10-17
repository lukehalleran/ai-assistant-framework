"""BC-97: the personal-claim read-time note names WHICH claim it flags.

2026-10-13: a generic marker was appended to nearly every stored assistant
turn, so it carried no information and a false date the assistant had stated
was repeated from [RECENT CONVERSATION] four times. The receipt now carries
the flagged spans; the one annotate function quotes them; the one stripper
removes them.

Original batch label: oct14e (renamed 2026-10-16; old path in tests/RENAMES.tsv).
"""
import asyncio
import json
from unittest.mock import MagicMock

from core.personal_claim_check import audit_personal_claims
from memory.corpus_manager import CorpusManager
from utils.personal_claim_provenance import (
    MARKER,
    annotate_personal_claim_memory,
    clean_personal_claim_receipt,
    marker_suffix,
    response_digest,
)
from utils.read_time_markers import (
    PERSONAL_CLAIM_SPAN_OPEN,
    personal_claim_span_marker,
    strip_machinery,
)

CLAIM = "COURSE B HW 4 isn't due until the 25th"
DISCUSSION = "Most students find a steady weekly schedule helpful"
RESPONSE = (
    f"Quick check on your week. {CLAIM}. Also, {DISCUSSION.lower()}.\n\n"
    "Let me know how it goes."
)


class _Model:
    def __init__(self, payload):
        self.payload = payload

    async def generate_once(self, prompt, **kwargs):
        return json.dumps(self.payload)


def _evidence():
    return [{"source_id": "u1", "role": "user", "text": "I have a few things due this month."}]


def _audit(response=RESPONSE, claims=None):
    claims = claims if claims is not None else [
        {"text": CLAIM, "status": "insufficient", "kind": "plan", "evidence": []},
        {"text": DISCUSSION, "status": "insufficient", "kind": "discussion", "evidence": []},
    ]
    return asyncio.run(audit_personal_claims(response, _evidence(), _Model({"claims": claims})))


def test_receipt_names_only_personal_kind_spans():
    result = _audit()
    assert result.status == "checked"
    receipt = result.receipt()
    assert receipt["insufficient_count"] == 2
    assert receipt["flagged_spans"] == [CLAIM]


def test_receipt_flagged_spans_present_and_empty_when_nothing_flagged():
    result = _audit(claims=[{"text": DISCUSSION, "status": "insufficient",
                             "kind": "discussion", "evidence": []}])
    assert result.receipt()["flagged_spans"] == []


def test_stored_turn_renders_span_marker_not_generic(tmp_path):
    result = _audit()
    corpus = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    corpus.add_entry("what's due?", RESPONSE, personal_claim_support=dict(
        result.receipt(), delivery="unchanged"))
    entry = corpus.corpus[-1]
    assert entry["personal_claim_support"]["flagged_spans"] == [CLAIM]
    marked = annotate_personal_claim_memory(entry)
    assert personal_claim_span_marker(CLAIM) in marked["response"]
    assert MARKER not in marked["response"]
    assert strip_machinery(marked["response"]) == RESPONSE.strip()


def test_deployed_formatter_render_path_shows_span_marker(tmp_path):
    from core.prompt.formatter import PromptFormatter

    result = _audit()
    corpus = CorpusManager(corpus_file=str(tmp_path / "corpus.json"))
    corpus.add_entry("what's due?", RESPONSE, personal_claim_support=dict(
        result.receipt(), delivery="unchanged"))
    token_mgr = MagicMock()
    token_mgr.count_tokens = MagicMock(return_value=10)
    fmt = PromptFormatter(token_manager=token_mgr, time_manager=None)
    rendered = fmt._format_memory(corpus.corpus[-1])
    assert f'{PERSONAL_CLAIM_SPAN_OPEN}{CLAIM}"' in rendered
    assert MARKER not in rendered


def _receipt(**over):
    base = {"status": "checked", "reason": "ok", "insufficient_count": 1,
            "delivery": "unchanged"}
    base.update(over)
    return base


def test_clean_drops_bad_elements_and_flags_it():
    out = clean_personal_claim_receipt(
        _receipt(flagged_spans=[CLAIM, "not in the response", 7, "", "x" * 300]),
        response=RESPONSE)
    assert out["flagged_spans"] == [CLAIM]
    assert out["flagged_spans_dropped"] is True
    assert out["status"] == "checked"


def test_clean_normalizes_newlines_on_both_sides():
    resp = "Line one starts here\nand continues over a break."
    out = clean_personal_claim_receipt(
        _receipt(flagged_spans=["starts here\nand continues"]), response=resp)
    assert out["flagged_spans"] == ["starts here and continues"]
    assert "flagged_spans_dropped" not in out


def test_clean_caps_at_three():
    resp = "Alpha one. Bravo two. Charlie three. Delta four."
    out = clean_personal_claim_receipt(
        _receipt(flagged_spans=["Alpha one.", "Bravo two.", "Charlie three.", "Delta four."]),
        response=resp)
    assert len(out["flagged_spans"]) == 3
    assert out["flagged_spans_dropped"] is True


def test_long_span_is_cut_at_whitespace_prefix():
    long_claim = " ".join(["word"] * 120)
    result = _audit(response=f"Start. {long_claim}. End.", claims=[
        {"text": long_claim, "status": "insufficient", "kind": "plan", "evidence": []}])
    span = result.receipt()["flagged_spans"][0]
    assert len(span) <= 240 and long_claim.startswith(span) and not span.endswith(" ")


def _item(receipt_spans=None, legacy=False, response=RESPONSE):
    receipt = {"status": "checked", "reason": "ok", "delivery": "unchanged",
               "insufficient_count": 2, "response_sha256": response_digest(response)}
    if not legacy:
        receipt["flagged_spans"] = receipt_spans or []
    return {"query": "q", "response": response, "personal_claim_support": receipt}


def test_empty_flagged_spans_yields_no_marker():
    assert annotate_personal_claim_memory(_item([]))["response"] == RESPONSE


def test_legacy_receipt_keeps_generic_marker():
    assert annotate_personal_claim_memory(_item(legacy=True))["response"].endswith(MARKER)


def test_idempotent_and_roundtrip():
    once = annotate_personal_claim_memory(_item([CLAIM]))
    assert strip_machinery(once["response"]) == RESPONSE.strip()
    twice = annotate_personal_claim_memory(once)
    assert twice["response"] == once["response"]
    legacy = annotate_personal_claim_memory(_item(legacy=True))
    assert strip_machinery(legacy["response"]) == RESPONSE.strip()


def test_content_branch_gets_same_block():
    item = _item([CLAIM])
    item["content"] = f"User: q\nAssistant: {RESPONSE.strip()}"
    marked = annotate_personal_claim_memory(item)
    assert personal_claim_span_marker(CLAIM) in marked["content"]


def test_marker_suffix_for_clipping_consumers():
    item = _item([CLAIM])
    marked = annotate_personal_claim_memory(item)
    assert marker_suffix(item["response"], marked["response"]) == personal_claim_span_marker(CLAIM)
    assert marker_suffix("a", "a") == ""


# --- referee addition (2026-10-14): spans are storage-only -------------------
def test_flagged_spans_reach_storage_provenance_but_not_telemetry_or_debug():
    """PLAN_20260915 keeps telemetry/debug excerpt-free; only the storage
    provenance (stored beside the same reply) carries the quoted spans."""
    from types import SimpleNamespace
    from gui import handlers

    ctx = SimpleNamespace(telemetry={}, debug_record={}, personal_claim_receipt=None)
    receipt = {"status": "checked", "reason": "ok", "candidate_count": 1,
               "insufficient_count": 1, "delivery": "unchanged",
               "flagged_spans": ["COURSE B HW 4 isn't due until the 25th"]}
    provenance = {}
    handlers._record_personal_claim_outcome(ctx, receipt, provenance)
    assert provenance["personal_claim_support"]["flagged_spans"] == receipt["flagged_spans"]
    assert "25th" not in json.dumps(ctx.telemetry)
    assert "25th" not in json.dumps(ctx.debug_record)


def test_revised_reply_receipt_carries_no_spans():
    """A reply revised in correct mode no longer contains the flagged text."""
    from types import SimpleNamespace
    from gui import handlers

    result = SimpleNamespace(receipt=lambda: {
        "status": "checked", "reason": "ok", "candidate_count": 1,
        "insufficient_count": 1, "flagged_spans": ["COURSE B HW 4 isn't due until the 25th"]})
    receipt = handlers._personal_claim_receipt(result, delivery="omitted")
    assert "flagged_spans" not in receipt
