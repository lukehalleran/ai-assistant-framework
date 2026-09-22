"""core.prompt.builder._topup_filler/_recency_floor_filler must dedup
through the canonical key.

Regression for 2026-09-22 (B2, class: BC-20, BC-24, BC-91): the memory
top-up block (core/prompt/builder.py ~1815-1860) used to key already-shown
items on raw `str(query) + str(response)` (lowercased, stripped) to decide
which extra recent conversations to fold in as filler "relevant memories".
`_get_recent_conversations()` (core/prompt/gatherer_memory.py) returns
shallow copies whose `response` may carry a read-time marker appended AFTER
generation (the personal-claim-check marker or the unverified-action-claim
marker), while the top-up's own backfill fetch (`_get_recent_conversations`
called a second time with a larger limit) returns the SAME turn without any
marker for a corpus entry that predates the marker's application order, or
vice versa across [RECENT CONVERSATION] vs [RELEVANT MEMORIES]. The raw key
diverged between the marked and unmarked copies of the identical turn, so
the top-up treated the marked copy as "not yet shown" and re-added it as a
duplicate. `_topup_filler` now dedups through `core.prompt.hygiene.
_canonical_turn_key`, which strips read-time machinery (utils.
read_time_markers.strip_machinery) before composing the key — the two
copies of the same turn collide regardless of which one carries the marker.

Live evidence: daemon_debug.log 14780->14791 ("CROSS-SECTION DEDUP
memories: 1 -> 0" then "MEMORY TOP-UP: Added 3 ... skipped 9 duplicates")
and 16579->16590 — a 14K-char paste rendered in both [RECENT CONVERSATION]
and [RELEVANT MEMORIES] across three turns.

A second, structurally identical site exists in the same file: the
pre-budget "Recent conversations floor" block (core/prompt/builder.py,
inside the "Complete recency candidates BEFORE budgeting" step) re-fetches
recent conversations and used to dedup them against the already-present
`recent_conversations` on the SAME raw `query + response` key — same bug,
same fix. `_recency_floor_filler` is the extracted, directly-testable form
of that block's dedup loop.
"""

from core.prompt.builder import _recency_floor_filler, _topup_filler
from core.prompt.gatherer_memory import _annotate_memory_item_claim
from utils.personal_claim_provenance import KEY, clean_personal_claim_receipt

QUERY = "I was walking about nine thousand steps a day for a week or so."
RESPONSE = "That's a completely reasonable line to hold, and honestly a responsible one."


def _marking_receipt(response: str) -> dict:
    """Build a receipt via the deployed `clean_personal_claim_receipt` that
    will cause `annotate_personal_claim_memory` to append the personal-claim
    marker to `response` (status checked, delivery unchanged, at least one
    insufficient/contradicted count, digest over the machinery-free text —
    matching how a real checker receipt is produced pre-suffix)."""
    value = dict(status="checked", delivery="unchanged", insufficient_count=1)
    return clean_personal_claim_receipt(value, response=response)


def test_topup_filler_dedupes_marked_recent_against_raw_backfill_copy():
    # X: the SAME turn appears twice — once already in `recents` with the
    # read-time marker applied (as `_get_recent_conversations` would return
    # it), and once raw in the backfill's `extra_recent` fetch. Y is a
    # genuinely different turn that must survive as filler.
    x_raw = {"query": QUERY, "response": RESPONSE, "timestamp": "2026-09-22T13:05:00"}
    x_raw_with_receipt = {**x_raw, KEY: _marking_receipt(RESPONSE)}
    x_marked = _annotate_memory_item_claim(x_raw_with_receipt)

    # The marker really must have applied, or this test proves nothing about
    # the mismatch it is regression-guarding against.
    assert x_marked["response"] != x_raw["response"], (
        "personal-claim marker did not apply via the deployed annotator; "
        "test setup is not exercising the marked-vs-raw mismatch"
    )

    y_raw = {
        "query": "Totally different question about FAISS?",
        "response": "FAISS is a vector index library.",
        "timestamp": "2026-09-22T13:10:00",
    }

    filler, skipped = _topup_filler(
        recents=[x_marked], mems=[], extra_recent=[x_raw, y_raw], needed=3,
    )

    assert filler == [y_raw]
    assert skipped == 1


def test_topup_filler_dedupes_against_mems_not_only_recents():
    x_raw = {"query": QUERY, "response": RESPONSE}
    x_raw_with_receipt = {**x_raw, KEY: _marking_receipt(RESPONSE)}
    x_marked = _annotate_memory_item_claim(x_raw_with_receipt)
    assert x_marked["response"] != x_raw["response"]

    filler, skipped = _topup_filler(
        recents=[], mems=[x_marked], extra_recent=[x_raw], needed=2,
    )

    assert filler == []
    assert skipped == 1


def test_topup_filler_needed_zero_returns_empty_and_true_skip_count():
    y_raw = {"query": "Unrelated.", "response": "Also unrelated."}
    filler, skipped = _topup_filler(recents=[], mems=[], extra_recent=[y_raw], needed=0)
    assert filler == []
    assert skipped == 0


def test_topup_filler_no_extra_recent_returns_empty():
    assert _topup_filler(recents=[], mems=[], extra_recent=[], needed=3) == ([], 0)


def test_topup_filler_distinct_turns_all_survive_as_filler():
    a = {"query": "First question.", "response": "First answer."}
    b = {"query": "Second question.", "response": "Second answer."}
    filler, skipped = _topup_filler(recents=[], mems=[], extra_recent=[a, b], needed=5)
    assert filler == [a, b]
    assert skipped == 0


# --- second site: the pre-budget recency-floor block --------------------

def test_recency_floor_filler_dedupes_marked_recent_against_raw_stored_copy():
    # X is already in `recent_convos` marked (as the gatherer would return
    # it); the fresh `stored_recent` floor fetch returns X raw plus a
    # genuinely different turn Y that must survive.
    x_raw = {"query": QUERY, "response": RESPONSE, "timestamp": "2026-09-22T13:05:00"}
    x_raw_with_receipt = {**x_raw, KEY: _marking_receipt(RESPONSE)}
    x_marked = _annotate_memory_item_claim(x_raw_with_receipt)
    assert x_marked["response"] != x_raw["response"], (
        "personal-claim marker did not apply via the deployed annotator; "
        "test setup is not exercising the marked-vs-raw mismatch"
    )

    y_raw = {
        "query": "Totally different question about FAISS?",
        "response": "FAISS is a vector index library.",
        "timestamp": "2026-09-22T13:10:00",
    }

    add = _recency_floor_filler(
        recent_convos=[x_marked], stored_recent=[x_raw, y_raw], needed=3,
    )

    assert add == [y_raw]


def test_recency_floor_filler_needed_zero_returns_empty():
    y_raw = {"query": "Unrelated.", "response": "Also unrelated."}
    assert _recency_floor_filler(recent_convos=[], stored_recent=[y_raw], needed=0) == []


def test_recency_floor_filler_stops_at_needed_and_dedupes_within_batch():
    a = {"query": "First question.", "response": "First answer."}
    a_dup = {"query": "First question.", "response": "First answer."}
    b = {"query": "Second question.", "response": "Second answer."}
    add = _recency_floor_filler(recent_convos=[], stored_recent=[a, a_dup, b], needed=1)
    assert add == [a]
