"""B4 item 3: the personal-claim marker names the ASSISTANT as author of the
doubted claims and is never read as the user's statement being unverified (BC-91)."""
from utils.read_time_markers import (
    LEGACY_PERSONAL_CLAIM_MARKERS, PERSONAL_CLAIM_MARKER, strip_machinery,
)


def test_marker_names_assistant_as_claim_author():
    m = PERSONAL_CLAIM_MARKER.lower()
    assert "assistant" in m and "added" in m
    assert "user's own words" in m
    assert "not unverified" in m or "says nothing against" in m
    # must not pose the user's account as the unverified thing
    assert "insufficient or conflicting user evidence" not in m


def test_legacy_marker_still_stripped_from_stored_turns():
    legacy = LEGACY_PERSONAL_CLAIM_MARKERS[0]
    stored = f"You work at Northwind as a data analyst.\n{legacy}"
    assert strip_machinery(stored) == "You work at Northwind as a data analyst."
    new = f"You work at Northwind.\n{PERSONAL_CLAIM_MARKER}"
    assert strip_machinery(new) == "You work at Northwind."
