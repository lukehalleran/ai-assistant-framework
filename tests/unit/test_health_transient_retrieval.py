"""
Tests that health-transient facts age out of BOTH read paths, and that
explicit supersession (is_current=False / superseded_by) drops facts from the
facts-collection retriever immediately.

Covers the bug behind "agent still thinks I'm getting over an illness": the
profile section expired health relations caught by the _status suffix but the
facts-collection retriever only checked the exact config list, and relations
like post_illness_recovery matched no pattern at all so they never expired.
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from memory.user_profile import UserProfile
from memory.user_profile_schema import ProfileCategory
from memory.memory_retriever import MemoryRetriever, _fact_ephemeral_ttl


# ==========================================================================
# Profile section (UserProfile.get_category)
# ==========================================================================

@pytest.fixture
def temp_profile():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def _current_values(profile):
    vals = set()
    for cat in ProfileCategory:
        for f in profile.get_category(cat):
            vals.add(f.get("value"))
    return vals


def test_profile_expires_stale_health_transient(temp_profile):
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    profile = UserProfile(temp_profile)
    old_ts = datetime.now() - timedelta(hours=H + 48)
    fresh_ts = datetime.now() - timedelta(hours=2)

    # Values avoid temporal words ("today"/weekday) so the profile's temporal
    # resolver doesn't rewrite them out from under the assertion.
    profile.add_fact("post_illness_recovery", "still recovering from illness",
                     confidence=0.9, category=ProfileCategory.HEALTH, timestamp=old_ts)
    profile.add_fact("symptoms", "queasy stomach",
                     confidence=0.9, category=ProfileCategory.HEALTH, timestamp=fresh_ts)

    current = _current_values(profile)
    assert "still recovering from illness" not in current  # aged out (> health TTL)
    assert "queasy stomach" in current                     # within health TTL


def test_profile_keeps_durable_fact_regardless_of_age(temp_profile):
    profile = UserProfile(temp_profile)
    very_old = datetime.now() - timedelta(days=400)
    profile.add_fact("name", "Luke", confidence=0.9,
                     category=ProfileCategory.IDENTITY, timestamp=very_old)
    assert "Luke" in _current_values(profile)


def test_profile_standard_ephemeral_uses_short_ttl(temp_profile):
    """A standard-ephemeral relation expires on the 24h horizon, not the
    longer health one — proving the two tiers are distinct."""
    from config.app_config import (
        PROFILE_EPHEMERAL_TTL_HOURS as E, PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H,
    )
    assert E < H  # precondition for a meaningful test
    profile = UserProfile(temp_profile)
    between = datetime.now() - timedelta(hours=(E + H) / 2)  # past E, within H
    profile.add_fact("current_activity", "walking to the store",
                     confidence=0.9, category=ProfileCategory.HOBBIES, timestamp=between)
    assert "walking to the store" not in _current_values(profile)


# ==========================================================================
# Facts-collection retriever (MemoryRetriever.get_facts)
# ==========================================================================

class _FakeColl:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


class _FakeStore:
    """Minimal MultiCollectionChromaStore stand-in for get_facts()."""
    def __init__(self, rows):
        self._rows = rows
        self.collections = {"facts": _FakeColl(len(rows))}

    def query_collection(self, name, query_text, n_results):
        return self._rows


def _row(content, *, ts=None, is_current=None, superseded_by=None, rel_score=0.8):
    meta = {"timestamp": (ts or datetime.now()).isoformat(), "confidence": 0.8}
    if is_current is not None:
        meta["is_current"] = is_current
    if superseded_by is not None:
        meta["superseded_by"] = superseded_by
    return {"id": content[:8], "content": content, "metadata": meta,
            "relevance_score": rel_score}


def _get_facts(rows, query="recovering from illness", limit=10):
    retr = MemoryRetriever(corpus_manager=MagicMock(), chroma_store=_FakeStore(rows))
    return asyncio.run(retr.get_facts(query, limit=limit))


def _contents(results):
    return {r["content"] for r in results}


def test_retriever_drops_explicitly_superseded():
    now = datetime.now()
    rows = [
        _row("user | post_illness_recovery | still recovering", ts=now, is_current=False),
        _row("user | post_illness_recovery | fighting a virus", ts=now, superseded_by="x"),
        _row("user | name | Luke", ts=now),
    ]
    out = _contents(_get_facts(rows))
    assert "user | name | Luke" in out
    assert not any("recovering" in c or "virus" in c for c in out)


def test_retriever_ages_out_stale_health_transient():
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    rows = [
        _row("user | health_status | recovering from illness",
             ts=datetime.now() - timedelta(hours=H + 72)),       # stale → drop
        _row("user | symptoms | nausea today",
             ts=datetime.now() - timedelta(hours=2)),            # fresh → keep
    ]
    out = _contents(_get_facts(rows))
    assert "user | symptoms | nausea today" in out
    assert "user | health_status | recovering from illness" not in out


def test_retriever_keeps_durable_fact_even_if_old():
    rows = [_row("user | brother_name | Dillion",
                 ts=datetime.now() - timedelta(days=300))]
    out = _contents(_get_facts(rows))
    assert "user | brother_name | Dillion" in out


def test_fact_ephemeral_ttl_helper():
    from config.app_config import (
        PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H, PROFILE_EPHEMERAL_TTL_HOURS as E,
    )
    assert _fact_ephemeral_ttl("user | post_illness_recovery | x") == float(H)
    assert _fact_ephemeral_ttl("user | current_activity | x") == float(E)
    assert _fact_ephemeral_ttl("user | name | Luke") is None
    assert _fact_ephemeral_ttl("no pipes here") is None


# ==========================================================================
# Read-time scorer: free-text health-framing decay (MemoryScorer.rank_memories)
# ==========================================================================
#
# The structured paths above (profile, facts) age out illness *relations*. But
# the "post-viral" framing the user complained about also lived as free text in
# conversations / reflections / notes, which carry no relation and so were never
# aged out. These tests cover the scorer-level decay that closes that gap.

from memory.memory_scorer import MemoryScorer


def _mem(content, collection, age_hours, rel_score=0.8):
    return {
        "content": content,
        "collection": collection,
        "relevance_score": rel_score,
        "timestamp": (datetime.now() - timedelta(hours=age_hours)).isoformat(),
        "metadata": {},
    }


def _score_one(mem):
    """Score a single memory and return its final_score."""
    ranked = MemoryScorer().rank_memories([dict(mem)], "how have i been feeling lately")
    return ranked[0]["final_score"]


def test_stale_health_framing_is_penalized():
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    stale = "You were fighting a virus — that post-viral drag is stubborn"
    fresh_score = _score_one(_mem(stale, "conversations", age_hours=2))          # within TTL
    stale_score = _score_one(_mem(stale, "conversations", age_hours=H + 400))    # well past TTL
    assert stale_score < fresh_score - 0.1  # meaningfully down-weighted


def test_non_health_narrative_not_penalized():
    """A stale non-health memory of the same age/collection is untouched, so the
    penalty is health-specific, not a generic recency effect (recency is handled
    separately and is identical for both)."""
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    health = _mem("still recovering from illness, feeling wiped", "conversations", H + 400)
    other = _mem("we shipped the retrieval overhaul and OAuth2 pipeline", "conversations", H + 400)
    assert _score_one(health) < _score_one(other)


def test_health_framing_decay_scoped_to_narrative_collections():
    """A factual wiki article mentioning illness is NOT user state and must not
    be penalized — collection scoping keeps the penalty off reference content."""
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    text = "Post viral fatigue syndrome is managed with rest and nutrition"
    wiki = _score_one(_mem(text, "wiki_knowledge", H + 400))     # not in scoped set
    convo = _score_one(_mem(text, "conversations", H + 400))     # scoped → penalized
    assert convo < wiki


def test_durable_condition_narrative_not_penalized():
    """Chronic/permanent condition language never ages out, even old and in a
    scoped collection (a disability is not an illness episode)."""
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    durable = _mem("Luke has a chronic autoimmune condition he manages daily",
                   "obsidian_notes", H + 400)
    transient = _mem("Luke is recovering from illness and feeling wiped",
                     "obsidian_notes", H + 400)
    assert _score_one(durable) > _score_one(transient)


def test_health_framing_decay_respects_disabled_flag(monkeypatch):
    import memory.memory_scorer as ms
    from config.app_config import PROFILE_HEALTH_TRANSIENT_TTL_HOURS as H
    stale = _mem("still post-viral and dragging", "conversations", H + 400)
    monkeypatch.setattr(ms, "HEALTH_FRAMING_DECAY_ENABLED", False)
    disabled_score = ms.MemoryScorer().rank_memories([dict(stale)], "q")[0]["final_score"]
    monkeypatch.setattr(ms, "HEALTH_FRAMING_DECAY_ENABLED", True)
    enabled_score = ms.MemoryScorer().rank_memories([dict(stale)], "q")[0]["final_score"]
    assert disabled_score > enabled_score
