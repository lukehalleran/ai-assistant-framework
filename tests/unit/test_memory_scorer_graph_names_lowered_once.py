"""MemoryScorer.rank_memories lowercases graph-related names once.

2026-10-17, class BC-41.  Live: one 9572 ms stall at the graph-proximity
genexpr (`name.lower() in blob_lower` -- every name lowered once per memory).
Drives THE deployed `rank_memories`; the graph seams are stubbed.  Scores must
be unchanged (golden deltas recorded from the base tree).
"""
from unittest.mock import MagicMock, patch

from memory.memory_scorer import MemoryScorer


class _CountingName(str):
    lower_calls = 0

    def lower(self):
        type(self).lower_calls += 1
        return super().lower()


NAMES = {_CountingName(n) for n in ("Rowan", "Tamsin", "Golden Retriever", "Quasar Lab")}


def _mem(text):
    return {
        "content": text,
        "relevance_score": 0.5,
        "truth_score": 0.9,
        "timestamp": "2026-10-16T10:00:00",
        "collection": "conversations",
        "metadata": {},
    }


def _rank(memories):
    scorer = MemoryScorer()
    scorer._graph_memory = MagicMock()
    scorer._graph_memory.node_count.return_value = 5
    scorer._entity_resolver = MagicMock()
    with patch("config.app_config.KNOWLEDGE_GRAPH_ENABLED", True), \
         patch("config.app_config.GRAPH_SCORING_BOOST_ENABLED", True), \
         patch("config.app_config.GRAPH_SCORING_BOOST_CAP", 0.15), \
         patch("memory.graph_utils.extract_graph_entities", return_value={"x"}), \
         patch("memory.graph_utils.get_related_display_names", return_value=set(NAMES)):
        return scorer.rank_memories(memories, current_query="tell me about the lab")


def _texts():
    return [
        "Rowan visited the Quasar Lab with Tamsin",   # 3 matches
        "The weather was nice today",                  # 0
        "A Golden Retriever near the lab",            # 1
    ] + [f"unrelated filler number {i}" for i in range(20)]


def test_names_are_lowered_once_not_per_memory():
    _CountingName.lower_calls = 0
    _rank([_mem(t) for t in _texts()])
    assert _CountingName.lower_calls <= len(NAMES), _CountingName.lower_calls


def test_graph_bonus_scores_unchanged():
    mems = [_mem(t) for t in _texts()]
    ranked = _rank(mems)
    by_text = {m["content"]: m for m in ranked}
    base = by_text["The weather was nice today"]["final_score"]
    three = by_text["Rowan visited the Quasar Lab with Tamsin"]["final_score"]
    one = by_text["A Golden Retriever near the lab"]["final_score"]
    # Golden deltas recorded from the base implementation (89dbbc5) on this fixture:
    # 3 matches -> 0.1575, 1 match -> 0.0575 over the 0-match memory.
    assert abs((three - base) - 0.1575) < 1e-4
    assert abs((one - base) - 0.0575) < 1e-4
