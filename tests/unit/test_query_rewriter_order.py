"""
Query rewriter must preserve word order (2026-08-23).

rewrite_query round-tripped keywords through a set() before joining, so the
string fed to the (order-sensitive) bge embedder was scrambled — the same
query produced different vectors run-to-run, and multi-word phrases lost
their structure entirely. These tests drive THE DEPLOYED rewrite_query.
"""

from utils.query_rewriter import (
    NOISE_WORDS,
    SYNONYM_GROUPS,
    _ordered_keywords,
    rewrite_query,
)


class TestOrderPreservation:
    def test_original_word_order_preserved(self):
        q = "gym session before work shift tonight"
        out = rewrite_query(q, use_topic_extraction=False)
        assert out.startswith("gym session before work shift tonight")

    def test_deterministic_across_calls(self):
        q = "tired after the gym and work today"
        outs = {rewrite_query(q, use_topic_extraction=False) for _ in range(5)}
        assert len(outs) == 1

    def test_expansions_append_after_originals(self):
        q = "cat scratched the couch"
        out = rewrite_query(q, use_topic_extraction=False).split()
        originals = _ordered_keywords(q)
        assert out[: len(originals)] == originals
        # everything after the originals came from a synonym group
        all_synonyms = {s for group in SYNONYM_GROUPS.values() for s in group}
        assert all(w in all_synonyms for w in out[len(originals):])

    def test_expansion_order_deterministic_group_order(self):
        # 'gym' and 'cat' both trigger groups; expansions must follow query
        # order (gym's group before cat's), each in group-list order.
        q = "gym with cat"
        out = rewrite_query(q, use_topic_extraction=False).split()
        originals = _ordered_keywords(q)
        extras = out[len(originals):]
        gym_extras = [s for s in SYNONYM_GROUPS['gym'] if s not in originals]
        expected_head = gym_extras[: len(extras)]
        assert extras[: len(expected_head)] == expected_head


class TestExistingBehaviorKept:
    def test_noise_words_removed(self):
        out = rewrite_query("lmao that was really crazy haha",
                            use_topic_extraction=False)
        for noise in ("lmao", "really", "haha"):
            assert noise not in out.split()

    def test_expansion_cap_ten(self):
        # single-word synonym groups only ('tired' has the two-token
        # 'worn out', which would make token-counting over-report the cap)
        q = "gym work health code cat body"
        out = rewrite_query(q, use_topic_extraction=False).split()
        originals = _ordered_keywords(q)
        assert len(out) - len(originals) <= 10

    def test_short_query_passthrough(self):
        assert rewrite_query("hi", use_topic_extraction=False) == "hi"

    def test_no_duplicate_keywords(self):
        q = "gym workout at the gym"
        out = rewrite_query(q, use_topic_extraction=False).split()
        assert len(out) == len(set(out))


class TestOrderedKeywordsHelper:
    def test_first_occurrence_dedup(self):
        assert _ordered_keywords("work then gym then work") == \
            ["work", "then", "gym"]

    def test_noise_filtered(self):
        assert _ordered_keywords("just really tired") == ["tired"]

    def test_noise_words_set_unchanged(self):
        # helper and extract_keywords share NOISE_WORDS — a rename would
        # silently stop filtering
        assert "lmao" in NOISE_WORDS
