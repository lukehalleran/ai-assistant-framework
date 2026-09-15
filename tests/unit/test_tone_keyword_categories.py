"""
H02a: MEDIUM_CRISIS_KEYWORDS / CONCERN_KEYWORDS -> MEDIUM_KEYWORD_CATEGORIES /
CONCERN_KEYWORD_CATEGORIES, dict[str, dict[str, frozenset[str]]] keyed
"affect"/"strain"/"stressor_topic" (H02b: affect/strain = strain evidence,
stressor_topic = domain anchors). BC-76: flattened sets + matchers stay
equal -- digest pin below, table in batches/H02a.md.
`import utils.tone_detector as td`, not `from ... import`: the category
names don't exist pre-edit, so a `from` import fails at COLLECTION time,
erroring every test. Deferred attribute access means only the tests
needing the new names fail pre-edit.
"""

import hashlib

import utils.tone_detector as td
from utils.trigger_match import compile_keyword_matcher

# Digest pin (BC-76): pins pre-H02a MEDIUM_CRISIS_KEYWORDS/CONCERN_KEYWORDS
# (len + sha256 of sorted-joined set); update only on a deliberate vocabulary
# change. Pinned 2026-09-14, run in H02a.md "Digest pin".
_DIGEST_CASES = (
    ("MEDIUM_CRISIS_KEYWORDS", 106, "f3e2e6d7c9d9171095664cb6a4edcdc146e372e145aa6255deb917d729112bc4"),
    ("CONCERN_KEYWORDS", 130, "f040e86a0a50e3aae05118681a77efdb63acb08b30f35c2e98c708e0329b07a1"),
)


def test_flattened_sets_unchanged():
    # Compute both actuals before asserting, so a failure prints BOTH.
    actual = {}
    for name, _len, _digest in _DIGEST_CASES:
        keyword_set = getattr(td, name)
        joined = "\n".join(sorted(keyword_set))
        actual[name] = (len(keyword_set), hashlib.sha256(joined.encode("utf-8")).hexdigest())
    expected = {name: (length, digest) for name, length, digest in _DIGEST_CASES}
    assert actual == expected, f"actual={actual!r} expected={expected!r}"


# Category structure tests must FAIL on the unedited source (see H02a.md).
_LEVELS = (
    ("MEDIUM", "MEDIUM_KEYWORD_CATEGORIES", "MEDIUM_CRISIS_KEYWORDS", "_MEDIUM_MATCHER"),
    ("CONCERN", "CONCERN_KEYWORD_CATEGORIES", "CONCERN_KEYWORDS", "_CONCERN_MATCHER"),
)
_MEDIUM_TOPIC_SPLIT = frozenset({"divorce", "breakup", "insomnia"})
_CONCERN_TOPIC_SPLIT = frozenset({
    "bills", "debt", "broke", "deadline", "pressure", "money problems",
    "financial stress", "work stress", "job stress", "school stress",
    "behind on bills", "behind on rent", "drowning in debt", "no days off",
})


def test_categories_flatten_to_sets():
    """Per level: union of every subgroup == the flattened keyword set."""
    for level_name, categories_attr, flat_attr, _m in _LEVELS:
        categories = getattr(td, categories_attr)
        flat = getattr(td, flat_attr)
        union = set().union(*(e for sg in categories.values() for e in sg.values()))
        assert union == flat, f"{level_name}: union != {flat_attr} (diff: {union ^ flat})"


def test_subgroups_disjoint_and_nonempty():
    """Keys affect/strain/stressor_topic; subgroups non-empty; no entry in two (dups keep ONE placement, H02a.md)."""
    for level_name, categories_attr, _f, _m in _LEVELS:
        categories = getattr(td, categories_attr)
        assert set(categories.keys()) == {"affect", "strain", "stressor_topic"}, level_name
        placement: dict[str, str] = {}
        for top_key, subgroups in categories.items():
            assert subgroups, f"{level_name}.{top_key} has no subgroups"
            for sub_key, entries in subgroups.items():
                assert entries, f"{level_name}.{top_key}.{sub_key} is empty"
                for entry in entries:
                    assert entry not in placement, f"{level_name}: {entry!r} in both {placement.get(entry)} and {top_key}.{sub_key}"
                    placement[entry] = f"{top_key}.{sub_key}"


def test_stressor_topic_matches_design_split():
    """stressor_topic union per level == the FIXED round-3 probe split: CONCERN 14, MEDIUM 3."""
    medium_topic = set().union(*td.MEDIUM_KEYWORD_CATEGORIES["stressor_topic"].values())
    assert medium_topic == _MEDIUM_TOPIC_SPLIT
    concern_topic = set().union(*td.CONCERN_KEYWORD_CATEGORIES["stressor_topic"].values())
    assert concern_topic == _CONCERN_TOPIC_SPLIT


# Non-trigger controls: plain narrative sharing no substring with any keyword.
_FILLER = (
    "We spent the afternoon watching clouds drift over the harbor",
    "Then we baked a loaf of bread from an old recipe and read on the porch",
)


def _corpus_for(categories):
    """Clean + wrapped/indented (BC-64) text: one entry per subgroup (BC-64
    wraps fall only BETWEEN sentences) plus filler."""
    picks = [sorted(e)[0] for sg in categories.values() for e in sg.values()]
    sentences = [_FILLER[0], *picks, _FILLER[1]]
    return ". ".join(sentences) + ".", "\n    ".join(sentences) + ".", picks


def test_matchers_equivalent():
    """Live matchers hit the same (keyword, start) pairs as a matcher built
    fresh from sorted(<flattened set>), on clean + wrapped (BC-64) text."""
    control_text = ". ".join(_FILLER) + "."
    for level_name, categories_attr, flat_attr, matcher_attr in _LEVELS:
        categories = getattr(td, categories_attr)
        live_matcher = getattr(td, matcher_attr)
        fresh_matcher = compile_keyword_matcher(sorted(getattr(td, flat_attr)))

        clean, wrapped, picks = _corpus_for(categories)
        for label, text in (("clean", clean), ("wrapped", wrapped)):
            lower = text.lower()
            live_hits = {(h.keyword, h.start) for h in live_matcher.iter_hits(lower)}
            fresh_hits = {(h.keyword, h.start) for h in fresh_matcher.iter_hits(lower)}
            assert live_hits == fresh_hits, f"{level_name} {label}: live={live_hits} fresh={fresh_hits}"
            missing = set(picks) - {kw for kw, _ in live_hits}
            assert not missing, f"{level_name} {label}: subgroup reps not matched: {missing}"

        control_hits = list(live_matcher.iter_hits(control_text.lower()))
        assert control_hits == [], f"{level_name}: control text matched {control_hits}"
