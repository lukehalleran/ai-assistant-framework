"""2026-08-29 music-session misfires (13:05–13:09, live-turn reproductions).

Three agentic loops on emotional turns in one session:

Turn 1 (13:05, 151s): a lyrics paste ran the memory loop — Tier 2 matched
graph entities {'tie','casey','bed','atlanta'} and the "recall signal" was a
'?' INSIDE the pasted lyrics ("am I just beaten so ?"). Fixes: Tier-2 arm
capped at TIER2_ENTITY_MAX_WORDS, and matched entities must appear TitleCase
in the raw text (or be multi-word ids) — _entity_mention_is_proper.

Turn 2 (13:07, 44s): Tier 2 fired on entity {'normal'} — a generic-word
graph node — plus the '?' in "more reactive then normal?". Killed by the
proper-mention filter.

Turn 3 (13:09, the reported error): every deterministic layer said no
(Tier 2 skipped, heuristic conf 0.00), then the Tier-4 LLM trigger read
"potentially useful info for Monday" literally and searched
"expected events for Monday August 31 2026" etc. — three Tavily calls that
returned Keene-NH construction news. Fix: _terms_are_temporal_generic —
when EVERY proposed term reduces to time tokens + generic filler, the
trigger is suppressed (plus a prompt rule: the user's own weekday/date
references are not news topics).
"""

import pytest

from core.agentic.gate import (
    TIER2_ENTITY_MAX_WORDS,
    _entity_mention_is_proper,
    _terms_are_temporal_generic,
)

LIVE_TURN2 = (
    "Thank you for pointing out. I think that means I must be more reactive "
    "then normal? These are not specific songs I am seeking out, but random "
    "ones that come on when I let my music play"
)

LIVE_TURN3_TERMS = [
    "expected events for Monday August 31 2026",
    "useful information for Monday August 31 2026",
    "news updates for Monday August 31 2026",
]


class TestTier2ProperMentionFilter:
    def test_generic_word_entities_filtered(self):
        # the live anchors: lowercase generic words in ordinary prose
        assert _entity_mention_is_proper(LIVE_TURN2, "normal") is False
        lyrics = "Tie the noose high-beaming, when I hung from ceilings"
        # "Tie" IS TitleCase at line start in the lyrics — that one passes
        # the mention test; the length cap is what excludes lyric pastes.
        assert _entity_mention_is_proper("i was in bed all day", "bed") is False
        assert _entity_mention_is_proper("tie my shoes", "tie") is False

    def test_proper_names_pass(self):
        text = "this song is making me think of when I moved to Atlanta with Casey"
        assert _entity_mention_is_proper(text, "casey") is True
        assert _entity_mention_is_proper(text, "atlanta") is True

    def test_multiword_entities_always_pass(self):
        assert _entity_mention_is_proper("the heat exchanger project",
                                         "phase_change_heat_exchanger_project") is True

    def test_lowercase_name_underfires(self):
        # deliberate under-fire: lowercase-typed names don't anchor Tier 2
        # (genuine recall queries carry Tier-1 keywords instead)
        assert _entity_mention_is_proper("do you remember casey", "casey") is False


class TestTier2LengthCap:
    def test_lyrics_paste_exceeds_cap(self):
        # the live turn-1 paste is far beyond the Tier-2 word cap
        paste = " ".join(["lyric"] * 400) + " am I just beaten so ?"
        assert len(paste.split()) > TIER2_ENTITY_MAX_WORDS

    def test_normal_recall_query_within_cap(self):
        q = "Was it Atlanta where I lived with Casey back then?"
        assert len(q.split()) <= TIER2_ENTITY_MAX_WORDS


class TestTemporalGenericTermGuard:
    def test_live_terms_suppressed(self):
        assert _terms_are_temporal_generic(LIVE_TURN3_TERMS) is True

    @pytest.mark.parametrize("terms", [
        ["Georgia Tech drop date August 2026"],          # content: georgia/tech/drop/date
        ["weather Schaumburg IL Monday"],                # content: weather/schaumburg/il
        ["news updates for Monday", "MGT 6203 syllabus"],  # ONE real term rescues
        ["kavarin itch benadryl interaction"],
    ])
    def test_real_topics_never_suppressed(self, terms):
        assert _terms_are_temporal_generic(terms) is False

    def test_empty_terms_not_suppressed(self):
        # no terms is the no-op case — other layers decide
        assert _terms_are_temporal_generic([]) is False

    @pytest.mark.parametrize("terms", [
        ["things happening this weekend"],
        ["upcoming events for tomorrow"],
        ["general news updates today"],
    ])
    def test_other_temporal_generic_shapes_suppressed(self, terms):
        assert _terms_are_temporal_generic(terms) is True


class TestTriggerPromptRule:
    def test_prompt_carries_personal_schedule_rule(self):
        import inspect
        import utils.web_search_trigger as wst
        src = inspect.getsource(wst)
        assert "user's Monday" in src
