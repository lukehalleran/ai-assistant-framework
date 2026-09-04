"""2026-09-03 cat-fetch session review — five defects from one four-turn dump.

Turn 4 ("Biscuit really likes chasing the ball too…") was answered as
"the cat fetches and the dog just runs track". Every signal that carried the
wrong species is pinned here:

1. [KNOWLEDGE GRAPH] rendered ``user|has_dog|biscuit`` / ``…|mochi`` — edges
   from the 2026-08-18 shutdown LLM run that pre-date the ingestion-time
   species guard, on nodes whose curated metadata says ``species: cat``.
   Fix: GraphMemory.edge_is_suppressed() at every read site.
2. The agentic gate ran an 18s memory loop: graph entity "Biscuit" + a bare
   ``what`` inside "doesn't know what to do" counted as a recall cue.
   Fix: bare interrogatives only count where they open a clause.
3. The planner turned a "Dog Behavior" topic label into "common in dogs".
   Fix: entity-attribute rule in the prompt + derived-section labels in the
   digest.
4. Song lyrics pasted on 08-29 minted lived_in=Atlanta + a partner name.
   Fix: lyrics/poem/quote turns yield no facts on either extractor path.
5. A pasted Aug-27 email ("I'm … enrolled in two courses") superseded the
   curated enrolled_in=MGT 6203 fact on Sep 2. Fix: pasted correspondence is
   never a claim span; the regex path sees the message with the block removed.
"""
from datetime import datetime

import pytest

from core.agentic.gate import _RECALL_SIGNAL_HIT, _recall_signal_hit
from core.response_planner import ResponsePlanner
from memory.fact_source import (
    _claim_spans,
    find_supporting_user_span,
    quoted_correspondence_lines,
    strip_quoted_correspondence,
)
from memory.graph_memory import GraphMemory
from memory.graph_models import GraphEdge, GraphNode
from memory.memory_storage import fact_extraction_skip_reason, fact_extraction_source_text
from memory.llm_fact_extractor import _entry_is_shared_content

LIVE_TURN4 = (
    "Biscuit really likes chasing the ball too. But mostly chasing it. He has "
    "picked it up a few times, but I believe he didn't plan that far and doesn't "
    "know what to do at that point and runs away"
)


# ── 1. graph read-time suppression ────────────────────────────────────────
def _graph_with_pets(tmp_path):
    gm = GraphMemory(persist_path=str(tmp_path / "g.json"))
    gm.add_entity(GraphNode(entity_id="user", display_name="User", entity_type="person"))
    gm.add_entity(GraphNode(entity_id="mochi", display_name="Mochi", entity_type="animal",
                            metadata={"species": "cat", "owner": "mom"}))
    gm.add_entity(GraphNode(entity_id="biscuit", display_name="Biscuit", entity_type="animal",
                            metadata={"species": "cat, black, big golden eyes"}))
    gm.add_entity(GraphNode(entity_id="daisy", display_name="Daisy", entity_type="animal",
                            metadata={"species": "cat"}))
    gm.add_entity(GraphNode(entity_id="rex", display_name="Rex", entity_type="animal"))
    now = datetime.now()
    # NetworkX DiGraph keeps ONE relation per node pair (last add wins), so
    # every relation here targets a distinct node.
    for rel, tgt in (("has_dog", "mochi"), ("has_dog", "biscuit"), ("has_cat", "daisy"),
                     ("has_dog", "rex")):
        gm.add_relation(GraphEdge(source_id="user", relation=rel, target_id=tgt, weight=1.0,
                                  first_seen=now, last_seen=now))
    return gm


class TestGraphSpeciesSuppression:
    def test_has_dog_on_curated_cat_is_suppressed(self, tmp_path):
        gm = _graph_with_pets(tmp_path)
        edges = {e.edge_key(): e for e in gm.get_relations("user", direction="out")}
        assert gm.edge_is_suppressed(edges["user|has_dog|mochi"])
        assert gm.edge_is_suppressed(edges["user|has_dog|biscuit"])
        assert not gm.edge_is_suppressed(edges["user|has_cat|daisy"])
        # no species metadata → never blocked (under-fires by design)
        assert not gm.edge_is_suppressed(edges["user|has_dog|rex"])

    def test_context_sentences_never_render_the_dog(self, tmp_path):
        gm = _graph_with_pets(tmp_path)
        text = " ".join(gm.get_context_sentences("user", depth=1)).lower()
        assert "dog mochi" not in text and "dog biscuit" not in text
        assert "cat daisy" in text
        assert "dog rex" in text

    def test_quarantine_flag_suppresses_any_edge(self, tmp_path):
        gm = _graph_with_pets(tmp_path)
        edge = {e.edge_key(): e for e in gm.get_relations("user", direction="out")}["user|has_dog|rex"]
        edge.metadata = {"curation_quarantined": True}
        assert gm.edge_is_suppressed(edge)
        assert "rex" not in " ".join(gm.get_context_sentences("user", depth=1)).lower()

    def test_surfacer_drops_suppressed_edges(self, tmp_path):
        from memory.context_surfacer import ContextSurfacer
        gm = _graph_with_pets(tmp_path)
        surfacer = ContextSurfacer.__new__(ContextSurfacer)
        surfacer._graph_memory = gm
        live = surfacer._live_edges(gm.get_relations("user", direction="out"))
        rels = {(e.relation, e.target_id) for e in live}
        assert ("has_dog", "mochi") not in rels
        assert ("has_dog", "biscuit") not in rels
        assert ("has_cat", "daisy") in rels


# ── 2. gate recall cue ────────────────────────────────────────────────────
class TestRecallCueClausePosition:
    def test_live_turn4_no_recall_signal(self):
        assert _RECALL_SIGNAL_HIT(LIVE_TURN4.lower()) is False

    @pytest.mark.parametrize("q", [
        "he doesn't know what to do with the ball",
        "i wonder how he learned that",
        "not sure when that started honestly",
        "she told me where she parked",
        "biscuit is the one who bites butts",
        "that is why mochi cries",
    ])
    def test_embedded_interrogatives_do_not_hit(self, q):
        assert _recall_signal_hit(q) is False

    @pytest.mark.parametrize("q", [
        "what did biscuit do last week",
        "how did mochi react when biscuit was at the vet",
        "ok. when did i last mention daisy",
        "so what happened with the vet bill",
        "remember when biscuit got poisoned",
        "tell me anything about my sleep patterns",
        "remind me what the deadline was",
    ])
    def test_clause_opening_and_phrase_cues_hit(self, q):
        assert _recall_signal_hit(q) is True

    @pytest.mark.parametrize("q", [
        "ok. i am in bathroom with shower running. i will login paste in syllabus here shortly.",
        "the showerhead is broken",
        "i was somewhat tired after the gym",
        "that got me nowhere yesterday",
    ])
    def test_substring_traps_still_do_not_hit(self, q):
        assert _recall_signal_hit(q) is False

    @pytest.mark.asyncio
    async def test_gate_does_not_route_turn4_to_memory(self):
        from unittest.mock import MagicMock, patch
        from core.agentic.gate import evaluate_agentic_gate
        with patch("memory.graph_utils.extract_graph_entities", return_value={"biscuit"}):
            d = await evaluate_agentic_gate(LIVE_TURN4, entity_resolver=MagicMock())
        assert "memory" not in d.modes


# ── 3. planner ────────────────────────────────────────────────────────────
class TestPlannerEntityAttributes:
    def test_digest_labels_derived_sections(self):
        digest, included = ResponsePlanner.build_context_digest({
            "memories": [{"content": "Biscuit is your black cat"}],
            "graph_context": ["User has dog Biscuit (from relationship data)"],
        })
        assert "[memories]" in digest
        assert "[graph_context — derived, not the user's words]" in digest
        assert included == ["memories", "graph_context"]

    def test_prompt_carries_entity_attribute_rule(self):
        import inspect
        src = inspect.getsource(ResponsePlanner)
        assert "Never assign or infer a species, gender, role, relationship" in src
        assert "wins over the Topics label" in src


# ── 4. lyrics never yield facts ───────────────────────────────────────────
LYRICS = "\n".join([
    "[Verse 1]",
    "Lately, I think I was over Time am I just beaten so",
    "Like the clouds, see color, I moved to Atlanta with Casey",
    "[Chorus]",
    "And I live in Atlanta now, oh Atlanta now",
    "And I live in Atlanta now, oh Atlanta now",
])


class TestSharedContentYieldsNoFacts:
    def test_per_turn_skip_reason(self):
        assert fact_extraction_skip_reason(LYRICS).startswith("shared content (lyrics")
        assert fact_extraction_skip_reason("I moved to Atlanta with Casey last week.") == ""

    def test_llm_path_drops_lyrics_entries(self):
        assert _entry_is_shared_content("User: " + LYRICS) is True
        assert _entry_is_shared_content("User: Started Zelphex today, 5 mg") is False

    def test_llm_prompt_omits_lyrics_turn(self):
        from memory.llm_fact_extractor import LLMFactExtractor
        ex = LLMFactExtractor.__new__(LLMFactExtractor)
        ex.max_input_chars = 9000
        prompt = ex._build_prompt([
            {"query": "Mochi stole my phone again lol"},
            {"query": LYRICS},
        ])
        assert "Mochi stole my phone" in prompt
        assert "I live in Atlanta now" not in prompt


# ── 5. pasted correspondence is not a claim ───────────────────────────────
PASTED_EMAIL_TURN = "\n".join([
    "Ok on a roll so might as well do this one",
    "Hi Morgan and Robin,",
    "",
    "I'm doing meaningfully better and am enrolled in two courses this fall (CSE 6040 and MGT 6203).",
    "The drop deadline is tomorrow.",
    "",
    "Thank you so much.",
    "",
    "Alex Rivers",
    "GTID: 000000000",
    "",
    "Hi Alex,",
    "",
    "I do not recommend enrolling in two courses this fall.",
    "",
    "Best,",
    "",
    "Morgan Reeves",
    "Academic Advising Manager, College of Continuing Studies",
    "Springfield Institute of Technology",
    "",
    "need to make sure I'm not missing anything, I dropped CSE 6040 and kept MGT 6203",
])


class TestPastedCorrespondenceNotEvidence:
    def test_blocks_cover_both_emails_and_signatures(self):
        lines = PASTED_EMAIL_TURN.splitlines()
        inside = quoted_correspondence_lines(PASTED_EMAIL_TURN)
        covered = [lines[i] for i in sorted(inside)]
        assert any("enrolled in two courses" in ln for ln in covered)
        assert any("I do not recommend" in ln for ln in covered)
        assert "Springfield Institute of Technology" in covered
        assert lines[0] not in covered
        assert lines[-1] not in covered

    def test_bare_chat_greeting_is_not_a_block(self):
        assert quoted_correspondence_lines("Hi,\nI moved to Atlanta in June.") == set()
        assert strip_quoted_correspondence("Hi,\nI moved to Atlanta in June.") == "Hi,\nI moved to Atlanta in June."

    def test_claim_spans_skip_the_quoted_enrollment(self):
        spans = list(_claim_spans(PASTED_EMAIL_TURN))
        assert not any("enrolled in two courses" in s for s in spans)
        assert any("dropped CSE 6040" in s for s in spans)

    def test_stale_enrollment_triple_finds_no_support(self):
        triple = {"subject": "user", "relation": "enrolled_in", "object": "CSE 6040 and MGT 6203"}
        assert find_supporting_user_span(triple, [PASTED_EMAIL_TURN]) is None
        live = {"subject": "user", "relation": "dropped", "object": "CSE 6040"}
        span = find_supporting_user_span(live, [PASTED_EMAIL_TURN])
        assert span is not None and "dropped CSE 6040" in span.text

    def test_regex_path_source_text_has_block_removed(self):
        src = fact_extraction_source_text(PASTED_EMAIL_TURN)
        assert "enrolled in two courses" not in src
        assert "dropped CSE 6040" in src
