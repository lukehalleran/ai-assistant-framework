"""2026-08-28 memory-ingest turn audit: a "DAEMON MEMORY INGEST" status paste
(crisis-resolution dump + forwarded advisor email + one-line check-in) misfired
two systems the day after the 08-27 paste-turn fixes shipped:

1. Agentic gate — COMPUTATION keyword 'solve' matched as a SUBSTRING of
   "resolution"/"unresolved" (same class as 'document'⊂"documented"), and both
   email-action arms fired on the advisor email addresses in the forwarded
   SIGNATURE plus incidental "email"/"write" narration → modes
   [computation, tools] → 49s agentic loop on a turn that requested nothing.
2. Visual gate — the 08-27 "visual nouns suffice at any length" rule was the
   new hole: "Screenshot saved." (narration of an OSCAR drop confirmation) and
   a literal "image" placeholder from the pasted email client fired the gate;
   the generic auto-learned alias "project" (bound to
   phase_change_heat_exchanger_project via a possessive mention) resolved from
   "group project" in the paste; the node's handwritten-notes photo was
   attached to the final synthesis and narrated ("those SVM notes are real
   work...") into a crisis-logistics reply.

Fixes under test:
- gate.py: _compile_keyword_matcher (left word boundary for bare-word
  keywords), email arms gated on EMAIL_ACTION_MAX_WORDS or a head-anchored
  send imperative (_EMAIL_COMMAND_RE).
- gatherer_knowledge.py: visual nouns sufficient only ≤ _VISUAL_INTENT_MAX_WORDS;
  long messages need a short request-shaped line naming imagery
  (_long_message_visual_request).
- graph_utils.py: generic admin nouns ("project", "notes", "email", ...)
  added to the single-word entity stoplist.
"""

import pytest

from core.agentic.gate import (
    _COMPUTATION_HIT,
    _EMAIL_COMMAND_RE,
    _TOOL_HIT,
    evaluate_agentic_gate,
)
from core.prompt.gatherer_knowledge import (
    _long_message_visual_request,
    _query_wants_visual,
)
from memory.graph_utils import extract_graph_entities


# Abridged but structurally faithful reproduction of the live turn: section
# headers, "Screenshot saved." narration, forwarded email with real-looking
# addresses and a signature, an inline "image" placeholder from the email
# client, "group project", "Emailed re:", "write that this issue", and a
# terse first-person tail. No question marks, no send imperative.
INGEST_PASTE = (
    "DAEMON MEMORY INGEST — 2026-08-28 — OMSA/Fall 2026 crisis resolution\n\n"
    "STATUS: Fall 2026 enrollment finalized. Crisis-mode decisions complete.\n\n"
    "DECISIONS MADE\n"
    "- Dropped CSE 6040 (clean drop, no W, no charge) via OSCAR 2026-08-28. "
    "Screenshot saved.\n"
    "- Enrolled in MGT 6203 only (3 hrs, part-time). Per advisor recommendation.\n\n"
    "KEY FACTS\n"
    "- Advisor: Morgan Reeves, Academic Advising Manager "
    "(morgan.reeves@lifetimelearning.gatech.edu). Authoritative on deadlines.\n"
    "- Incomplete from last term: unresolved. AWAITING REPLY.\n"
    "- MGT 6203: light course. Check syllabus for group project and the "
    "percent of grade after Oct 31.\n\n"
    "FINANCIAL\n"
    "- Dean of Students contact: sheree.gibson@gatech.edu, CC "
    "april.glover@gatech.edu. Emailed re: retro medical W and petition "
    "standards.\n\n"
    "HEALTH\n"
    "- New provider appointment Monday. My provider will likely be willing to "
    "write that this issue was caused by improper prescribing.\n\n"
    "I received this from Morgan. Hi Luke, I would like to correct your "
    "understanding of the registration deadline for Fall 2026. Best, Morgan "
    "Reeves, Georgia Institute of Technology, "
    "morgan.reeves@lifetimelearning.gatech.edu\n"
    "image\n"
    "Book time to meet with me\n"
    "I woke up at maybe 1015 and took 5 mg extra Dexivar today. I dropped cse "
    "6040 before the deadline so that's done. Ugh"
)


# ===========================================================================
# 1. Keyword word-bounding
# ===========================================================================

class TestKeywordWordBounding:

    def test_solve_does_not_match_resolution(self):
        assert not _COMPUTATION_HIT("omsa fall 2026 crisis resolution")
        assert not _COMPUTATION_HIT("the incomplete from last term is unresolved")
        assert not _COMPUTATION_HIT("we reached a resolution yesterday")

    def test_solve_still_matches_real_requests(self):
        assert _COMPUTATION_HIT("solve this equation for x")
        assert _COMPUTATION_HIT("can you solve the integral")
        # Left-open boundary keeps inflections.
        assert _COMPUTATION_HIT("solving the system of equations")

    def test_other_left_boundary_cases(self):
        # 'actions' must not match "transactions"; 'integrate' not "reintegrated".
        assert not _TOOL_HIT("my bank transactions look weird")
        assert not _COMPUTATION_HIT("he reintegrated into the group")
        assert _TOOL_HIT("check the github actions runs")

    def test_phrase_keywords_keep_substring_semantics(self):
        from core.agentic.gate import _WEB_SEARCH_HIT
        assert _WEB_SEARCH_HIT("go to https://example.com and check")
        assert _WEB_SEARCH_HIT("do a web search for the deadline")


# ===========================================================================
# 2. Email-action arms
# ===========================================================================

class TestEmailArmGating:

    def test_head_anchored_send_command_matches(self):
        assert _EMAIL_COMMAND_RE.search("Send this to morgan@gatech.edu: draft below")
        assert _EMAIL_COMMAND_RE.search("ok, email Meagan the update")
        assert _EMAIL_COMMAND_RE.search("please draft an email to the bursar")

    def test_mid_paste_verbs_do_not_match(self):
        assert not _EMAIL_COMMAND_RE.search(
            "I received this from Morgan. She said to write back soon."
        )
        assert not _EMAIL_COMMAND_RE.search(
            "Emailed re: retro medical W and petition standards."
        )

    @pytest.mark.asyncio
    async def test_short_email_command_still_routes_to_tools(self):
        d = await evaluate_agentic_gate(user_text="email Morgan the update about the drop")
        assert d.should_trigger is True
        assert "tools" in (d.modes or [])

    @pytest.mark.asyncio
    async def test_long_anchored_draft_still_routes_to_tools(self):
        d = await evaluate_agentic_gate(
            user_text="Send this to morgan.reeves@lifetimelearning.gatech.edu: "
                      + ("thank you for your guidance " * 20)
        )
        assert d.should_trigger is True
        assert "tools" in (d.modes or [])


# ===========================================================================
# 3. Live-turn gate reproduction
# ===========================================================================

class TestIngestGateReproduction:

    @pytest.mark.asyncio
    async def test_ingest_paste_does_not_trigger(self):
        """The live turn: signature addresses + 'resolution' + narration must
        not route a memory-ingest status dump into the tool loop."""
        d = await evaluate_agentic_gate(user_text=INGEST_PASTE)
        assert d.should_trigger is False
        assert "computation" not in (d.modes or [])
        assert "tools" not in (d.modes or [])


# ===========================================================================
# 4. Visual gate — narration nouns in long pastes
# ===========================================================================

class TestVisualNarrationNouns:

    def test_ingest_paste_blocked(self):
        assert not _query_wants_visual(INGEST_PASTE, None)
        assert not _query_wants_visual(INGEST_PASTE, "factual_recall")

    def test_screenshot_saved_narration_is_not_a_request(self):
        assert not _long_message_visual_request(
            "lots of words here. Screenshot saved. more words follow"
        )

    def test_appended_request_line_passes(self):
        assert _query_wants_visual(INGEST_PASTE + "\nShow me the screenshot", None)
        assert _query_wants_visual(
            INGEST_PASTE + "\ncan I see that photo again?", None
        )

    def test_short_noun_messages_unchanged(self):
        assert _query_wants_visual("pull up the screenshot", None)
        assert _query_wants_visual("do you still have that photo of Mochi", None)


# ===========================================================================
# 5. Generic-alias single-word stoplist
# ===========================================================================

class _FakeResolver:
    """Resolver with a junk generic alias, mirroring the live graph state
    (possessive alias "project" → phase_change_heat_exchanger_project)."""

    def __init__(self):
        self.aliases = {
            "project": "phase_change_heat_exchanger_project",
            "notes": "phase_change_heat_exchanger_project",
            "morgan": "morgan_reeves",
            "phase change project": "phase_change_heat_exchanger_project",
        }

    def resolve(self, mention):
        return self.aliases.get(mention)


class TestGenericAliasStoplist:

    def test_single_word_project_does_not_resolve(self):
        ents = extract_graph_entities(
            "check syllabus for group project details", _FakeResolver()
        )
        assert "phase_change_heat_exchanger_project" not in ents

    def test_named_entities_still_resolve(self):
        ents = extract_graph_entities(
            "did Morgan reply about the deadline", _FakeResolver()
        )
        assert "morgan_reeves" in ents

    def test_multiword_mention_still_resolves(self):
        # The stoplist only guards SINGLE-word matches — a real multi-word
        # mention of the entity still resolves via the n-gram pass.
        ents = extract_graph_entities(
            "how is the phase change project going", _FakeResolver()
        )
        assert "phase_change_heat_exchanger_project" in ents


# ===========================================================================
# 6. Generic-word alias binding guard (write side + load-time neutralization)
# ===========================================================================

class TestGenericAliasBindingGuard:

    def _graph(self, tmp_path):
        from memory.graph_memory import GraphMemory
        return GraphMemory(persist_path=str(tmp_path / "kg.json"))

    def test_register_alias_refuses_generic_word(self, tmp_path):
        from memory.graph_models import GraphNode
        g = self._graph(tmp_path)
        g.add_entity(GraphNode(entity_id="phase_change_heat_exchanger_project",
                               display_name="Phase Change Project",
                               entity_type="other"))
        g.register_alias("project", "phase_change_heat_exchanger_project")
        assert g.resolve_entity("project") is None

    def test_multiword_possessive_alias_still_binds(self, tmp_path):
        from memory.graph_models import GraphNode
        g = self._graph(tmp_path)
        g.add_entity(GraphNode(entity_id="phase_change_heat_exchanger_project",
                               display_name="Phase Change Project",
                               entity_type="other"))
        g.register_alias("my project", "phase_change_heat_exchanger_project")
        assert g.resolve_entity("my project") == "phase_change_heat_exchanger_project"

    def test_named_alias_still_binds(self, tmp_path):
        from memory.graph_models import GraphNode
        g = self._graph(tmp_path)
        g.add_entity(GraphNode(entity_id="morgan_reeves",
                               display_name="Morgan Reeves",
                               entity_type="person"))
        g.register_alias("morgan", "morgan_reeves")
        assert g.resolve_entity("morgan") == "morgan_reeves"

    def test_add_entity_drops_generic_alias(self, tmp_path):
        from memory.graph_models import GraphNode
        g = self._graph(tmp_path)
        g.add_entity(GraphNode(entity_id="gou", display_name="gòu",
                               entity_type="other", aliases=["meeting", "gou zhang"]))
        assert g.resolve_entity("meeting") is None
        assert g.resolve_entity("gou zhang") == "gou"

    def test_load_neutralizes_historical_junk_alias(self, tmp_path):
        """Historical node data carrying a generic alias must not rebuild the
        binding at load — the live-graph state behind the 08-28 misfire."""
        import json
        path = tmp_path / "kg.json"
        payload = {
            "schema_version": 1,
            "nodes": {
                "phase_change_heat_exchanger_project": {
                    "display_name": "phase_change_heat_exchanger_project",
                    "entity_type": "other",
                    "aliases": ["project"],
                    "first_seen": "2026-05-07T00:00:00",
                    "last_seen": "2026-08-21T00:00:00",
                    "mention_count": 2,
                    "metadata": {},
                },
            },
            "edges": [],
        }
        path.write_text(json.dumps(payload))
        from memory.graph_memory import GraphMemory
        g = GraphMemory(persist_path=str(path))
        assert g.resolve_entity("project") is None
        # Node data itself is untouched — no silent store mutation.
        assert "project" in g.graph.nodes["phase_change_heat_exchanger_project"]["aliases"]
