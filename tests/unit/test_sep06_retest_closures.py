"""2026-09-06 retest closures (turn 3 evidence quality), deployed functions only."""

from core.insight.sweep import exclude_assistant_directed_items, fact_triple_is_junk
from core.insight.synthesizer import _BASE_SYSTEM, _CONVERSATION_SYSTEM
from core.insight.types import EvidenceItem


def _u(text, coll="conversations", speaker="user"):
    return EvidenceItem(doc_id=text[:12], collection=coll, speaker=speaker, date="2026-09-05", text=text)


class TestAssistantDirectedExclusion:
    LIVE_REQUESTS = [
        "Give me a detailed analysis in a table of what my record can establish about medication gaps.",
        "Does my history actually support scheduling occasional rest days off the medication? Weigh both sides.",
        "…I received from them in outlook and tell me what deadlines it mentioned? I want to make sure I am not missing anything before the weekend.",
        "Ok so I guess I took meds at 9, can you search my claim and fact check that",
    ]

    def test_live_requests_dropped(self):
        kept = exclude_assistant_directed_items([_u(t) for t in self.LIVE_REQUESTS])
        assert kept == []

    def test_observations_kept(self):
        obs = [
            _u("Yeah fuck me I feel like shit. Idk. I took meds at 10 again and have been out of bed 2 hours"),
            _u("I did not take Zelphex today. I think I need to actually take a semi break today", coll="corpus"),
            _u("It seems like the ends of the days are weird. Before I take my ADHD meds and after they wear off"),
        ]
        assert exclude_assistant_directed_items(obs) == obs

    def test_non_user_items_never_touched(self):
        items = [
            _u("Can you check the log?", coll="conversations", speaker="assistant"),
            EvidenceItem(doc_id="n", collection="obsidian_notes", date="2026-08-31", text="Give me a break — note title"),
            EvidenceItem(doc_id="f", collection="facts", date="2026-08-31", text="user | wants_to | take a semi break"),
        ]
        assert exclude_assistant_directed_items(items) == items

    def test_long_request_framed_report_kept(self):
        long = "Can you help me think through this: " + "I slept badly and took my meds late again, " * 12
        assert exclude_assistant_directed_items([_u(long)]) != []


class TestFactTripleJunk:
    def test_live_junk_fact(self):
        assert fact_triple_is_junk("psychologist | is | tuesday") is True

    def test_real_fact_kept(self):
        assert fact_triple_is_junk("user | wants_to | take a semi break from Zelphex") is False
        assert fact_triple_is_junk("user | concerned_about | going more than 2 or 3 days off ADHD meds") is False

    def test_non_triple_text_not_junk(self):
        assert fact_triple_is_junk("just some prose without pipes") is False


def test_synthesis_rules_as_written_and_no_labels():
    for block in (_BASE_SYSTEM, _CONVERSATION_SYSTEM):
        low = block.lower()
        assert "as written" in low or "as\nwritten" in low
        assert "never guess" in low
        assert "label" in low and "did not use" in low


class TestSecondRetestClosures:
    """15:37 retest: 9 of 58 items rendered (one oversized item stopped both the
    layout walker and the renderer) and PubMed junk returned because the
    planner's axis keys arrived as identifiers ("rest_days") and the outcome
    axes were absent from the query terms."""

    def test_pubmed_identifier_keys_and_outcome_axes_are_mandatory(self):
        from knowledge.pubmed_search import rank_pubmed_rows, build_pubmed_query_ladder
        q = "rest days off medication effects"
        rows = [
            {"pmid": "fundo", "title": "Laparoscopic fundoplication",
             "abstract": "Visick grade IV off medication. the rest had a Nissen. days"},
            {"pmid": "holiday", "title": "Planned drug holidays",
             "abstract": "planned days off medication in adults: effects on symptoms and well-being"},
            {"pmid": "pitch", "title": "Pitcher workload and days of rest",
             "abstract": "days of rest between outings; symptoms of fatigue"},
        ]
        syn = {"well-being": ["health", "wellness"], "symptoms": ["issues", "problems"],
               "rest_days": ["rest days", "days off"], "medication_use": ["medication"]}
        anchors = ["rest_days", "medication_use", "well-being", "symptoms"]
        ranked = rank_pubmed_rows(rows, q, anchor_terms=anchors, concept_synonyms=syn)
        assert [r["pmid"] for r in ranked] == ["holiday"]
        first_rung = build_pubmed_query_ladder(q, anchor_terms=anchors, concept_synonyms=syn)[0]
        assert "well-being" in first_rung and "symptoms" in first_rung
        assert '"well being"' not in first_rung

    def test_oversized_item_does_not_hide_the_rest(self):
        from core.insight.evidence_layout import layout_evidence_with_report, clip_evidence_texts
        from core.insight.provenance import render_evidence_block
        from core.insight.types import EvidenceItem
        small = [EvidenceItem(doc_id=f"s{i}", collection="conversations", speaker="user",
                              date=f"2026-08-{10+i:02d}", text="short observation " * 5) for i in range(6)]
        huge = EvidenceItem(doc_id="huge", collection="obsidian_notes", date="2026-08-05",
                            text="x" * 20000)
        items = small[:2] + [huge] + small[2:]
        out, rep = layout_evidence_with_report(items, max_chars=3000)
        assert rep.personal_in_zone == 6 and rep.personal_tail == 1
        block = render_evidence_block(out, max_chars=3000)
        rendered = [l for l in block.splitlines() if l.startswith("[E")]
        assert len(rendered) == 6
        # And the renderer alone (no layout) also skips instead of stopping.
        block2 = render_evidence_block(items, max_chars=3000)
        assert len([l for l in block2.splitlines() if l.startswith("[E")]) == 6
        # Clip helper caps the text so it fits in the first place.
        clip_evidence_texts([huge], max_chars=560)
        assert len(huge.text) <= 561


def test_handler_source_reapplies_exclusions_after_pattern_merge():
    """The deliberation merge (`_pattern_evidence + evidence`) happens after
    the first exclusion pass; the exclusions must run again post-merge."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2].joinpath("gui", "handlers.py").read_text()
    merge = src.index("evidence = _pattern_evidence + evidence")
    tail = src[merge:merge + 900]
    assert "exclude_current_request_evidence(" in tail
    assert "exclude_assistant_directed_items(evidence)" in tail


class TestThirdRetestClosures:
    def test_search_pubmed_threads_anchors_to_ranker(self):
        """search_pubmed ranked each rung with the rung's own terms — anchors
        never reached production. Now the signature accepts them and the
        ranker call forwards them (source-level + signature check)."""
        import inspect
        from pathlib import Path
        from knowledge.pubmed_search import search_pubmed
        params = inspect.signature(search_pubmed).parameters
        assert "anchor_terms" in params and "concept_synonyms" in params
        src = Path(__file__).resolve().parents[2].joinpath("knowledge", "pubmed_search.py").read_text()
        call = src[src.index("return rank_pubmed_rows(parse_pubmed_articles"):][:300]
        assert "anchor_terms=anchor_terms" in call and "concept_synonyms=concept_synonyms" in call

    def test_coordinator_passes_anchor_kwargs_only_to_adapters_that_accept_them(self):
        import asyncio
        from core.insight.coordinator import LongitudinalDeliberationCoordinator
        seen = {}

        async def anchored(q, anchor_terms=None, concept_synonyms=None):
            seen["anchored"] = (anchor_terms, concept_synonyms)
            return []

        async def plain(q):
            seen["plain"] = True
            return []

        coord = LongitudinalDeliberationCoordinator.__new__(LongitudinalDeliberationCoordinator)
        coord.adapters = {"pubmed": anchored, "web": plain}
        coord.adapter_timeout_s = 5
        coord._research_anchor = {"anchor_terms": ["x"], "concept_synonyms": {"x": ["y"]}}
        asyncio.run(coord._call_adapter("pubmed", "q"))
        asyncio.run(coord._call_adapter("web", "q"))
        assert seen["anchored"] == (["x"], {"x": ["y"]}) and seen["plain"] is True

    def test_junk_subject_and_bare_quantity_facts(self):
        from core.insight.sweep import fact_triple_is_junk
        assert fact_triple_is_junk("and | is | reported as evidence") is True
        assert fact_triple_is_junk("means | is | plausibly similar at the confidence level of") is True
        assert fact_triple_is_junk("user | belief | 3 months") is True
        assert fact_triple_is_junk("user | medication_time | 10:15 AM") is False
        # A bare quantity as the WHOLE object is not a claim on a non-schedule
        # relation; schedule/duration relations keep it.
        assert fact_triple_is_junk("user | works_out | 3 days") is True
        assert fact_triple_is_junk("user | streak_days | 3 days") is False
        # Defer to THE deployed extractor predicate for ordinary objects.
        from memory.fact_extractor import _is_junk_object
        assert fact_triple_is_junk("user | medications_status | lost meds") is bool(
            _is_junk_object("lost meds", "medications_status"))

    def test_greeting_and_ack_turns_are_not_evidence(self):
        from core.insight.sweep import _user_side_is_greeting_or_ack
        assert _user_side_is_greeting_or_ack("User: Hey\nAssistant: Hey, how's it going?") is True
        assert _user_side_is_greeting_or_ack("User: ok thanks\nAssistant: Sure.") is True
        assert _user_side_is_greeting_or_ack("User: I took my meds at 10 and feel off\nAssistant: …") is False
