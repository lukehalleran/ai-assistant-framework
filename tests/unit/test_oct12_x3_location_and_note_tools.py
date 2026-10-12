"""Oct-12 X3: (1) an in-session "moved to Chicago" becomes the location for a
later local-conditions query; (2) the daily note is told which tools ACTUALLY
ran and a log-only post-check flags a tool the note names that ran in no turn.

class: BC-46 (stated fact never reaches its consumer), BC-04 (resolver shape
check rejects the extractor's own output), BC-16/BC-58 (note attributes work
to a tool that did not run).

Deployed functions throughout: FactExtractor.extract_facts -> UserProfile ->
LocationResolver; DailyNotesGenerator.generate_for_date with a fake model.
"""
import asyncio
import json
import os
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import utils.location_resolver as lr
from utils import daily_notes_generator as dng
from utils.daily_notes_generator import DailyNotesGenerator

# ----------------------------------------------------------------------------
# Item 1: in-session location
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor():
    from memory.fact_extractor import FactExtractor
    return FactExtractor(use_rebel=False)


def _lives_in(extractor, text):
    nodes = asyncio.run(extractor.extract_facts(text, ""))
    return [str(n.content) for n in nodes if "lives_in" in str(n.content)]


class TestExtractor:
    def test_elided_subject_move_clause_mints_current_city(self, extractor):
        got = _lives_in(extractor, "I used to live in Denver, moved to Chicago last year")
        assert got == ["user | lives_in | chicago"]

    def test_prior_city_alone_is_not_minted(self, extractor):
        assert _lives_in(extractor, "I used to live in Denver") == []

    def test_third_person_move_not_minted(self, extractor):
        assert _lives_in(extractor, "My sister, moved to Austin") == []


@pytest.fixture
def no_ip(monkeypatch):
    monkeypatch.setattr(lr, "LOCATION_ENABLED", True)
    monkeypatch.setattr(lr, "LOCATION_IP_LOOKUP_ENABLED", False)
    monkeypatch.setattr(lr, "LOCATION_OVERRIDE", "")


def _profile_with(tmp_path, value):
    from memory.user_profile import UserProfile
    path = str(tmp_path / "up.json")
    up = UserProfile(profile_path=path)
    up.add_fact(relation="lives_in", value=value, confidence=0.8)
    up.save()
    return path


class TestResolver:
    def test_extracted_city_flows_through_profile_to_resolver(self, extractor, tmp_path, no_ip):
        triple = _lives_in(extractor, "I used to live in Denver, moved to Chicago last year")[0]
        value = triple.split(" | ")[2]
        resolver = lr.LocationResolver(profile_path=_profile_with(tmp_path, value))
        got = resolver.get_location()
        assert got and got.lower() == "chicago"

    @pytest.mark.parametrize("junk", ["joke state", "a bad mood", "home", "your time",
                                      "empty house", "work", "here"])
    def test_junk_and_generic_values_still_rejected(self, tmp_path, no_ip, junk):
        resolver = lr.LocationResolver(profile_path=_profile_with(tmp_path, junk))
        assert resolver.get_location() is None

    def test_city_region_shape_unchanged(self, tmp_path, no_ip):
        resolver = lr.LocationResolver(profile_path=_profile_with(tmp_path, "Springfield, IL"))
        assert resolver.get_location() == "Springfield, IL"

    def test_stated_city_is_not_a_placeholder_unknown(self, tmp_path, no_ip):
        """B5's LOCATION_UNKNOWN path stays: no fact -> None; fact -> a place."""
        empty = lr.LocationResolver(profile_path=str(tmp_path / "none.json"))
        assert empty.get_location() is None
        resolver = lr.LocationResolver(profile_path=_profile_with(tmp_path, "chicago"))
        assert resolver.get_location() == "chicago"


# ----------------------------------------------------------------------------
# Item 2: daily-note tool attribution
# ----------------------------------------------------------------------------

DAY = date(2026, 10, 11)


def _write_turns(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for ts, query, tools in rows:
            f.write(json.dumps({
                "ts": ts, "query": query,
                "tool_calls": [{"tool": t, "status": "ok", "detail": ""} for t in tools],
            }) + "\n")


@pytest.fixture
def telemetry(tmp_path):
    p = str(tmp_path / "turn_records.jsonl")
    _write_turns(p, [
        ("2026-10-11T22:20:40-05:00", "do a web search on the news", ["search_memory", "web_search", "wolfram"]),
        ("2026-10-11T22:32:21-05:00", "What's a 20% tip on $86.40, split 3 ways", []),
        ("2026-10-10T09:00:00-05:00", "yesterday only", ["github"]),
    ])
    return p


def _convos():
    def c(h, m, q):
        return {"timestamp": datetime(2026, 10, 11, h, m, 0), "query": q, "response": "ok " * 5}
    return [c(22, 20, "do a web search on the news"),
            c(22, 32, "What's a 20% tip on $86.40, split 3 ways")]


_NOTE = ("## Summary\nA short synthetic day.\n\n## Main Quest: Tips\n"
         "- Wolfram computed the tip split three ways\n\n## Side Quests\nNone.\n\n"
         "## Life Events\n- **Work**: Not discussed today.\n\n## Emotional State\nCalm.\n\n"
         "## Key Decisions\nNone.\n\n## Knowledge Gained\nNone.\n\n## Open Threads\nNone.\n\n## Intensity: 2/10\n")


class _Model:
    def __init__(self, response):
        self.prompt = None
        self.response = response

    async def generate_once(self, prompt, **kw):
        self.prompt = prompt
        return self.response


def _gen(tmp_path, model):
    corpus = MagicMock()
    corpus.corpus = _convos()
    gen = DailyNotesGenerator(corpus_manager=corpus, model_manager=model,
                              vault_path=str(tmp_path / "vault"), user_profile=MagicMock(get_current_view=lambda: {}))
    gen.tag_generation_enabled = False
    gen._trigger_narrative_refresh = AsyncMock(return_value=None)
    return gen


class TestReceipts:
    def test_loader_filters_to_the_day_and_dedupes_tools(self, telemetry):
        rows = dng.load_turn_tool_receipts(DAY, telemetry)
        assert [r["tools"] for r in rows] == [["search_memory", "web_search", "wolfram"], []]

    def test_loader_unavailable_is_none_not_empty(self, tmp_path):
        assert dng.load_turn_tool_receipts(DAY, str(tmp_path / "missing.jsonl")) is None

    def test_registry_names_come_from_the_agentic_definitions(self):
        names = dng.known_tool_names()
        assert {"wolfram_alpha", "web_search", "search_memory", "wolfram"} <= set(names)
        assert "done_searching" not in names

    def test_block_lists_tools_per_exchange(self, telemetry):
        block = dng.build_tool_use_record(_convos(), dng.load_turn_tool_receipts(DAY, telemetry))
        assert "Exchange 1: search_memory, web_search, wolfram" in block
        assert "Exchange 2" not in block
        assert "only if it is listed" in block

    def test_unavailable_record_forbids_naming_tools(self):
        assert "Do NOT name any tool" in dng.build_tool_use_record(_convos(), None)


class TestPostCheck:
    def test_unrun_tool_flagged_only_when_absent_from_every_turn(self, telemetry):
        # Wolfram ran on exchange 1 that day, so a day-level check does not flag it...
        rows = dng.load_turn_tool_receipts(DAY, telemetry)
        assert dng.find_unrun_tool_mentions(_NOTE, rows) == []
        # ...but on a day where no turn ran it, the same sentence is flagged.
        none_ran = [{"ts": "x", "query": "q", "tools": ["web_search"]}]
        assert dng.find_unrun_tool_mentions(_NOTE, none_ran) == ["wolfram"]

    def test_registry_spellings_match(self):
        rows = [{"ts": "x", "query": "q", "tools": ["wolfram"]}]
        assert dng.find_unrun_tool_mentions("Wolfram Alpha answered.", rows) == []
        rows = [{"ts": "x", "query": "q", "tools": ["search_memory"]}]
        assert dng.find_unrun_tool_mentions("ran a memory search", rows) == []

    def test_unavailable_record_flags_nothing(self):
        assert dng.find_unrun_tool_mentions(_NOTE, None) == []


class TestGenerateForDate:
    def _run(self, tmp_path, monkeypatch, telemetry_path, response):
        import config.app_config as ac
        monkeypatch.setattr(ac, "TURN_TELEMETRY_PATH", telemetry_path)
        model = _Model(response)
        gen = _gen(tmp_path, model)
        result = asyncio.run(gen.generate_for_date(DAY, force=True))
        return model, result

    def test_receipts_reach_the_prompt(self, tmp_path, monkeypatch, telemetry):
        model, result = self._run(tmp_path, monkeypatch, telemetry, _NOTE)
        assert result.success
        assert "TOOLS THAT ACTUALLY RAN" in model.prompt
        assert "Exchange 1: search_memory, web_search, wolfram" in model.prompt
        assert "ONLY if the" in model.prompt

    def test_misbehaving_model_naming_unrun_tool_is_flagged_log_only(self, tmp_path, monkeypatch):
        p = str(tmp_path / "t.jsonl")
        _write_turns(p, [("2026-10-11T22:20:40-05:00", "do a web search on the news", ["web_search"]),
                         ("2026-10-11T22:32:21-05:00", "What's a 20% tip on $86.40, split 3 ways", [])])
        model, result = self._run(tmp_path, monkeypatch, p, _NOTE)
        assert result.success
        assert "wolfram" in result.unrun_tool_mentions
        # log-only: the note on disk is exactly what the model wrote
        assert "Wolfram computed the tip" in result.output_path.read_text(encoding="utf-8")

    def test_honest_note_not_flagged(self, tmp_path, monkeypatch, telemetry):
        honest = _NOTE.replace("Wolfram computed the tip split three ways", "Split the tip three ways")
        _, result = self._run(tmp_path, monkeypatch, telemetry, honest)
        assert result.unrun_tool_mentions == []
