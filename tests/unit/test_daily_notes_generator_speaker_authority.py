"""B4 item 7: daily-note + narrative prompts label speakers and carry the
speaker-authority rules (user statements are never 'unconfirmed'; assistant
self-corrections are not evidence; no tool named unless it ran). class: BC-46, BC-75, BC-91.

Original batch label: oct12_b4 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

from memory.memory_consolidator import MemoryConsolidator
from utils.daily_notes_generator import DailyNotesGenerator

# A misbehaving model: the note it would write is exactly the beta defect.
_BAD_NOTE = (
    "## Summary\nA scattered session; the user mentioned a job at Northwind but it was test data.\n\n"
    "## Main Quest: Memory checks\n- Actual workplace unknown, treating it as unverified\n"
    "- Wolfram computed the tip\n\n## Side Quests\nNone today.\n\n## Life Events\n- x\n\n"
    "## Emotional State\nTired.\n\n## Key Decisions\nNone.\n\n## Knowledge Gained\nNone.\n\n"
    "## Open Threads\nNone.\n\n## Intensity: 3/10\n"
)


class _MM:
    def __init__(self):
        self.prompts = []

    async def generate_once(self, prompt, **kw):
        self.prompts.append(prompt)
        return _BAD_NOTE


def _gen(tmp_path, convos):
    corpus = MagicMock()
    corpus.corpus = convos
    gen = DailyNotesGenerator(corpus_manager=corpus, model_manager=_MM(),
                              vault_path=str(tmp_path / "vault"), user_profile=MagicMock(
                                  get_current_view=lambda: {}))
    gen.tag_generation_enabled = False
    gen._trigger_narrative_refresh = AsyncMock(return_value=None)
    return gen


def test_daily_note_prompt_labels_speakers_and_carries_authority_rules(tmp_path):
    d = date(2026, 10, 11)
    convos = [
        {"timestamp": datetime(2026, 10, 11, 22, 25), "query": "I just started a new job at Northwind as a data analyst",
         "response": "Congrats!"},
        {"timestamp": datetime(2026, 10, 11, 22, 34), "query": "where do i work?",
         "response": "Northwind was my own guess, you never told me."},
    ]
    gen = _gen(tmp_path, convos)
    gen._get_conversations_for_date = lambda target: convos
    res = asyncio.run(gen.generate_for_date(d, force=True))
    prompt = gen.model_manager.prompts[0]
    assert "User: I just started a new job at Northwind as a data analyst" in prompt
    assert "Daemon: Northwind was my own guess" in prompt
    assert "SPEAKER AUTHORITY" in prompt
    assert "NEVER describe a user statement as unconfirmed" in prompt
    assert "not evidence against the user" in prompt
    assert "ONLY if the source text explicitly says that tool ran" in prompt
    assert res is not None


def test_narrative_prompt_carries_the_same_rules_and_still_formats():
    p = MemoryConsolidator.NARRATIVE_SYNTHESIS_PROMPT
    assert "SPEAKER AUTHORITY" in p and "NEVER describe a user statement as unconfirmed" in p
    out = p.format(today="x", monthly_summaries="", weekly_summaries="", daily_notes="",
                   corpus_summaries="", authoritative_status_facts="", streak_ledger="")
    assert "SPEAKER AUTHORITY" in out
