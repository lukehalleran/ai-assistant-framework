"""B4 item 6: a paste-sized turn must not end the shutdown LLM fact pass's walk
and hide the user's older one-line facts (class: BC-46, BC-75).

Original batch label: oct12_b4 (renamed 2026-10-16; old path in tests/RENAMES.tsv)."""
import asyncio
import json

from memory.llm_fact_extractor import LLMFactExtractor

FACT = "I just started a new job at Northwind as a data analyst"
MOVE = "I used to live in Denver, moved to Chicago last year"


class _MM:
    def __init__(self, reply="[]"):
        self.prompts = []
        self.reply = reply

    async def generate_once(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.reply


def _pairs():
    huge = "x " * 57_000  # 114K chars, the beta-session paste
    rows = [FACT, MOVE, "I have a dentist appointment Thursday at 3.", huge,
            "What's a 20% tip on $86.40, split 3 ways?", "ough day, kinda tired of everything."]
    return [{"query": q, "response": "r", "turn_id": f"2026-10-11T22:{25 + i:02d}:00"}
            for i, q in enumerate(rows)]


def test_older_user_facts_survive_an_oversized_paste_turn():
    mm = _MM()
    ex = LLMFactExtractor(mm, max_input_chars=9000)
    prompt = ex._build_prompt(_pairs())
    assert FACT in prompt and MOVE in prompt
    assert "ough day" in prompt  # newer turns still present
    assert "x x x x x x x x" not in prompt  # the paste itself is skipped


def test_extractor_end_to_end_keeps_the_job_triple():
    reply = json.dumps([{"subject": "user", "relation": "employer", "object": "Northwind",
                         "category": "career", "confidence": 0.9, "stance": "objective"}])
    mm = _MM(reply)
    ex = LLMFactExtractor(mm, max_input_chars=9000)
    triples = asyncio.run(ex.extract_triples(_pairs()))
    assert any(FACT in p for p in mm.prompts)
    assert any(t.get("object") == "Northwind" for t in triples), triples
