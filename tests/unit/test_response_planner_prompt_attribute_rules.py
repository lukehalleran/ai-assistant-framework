"""Behavioural sibling for ``test_prompt_carries_entity_attribute_rule`` in
tests/unit/test_graph_memory_species_suppression_and_evidence.py (BC-63: it
grepped ``inspect.getsource(ResponsePlanner)`` for the rule text, which would
also pass if the text sat in a dead branch or a comment).

Incident date: 2026-09-03 (a cat was called a dog; the planner prompt must
tell the planner not to infer species/gender/role from the Topics label).
Bug classes: BC-63, BC-62. THE deployed ``create_plan`` is driven with a
recording fake model manager; the rule must be in the prompt actually SENT,
on every call shape (with and without a previous exchange, wrapped query).
"""
import pytest

from core.response_planner import ResponsePlanner

RULE_PARTS = (
    "Never assign or infer a species, gender, role, relationship",
    "wins over the Topics label",
)


class _MM:
    def __init__(self):
        self.prompts = []

    async def generate_once(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return ('{"key_points": ["a"], "tone": "warm", "avoid": ["b"], '
                '"strategy": "c"}')


class _Ctx:
    intent = None
    tone_level = None
    topics = ["Pets"]
    thread_context = None

    def __init__(self, last_exchange=None):
        self.last_exchange = last_exchange


async def _sent_prompt(query, last_exchange=None):
    mm = _MM()
    plan = await ResponsePlanner(mm).create_plan(query, _Ctx(last_exchange))
    assert plan is not None
    return " ".join(mm.prompts[0].split())


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "Biscuit is asleep on the couch",
    "    Biscuit is asleep\n    on the couch\n   ",
])
async def test_attribute_rule_is_in_the_sent_prompt(query):
    prompt = await _sent_prompt(query)
    for part in RULE_PARTS:
        assert part in prompt


@pytest.mark.asyncio
async def test_rule_survives_a_previous_exchange_block():
    prompt = await _sent_prompt(
        "Biscuit is asleep on the couch",
        last_exchange={"query": "my cat is named Biscuit", "response": "Nice."})
    assert "Previous exchange" in prompt
    for part in RULE_PARTS:
        assert part in prompt


@pytest.mark.asyncio
async def test_query_stays_authoritative_over_the_label():
    prompt = await _sent_prompt("Biscuit is asleep on the couch")
    assert "Query: Biscuit is asleep on the couch" in prompt
    assert prompt.index("Query:") < prompt.index("Topics: Pets")
