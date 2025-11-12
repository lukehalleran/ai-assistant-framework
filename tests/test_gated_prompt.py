#tests/test_gated_prompt.py
"""Test the gated prompt builder's multi-stage filtering."""
import asyncio
from models.model_manager import ModelManager
from processing.gate_system import GatedPromptBuilder
from core.prompt import PromptBuilder

# Set up dummy memory data
dummy_memories = [
    {"content": "The mitochondria is the powerhouse of the cell.", "metadata": {"source": "test"}},
    {"content": "Luke's project uses cosine similarity and reranking for filtering memories.", "metadata": {"source": "test"}},
    {"content": "Bananas are yellow and contain potassium.", "metadata": {"source": "test"}},
]

async def test_gated_builder():
    """Test that GatedPromptBuilder can build a prompt with memories."""
    model_manager = ModelManager()

    # Create base and gated builders
    base_pb = PromptBuilder(model_manager)
    gated_pb = GatedPromptBuilder(base_pb, model_manager)

    # Build a gated prompt
    query = "How do cosine similarity gates work in memory filtering?"

    print("Building gated prompt...\n")
    prompt = await gated_pb.build_gated_prompt(
        user_input=query,
        memories=dummy_memories,
        summaries=[],
        dreams=[],
        wiki_snippet="",
        semantic_chunks=None,
        semantic_memory_results=None,
        time_context=None,
        recent_conversations=None,
    )

    print("Gated Prompt Built Successfully!")
    print(f"Prompt length: {len(prompt)} chars")

    # Basic assertions
    assert len(prompt) > 0, "Prompt should not be empty"
    assert query in prompt, "Query should appear in prompt"

    print("\n✅ Test passed!")

if __name__ == "__main__":
    asyncio.run(test_gated_builder())
