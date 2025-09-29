#tests/test_gated_prompt.py
import asyncio
from core.prompt import PromptBuilder
from core.model_manager import ModelManager
from core.file_processor import FileProcessor
from core.response_generator import ResponseGenerator
from core.orchestrator import DaemonOrchestrator
from memory.multi_collection_store import MultiCollectionChromaStore
from core.prompt_builder import UnifiedHierarchicalPromptBuilder
from core.llm_gate_module import GatedPromptBuilder

# Set up dummy memory data
dummy_memories = [
    {"content": "The mitochondria is the powerhouse of the cell."},
    {"content": "Luke’s project uses cosine similarity and reranking for filtering memories."},
    {"content": "Bananas are yellow and contain potassium."},
]

async def run_test():
    model_manager = ModelManager()
    model_manager.load_openai_model("gpt-4-turbo", "gpt-4-turbo")
    model_manager.switch_model("gpt-4-turbo")

    base_pb = PromptBuilder(model_manager)
    gated_pb = GatedPromptBuilder(base_pb, model_manager)

    chroma_store = MultiCollectionChromaStore(persist_directory="daemon_memory")

    prompt_builder = UnifiedHierarchicalPromptBuilder(
        prompt_builder=gated_pb,
        model_manager=model_manager,
        chroma_store=chroma_store
    )

    file_processor = FileProcessor()
    response_generator = ResponseGenerator(model_manager)

    orchestrator = DaemonOrchestrator(
        model_manager=model_manager,
        response_generator=response_generator,
        file_processor=file_processor,
        prompt_builder=prompt_builder
    )

    query = "How do cosine similarity gates work in memory filtering?"

    print("Sending query to orchestrator...\n")
    reply = await orchestrator.generate_reply(query, injected_context={"memories": dummy_memories})

    print("📤 Final Response:\n")
    print(reply)

asyncio.run(run_test())
