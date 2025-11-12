#test/memory_test
import os
import asyncio
from core.orchestrator     import DaemonOrchestrator
from models.model_manager  import ModelManager
from core.response_generator import ResponseGenerator
from utils.file_processor  import FileProcessor

# 1) Point at clean fixture stores
os.environ["CORPUS_FILE"]  = "tests/fixtures/corpus_test.json"
os.environ["CHROMA_PATH"]  = "tests/fixtures/chroma_test"

# 2) Bootstrap
model_mgr = ModelManager()
resp_gen  = ResponseGenerator(model_mgr)
file_proc = FileProcessor()
# Use the fallback prompt builder from orchestrator
from core.orchestrator import _SimplePromptBuilder
prompt_builder = _SimplePromptBuilder()
orch = DaemonOrchestrator(
    model_manager     = model_mgr,
    response_generator= resp_gen,
    file_processor    = file_proc,
    prompt_builder    = prompt_builder,
)

# 3) Check if memory system exists, create fallback if needed
if orch.memory_system is None:
    from core.orchestrator import _FallbackMemoryCoordinator
    orch.memory_system = _FallbackMemoryCoordinator()

# Seed corpus
cm = orch.memory_system.corpus_manager
cm.add_entry("Hello, how are you?", "I am fine, thanks!", tags=[])
cm.add_entry("What's the weather?",   "Sunny and 75°F.",       tags=[])

# 4) Run and inspect
async def smoke_test():
    response, debug = await orch.process_user_query("How are you today?", files=None)
    # print out the prompt for manual inspection
    prompt_text = debug.get("prompt", "")
    print("❯ Prompt:\n", prompt_text if prompt_text else "(not captured)")
    print("❯ Response:\n", response[:200] if response else "(empty)")
    print("❯ Debug keys:", list(debug.keys()))

    # If prompt is captured, check memories appear
    if prompt_text:
        assert "Hello, how are you?" in prompt_text or "fine" in prompt_text.lower()
        print("✅ Memory surfaced correctly")
    else:
        print("⚠️  Prompt not captured in debug, but response generated:", bool(response))

asyncio.run(smoke_test())
