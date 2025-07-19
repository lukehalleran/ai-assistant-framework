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
orch = DaemonOrchestrator(
    model_manager     = model_mgr,
    response_generator= resp_gen,
    file_processor    = file_proc,
)

# 3) Seed corpus
cm = orch.memory_system.corpus_manager
cm.add_entry("Hello, how are you?", "I am fine, thanks!", tags=[])
cm.add_entry("What’s the weather?",   "Sunny and 75°F.",       tags=[])

# 4) Run and inspect
async def smoke_test():
    response, debug = await orch.process_user_query("How are you today?", files=None)
    # print out the prompt for manual inspection
    print("❯ Prompt:\n", debug.get("prompt", "(not captured)"))
    # basic assertions
    assert "Hello, how are you?" in debug["prompt"]
    assert "Sunny and 75°F."      in debug["prompt"]
    print("✅ Memory surfaced correctly")

asyncio.run(smoke_test())
